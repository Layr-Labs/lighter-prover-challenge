// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Fail-closed runtime orchestration for the phantom spot-session splice.
//!
//! Discovery and witness construction remain off-side. The source block is
//! changed only after the materialized pair, its boundary roots, and the
//! rechunker's independently-derived lane arithmetic all agree.

use std::cmp::Ordering;

use circuit::block::Block;
use circuit::tx::Tx;
use circuit::types::config::F;
use circuit::types::constants::{FEE_ACCOUNT_ID, TREASURY_ACCOUNT_INDEX, TX_HEAVY, TX_LIGHT};
use plonky2::hash::hash_types::HashOut;

use crate::api::PROVER_THREAD_STACK_BYTES;
use crate::phantom_materialize::{MaterializedPhantomSpotPair, materialize_scanned_candidate};
use crate::phantom_rechunk::{
    HEAVY_TXS_PER_GROUP, InclusiveInterval, LIGHT_TXS_PER_GROUP, PreparedSplice, ReplacementPair,
    SpliceReport, prepare_verified_pair,
};
use crate::phantom_spot::{Candidate, scan};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Unchanged,
    Applied(SpliceReport),
}

/// Try every scanner-certified candidate in deterministic score order.
///
/// A dedicated stack is used because a `Tx` contains several large fixed-size
/// witness arrays and materialization owns two of them at once. Thread-spawn
/// failure and every candidate-local validation error are ordinary fallbacks.
/// No diagnostic is emitted on the ranked path.
pub fn try_apply_best(block: &mut Block<F>, state_metadata_hash: HashOut<F>) -> Outcome {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("phantom-splice".into())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn_scoped(scope, || try_apply_best_inner(block, state_metadata_hash));
        match worker {
            Ok(worker) => worker.join().unwrap_or(Outcome::Unchanged),
            Err(_) => Outcome::Unchanged,
        }
    })
}

fn try_apply_best_inner(block: &mut Block<F>, state_metadata_hash: HashOut<F>) -> Outcome {
    // This is the sole discovery pass. Materialization consumes the resulting
    // candidate and revalidates its donors without invoking the scanner again.
    let mut scan_report = scan(block);
    let Some((total_heavy, total_light)) = active_lane_counts(block) else {
        return Outcome::Unchanged;
    };
    if total_heavy.checked_add(total_light) != Some(scan_report.active_tx_count) {
        return Outcome::Unchanged;
    }

    rank_candidates(&mut scan_report.candidates, total_heavy);
    for candidate in &scan_report.candidates {
        let Some(prepared) = prepare_candidate(
            block,
            candidate,
            state_metadata_hash,
            total_heavy,
            total_light,
        ) else {
            continue;
        };
        let report = prepared.commit(block);
        return Outcome::Applied(report);
    }
    Outcome::Unchanged
}

fn prepare_candidate(
    block: &Block<F>,
    candidate: &Candidate,
    state_metadata_hash: HashOut<F>,
    total_heavy: usize,
    total_light: usize,
) -> Option<PreparedSplice> {
    let removed_u64 = candidate
        .end_tx_index
        .checked_sub(candidate.start_tx_index)?;
    let removed_tx_count = usize::try_from(removed_u64).ok()?;
    if removed_tx_count < 3 || removed_tx_count != candidate.replaced_tx_count {
        return None;
    }
    if candidate
        .replaced_heavy_count
        .checked_add(candidate.replaced_light_count)
        != Some(removed_tx_count)
    {
        return None;
    }
    let heavy_after = total_heavy
        .checked_sub(candidate.replaced_heavy_count)?
        .checked_add(1)?;
    let light_after = total_light
        .checked_sub(candidate.replaced_light_count)?
        .checked_add(1)?;
    let expected_shift = removed_u64.checked_sub(2)?;
    let last_removed = candidate.end_tx_index.checked_sub(1)?;

    let left = unique_active_tx(block, candidate.start_tx_index)?;
    let right = unique_active_tx(block, candidate.end_tx_index)?;
    let materialized = materialize_scanned_candidate(block, candidate, state_metadata_hash).ok()?;
    if !materialized_matches_boundaries(&materialized, candidate, left, right, expected_shift) {
        return None;
    }

    let pair = ReplacementPair::new(materialized.light_insert, materialized.heavy_fill);
    let prepared = prepare_verified_pair(
        block,
        InclusiveInterval::new(candidate.start_tx_index, last_removed),
        pair,
    )
    .ok()?;
    let report = prepared.report();
    if report.removed_tx_count != removed_tx_count
        || report.removed_heavy_count != candidate.replaced_heavy_count
        || report.removed_light_count != candidate.replaced_light_count
        || report.suffix_index_shift != expected_shift
        || report.saved_group_count != candidate.saved_lane_proof_count
        || report.groups_before.heavy != lane_groups(total_heavy, HEAVY_TXS_PER_GROUP)
        || report.groups_before.light != lane_groups(total_light, LIGHT_TXS_PER_GROUP)
        || report.groups_after.heavy != lane_groups(heavy_after, HEAVY_TXS_PER_GROUP)
        || report.groups_after.light != lane_groups(light_after, LIGHT_TXS_PER_GROUP)
    {
        return None;
    }
    Some(prepared)
}

fn materialized_matches_boundaries(
    pair: &MaterializedPhantomSpotPair,
    candidate: &Candidate,
    left: &Tx<F>,
    right: &Tx<F>,
    expected_shift: u64,
) -> bool {
    if pair.first_removed_tx_index != candidate.start_tx_index
        || pair.right_boundary_tx_index != candidate.end_tx_index
        || pair.following_tx_index_shift != expected_shift
        || pair.light_insert.tx_index != candidate.start_tx_index
        || pair.heavy_fill.tx_index != candidate.start_tx_index.checked_add(1).unwrap_or(u64::MAX)
    {
        return false;
    }

    let expected_assets = [candidate.base_asset_index, candidate.quote_asset_index];
    if pair.light_insert.asset_indices != expected_assets
        || pair.heavy_fill.asset_indices != expected_assets
        || pair.light_insert.accounts_before[0].account_index != candidate.seller_account_index
        || pair.heavy_fill.accounts_before[0].account_index != candidate.buyer_account_index
        || pair.heavy_fill.accounts_before[1].account_index != candidate.seller_account_index
        || pair.heavy_fill.accounts_before[FEE_ACCOUNT_ID].account_index
            != TREASURY_ACCOUNT_INDEX as i64
    {
        return false;
    }

    // The light witness must start at the original left boundary, the heavy
    // witness at the materializer's authenticated intermediate boundary, and
    // the heavy result must land on the original right boundary.
    same_old_roots(&pair.light_insert, left)
        && pair.heavy_fill.old_validium_root == pair.after_light_validium_root
        && pair.heavy_fill.old_state_root == pair.after_light_state_root
        && pair.heavy_fill.old_account_delta_tree_root == pair.after_light_delta_root
        && pair.final_validium_root == right.old_validium_root
        && pair.final_state_root == right.old_state_root
        && pair.final_delta_root == right.old_account_delta_tree_root
}

fn same_old_roots(left: &Tx<F>, right: &Tx<F>) -> bool {
    left.old_account_tree_root == right.old_account_tree_root
        && left.old_account_pub_data_tree_root == right.old_account_pub_data_tree_root
        && left.old_account_delta_tree_root == right.old_account_delta_tree_root
        && left.old_market_details_tree_root == right.old_market_details_tree_root
        && left.old_market_tree_root == right.old_market_tree_root
        && left.old_validium_root == right.old_validium_root
        && left.old_state_root == right.old_state_root
}

fn unique_active_tx(block: &Block<F>, tx_index: u64) -> Option<&Tx<F>> {
    let mut matches = block
        .tx_chunks
        .iter()
        .flatten()
        .filter(|tx| !tx.is_empty() && tx.tx_index == tx_index);
    let tx = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(tx.as_ref())
}

fn active_lane_counts(block: &Block<F>) -> Option<(usize, usize)> {
    let mut heavy = 0_usize;
    let mut light = 0_usize;
    for tx in block.tx_chunks.iter().flatten().filter(|tx| !tx.is_empty()) {
        match tx.tx_circuit_type {
            TX_HEAVY => heavy = heavy.checked_add(1)?,
            TX_LIGHT => light = light.checked_add(1)?,
            _ => return None,
        }
    }
    Some((heavy, light))
}

fn rank_candidates(candidates: &mut [Candidate], total_heavy: usize) {
    candidates.sort_by(|left, right| {
        right
            .saved_lane_proof_count
            .cmp(&left.saved_lane_proof_count)
            .then_with(|| {
                heavy_lane_group_saving(total_heavy, right)
                    .cmp(&heavy_lane_group_saving(total_heavy, left))
            })
            .then_with(|| right.replaced_tx_count.cmp(&left.replaced_tx_count))
            .then_with(|| left.start_tx_index.cmp(&right.start_tx_index))
            .then_with(|| left.end_tx_index.cmp(&right.end_tx_index))
            .then_with(|| deterministic_candidate_tail(left, right))
    });
}

fn deterministic_candidate_tail(left: &Candidate, right: &Candidate) -> Ordering {
    left.replaced_heavy_count
        .cmp(&right.replaced_heavy_count)
        .then_with(|| left.replaced_light_count.cmp(&right.replaced_light_count))
        .then_with(|| left.seller_account_index.cmp(&right.seller_account_index))
        .then_with(|| left.buyer_account_index.cmp(&right.buyer_account_index))
        .then_with(|| left.base_asset_index.cmp(&right.base_asset_index))
        .then_with(|| left.quote_asset_index.cmp(&right.quote_asset_index))
        .then_with(|| left.base_amount.cmp(&right.base_amount))
        .then_with(|| left.price.cmp(&right.price))
}

fn heavy_lane_group_saving(total_heavy: usize, candidate: &Candidate) -> usize {
    let Some(after) = total_heavy
        .checked_sub(candidate.replaced_heavy_count)
        .and_then(|count| count.checked_add(1))
    else {
        return 0;
    };
    lane_groups(total_heavy, HEAVY_TXS_PER_GROUP)
        .saturating_sub(lane_groups(after, HEAVY_TXS_PER_GROUP))
}

fn lane_groups(tx_count: usize, group_size: usize) -> usize {
    tx_count.div_ceil(group_size).max(1)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn candidate(
        start: u64,
        end: u64,
        replaced_heavy: usize,
        replaced_light: usize,
        saved: usize,
    ) -> Candidate {
        Candidate {
            start_tx_index: start,
            end_tx_index: end,
            replaced_tx_count: replaced_heavy + replaced_light,
            replaced_light_count: replaced_light,
            replaced_heavy_count: replaced_heavy,
            saved_lane_proof_count: saved,
            seller_account_index: 11,
            buyer_account_index: 12,
            base_asset_index: 1,
            quote_asset_index: 2,
            base_amount: 3,
            price: 4,
            maker_account_orders_path_at_phantom_key: true,
            maker_api_path_from_padding: true,
            taker_api_path_from_padding: true,
            fee_paths_natively_verified: true,
            nil_market_path_from_final_padding: true,
            nil_market_details_path_from_final_padding: true,
        }
    }

    #[test]
    fn ranking_prefers_total_then_heavy_saving_then_length_then_position() {
        let mut candidates = vec![
            candidate(9, 13, 3, 1, 1),
            candidate(4, 8, 2, 2, 2),
            candidate(5, 10, 3, 2, 1),
            candidate(1, 5, 3, 1, 1),
        ];
        rank_candidates(&mut candidates, 10);
        let intervals = candidates
            .iter()
            .map(|candidate| (candidate.start_tx_index, candidate.end_tx_index))
            .collect::<Vec<_>>();
        assert_eq!(intervals, vec![(4, 8), (5, 10), (1, 5), (9, 13)]);
    }

    #[test]
    fn public_fixture_fallback_preserves_every_arc_identity() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let mut block =
                    Block::<F>::from_json(include_bytes!("../bench_test.json"), 4, 10)
                    .expect("public fixture must parse");
                let before = block
                    .tx_chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|tx| Arc::as_ptr(tx) as usize)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                assert_eq!(
                    try_apply_best(&mut block, HashOut::ZERO),
                    Outcome::Unchanged
                );
                let after = block
                    .tx_chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|tx| Arc::as_ptr(tx) as usize)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(after, before);
            })
            .expect("runtime test thread must start")
            .join()
            .expect("runtime test thread must finish");
    }
}
