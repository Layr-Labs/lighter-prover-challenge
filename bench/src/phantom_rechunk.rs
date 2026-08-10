// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Transactional replacement and rechunking for an already-verified phantom pair.
//!
//! This module deliberately knows nothing about how the pair was discovered or
//! materialized. It validates only the structural contract needed to splice two
//! `INTERNAL_CLAIM_ORDER` transactions into an inclusive execution interval.
//! Every fallible operation completes against off-side data before `tx_chunks`
//! is replaced, so an error cannot partially mutate the input block.

use std::fmt;
use std::sync::Arc;

use circuit::block::Block;
use circuit::tx::Tx;
use circuit::types::config::F;
use circuit::types::constants::{TX_HEAVY, TX_LIGHT, TX_TYPE_EMPTY, TX_TYPE_INTERNAL_CLAIM_ORDER};
use plonky2::field::types::{Field, PrimeField64};

pub const HEAVY_TXS_PER_GROUP: usize = 4;
pub const LIGHT_TXS_PER_GROUP: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InclusiveInterval {
    pub first_tx_index: u64,
    pub last_tx_index: u64,
}

impl InclusiveInterval {
    pub const fn new(first_tx_index: u64, last_tx_index: u64) -> Self {
        Self {
            first_tx_index,
            last_tx_index,
        }
    }
}

/// Neutral handoff from a verifier/materializer. The execution order is fixed:
/// the light insertion transaction precedes the heavy fill transaction.
#[derive(Clone, Debug)]
pub struct ReplacementPair {
    pub light_insert: Tx<F>,
    pub heavy_fill: Tx<F>,
}

impl ReplacementPair {
    pub fn new(light_insert: Tx<F>, heavy_fill: Tx<F>) -> Self {
        Self {
            light_insert,
            heavy_fill,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaneGroupCount {
    pub heavy: usize,
    pub light: usize,
}

impl LaneGroupCount {
    pub const fn total(self) -> usize {
        self.heavy + self.light
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpliceReport {
    pub removed_tx_count: usize,
    pub removed_heavy_count: usize,
    pub removed_light_count: usize,
    pub suffix_index_shift: u64,
    pub groups_before: LaneGroupCount,
    pub groups_after: LaneGroupCount,
    pub saved_group_count: usize,
}

/// A complete off-side rechunk plan. Constructing this value never changes the
/// source block; committing it is the exploit path's sole mutation point.
#[derive(Clone, Debug)]
pub struct PreparedSplice {
    replacement_chunks: Vec<Vec<Arc<Tx<F>>>>,
    report: SpliceReport,
}

impl PreparedSplice {
    pub const fn report(&self) -> SpliceReport {
        self.report
    }

    pub fn commit(self, block: &mut Block<F>) -> SpliceReport {
        block.tx_chunks = self.replacement_chunks;
        self.report
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpliceError {
    MissingPaddingTemplate,
    NoActiveTransactions,
    InvalidActiveCircuitType {
        tx_index: u64,
        circuit_type: u8,
    },
    NonContiguousActiveIndex {
        expected: u64,
        found: u64,
    },
    ReversedInterval {
        first: u64,
        last: u64,
    },
    IntervalOutOfBounds {
        first: u64,
        last: u64,
        active_tx_count: usize,
    },
    IntervalTooShort {
        removed_tx_count: usize,
    },
    InvalidPairIndex {
        member: PairMember,
        expected: u64,
        found: u64,
    },
    InvalidPairTxType {
        member: PairMember,
        found: u8,
    },
    InvalidPairCircuitType {
        member: PairMember,
        expected: u8,
        found: u8,
    },
    CountArithmeticOverflow,
    NoExactLaneGroupSaving {
        before: LaneGroupCount,
        after: LaneGroupCount,
    },
    RechunkInvariant {
        expected: LaneGroupCount,
        actual: LaneGroupCount,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairMember {
    LightInsert,
    HeavyFill,
}

impl fmt::Display for SpliceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPaddingTemplate => {
                write!(formatter, "block has no empty padding template")
            }
            Self::NoActiveTransactions => write!(formatter, "block has no active transactions"),
            Self::InvalidActiveCircuitType {
                tx_index,
                circuit_type,
            } => write!(
                formatter,
                "active transaction {tx_index} has invalid circuit type {circuit_type}"
            ),
            Self::NonContiguousActiveIndex { expected, found } => write!(
                formatter,
                "active execution indices are not contiguous: expected {expected}, found {found}"
            ),
            Self::ReversedInterval { first, last } => {
                write!(
                    formatter,
                    "inclusive interval is reversed: {first}..={last}"
                )
            }
            Self::IntervalOutOfBounds {
                first,
                last,
                active_tx_count,
            } => write!(
                formatter,
                "inclusive interval {first}..={last} is outside {active_tx_count} active transactions"
            ),
            Self::IntervalTooShort { removed_tx_count } => write!(
                formatter,
                "replacement pair cannot replace only {removed_tx_count} transaction(s)"
            ),
            Self::InvalidPairIndex {
                member,
                expected,
                found,
            } => write!(
                formatter,
                "{member:?} index mismatch: expected {expected}, found {found}"
            ),
            Self::InvalidPairTxType { member, found } => write!(
                formatter,
                "{member:?} must be INTERNAL_CLAIM_ORDER, found tx type {found}"
            ),
            Self::InvalidPairCircuitType {
                member,
                expected,
                found,
            } => write!(
                formatter,
                "{member:?} circuit type mismatch: expected {expected}, found {found}"
            ),
            Self::CountArithmeticOverflow => {
                write!(
                    formatter,
                    "transaction count or index arithmetic overflowed"
                )
            }
            Self::NoExactLaneGroupSaving { before, after } => write!(
                formatter,
                "replacement does not save a lane group: {before:?} -> {after:?}"
            ),
            Self::RechunkInvariant { expected, actual } => write!(
                formatter,
                "rechunked lane groups differ from the precomputed plan: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for SpliceError {}

/// Replace an inclusive active-transaction interval with a verified two-step
/// phantom pair. The block is mutated exactly once, after every validation and
/// the complete replacement chunk vector have succeeded.
#[cfg(test)]
pub fn splice_verified_pair(
    block: &mut Block<F>,
    interval: InclusiveInterval,
    pair: ReplacementPair,
) -> Result<SpliceReport, SpliceError> {
    Ok(prepare_verified_pair(block, interval, pair)?.commit(block))
}

/// Validate and construct a replacement without touching `block`. Callers may
/// cross-check the returned report against stronger witness-level invariants
/// before consuming [`PreparedSplice::commit`].
pub fn prepare_verified_pair(
    block: &Block<F>,
    interval: InclusiveInterval,
    pair: ReplacementPair,
) -> Result<PreparedSplice, SpliceError> {
    let (active, padding_template) = flatten_active_in_execution_order(block)?;
    let active_count = active.len();
    if active_count == 0 {
        return Err(SpliceError::NoActiveTransactions);
    }

    let (first, last) = (interval.first_tx_index, interval.last_tx_index);
    if first > last {
        return Err(SpliceError::ReversedInterval { first, last });
    }
    let first_position = usize::try_from(first).map_err(|_| SpliceError::IntervalOutOfBounds {
        first,
        last,
        active_tx_count: active_count,
    })?;
    let last_position = usize::try_from(last).map_err(|_| SpliceError::IntervalOutOfBounds {
        first,
        last,
        active_tx_count: active_count,
    })?;
    if first_position >= active_count || last_position >= active_count {
        return Err(SpliceError::IntervalOutOfBounds {
            first,
            last,
            active_tx_count: active_count,
        });
    }
    let removed_tx_count = last_position
        .checked_sub(first_position)
        .and_then(|distance| distance.checked_add(1))
        .ok_or(SpliceError::CountArithmeticOverflow)?;
    if removed_tx_count < 2 {
        return Err(SpliceError::IntervalTooShort { removed_tx_count });
    }

    validate_pair(&pair, first)?;

    let (heavy_before, light_before) = lane_tx_counts(&active);
    let (removed_heavy, removed_light) = lane_tx_counts(&active[first_position..=last_position]);
    let heavy_after = heavy_before
        .checked_sub(removed_heavy)
        .and_then(|count| count.checked_add(1))
        .ok_or(SpliceError::CountArithmeticOverflow)?;
    let light_after = light_before
        .checked_sub(removed_light)
        .and_then(|count| count.checked_add(1))
        .ok_or(SpliceError::CountArithmeticOverflow)?;
    let groups_before = lane_group_count(heavy_before, light_before);
    let groups_after = lane_group_count(heavy_after, light_after);
    if groups_after.total() >= groups_before.total() {
        return Err(SpliceError::NoExactLaneGroupSaving {
            before: groups_before,
            after: groups_after,
        });
    }

    let suffix_index_shift =
        u64::try_from(removed_tx_count - 2).map_err(|_| SpliceError::CountArithmeticOverflow)?;
    let new_active_count = active_count
        .checked_sub(removed_tx_count)
        .and_then(|count| count.checked_add(2))
        .ok_or(SpliceError::CountArithmeticOverflow)?;
    let mut replacement = Vec::with_capacity(new_active_count);
    replacement.extend(
        active[..first_position]
            .iter()
            .map(|tx| tx.as_ref().clone()),
    );
    replacement.push(pair.light_insert);
    replacement.push(pair.heavy_fill);
    for tx in &active[last_position + 1..] {
        let mut shifted = tx.as_ref().clone();
        shifted.tx_index = shifted
            .tx_index
            .checked_sub(suffix_index_shift)
            .ok_or(SpliceError::CountArithmeticOverflow)?;
        replacement.push(shifted);
    }
    debug_assert_eq!(replacement.len(), new_active_count);
    for (expected, tx) in (0_u64..).zip(&replacement) {
        if tx.tx_index != expected {
            return Err(SpliceError::NonContiguousActiveIndex {
                expected,
                found: tx.tx_index,
            });
        }
    }

    let replacement_chunks = rechunk(replacement, padding_template);
    let actual_groups = group_count_from_chunks(&replacement_chunks);
    if actual_groups != groups_after {
        return Err(SpliceError::RechunkInvariant {
            expected: groups_after,
            actual: actual_groups,
        });
    }

    let saved_group_count = groups_before.total() - groups_after.total();
    let report = SpliceReport {
        removed_tx_count,
        removed_heavy_count: removed_heavy,
        removed_light_count: removed_light,
        suffix_index_shift,
        groups_before,
        groups_after,
        saved_group_count,
    };
    Ok(PreparedSplice {
        replacement_chunks,
        report,
    })
}

fn flatten_active_in_execution_order(
    block: &Block<F>,
) -> Result<(Vec<Arc<Tx<F>>>, Tx<F>), SpliceError> {
    let mut active = Vec::new();
    let mut padding_template = None;
    for tx in block.tx_chunks.iter().flatten() {
        if tx.tx_type == TX_TYPE_EMPTY {
            if padding_template.is_none() {
                padding_template = Some(tx.as_ref().clone());
            }
            continue;
        }
        if !matches!(tx.tx_circuit_type, TX_HEAVY | TX_LIGHT) {
            return Err(SpliceError::InvalidActiveCircuitType {
                tx_index: tx.tx_index,
                circuit_type: tx.tx_circuit_type,
            });
        }
        active.push(Arc::clone(tx));
    }
    let padding_template = padding_template.ok_or(SpliceError::MissingPaddingTemplate)?;
    active.sort_unstable_by_key(|tx| tx.tx_index);
    for (expected, tx) in (0_u64..).zip(&active) {
        if tx.tx_index != expected {
            return Err(SpliceError::NonContiguousActiveIndex {
                expected,
                found: tx.tx_index,
            });
        }
    }
    Ok((active, padding_template))
}

fn validate_pair(pair: &ReplacementPair, first_index: u64) -> Result<(), SpliceError> {
    let second_index = first_index
        .checked_add(1)
        .ok_or(SpliceError::CountArithmeticOverflow)?;
    for (member, tx, expected_index, expected_circuit) in [
        (
            PairMember::LightInsert,
            &pair.light_insert,
            first_index,
            TX_LIGHT,
        ),
        (
            PairMember::HeavyFill,
            &pair.heavy_fill,
            second_index,
            TX_HEAVY,
        ),
    ] {
        if tx.tx_index != expected_index {
            return Err(SpliceError::InvalidPairIndex {
                member,
                expected: expected_index,
                found: tx.tx_index,
            });
        }
        if tx.tx_type != TX_TYPE_INTERNAL_CLAIM_ORDER {
            return Err(SpliceError::InvalidPairTxType {
                member,
                found: tx.tx_type,
            });
        }
        if tx.tx_circuit_type != expected_circuit {
            return Err(SpliceError::InvalidPairCircuitType {
                member,
                expected: expected_circuit,
                found: tx.tx_circuit_type,
            });
        }
    }
    Ok(())
}

fn lane_tx_counts<T>(txs: &[T]) -> (usize, usize)
where
    T: AsRef<Tx<F>>,
{
    let light = txs
        .iter()
        .filter(|tx| tx.as_ref().tx_circuit_type == TX_LIGHT)
        .count();
    (txs.len() - light, light)
}

fn lane_group_count(heavy: usize, light: usize) -> LaneGroupCount {
    LaneGroupCount {
        heavy: heavy.div_ceil(HEAVY_TXS_PER_GROUP).max(1),
        light: light.div_ceil(LIGHT_TXS_PER_GROUP).max(1),
    }
}

fn rechunk(txs: Vec<Tx<F>>, empty_template: Tx<F>) -> Vec<Vec<Arc<Tx<F>>>> {
    let mut chunks = Vec::new();
    let mut heavy = Vec::with_capacity(HEAVY_TXS_PER_GROUP);
    let mut light = Vec::with_capacity(LIGHT_TXS_PER_GROUP);
    let mut has_heavy = false;
    let mut has_light = false;
    let mut last_heavy_index = F::NEG_ONE.to_canonical_u64();
    let mut last_light_index = F::NEG_ONE.to_canonical_u64();

    for tx in txs {
        let (buffer, group_size) = if tx.tx_circuit_type == TX_LIGHT {
            has_light = true;
            last_light_index = tx.tx_index;
            (&mut light, LIGHT_TXS_PER_GROUP)
        } else {
            has_heavy = true;
            last_heavy_index = tx.tx_index;
            (&mut heavy, HEAVY_TXS_PER_GROUP)
        };
        buffer.push(Arc::new(tx));
        if buffer.len() == group_size {
            chunks.push(std::mem::take(buffer));
        }
    }

    // A single payload template is cloned once for the heavy lane and consumed
    // for the light lane. Within each lane every padding slot shares one Arc.
    let mut heavy_padding = empty_template.clone();
    heavy_padding.tx_circuit_type = TX_HEAVY;
    heavy_padding.tx_index = last_heavy_index;
    let mut light_padding = empty_template;
    light_padding.tx_circuit_type = TX_LIGHT;
    light_padding.tx_index = last_light_index;
    for (buffer, group_size, has_txs, padding) in [
        (
            &mut heavy,
            HEAVY_TXS_PER_GROUP,
            has_heavy,
            Arc::new(heavy_padding),
        ),
        (
            &mut light,
            LIGHT_TXS_PER_GROUP,
            has_light,
            Arc::new(light_padding),
        ),
    ] {
        if buffer.is_empty() && has_txs {
            continue;
        }
        while buffer.len() < group_size {
            buffer.push(Arc::clone(&padding));
        }
        chunks.push(std::mem::take(buffer));
    }
    chunks
}

fn group_count_from_chunks(chunks: &[Vec<Arc<Tx<F>>>]) -> LaneGroupCount {
    let mut result = LaneGroupCount { heavy: 0, light: 0 };
    for chunk in chunks {
        let is_light = chunk
            .first()
            .is_some_and(|tx| tx.tx_circuit_type == TX_LIGHT);
        if is_light {
            result.light += 1;
        } else {
            result.heavy += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::fmt::Write as _;
    use std::hash::Hasher;

    use circuit::types::constants::{TX_TYPE_INTERNAL_CANCEL_ORDER, TX_TYPE_INTERNAL_CLAIM_ORDER};

    use super::*;

    struct HashWriter<'a>(&'a mut DefaultHasher);

    impl fmt::Write for HashWriter<'_> {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            self.0.write(value.as_bytes());
            Ok(())
        }
    }

    fn tx_payload_digest_without_index(tx: &Tx<F>) -> u64 {
        let mut normalized = tx.clone();
        normalized.tx_index = 0;
        let mut hasher = DefaultHasher::new();
        write!(HashWriter(&mut hasher), "{normalized:?}").unwrap();
        hasher.finish()
    }

    fn fixture_block_and_padding() -> (Block<F>, Tx<F>) {
        let block = Block::<F>::from_json(include_bytes!("../bench_test.json"), 4, 10)
            .expect("public fixture must parse");
        let padding = block
            .tx_chunks
            .iter()
            .flatten()
            .find(|tx| tx.tx_type == TX_TYPE_EMPTY)
            .expect("fixture has padding")
            .as_ref()
            .clone();
        (block, padding)
    }

    fn active_tx(template: &Tx<F>, tx_index: u64, circuit_type: u8) -> Tx<F> {
        let mut tx = template.clone();
        tx.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
        tx.tx_circuit_type = circuit_type;
        tx.tx_index = tx_index;
        tx.nonce = i64::try_from(10_000_u64 + tx_index).unwrap();
        tx
    }

    fn replacement_pair(template: &Tx<F>, first_index: u64) -> ReplacementPair {
        let mut light = template.clone();
        light.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;
        light.tx_circuit_type = TX_LIGHT;
        light.tx_index = first_index;
        light.nonce = -101;

        let mut heavy = template.clone();
        heavy.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;
        heavy.tx_circuit_type = TX_HEAVY;
        heavy.tx_index = first_index + 1;
        heavy.nonce = -102;
        ReplacementPair::new(light, heavy)
    }

    fn install_active(block: &mut Block<F>, padding: &Tx<F>, lane_types: &[u8]) {
        let mut chunk = lane_types
            .iter()
            .copied()
            .enumerate()
            .map(|(index, circuit_type)| Arc::new(active_tx(padding, index as u64, circuit_type)))
            .collect::<Vec<_>>();
        chunk.push(Arc::new(padding.clone()));
        block.tx_chunks = vec![chunk];
    }

    fn sorted_active(block: &Block<F>) -> Vec<Arc<Tx<F>>> {
        let mut result = block
            .tx_chunks
            .iter()
            .flatten()
            .filter(|tx| tx.tx_type != TX_TYPE_EMPTY)
            .cloned()
            .collect::<Vec<_>>();
        result.sort_unstable_by_key(|tx| tx.tx_index);
        result
    }

    fn tx_chunk_snapshot(block: &Block<F>) -> Vec<Vec<(usize, u64, u64)>> {
        block
            .tx_chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|tx| {
                        (
                            Arc::as_ptr(tx) as usize,
                            tx.tx_index,
                            tx_payload_digest_without_index(tx),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    fn with_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(test)
            .expect("rechunk test thread must start")
            .join()
            .expect("rechunk test thread must finish");
    }

    #[test]
    fn ranked_10h_490l_case_saves_exactly_one_group_and_preserves_suffix_payloads() {
        with_large_stack(|| {
            let (mut block, padding) = fixture_block_and_padding();
            let mut lane_types = vec![TX_HEAVY, TX_HEAVY, TX_HEAVY, TX_LIGHT];
            lane_types.extend(std::iter::repeat_n(TX_HEAVY, 7));
            lane_types.extend(std::iter::repeat_n(TX_LIGHT, 489));
            assert_eq!(lane_types.len(), 500);
            install_active(&mut block, &padding, &lane_types);

            let suffix_before = sorted_active(&block)
                .into_iter()
                .filter(|tx| tx.tx_index >= 4)
                .map(|tx| {
                    (
                        tx.tx_index,
                        tx.tx_circuit_type,
                        tx_payload_digest_without_index(&tx),
                    )
                })
                .collect::<Vec<_>>();
            let report = splice_verified_pair(
                &mut block,
                InclusiveInterval::new(0, 3),
                replacement_pair(&padding, 0),
            )
            .expect("ranked replacement must save one group");
            assert_eq!(
                report,
                SpliceReport {
                    removed_tx_count: 4,
                    removed_heavy_count: 3,
                    removed_light_count: 1,
                    suffix_index_shift: 2,
                    groups_before: LaneGroupCount {
                        heavy: 3,
                        light: 49,
                    },
                    groups_after: LaneGroupCount {
                        heavy: 2,
                        light: 49,
                    },
                    saved_group_count: 1,
                }
            );

            let active_after = sorted_active(&block);
            assert_eq!(active_after.len(), 498);
            for (expected, tx) in (0_u64..).zip(&active_after) {
                assert_eq!(tx.tx_index, expected);
            }
            assert_eq!(active_after[0].tx_circuit_type, TX_LIGHT);
            assert_eq!(active_after[1].tx_circuit_type, TX_HEAVY);
            assert_eq!(active_after[0].nonce, -101);
            assert_eq!(active_after[1].nonce, -102);
            for ((old_index, circuit_type, payload), shifted) in
                suffix_before.into_iter().zip(&active_after[2..])
            {
                assert_eq!(shifted.tx_index, old_index - 2);
                assert_eq!(shifted.tx_circuit_type, circuit_type);
                assert_eq!(tx_payload_digest_without_index(shifted), payload);
            }

            assert_eq!(
                group_count_from_chunks(&block.tx_chunks),
                report.groups_after
            );
            assert!(block.tx_chunks.iter().all(|chunk| {
                let expected = if chunk[0].tx_circuit_type == TX_LIGHT {
                    LIGHT_TXS_PER_GROUP
                } else {
                    HEAVY_TXS_PER_GROUP
                };
                chunk.len() == expected
                    && chunk
                        .iter()
                        .all(|tx| tx.tx_circuit_type == chunk[0].tx_circuit_type)
            }));
        });
    }

    #[test]
    fn cross_lane_execution_order_and_shared_padding_are_preserved() {
        with_large_stack(|| {
            let (mut block, padding) = fixture_block_and_padding();
            install_active(
                &mut block,
                &padding,
                &[TX_HEAVY, TX_HEAVY, TX_HEAVY, TX_LIGHT, TX_HEAVY, TX_HEAVY],
            );
            let report = splice_verified_pair(
                &mut block,
                InclusiveInterval::new(0, 2),
                replacement_pair(&padding, 0),
            )
            .expect("three-heavy replacement saves one group");
            assert_eq!(report.groups_after, LaneGroupCount { heavy: 1, light: 1 });
            let active = sorted_active(&block);
            assert_eq!(
                active
                    .iter()
                    .map(|tx| (tx.tx_index, tx.tx_circuit_type))
                    .collect::<Vec<_>>(),
                vec![
                    (0, TX_LIGHT),
                    (1, TX_HEAVY),
                    (2, TX_LIGHT),
                    (3, TX_HEAVY),
                    (4, TX_HEAVY),
                ]
            );

            for lane in [TX_HEAVY, TX_LIGHT] {
                let padding_slots = block
                    .tx_chunks
                    .iter()
                    .filter(|chunk| chunk[0].tx_circuit_type == lane)
                    .flatten()
                    .filter(|tx| tx.tx_type == TX_TYPE_EMPTY)
                    .collect::<Vec<_>>();
                assert!(!padding_slots.is_empty());
                assert!(
                    padding_slots[1..]
                        .iter()
                        .all(|tx| Arc::ptr_eq(padding_slots[0], tx))
                );
            }
        });
    }

    #[test]
    fn invalid_request_is_strictly_transactional() {
        with_large_stack(|| {
            let (mut block, padding) = fixture_block_and_padding();
            install_active(
                &mut block,
                &padding,
                &[TX_HEAVY, TX_HEAVY, TX_HEAVY, TX_LIGHT, TX_HEAVY],
            );
            let before = tx_chunk_snapshot(&block);

            let mut invalid_pair = replacement_pair(&padding, 0);
            invalid_pair.heavy_fill.tx_index = 7;
            assert!(matches!(
                splice_verified_pair(&mut block, InclusiveInterval::new(0, 2), invalid_pair,),
                Err(SpliceError::InvalidPairIndex {
                    member: PairMember::HeavyFill,
                    expected: 1,
                    found: 7,
                })
            ));
            assert_eq!(tx_chunk_snapshot(&block), before);

            assert!(matches!(
                splice_verified_pair(
                    &mut block,
                    InclusiveInterval::new(2, 99),
                    replacement_pair(&padding, 2),
                ),
                Err(SpliceError::IntervalOutOfBounds { .. })
            ));
            assert_eq!(tx_chunk_snapshot(&block), before);
        });
    }
}
