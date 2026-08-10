// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Read-only, fail-closed discovery for the phantom spot-session soundness break.
//!
//! This first pass intentionally recognizes only intervals whose complete endpoint
//! state is exposed by the witnesses immediately before both interval boundaries.
//! It does not mutate the block and it does not guess or reconstruct missing paths.

use std::collections::BTreeSet;

use circuit::block::Block;
use circuit::tx::Tx;
use circuit::types::account::Account;
use circuit::types::account_delta::AccountDelta;
use circuit::types::asset::Asset;
use circuit::types::config::F;
use circuit::types::constants::{
    ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE, EMPTY_ACCOUNT_ORDERS_TREE_ROOT, EMPTY_API_KEY_TREE_ROOT,
    EMPTY_POSITION_DELTA_TREE_ROOT, FEE_ACCOUNT_ID, GTT, INSERT_ORDER,
    INSURANCE_FUND_ACCOUNT_TYPE, IOC, LIMIT_ORDER, MARKET_TYPE_SPOT, MASTER_ACCOUNT_TYPE,
    MIN_ORDER_EXPIRY_PERIOD, NIL_ACCOUNT_INDEX, NIL_MARKET_INDEX, ORDER_BASE_AMOUNT_BITS,
    ORDER_NONCE_BITS, ORDER_PRICE_BITS, QUOTE_SUM_BITS, SUB_ACCOUNT_TYPE, TIMESTAMP_BITS,
    TREASURY_ACCOUNT_INDEX, TX_LIGHT,
    TX_TYPE_INTERNAL_CANCEL_ALL_ORDERS, TX_TYPE_INTERNAL_CANCEL_ORDER,
    TX_TYPE_INTERNAL_CLAIM_ORDER, TX_TYPE_INTERNAL_CREATE_ORDER, TX_TYPE_L2_CANCEL_ALL_ORDERS,
    TX_TYPE_L2_CANCEL_ORDER, TX_TYPE_L2_CREATE_GROUPED_ORDERS, TX_TYPE_L2_CREATE_ORDER,
    TX_TYPE_L2_MODIFY_ORDER,
};
use circuit::types::margined_asset::MarginedAsset;
use circuit::types::system_config::SystemConfig;
use num::{BigInt, BigUint, ToPrimitive, Zero};
use plonky2::hash::hash_types::HashOut;

use crate::phantom_native::{
    account_asset_hash, account_delta_hash, account_hash, apply_one_update_to_proof,
    bigint_leaf_hash, fee_account_delta_hash, fee_account_hash, merkle_root, rewind_one_update,
};

const MIN_REPLACED_TXS: usize = 3;
const HEAVY_TXS_PER_PROOF: usize = 4;
const LIGHT_TXS_PER_PROOF: usize = 10;
const PHANTOM_MAKER_NONCE: i64 = 11;
const PHANTOM_TAKER_NONCE: i64 = 12;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BilateralSwap {
    pub seller_slot: usize,
    pub buyer_slot: usize,
    pub base_asset_slot: usize,
    pub quote_asset_slot: usize,
    pub base_amount: i64,
    pub price: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    /// Original execution indices; the interval is `[start_tx_index, end_tx_index)`.
    pub start_tx_index: u64,
    pub end_tx_index: u64,
    pub replaced_tx_count: usize,
    pub replaced_light_count: usize,
    pub replaced_heavy_count: usize,
    pub saved_lane_proof_count: usize,

    pub seller_account_index: i64,
    pub buyer_account_index: i64,
    pub base_asset_index: i16,
    pub quote_asset_index: i16,
    pub base_amount: i64,
    pub price: i64,

    /// These are donor facts, not a claim that a rewriter already exists.
    /// The canonical-empty tree permits constructing the sibling path for the
    /// chosen phantom `(order_index, client_order_index)` itself. A zero leaf at
    /// some unrelated key in a nonempty tree is never accepted as a donor.
    pub maker_account_orders_path_at_phantom_key: bool,
    pub maker_api_path_from_padding: bool,
    pub taker_api_path_from_padding: bool,
    pub fee_paths_natively_verified: bool,
    pub nil_market_path_from_final_padding: bool,
    pub nil_market_details_path_from_final_padding: bool,
}

impl Candidate {
    pub fn has_direct_v1_donors(&self) -> bool {
        self.maker_account_orders_path_at_phantom_key
            && self.maker_api_path_from_padding
            && self.taker_api_path_from_padding
            && self.fee_paths_natively_verified
            && self.nil_market_path_from_final_padding
            && self.nil_market_details_path_from_final_padding
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanReport {
    pub active_tx_count: usize,
    pub candidates: Vec<Candidate>,
}

/// Find endpoint-equivalent intervals which the two-transaction phantom session
/// could replace. The scan is conservative: a false negative is preferred to a
/// candidate whose endpoint or proof-donor contract has not been observed.
pub fn scan(block: &Block<F>) -> ScanReport {
    let mut active = block
        .tx_chunks
        .iter()
        .flatten()
        .filter(|tx| !tx.is_empty())
        .map(|tx| tx.as_ref())
        .collect::<Vec<_>>();
    active.sort_unstable_by_key(|tx| tx.tx_index);

    let padding = block
        .tx_chunks
        .iter()
        .flatten()
        .find(|tx| tx.is_empty())
        .map(|tx| tx.as_ref());

    let mut report = ScanReport {
        active_tx_count: active.len(),
        candidates: Vec::new(),
    };
    let total_light = active
        .iter()
        .filter(|tx| tx.tx_circuit_type == TX_LIGHT)
        .count();
    let total_heavy = active.len() - total_light;
    // These snapshots are block-static in the production witness. Validate
    // that invariant once with exact field equality; rejecting the whole scan
    // on a mismatch is conservative and avoids rewalking large arrays for each
    // of the O(n^2) candidate intervals.
    if active.first().is_some_and(|first| {
        active
            .iter()
            .skip(1)
            .any(|tx| !static_snapshot_equal(first, tx))
    }) {
        return report;
    }

    // Repeated or discontinuous execution indices mean the reconstructed order
    // is ambiguous. Padding indices are irrelevant because padding was removed.
    if active
        .windows(2)
        .any(|pair| pair[0].tx_index.checked_add(1) != Some(pair[1].tx_index))
    {
        return report;
    }

    let mut disallowed_prefix = Vec::with_capacity(active.len() + 1);
    let mut light_prefix = Vec::with_capacity(active.len() + 1);
    disallowed_prefix.push(0_usize);
    light_prefix.push(0_usize);
    for tx in &active {
        disallowed_prefix.push(
            disallowed_prefix.last().copied().unwrap()
                + usize::from(!is_pubdata_free_order_tx(tx.tx_type)),
        );
        light_prefix.push(
            light_prefix.last().copied().unwrap()
                + usize::from(tx.tx_circuit_type == TX_LIGHT),
        );
    }

    // `right` is the witness immediately after the replaced interval, so the
    // terminal block boundary is not considered until post-state leaves can be
    // reconstructed rather than guessed from padding's NIL accounts.
    for start in 0..active.len() {
        for end in start + MIN_REPLACED_TXS..active.len() {
            if disallowed_prefix[end] != disallowed_prefix[start] {
                continue;
            }
            let left = active[start];
            let right = active[end];
            let replaced_light = light_prefix[end] - light_prefix[start];
            let replaced_heavy = end - start - replaced_light;
            let Some(saved_lane_proof_count) =
                lane_proof_saving(total_heavy, total_light, replaced_heavy, replaced_light)
            else {
                continue;
            };

            if !cheap_global_endpoint_filter(left, right) {
                continue;
            }
            let common_accounts = common_user_accounts(left, right);
            if common_accounts.len() != 2 {
                continue;
            }

            let Some((swap, left_slots, right_slots)) =
                classify_endpoint(left, right, &common_accounts)
            else {
                continue;
            };
            if !verify_all_affected_roots(left, right, left_slots, right_slots) {
                continue;
            }
            if left_slots
                .into_iter()
                .zip(right_slots)
                .any(|(left_slot, right_slot)| {
                    left.accounts_delta_before[left_slot].position_delta_root
                        != EMPTY_POSITION_DELTA_TREE_ROOT
                        || right.accounts_delta_before[right_slot].position_delta_root
                            != EMPTY_POSITION_DELTA_TREE_ROOT
                })
            {
                continue;
            }

            let seller_left = &left.accounts_before[left_slots[swap.seller_slot]];
            let buyer_left = &left.accounts_before[left_slots[swap.buyer_slot]];
            if !claim_roles_are_materializable(seller_left, buyer_left)
                || !materialization_arithmetic_is_safe(block.created_at, left, &swap, left_slots)
            {
                continue;
            }
            let (nil_market_path_from_final_padding, nil_market_details_path_from_final_padding) =
                padding
                    .map(|pad| direct_nil_padding_paths(left, pad))
                    .unwrap_or((false, false));
            let maker_account_orders_path_at_phantom_key =
                can_synthesize_phantom_order_path(seller_left);
            let maker_api_path_from_padding = seller_left.api_key_root == EMPTY_API_KEY_TREE_ROOT;
            let taker_api_path_from_padding = buyer_left.api_key_root == EMPTY_API_KEY_TREE_ROOT;
            if !maker_account_orders_path_at_phantom_key
                || !maker_api_path_from_padding
                || !taker_api_path_from_padding
                || !nil_market_path_from_final_padding
                || !nil_market_details_path_from_final_padding
            {
                continue;
            }

            report.candidates.push(Candidate {
                start_tx_index: left.tx_index,
                end_tx_index: right.tx_index,
                replaced_tx_count: end - start,
                replaced_light_count: replaced_light,
                replaced_heavy_count: replaced_heavy,
                saved_lane_proof_count,
                seller_account_index: seller_left.account_index,
                buyer_account_index: buyer_left.account_index,
                base_asset_index: left.asset_indices[swap.base_asset_slot],
                quote_asset_index: left.asset_indices[swap.quote_asset_slot],
                base_amount: swap.base_amount,
                price: swap.price,
                maker_account_orders_path_at_phantom_key,
                maker_api_path_from_padding,
                taker_api_path_from_padding,
                fee_paths_natively_verified: true,
                nil_market_path_from_final_padding,
                nil_market_details_path_from_final_padding,
            });
        }
    }

    report
}

/// Exact lane arithmetic used by `Block::chunk_txs`: both paths retain at
/// least one chunk, and a phantom replacement contributes one light and one
/// heavy transaction. `None` means it cannot reduce the recursive proof count.
pub fn lane_proof_saving(
    total_heavy: usize,
    total_light: usize,
    replaced_heavy: usize,
    replaced_light: usize,
) -> Option<usize> {
    if replaced_heavy > total_heavy || replaced_light > total_light {
        return None;
    }
    let before = lane_chunks(total_heavy, HEAVY_TXS_PER_PROOF)
        + lane_chunks(total_light, LIGHT_TXS_PER_PROOF);
    let after_heavy = total_heavy - replaced_heavy + 1;
    let after_light = total_light - replaced_light + 1;
    let after = lane_chunks(after_heavy, HEAVY_TXS_PER_PROOF)
        + lane_chunks(after_light, LIGHT_TXS_PER_PROOF);
    (after < before).then_some(before - after)
}

fn lane_chunks(tx_count: usize, txs_per_proof: usize) -> usize {
    tx_count.div_ceil(txs_per_proof).max(1)
}

fn can_synthesize_phantom_order_path(account: &Account<F>) -> bool {
    account.account_orders_root == EMPTY_ACCOUNT_ORDERS_TREE_ROOT
}

fn is_pubdata_free_order_tx(tx_type: u8) -> bool {
    matches!(
        tx_type,
        TX_TYPE_L2_CREATE_ORDER
            | TX_TYPE_L2_CANCEL_ORDER
            | TX_TYPE_L2_CANCEL_ALL_ORDERS
            | TX_TYPE_L2_MODIFY_ORDER
            | TX_TYPE_L2_CREATE_GROUPED_ORDERS
            | TX_TYPE_INTERNAL_CLAIM_ORDER
            | TX_TYPE_INTERNAL_CANCEL_ORDER
            | TX_TYPE_INTERNAL_CANCEL_ALL_ORDERS
            | TX_TYPE_INTERNAL_CREATE_ORDER
    )
}

fn cheap_global_endpoint_filter(left: &Tx<F>, right: &Tx<F>) -> bool {
    left.asset_indices == right.asset_indices
        && left.asset_indices[0] != left.asset_indices[1]
        && left.old_market_tree_root == right.old_market_tree_root
        && left.old_market_details_tree_root == right.old_market_details_tree_root
        && register_stack_is_canonical_empty(left)
        && register_stack_is_canonical_empty(right)
}

#[cfg(test)]
fn strict_global_endpoint_filter(left: &Tx<F>, right: &Tx<F>) -> bool {
    static_snapshot_equal(left, right)
}

fn static_snapshot_equal(left: &Tx<F>, right: &Tx<F>) -> bool {
    fn system_equal(left: &SystemConfig, right: &SystemConfig) -> bool {
        left.liquidity_pool_index == right.liquidity_pool_index
            && left.staking_pool_index == right.staking_pool_index
            && left.liquidity_pool_cooldown_period == right.liquidity_pool_cooldown_period
            && left.staking_pool_lockup_period == right.staking_pool_lockup_period
            && left.max_integrator_spot_taker_fee == right.max_integrator_spot_taker_fee
            && left.max_integrator_spot_maker_fee == right.max_integrator_spot_maker_fee
            && left.max_integrator_perps_taker_fee == right.max_integrator_perps_taker_fee
            && left.max_integrator_perps_maker_fee == right.max_integrator_perps_maker_fee
    }
    fn asset_equal(left: &Asset, right: &Asset) -> bool {
        left.asset_index == right.asset_index
            && left.extension_multiplier == right.extension_multiplier
            && left.min_transfer_amount == right.min_transfer_amount
            && left.min_withdrawal_amount == right.min_withdrawal_amount
            && left.margin_mode == right.margin_mode
            && left.margin_index == right.margin_index
    }
    fn margined_asset_equal(left: &MarginedAsset, right: &MarginedAsset) -> bool {
        left.margin_index == right.margin_index
            && left.asset_index == right.asset_index
            && left.loan_to_value == right.loan_to_value
            && left.liquidation_threshold == right.liquidation_threshold
            && left.liquidation_factor == right.liquidation_factor
            && left.liquidation_fee == right.liquidation_fee
            && left.index_price == right.index_price
            && left.index_price_divider == right.index_price_divider
            && left.global_supply_cap == right.global_supply_cap
            && left.user_supply_cap == right.user_supply_cap
            && left.total_supplied_amount == right.total_supplied_amount
    }

    system_equal(&left.system_config_before, &right.system_config_before)
        && left
            .all_assets_before
            .iter()
            .zip(&right.all_assets_before)
            .all(|(left, right)| asset_equal(left, right))
        && left
            .all_margined_assets_before
            .iter()
            .zip(&right.all_margined_assets_before)
            .all(|(left, right)| margined_asset_equal(left, right))
        && left.all_market_risk_details_before == right.all_market_risk_details_before
}

fn register_stack_is_canonical_empty(tx: &Tx<F>) -> bool {
    tx.register_stack_before.count == 0
        && tx
            .register_stack_before
            .stack
            .iter()
            .all(|reg| reg.is_empty())
}

fn market_leaf_is_canonical_zero(tx: &Tx<F>) -> bool {
    let market = &tx.market_before;
    market.market_index == u16::from(NIL_MARKET_INDEX)
        && market.status == 0
        && market.market_type == 0
        && market.base_asset_id == 0
        && market.quote_asset_id == 0
        && market.ask_nonce == 0
        && market.bid_nonce == 0
        && market.taker_fee == 0
        && market.maker_fee == 0
        && market.liquidation_fee == 0
        && market.size_extension_multiplier == 0
        && market.quote_extension_multiplier == 0
        && market.total_order_count == 0
        && market.min_base_amount == 0
        && market.min_quote_amount == 0
        && market.order_quote_limit == 0
}

fn market_details_leaf_is_canonical_zero(tx: &Tx<F>) -> bool {
    let details = &tx.market_details_before;
    details.market_index == u16::from(NIL_MARKET_INDEX)
        && details.interest_rate == 0
        && details.aggregate_premium_sum == 0
        && details.impact_bid_price == 0
        && details.impact_ask_price == 0
        && details.impact_price == 0
        && details.open_interest == 0
        && details.index_price == 0
        && details.funding_clamp_small == 0
        && details.funding_clamp_big == 0
        && details.open_interest_limit == 0
        && details.market_flags == 0
        && details.funding_premium_multiplier == 0
}

fn direct_nil_padding_paths(left: &Tx<F>, padding: &Tx<F>) -> (bool, bool) {
    if !padding.is_empty() {
        return (false, false);
    }
    let nil_index = u64::from(NIL_MARKET_INDEX);
    let market_ok = market_leaf_is_canonical_zero(padding)
        && merkle_root(HashOut::ZERO, nil_index, &padding.market_tree_merkle_proof)
            == padding.old_market_tree_root
        && left.old_market_tree_root == padding.old_market_tree_root;
    let details_ok = market_details_leaf_is_canonical_zero(padding)
        && merkle_root(
            HashOut::ZERO,
            nil_index,
            &padding.market_details_tree_merkle_proof,
        ) == padding.old_market_details_tree_root
        && left.old_market_details_tree_root == padding.old_market_details_tree_root;
    (market_ok, details_ok)
}

fn claim_roles_are_materializable(seller: &Account<F>, buyer: &Account<F>) -> bool {
    fn regular_simple_account(account: &Account<F>) -> bool {
        account.account_index != NIL_ACCOUNT_INDEX
            && account.account_index != TREASURY_ACCOUNT_INDEX as i64
            && account.account_type != INSURANCE_FUND_ACCOUNT_TYPE
            && matches!(account.account_type, MASTER_ACCOUNT_TYPE | SUB_ACCOUNT_TYPE)
            && account.account_trading_mode == ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE
    }

    seller.account_index != buyer.account_index
        && regular_simple_account(seller)
        && regular_simple_account(buyer)
}

fn materialization_arithmetic_is_safe(
    block_created_at: i64,
    tx: &Tx<F>,
    swap: &BilateralSwap,
    account_slots: [usize; 2],
) -> bool {
    fn fits_unsigned(value: i64, bits: usize) -> bool {
        value >= 0 && (value as u128) < (1_u128 << bits)
    }
    fn checked_positive_sum(values: &[i64]) -> Option<i64> {
        values
            .iter()
            .try_fold(0_i64, |sum, value| sum.checked_add(*value))
            .filter(|sum| *sum >= 0)
    }
    fn checked_negative_sum(values: &[i64]) -> Option<i64> {
        checked_positive_sum(values)?.checked_neg()
    }

    let base_index = tx.asset_indices[swap.base_asset_slot];
    let quote_index = tx.asset_indices[swap.quote_asset_slot];
    if base_index < 0 || quote_index < 0 {
        return false;
    }
    let Some(base_asset) = usize::try_from(base_index)
        .ok()
        .and_then(|index| tx.all_assets_before.get(index))
        .filter(|asset| asset.asset_index == base_index)
    else {
        return false;
    };
    let Some(quote_asset) = usize::try_from(quote_index)
        .ok()
        .and_then(|index| tx.all_assets_before.get(index))
        .filter(|asset| asset.asset_index == quote_index)
    else {
        return false;
    };
    let (quantity, price) = (swap.base_amount, swap.price);
    if !fits_unsigned(quantity, ORDER_BASE_AMOUNT_BITS)
        || !fits_unsigned(price, ORDER_PRICE_BITS)
        || !fits_unsigned(PHANTOM_MAKER_NONCE, ORDER_NONCE_BITS)
        || !fits_unsigned(PHANTOM_TAKER_NONCE, ORDER_NONCE_BITS)
        || !fits_unsigned(base_asset.extension_multiplier, 48)
        || !fits_unsigned(quote_asset.extension_multiplier, 48)
        || base_asset.extension_multiplier == 0
        || quote_asset.extension_multiplier == 0
    {
        return false;
    }

    let Some(quote_amount) = quantity.checked_mul(price) else {
        return false;
    };
    if !fits_unsigned(quote_amount, QUOTE_SUM_BITS) {
        return false;
    }
    let u96_limit = 1_u128 << 96;
    let Some(actual_base) = (quantity as u128).checked_mul(base_asset.extension_multiplier as u128)
    else {
        return false;
    };
    let Some(actual_quote) =
        (quote_amount as u128).checked_mul(quote_asset.extension_multiplier as u128)
    else {
        return false;
    };
    if actual_base >= u96_limit || actual_quote >= u96_limit {
        return false;
    }

    if !fits_unsigned(block_created_at, TIMESTAMP_BITS) {
        return false;
    }
    let Some(expiry) = block_created_at.checked_add(MIN_ORDER_EXPIRY_PERIOD) else {
        return false;
    };
    if !fits_unsigned(expiry, TIMESTAMP_BITS) {
        return false;
    }

    let market = i64::from(NIL_MARKET_INDEX);
    let maker = account_slots[swap.seller_slot];
    let buyer = account_slots[swap.buyer_slot];
    let maker_account = tx.accounts_before[maker].account_index;
    let buyer_account = tx.accounts_before[buyer].account_index;
    if !fits_unsigned(maker_account, 48) || !fits_unsigned(buyer_account, 48) {
        return false;
    }

    // Every deliberately negative witness value is formed by cancelling a
    // positive sum in an `is_empty` check. Checked construction ensures host
    // arithmetic and the Goldilocks sum agree rather than wrapping first.
    let cancellation_sums = [
        checked_negative_sum(&[
            quantity,
            quantity,
            price,
            PHANTOM_MAKER_NONCE,
            1,
            i64::from(LIMIT_ORDER),
            i64::from(GTT),
        ]),
        checked_negative_sum(&[
            quantity,
            quantity,
            price,
            PHANTOM_TAKER_NONCE,
            i64::from(LIMIT_ORDER),
            i64::from(IOC),
        ]),
        checked_negative_sum(&[i64::from(INSERT_ORDER), market, maker_account]),
        checked_negative_sum(&[i64::from(INSERT_ORDER), market, buyer_account]),
        checked_negative_sum(&[
            quantity,
            price,
            PHANTOM_MAKER_NONCE,
            quantity,
            1,
            i64::from(LIMIT_ORDER),
            i64::from(GTT),
            expiry,
        ]),
        checked_negative_sum(&[
            MARKET_TYPE_SPOT as i64,
            i64::from(base_index),
            i64::from(quote_index),
            base_asset.extension_multiplier,
            quote_asset.extension_multiplier,
        ]),
    ];
    if cancellation_sums.iter().any(Option::is_none) {
        return false;
    }

    let account_order_index = (i128::from(market) + 1)
        .checked_shl(ORDER_NONCE_BITS as u32)
        .and_then(|prefix| prefix.checked_add(i128::from(PHANTOM_MAKER_NONCE)));
    let order_book_key = (price as u128)
        .checked_shl(ORDER_NONCE_BITS as u32)
        .and_then(|prefix| prefix.checked_add(PHANTOM_MAKER_NONCE as u128));
    matches!(account_order_index, Some(index) if index >= 0 && index <= i128::from(i64::MAX))
        && matches!(order_book_key, Some(index) if index < (1_u128 << (ORDER_PRICE_BITS + ORDER_NONCE_BITS)))
}

fn debug_equal<T: std::fmt::Debug>(left: &T, right: &T) -> bool {
    format!("{left:?}") == format!("{right:?}")
}

fn as_merkle_index(index: i64, levels: usize) -> Option<u64> {
    let index = u64::try_from(index).ok()?;
    ((index as u128) < (1_u128 << levels)).then_some(index)
}

pub fn verify_two_leaf_transition(
    indices: [u64; 2],
    old_leaves: [HashOut<F>; 2],
    new_leaves: [HashOut<F>; 2],
    proof_0_before: &[HashOut<F>],
    proof_1_after_original_0: &[HashOut<F>],
    old_root: HashOut<F>,
    new_root: HashOut<F>,
) -> bool {
    let Some(proof_1_before) = rewind_one_update(
        indices[0],
        old_leaves[0],
        proof_0_before,
        indices[1],
        proof_1_after_original_0,
    ) else {
        return false;
    };
    if merkle_root(old_leaves[0], indices[0], proof_0_before) != old_root
        || merkle_root(old_leaves[1], indices[1], &proof_1_before) != old_root
    {
        return false;
    }

    let root_after_0 = merkle_root(new_leaves[0], indices[0], proof_0_before);
    let Some(proof_1_after_0) = apply_one_update_to_proof(
        indices[0],
        new_leaves[0],
        proof_0_before,
        indices[1],
        &proof_1_before,
    ) else {
        return false;
    };
    if merkle_root(old_leaves[1], indices[1], &proof_1_after_0) != root_after_0 {
        return false;
    }
    let final_root = merkle_root(new_leaves[1], indices[1], &proof_1_after_0);
    if final_root != new_root {
        return false;
    }

    let Some(proof_0_after_1) = apply_one_update_to_proof(
        indices[1],
        new_leaves[1],
        &proof_1_after_0,
        indices[0],
        proof_0_before,
    ) else {
        return false;
    };
    merkle_root(new_leaves[0], indices[0], &proof_0_after_1) == new_root
}

fn verify_account_inner_trees(
    left: &Tx<F>,
    right: &Tx<F>,
    left_slot: usize,
    right_slot: usize,
) -> bool {
    let Some(indices) = left
        .asset_indices
        .map(|index| {
            as_merkle_index(
                i64::from(index),
                left.asset_tree_merkle_proofs[left_slot][0].len(),
            )
        })
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .and_then(|indices| indices.try_into().ok())
    else {
        return false;
    };

    let Some(old_assets) = left.account_assets_before[left_slot]
        .iter()
        .map(account_asset_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    let Some(new_assets) = right.account_assets_before[right_slot]
        .iter()
        .map(account_asset_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    if !verify_two_leaf_transition(
        indices,
        old_assets,
        new_assets,
        &left.asset_tree_merkle_proofs[left_slot][0],
        &left.asset_tree_merkle_proofs[left_slot][1],
        left.accounts_before[left_slot].asset_root,
        right.accounts_before[right_slot].asset_root,
    ) {
        return false;
    }

    let Some(old_balances) = left.accounts_before[left_slot]
        .aggregated_balances
        .iter()
        .map(bigint_leaf_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    let Some(new_balances) = right.accounts_before[right_slot]
        .aggregated_balances
        .iter()
        .map(bigint_leaf_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    if !verify_two_leaf_transition(
        indices,
        old_balances,
        new_balances,
        &left.public_asset_tree_merkle_proofs[left_slot][0],
        &left.public_asset_tree_merkle_proofs[left_slot][1],
        left.accounts_before[left_slot].aggregated_balances_root,
        right.accounts_before[right_slot].aggregated_balances_root,
    ) {
        return false;
    }

    let Some(old_deltas) = left.accounts_delta_before[left_slot]
        .aggregated_asset_deltas
        .iter()
        .map(bigint_leaf_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    let Some(new_deltas) = right.accounts_delta_before[right_slot]
        .aggregated_asset_deltas
        .iter()
        .map(bigint_leaf_hash)
        .collect::<Option<Vec<_>>>()
        .and_then(|leaves| leaves.try_into().ok())
    else {
        return false;
    };
    verify_two_leaf_transition(
        indices,
        old_deltas,
        new_deltas,
        &left.asset_delta_tree_merkle_proofs[left_slot][0],
        &left.asset_delta_tree_merkle_proofs[left_slot][1],
        left.accounts_delta_before[left_slot].asset_delta_root,
        right.accounts_delta_before[right_slot].asset_delta_root,
    )
}

pub fn verify_three_leaf_transition(
    indices: [u64; 3],
    old_leaves: [HashOut<F>; 3],
    new_leaves: [HashOut<F>; 3],
    sequential_proofs: [&[HashOut<F>]; 3],
    old_root: HashOut<F>,
    new_root: HashOut<F>,
) -> bool {
    if indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2] {
        return false;
    }
    let Some(proof_1_before) = rewind_one_update(
        indices[0],
        old_leaves[0],
        sequential_proofs[0],
        indices[1],
        sequential_proofs[1],
    ) else {
        return false;
    };
    let Some(proof_2_after_0) = rewind_one_update(
        indices[1],
        old_leaves[1],
        sequential_proofs[1],
        indices[2],
        sequential_proofs[2],
    ) else {
        return false;
    };
    let Some(proof_2_before) = rewind_one_update(
        indices[0],
        old_leaves[0],
        sequential_proofs[0],
        indices[2],
        &proof_2_after_0,
    ) else {
        return false;
    };
    if merkle_root(old_leaves[0], indices[0], sequential_proofs[0]) != old_root
        || merkle_root(old_leaves[1], indices[1], &proof_1_before) != old_root
        || merkle_root(old_leaves[2], indices[2], &proof_2_before) != old_root
    {
        return false;
    }

    let root_after_0 = merkle_root(new_leaves[0], indices[0], sequential_proofs[0]);
    let Some(proof_1_after_0) = apply_one_update_to_proof(
        indices[0],
        new_leaves[0],
        sequential_proofs[0],
        indices[1],
        &proof_1_before,
    ) else {
        return false;
    };
    if merkle_root(old_leaves[1], indices[1], &proof_1_after_0) != root_after_0 {
        return false;
    }
    let final_root = merkle_root(new_leaves[1], indices[1], &proof_1_after_0);
    if final_root != new_root {
        return false;
    }

    let Some(proof_2_after_0) = apply_one_update_to_proof(
        indices[0],
        new_leaves[0],
        sequential_proofs[0],
        indices[2],
        &proof_2_before,
    ) else {
        return false;
    };
    let Some(proof_2_after_1) = apply_one_update_to_proof(
        indices[1],
        new_leaves[1],
        &proof_1_after_0,
        indices[2],
        &proof_2_after_0,
    ) else {
        return false;
    };
    if merkle_root(new_leaves[2], indices[2], &proof_2_after_1) != new_root {
        return false;
    }

    let Some(proof_0_after_1) = apply_one_update_to_proof(
        indices[1],
        new_leaves[1],
        &proof_1_after_0,
        indices[0],
        sequential_proofs[0],
    ) else {
        return false;
    };
    merkle_root(new_leaves[0], indices[0], &proof_0_after_1) == new_root
}

fn verify_all_affected_roots(
    left: &Tx<F>,
    right: &Tx<F>,
    left_slots: [usize; 2],
    right_slots: [usize; 2],
) -> bool {
    let mut left_user_slots = left_slots;
    let mut right_user_slots = right_slots;
    left_user_slots.sort_unstable();
    right_user_slots.sort_unstable();
    if left_user_slots != [0, 1]
        || right_user_slots != [0, 1]
        || left.accounts_before[FEE_ACCOUNT_ID].account_index != TREASURY_ACCOUNT_INDEX as i64
        || right.accounts_before[FEE_ACCOUNT_ID].account_index != TREASURY_ACCOUNT_INDEX as i64
        || left.accounts_delta_before[FEE_ACCOUNT_ID].account_index != TREASURY_ACCOUNT_INDEX as i64
        || right.accounts_delta_before[FEE_ACCOUNT_ID].account_index
            != TREASURY_ACCOUNT_INDEX as i64
        || !debug_equal(
            &left.accounts_before[FEE_ACCOUNT_ID],
            &right.accounts_before[FEE_ACCOUNT_ID],
        )
        || !debug_equal(
            &left.account_assets_before[FEE_ACCOUNT_ID],
            &right.account_assets_before[FEE_ACCOUNT_ID],
        )
        || !debug_equal(
            &left.accounts_delta_before[FEE_ACCOUNT_ID],
            &right.accounts_delta_before[FEE_ACCOUNT_ID],
        )
    {
        return false;
    }

    let mut right_for_left = [usize::MAX; 2];
    for position in 0..2 {
        right_for_left[left_slots[position]] = right_slots[position];
    }
    for left_slot in 0..2 {
        let right_slot = right_for_left[left_slot];
        if right_slot >= 2
            || left.accounts_before[left_slot].account_index
                != right.accounts_before[right_slot].account_index
            || !verify_account_inner_trees(left, right, left_slot, right_slot)
        {
            return false;
        }
    }
    if !verify_account_inner_trees(left, right, FEE_ACCOUNT_ID, FEE_ACCOUNT_ID) {
        return false;
    }

    let Some(indices) = left
        .accounts_before
        .iter()
        .map(|account| {
            as_merkle_index(
                account.account_index,
                left.account_tree_merkle_proofs[0].len(),
            )
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|indices| indices.try_into().ok())
    else {
        return false;
    };

    let Some(old_user_hashes) = (0..2)
        .map(|slot| account_hash(&left.accounts_before[slot]))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(new_user_hashes) = (0..2)
        .map(|slot| account_hash(&right.accounts_before[right_for_left[slot]]))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(old_fee_hash) = fee_account_hash(&left.accounts_before[FEE_ACCOUNT_ID]) else {
        return false;
    };
    let Some(new_fee_hash) = fee_account_hash(&right.accounts_before[FEE_ACCOUNT_ID]) else {
        return false;
    };

    let old_full = [old_user_hashes[0].0, old_user_hashes[1].0, old_fee_hash.0];
    let new_full = [new_user_hashes[0].0, new_user_hashes[1].0, new_fee_hash.0];
    if !verify_three_leaf_transition(
        indices,
        old_full,
        new_full,
        [
            &left.account_tree_merkle_proofs[0],
            &left.account_tree_merkle_proofs[1],
            &left.account_tree_merkle_proofs[FEE_ACCOUNT_ID],
        ],
        left.old_account_tree_root,
        right.old_account_tree_root,
    ) {
        return false;
    }

    let old_public = [old_user_hashes[0].1, old_user_hashes[1].1, old_fee_hash.1];
    let new_public = [new_user_hashes[0].1, new_user_hashes[1].1, new_fee_hash.1];
    if !verify_three_leaf_transition(
        indices,
        old_public,
        new_public,
        [
            &left.account_pub_data_tree_merkle_proofs[0],
            &left.account_pub_data_tree_merkle_proofs[1],
            &left.account_pub_data_tree_merkle_proofs[FEE_ACCOUNT_ID],
        ],
        left.old_account_pub_data_tree_root,
        right.old_account_pub_data_tree_root,
    ) {
        return false;
    }

    let Some(old_delta_0) = account_delta_hash(&left.accounts_delta_before[0]) else {
        return false;
    };
    let Some(old_delta_1) = account_delta_hash(&left.accounts_delta_before[1]) else {
        return false;
    };
    let Some(new_delta_0) = account_delta_hash(&right.accounts_delta_before[right_for_left[0]])
    else {
        return false;
    };
    let Some(new_delta_1) = account_delta_hash(&right.accounts_delta_before[right_for_left[1]])
    else {
        return false;
    };
    let Some(old_fee_delta) = fee_account_delta_hash(&left.accounts_delta_before[FEE_ACCOUNT_ID])
    else {
        return false;
    };
    let Some(new_fee_delta) = fee_account_delta_hash(&right.accounts_delta_before[FEE_ACCOUNT_ID])
    else {
        return false;
    };
    verify_three_leaf_transition(
        indices,
        [old_delta_0, old_delta_1, old_fee_delta],
        [new_delta_0, new_delta_1, new_fee_delta],
        [
            &left.account_delta_tree_merkle_proofs[0],
            &left.account_delta_tree_merkle_proofs[1],
            &left.account_delta_tree_merkle_proofs[FEE_ACCOUNT_ID],
        ],
        left.old_account_delta_tree_root,
        right.old_account_delta_tree_root,
    )
}

fn common_user_accounts(left: &Tx<F>, right: &Tx<F>) -> Vec<i64> {
    let left_indices = left
        .accounts_before
        .iter()
        .map(|account| account.account_index)
        .filter(|&index| index != NIL_ACCOUNT_INDEX && index != TREASURY_ACCOUNT_INDEX as i64)
        .collect::<BTreeSet<_>>();
    let right_indices = right
        .accounts_before
        .iter()
        .map(|account| account.account_index)
        .collect::<BTreeSet<_>>();
    left_indices.intersection(&right_indices).copied().collect()
}

/// Returns swap parameters plus the account-array slots at the left and right
/// boundaries, in the order supplied by `account_indices`.
fn classify_endpoint(
    left: &Tx<F>,
    right: &Tx<F>,
    account_indices: &[i64],
) -> Option<(BilateralSwap, [usize; 2], [usize; 2])> {
    let left_slots = [
        account_slot(left, account_indices[0])?,
        account_slot(left, account_indices[1])?,
    ];
    let right_slots = [
        account_slot(right, account_indices[0])?,
        account_slot(right, account_indices[1])?,
    ];

    let aggregate_deltas = [
        account_aggregate_delta(
            &left.accounts_before[left_slots[0]],
            &right.accounts_before[right_slots[0]],
        ),
        account_aggregate_delta(
            &left.accounts_before[left_slots[1]],
            &right.accounts_before[right_slots[1]],
        ),
    ];
    let swap = classify_bilateral(&aggregate_deltas[0], &aggregate_deltas[1])?;

    for account_position in 0..2 {
        let left_slot = left_slots[account_position];
        let right_slot = right_slots[account_position];
        let expected = &aggregate_deltas[account_position];

        if normalized_account_debug(&left.accounts_before[left_slot])
            != normalized_account_debug(&right.accounts_before[right_slot])
        {
            return None;
        }

        let delta_before = &left.accounts_delta_before[left_slot];
        let delta_after = &right.accounts_delta_before[right_slot];
        if delta_before.account_index != account_indices[account_position]
            || delta_after.account_index != account_indices[account_position]
            || normalized_account_delta_debug(delta_before)
                != normalized_account_delta_debug(delta_after)
        {
            return None;
        }
        let observed_delta = [
            &delta_after.aggregated_asset_deltas[0] - &delta_before.aggregated_asset_deltas[0],
            &delta_after.aggregated_asset_deltas[1] - &delta_before.aggregated_asset_deltas[1],
        ];
        if &observed_delta != expected {
            return None;
        }

        for asset_slot in 0..2 {
            let before = &left.account_assets_before[left_slot][asset_slot];
            let after = &right.account_assets_before[right_slot][asset_slot];
            let asset_index = left.asset_indices[asset_slot];
            if before.index_0 != asset_index as i64
                || after.index_0 != asset_index as i64
                || before.locked_balance != after.locked_balance
            {
                return None;
            }
            let multiplier = left
                .all_assets_before
                .get(usize::try_from(asset_index).ok()?)
                .filter(|asset| asset.asset_index == asset_index)?
                .extension_multiplier;
            if multiplier <= 0 {
                return None;
            }
            let actual_delta = biguint_delta(&before.balance, &after.balance);
            if actual_delta != &expected[asset_slot] * BigInt::from(multiplier) {
                return None;
            }
        }
    }

    Some((swap, left_slots, right_slots))
}

fn account_slot(tx: &Tx<F>, account_index: i64) -> Option<usize> {
    let mut slots = tx
        .accounts_before
        .iter()
        .enumerate()
        .filter(|(_, account)| account.account_index == account_index)
        .map(|(slot, _)| slot);
    let slot = slots.next()?;
    slots.next().is_none().then_some(slot)
}

fn account_aggregate_delta(before: &Account<F>, after: &Account<F>) -> [BigInt; 2] {
    [
        &after.aggregated_balances[0] - &before.aggregated_balances[0],
        &after.aggregated_balances[1] - &before.aggregated_balances[1],
    ]
}

fn normalized_account_debug(account: &Account<F>) -> String {
    let mut normalized = account.clone();
    normalized.aggregated_balances = [BigInt::zero(), BigInt::zero()];
    normalized.aggregated_balances_root = HashOut::ZERO;
    normalized.asset_root = HashOut::ZERO;
    format!("{normalized:?}")
}

fn normalized_account_delta_debug(delta: &AccountDelta<F>) -> String {
    let mut normalized = delta.clone();
    normalized.aggregated_asset_deltas = [BigInt::zero(), BigInt::zero()];
    normalized.asset_delta_root = HashOut::ZERO;
    format!("{normalized:?}")
}

fn biguint_delta(before: &BigUint, after: &BigUint) -> BigInt {
    BigInt::from(after.clone()) - BigInt::from(before.clone())
}

/// Classify two opposite normalized two-asset deltas as an exact integer-price
/// spot swap. Either asset slot may be the base asset.
pub fn classify_bilateral(first: &[BigInt; 2], second: &[BigInt; 2]) -> Option<BilateralSwap> {
    if first[0] != -&second[0] || first[1] != -&second[1] {
        return None;
    }

    for (seller_slot, seller) in [(0, first), (1, second)] {
        for base_asset_slot in 0..2 {
            let quote_asset_slot = 1 - base_asset_slot;
            if seller[base_asset_slot] >= BigInt::zero()
                || seller[quote_asset_slot] <= BigInt::zero()
            {
                continue;
            }
            let base_amount = -&seller[base_asset_slot];
            let quote_amount = &seller[quote_asset_slot];
            if quote_amount % &base_amount != BigInt::zero() {
                continue;
            }
            let price = quote_amount / &base_amount;
            let (Some(base_amount), Some(price)) = (base_amount.to_i64(), price.to_i64()) else {
                continue;
            };
            if base_amount <= 0 || price <= 0 {
                continue;
            }
            return Some(BilateralSwap {
                seller_slot,
                buyer_slot: 1 - seller_slot,
                base_asset_slot,
                quote_asset_slot,
                base_amount,
                price,
            });
        }
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;

    use circuit::poseidon2::Poseidon2Hash;
    use circuit::types::account_asset::AccountAsset;
    use circuit::types::asset::Asset;
    use circuit::types::constants::{
        ACCOUNT_MERKLE_LEVELS, ASSET_MERKLE_LEVELS, EMPTY_ASSET_TREE_ROOT,
        EMPTY_POSITION_DELTA_TREE_ROOT, TX_HEAVY,
    };
    use circuit::types::state_metadata::StateMetadata;
    use plonky2::field::types::Field;
    use plonky2::plonk::config::Hasher;

    use super::*;
    use crate::phantom_materialize::materialize_scanned_candidate;
    use crate::phantom_native::{
        all_assets_hash, all_margined_assets_hash, all_market_details_hashes,
        register_stack_hash, system_config_hash, validium_and_state_root,
    };
    use crate::phantom_rechunk::{
        InclusiveInterval, ReplacementPair, prepare_verified_pair,
    };

    fn sparse_proofs<const L: usize>(
        indices: &[u64],
        leaves: &[HashOut<F>],
    ) -> (Vec<[HashOut<F>; L]>, HashOut<F>) {
        assert_eq!(indices.len(), leaves.len());
        let mut empty_roots = Vec::with_capacity(L + 1);
        empty_roots.push(HashOut::ZERO);
        for level in 0..L {
            empty_roots.push(Poseidon2Hash::two_to_one(
                empty_roots[level],
                empty_roots[level],
            ));
        }

        let mut levels = vec![HashMap::<u64, HashOut<F>>::new(); L + 1];
        for (&index, &leaf) in indices.iter().zip(leaves) {
            assert!(index < (1_u64 << L));
            if leaf != empty_roots[0] {
                levels[0].insert(index, leaf);
            }
        }
        for level in 0..L {
            let parents = levels[level]
                .keys()
                .map(|index| index >> 1)
                .collect::<BTreeSet<_>>();
            for parent in parents {
                let left = levels[level]
                    .get(&(parent << 1))
                    .copied()
                    .unwrap_or(empty_roots[level]);
                let right = levels[level]
                    .get(&((parent << 1) | 1))
                    .copied()
                    .unwrap_or(empty_roots[level]);
                let hash = Poseidon2Hash::two_to_one(left, right);
                if hash != empty_roots[level + 1] {
                    levels[level + 1].insert(parent, hash);
                }
            }
        }

        let proofs = indices
            .iter()
            .map(|index| {
                core::array::from_fn(|level| {
                    levels[level]
                        .get(&(((index >> level) ^ 1) as u64))
                        .copied()
                        .unwrap_or(empty_roots[level])
                })
            })
            .collect::<Vec<[HashOut<F>; L]>>();
        let root = levels[L].get(&0).copied().unwrap_or(empty_roots[L]);
        for ((index, leaf), proof) in indices.iter().zip(leaves).zip(&proofs) {
            assert_eq!(merkle_root(*leaf, *index, proof), root);
        }
        (proofs, root)
    }

    fn account(index: i64, balances: [i64; 2]) -> Account<F> {
        Account {
            account_index: index,
            aggregated_balances: balances.map(BigInt::from),
            api_key_root: EMPTY_API_KEY_TREE_ROOT,
            account_orders_root: EMPTY_ACCOUNT_ORDERS_TREE_ROOT,
            aggregated_balances_root: EMPTY_ASSET_TREE_ROOT,
            asset_root: EMPTY_ASSET_TREE_ROOT,
            ..Account::default()
        }
    }

    fn account_delta(index: i64, deltas: [i64; 2], fee: bool) -> AccountDelta<F> {
        AccountDelta {
            account_index: index,
            aggregated_asset_deltas: deltas.map(BigInt::from),
            asset_delta_root: EMPTY_ASSET_TREE_ROOT,
            position_delta_root: if fee {
                HashOut::ZERO
            } else {
                EMPTY_POSITION_DELTA_TREE_ROOT
            },
            ..AccountDelta::default()
        }
    }

    fn account_assets(balances: [u64; 2]) -> [AccountAsset; 2] {
        [
            AccountAsset {
                index_0: 1,
                balance: BigUint::from(balances[0]),
                locked_balance: BigUint::zero(),
            },
            AccountAsset {
                index_0: 2,
                balance: BigUint::from(balances[1]),
                locked_balance: BigUint::zero(),
            },
        ]
    }

    fn install_inner_trees(left: &mut Tx<F>, right: &mut Tx<F>, slot: usize) {
        let indices = [1_u64, 2_u64];

        let old_assets = left.account_assets_before[slot]
            .iter()
            .map(|asset| account_asset_hash(asset).unwrap())
            .collect::<Vec<_>>();
        let new_assets = right.account_assets_before[slot]
            .iter()
            .map(|asset| account_asset_hash(asset).unwrap())
            .collect::<Vec<_>>();
        let (old_proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_assets);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_assets);
        left.asset_tree_merkle_proofs[slot] = old_proofs.try_into().unwrap();
        left.accounts_before[slot].asset_root = old_root;
        right.accounts_before[slot].asset_root = new_root;
        assert!(
            verify_two_leaf_transition(
                indices,
                old_assets.clone().try_into().unwrap(),
                new_assets.clone().try_into().unwrap(),
                &left.asset_tree_merkle_proofs[slot][0],
                &left.asset_tree_merkle_proofs[slot][1],
                old_root,
                new_root,
            ),
            "asset transition for slot {slot}"
        );

        let old_balances = left.accounts_before[slot]
            .aggregated_balances
            .iter()
            .map(|balance| bigint_leaf_hash(balance).unwrap())
            .collect::<Vec<_>>();
        let new_balances = right.accounts_before[slot]
            .aggregated_balances
            .iter()
            .map(|balance| bigint_leaf_hash(balance).unwrap())
            .collect::<Vec<_>>();
        let (old_proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_balances);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_balances);
        left.public_asset_tree_merkle_proofs[slot] = old_proofs.try_into().unwrap();
        left.accounts_before[slot].aggregated_balances_root = old_root;
        right.accounts_before[slot].aggregated_balances_root = new_root;
        assert!(
            verify_two_leaf_transition(
                indices,
                old_balances.clone().try_into().unwrap(),
                new_balances.clone().try_into().unwrap(),
                &left.public_asset_tree_merkle_proofs[slot][0],
                &left.public_asset_tree_merkle_proofs[slot][1],
                old_root,
                new_root,
            ),
            "balance transition for slot {slot}"
        );

        let old_deltas = left.accounts_delta_before[slot]
            .aggregated_asset_deltas
            .iter()
            .map(|delta| bigint_leaf_hash(delta).unwrap())
            .collect::<Vec<_>>();
        let new_deltas = right.accounts_delta_before[slot]
            .aggregated_asset_deltas
            .iter()
            .map(|delta| bigint_leaf_hash(delta).unwrap())
            .collect::<Vec<_>>();
        let (old_proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_deltas);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_deltas);
        left.asset_delta_tree_merkle_proofs[slot] = old_proofs.try_into().unwrap();
        left.accounts_delta_before[slot].asset_delta_root = old_root;
        right.accounts_delta_before[slot].asset_delta_root = new_root;
        assert!(
            verify_two_leaf_transition(
                indices,
                old_deltas.try_into().unwrap(),
                new_deltas.try_into().unwrap(),
                &left.asset_delta_tree_merkle_proofs[slot][0],
                &left.asset_delta_tree_merkle_proofs[slot][1],
                old_root,
                new_root,
            ),
            "delta transition for slot {slot}"
        );
    }

    fn install_outer_trees(left: &mut Tx<F>, right: &mut Tx<F>) {
        let indices = [17_u64, 23_u64, TREASURY_ACCOUNT_INDEX as u64];

        let old_users = [
            account_hash(&left.accounts_before[0]).unwrap(),
            account_hash(&left.accounts_before[1]).unwrap(),
        ];
        let new_users = [
            account_hash(&right.accounts_before[0]).unwrap(),
            account_hash(&right.accounts_before[1]).unwrap(),
        ];
        let old_fee = fee_account_hash(&left.accounts_before[2]).unwrap();
        let new_fee = fee_account_hash(&right.accounts_before[2]).unwrap();

        let old_full = [old_users[0].0, old_users[1].0, old_fee.0];
        let new_full = [new_users[0].0, new_users[1].0, new_fee.0];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &old_full);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &new_full);
        left.account_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_tree_root = root;
        right.old_account_tree_root = new_root;

        let old_public = [old_users[0].1, old_users[1].1, old_fee.1];
        let new_public = [new_users[0].1, new_users[1].1, new_fee.1];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &old_public);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &new_public);
        left.account_pub_data_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_pub_data_tree_root = root;
        right.old_account_pub_data_tree_root = new_root;

        let old_delta = [
            account_delta_hash(&left.accounts_delta_before[0]).unwrap(),
            account_delta_hash(&left.accounts_delta_before[1]).unwrap(),
            fee_account_delta_hash(&left.accounts_delta_before[2]).unwrap(),
        ];
        let new_delta = [
            account_delta_hash(&right.accounts_delta_before[0]).unwrap(),
            account_delta_hash(&right.accounts_delta_before[1]).unwrap(),
            fee_account_delta_hash(&right.accounts_delta_before[2]).unwrap(),
        ];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &old_delta);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&indices, &new_delta);
        left.account_delta_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_delta_tree_root = root;
        right.old_account_delta_tree_root = new_root;
    }

    fn install_global_roots(tx: &mut Tx<F>, metadata: HashOut<F>) {
        let (market_risk, public_market) =
            all_market_details_hashes(&tx.all_market_risk_details_before).unwrap();
        let (validium, state) = validium_and_state_root(
            system_config_hash(&tx.system_config_before).unwrap(),
            all_assets_hash(&tx.all_assets_before).unwrap(),
            all_margined_assets_hash(&tx.all_margined_assets_before).unwrap(),
            market_risk,
            public_market,
            register_stack_hash(&tx.register_stack_before),
            tx.old_account_tree_root,
            tx.old_account_pub_data_tree_root,
            tx.old_market_details_tree_root,
            tx.old_market_tree_root,
            metadata,
        )
        .unwrap();
        tx.old_validium_root = validium;
        tx.old_state_root = state;
    }

    pub(crate) fn synthetic_candidate_block() -> Block<F> {
        let mut block = Block::<F>::from_json(include_bytes!("../bench_test.json"), 4, 10)
            .expect("public fixture must parse");
        let padding = block
            .tx_chunks
            .iter()
            .flatten()
            .find(|tx| tx.is_empty())
            .unwrap()
            .clone();
        let mut left = padding.as_ref().clone();
        let mut right = padding.as_ref().clone();

        left.asset_indices = [1, 2];
        right.asset_indices = [1, 2];
        for tx in [&mut left, &mut right] {
            tx.all_assets_before[1] = Asset {
                asset_index: 1,
                extension_multiplier: 1,
                ..Asset::default()
            };
            tx.all_assets_before[2] = Asset {
                asset_index: 2,
                extension_multiplier: 1,
                ..Asset::default()
            };
        }

        left.accounts_before = [
            account(17, [100, 0]),
            account(23, [0, 100]),
            account(TREASURY_ACCOUNT_INDEX as i64, [0, 0]),
        ];
        right.accounts_before = [
            account(17, [90, 70]),
            account(23, [10, 30]),
            account(TREASURY_ACCOUNT_INDEX as i64, [0, 0]),
        ];
        left.account_assets_before = [
            account_assets([100, 0]),
            account_assets([0, 100]),
            account_assets([0, 0]),
        ];
        right.account_assets_before = [
            account_assets([90, 70]),
            account_assets([10, 30]),
            account_assets([0, 0]),
        ];
        left.accounts_delta_before = [
            account_delta(17, [0, 0], false),
            account_delta(23, [0, 0], false),
            account_delta(TREASURY_ACCOUNT_INDEX as i64, [0, 0], true),
        ];
        right.accounts_delta_before = [
            account_delta(17, [-10, 70], false),
            account_delta(23, [10, -70], false),
            account_delta(TREASURY_ACCOUNT_INDEX as i64, [0, 0], true),
        ];

        for slot in 0..3 {
            install_inner_trees(&mut left, &mut right, slot);
        }
        install_outer_trees(&mut left, &mut right);
        let metadata = StateMetadata::empty().hash();
        install_global_roots(&mut left, metadata);
        install_global_roots(&mut right, metadata);

        let mut active = Vec::new();
        for tx_index in 0..5 {
            let mut tx = if tx_index < 3 {
                left.clone()
            } else {
                right.clone()
            };
            tx.tx_index = tx_index;
            tx.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
            tx.tx_circuit_type = TX_HEAVY;
            active.push(Arc::new(tx));
        }
        active.push(padding);
        block.tx_chunks = vec![active];
        block
    }

    fn with_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(test)
            .expect("scanner test thread must start")
            .join()
            .expect("scanner test thread must finish");
    }

    #[test]
    fn exact_bilateral_swap_is_classified() {
        let maker = [BigInt::from(-10), BigInt::from(70)];
        let taker = [BigInt::from(10), BigInt::from(-70)];
        assert_eq!(
            classify_bilateral(&maker, &taker),
            Some(BilateralSwap {
                seller_slot: 0,
                buyer_slot: 1,
                base_asset_slot: 0,
                quote_asset_slot: 1,
                base_amount: 10,
                price: 7,
            })
        );
    }

    #[test]
    fn imbalanced_or_fractional_price_is_rejected() {
        assert!(
            classify_bilateral(
                &[BigInt::from(-10), BigInt::from(71)],
                &[BigInt::from(10), BigInt::from(-71)]
            )
            .is_none()
        );
        assert!(
            classify_bilateral(
                &[BigInt::from(-10), BigInt::from(70)],
                &[BigInt::from(9), BigInt::from(-70)]
            )
            .is_none()
        );
    }

    #[test]
    fn protected_lane_count_gate_requires_a_real_chunk_saving() {
        assert_eq!(lane_proof_saving(10, 490, 3, 1), Some(1));
        assert_eq!(lane_proof_saving(10, 490, 2, 2), None);
        assert_eq!(lane_proof_saving(10, 490, 4, 0), None);
        assert_eq!(lane_proof_saving(10, 490, 10, 490), Some(50));
    }

    #[test]
    fn donors_fail_closed_and_order_path_requires_the_empty_tree() {
        let mut account = Account::<F>::default();
        account.account_orders_root = HashOut::ZERO;
        assert!(!can_synthesize_phantom_order_path(&account));
        account.account_orders_root = EMPTY_ACCOUNT_ORDERS_TREE_ROOT;
        assert!(can_synthesize_phantom_order_path(&account));

        let mut candidate = Candidate {
            start_tx_index: 0,
            end_tx_index: 4,
            replaced_tx_count: 4,
            replaced_light_count: 1,
            replaced_heavy_count: 3,
            saved_lane_proof_count: 1,
            seller_account_index: 17,
            buyer_account_index: 23,
            base_asset_index: 1,
            quote_asset_index: 2,
            base_amount: 10,
            price: 7,
            maker_account_orders_path_at_phantom_key: true,
            maker_api_path_from_padding: true,
            taker_api_path_from_padding: true,
            fee_paths_natively_verified: true,
            nil_market_path_from_final_padding: false,
            nil_market_details_path_from_final_padding: false,
        };
        assert!(!candidate.has_direct_v1_donors());
        candidate.nil_market_path_from_final_padding = true;
        assert!(!candidate.has_direct_v1_donors());
        candidate.nil_market_details_path_from_final_padding = true;
        assert!(candidate.has_direct_v1_donors());
    }

    #[test]
    fn fully_reconstructed_synthetic_interval_is_discovered() {
        with_large_stack(|| {
            let block = synthetic_candidate_block();
            let left = block.tx_chunks[0][0].as_ref();
            let right = block.tx_chunks[0][3].as_ref();
            let padding = block.tx_chunks[0].last().unwrap().as_ref();
            assert!(cheap_global_endpoint_filter(left, right), "cheap filter");
            let accounts = common_user_accounts(left, right);
            assert_eq!(accounts.len(), 2, "common accounts");
            let (swap, left_slots, right_slots) =
                classify_endpoint(left, right, &accounts).expect("endpoint classification");
            assert!(
                verify_account_inner_trees(left, right, 0, 0),
                "slot-0 inner trees"
            );
            assert!(
                verify_account_inner_trees(left, right, 1, 1),
                "slot-1 inner trees"
            );
            assert!(
                verify_account_inner_trees(left, right, 2, 2),
                "Treasury inner trees"
            );
            assert!(
                verify_all_affected_roots(left, right, left_slots, right_slots),
                "native root replay"
            );
            assert!(strict_global_endpoint_filter(left, right), "strict filter");
            assert!(
                claim_roles_are_materializable(
                    &left.accounts_before[left_slots[swap.seller_slot]],
                    &left.accounts_before[left_slots[swap.buyer_slot]],
                ),
                "role eligibility"
            );
            assert!(
                materialization_arithmetic_is_safe(block.created_at, left, &swap, left_slots,),
                "materialization arithmetic"
            );
            assert_eq!(
                direct_nil_padding_paths(left, padding),
                (true, true),
                "padding donors"
            );

            let report = scan(&block);
            assert!(!report.candidates.is_empty());
            assert!(
                report
                    .candidates
                    .iter()
                    .all(Candidate::has_direct_v1_donors)
            );
        });
    }

    #[test]
    fn scanned_candidate_materializes_and_prepares_an_exact_group_saving() {
        with_large_stack(|| {
            let mut block = synthetic_candidate_block();
            let metadata = StateMetadata::empty().hash();
            let candidate = scan(&block)
                .candidates
                .into_iter()
                .max_by_key(|candidate| candidate.replaced_tx_count)
                .expect("synthetic block must expose a materializable interval");
            let right = block.tx_chunks[0]
                .iter()
                .find(|tx| tx.tx_index == candidate.end_tx_index)
                .unwrap()
                .clone();

            let materialized = materialize_scanned_candidate(&block, &candidate, metadata)
                .expect("scanner-certified interval must materialize");
            assert_eq!(
                materialized.final_validium_root,
                right.old_validium_root
            );
            assert_eq!(materialized.final_state_root, right.old_state_root);
            assert_eq!(
                materialized.final_delta_root,
                right.old_account_delta_tree_root
            );

            let last_removed = candidate.end_tx_index.checked_sub(1).unwrap();
            let prepared = prepare_verified_pair(
                &block,
                InclusiveInterval::new(candidate.start_tx_index, last_removed),
                ReplacementPair::new(materialized.light_insert, materialized.heavy_fill),
            )
            .expect("verified pair must prepare transactionally");
            let report = prepared.report();
            assert_eq!(report.removed_tx_count, candidate.replaced_tx_count);
            assert_eq!(report.saved_group_count, 1);
            assert_eq!(report.groups_before.total(), 3);
            assert_eq!(report.groups_after.total(), 2);

            prepared.commit(&mut block);
            let mut active = block
                .tx_chunks
                .iter()
                .flatten()
                .filter(|tx| !tx.is_empty())
                .collect::<Vec<_>>();
            active.sort_unstable_by_key(|tx| tx.tx_index);
            assert!(
                (0_u64..).zip(active).all(|(expected, tx)| tx.tx_index == expected),
                "committed execution indices must remain contiguous"
            );
        });
    }

    #[test]
    fn fee_market_details_and_outer_root_mismatches_return_no_candidate() {
        with_large_stack(|| {
            let mut fee_mismatch = synthetic_candidate_block();
            for index in [3, 4] {
                Arc::make_mut(&mut fee_mismatch.tx_chunks[0][index]).accounts_before
                    [FEE_ACCOUNT_ID]
                    .total_order_count = 1;
            }
            assert!(scan(&fee_mismatch).candidates.is_empty());

            let mut details_mismatch = synthetic_candidate_block();
            let padding_index = details_mismatch.tx_chunks[0].len() - 1;
            Arc::make_mut(&mut details_mismatch.tx_chunks[0][padding_index])
                .old_market_details_tree_root
                .elements[0] += F::ONE;
            assert!(scan(&details_mismatch).candidates.is_empty());

            let mut root_mismatch = synthetic_candidate_block();
            for index in [3, 4] {
                Arc::make_mut(&mut root_mismatch.tx_chunks[0][index])
                    .old_account_tree_root
                    .elements[0] += F::ONE;
            }
            assert!(scan(&root_mismatch).candidates.is_empty());

            // The circuit random-accesses the fixed asset array by asset index;
            // the informational `asset_index` tag cannot be used as a lookup.
            let mut shuffled_asset_array = synthetic_candidate_block();
            for tx in shuffled_asset_array.tx_chunks[0]
                .iter_mut()
                .filter(|tx| !tx.is_empty())
            {
                let tx = Arc::make_mut(tx);
                tx.all_assets_before[1].asset_index = 2;
                tx.all_assets_before[2].asset_index = 1;
            }
            assert!(scan(&shuffled_asset_array).candidates.is_empty());

            let mut overflowing_indices = synthetic_candidate_block();
            for index in [3, 4] {
                Arc::make_mut(&mut overflowing_indices.tx_chunks[0][index]).tx_index = u64::MAX;
            }
            assert!(scan(&overflowing_indices).candidates.is_empty());
        });
    }

    #[test]
    fn public_padding_fixture_has_no_candidate() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let block = Block::<F>::from_json(include_bytes!("../bench_test.json"), 4, 10)
                    .expect("public fixture must parse");
                let report = scan(&block);
                assert_eq!(report.active_tx_count, 0);
                assert!(report.candidates.is_empty());
            })
            .expect("scanner test thread must start")
            .join()
            .expect("scanner test thread must finish");
    }
}
