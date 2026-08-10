// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Native mirrors of the leaf hashes needed by the phantom-session scanner.
//! Keep these formulas byte-for-byte ordered like their circuit counterparts;
//! every caller treats `None` as a hard rejection.

use circuit::poseidon2::Poseidon2Hash;
use circuit::types::account::Account;
use circuit::types::account_asset::AccountAsset;
use circuit::types::account_delta::AccountDelta;
use circuit::types::account_order::AccountOrder;
use circuit::types::account_position::AccountPosition;
use circuit::types::asset::Asset;
use circuit::types::config::F;
use circuit::types::constants::{
    ASSET_LIST_SIZE, EMPTY_ACCOUNT_HASH, EMPTY_ASSET_TREE_ROOT, EMPTY_POSITION_DELTA_TREE_ROOT,
    MARGINED_ASSET_LIST_SIZE, MAX_ASSET_INDEX, MIN_ASSET_INDEX, POSITION_HASH_BUCKET_COUNT,
    POSITION_HASH_BUCKET_SIZE, POSITION_LIST_SIZE, REGISTER_STACK_SIZE, TREASURY_ACCOUNT_INDEX,
};
use circuit::types::margined_asset::MarginedAsset;
use circuit::types::market::Market;
use circuit::types::market_details::{MarketDetails, MarketRiskDetails};
use circuit::types::register::{BaseRegisterInfo, RegisterStack};
use circuit::types::system_config::SystemConfig;
use num::bigint::Sign;
use num::{BigInt, BigUint, Signed, Zero};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::config::Hasher;

const U96_LIMBS: usize = 3;
const U160_LIMBS: usize = 5;
const U16_U64_LIMBS: usize = 4;

fn hash(elements: Vec<F>) -> HashOut<F> {
    Poseidon2Hash::hash_no_pad(&elements)
}

fn field_i64(value: i64) -> F {
    F::from_noncanonical_i64(value)
}

fn field_sign(value: &BigInt) -> F {
    match value.sign() {
        Sign::Minus => F::NEG_ONE,
        Sign::NoSign => F::ZERO,
        Sign::Plus => F::ONE,
    }
}

fn u32_limbs(value: &BigUint, count: usize) -> Option<Vec<F>> {
    let words = value.to_u32_digits();
    if words.len() > count {
        return None;
    }
    Some(
        (0..count)
            .map(|index| F::from_canonical_u32(words.get(index).copied().unwrap_or(0)))
            .collect(),
    )
}

fn u16_limbs(value: &BigUint, count: usize) -> Option<Vec<F>> {
    let words = value.to_u32_digits();
    let mut limbs = Vec::with_capacity(words.len() * 2);
    for word in words {
        limbs.push((word & 0xffff) as u16);
        limbs.push((word >> 16) as u16);
    }
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    if limbs.len() > count {
        return None;
    }
    Some(
        (0..count)
            .map(|index| F::from_canonical_u16(limbs.get(index).copied().unwrap_or(0)))
            .collect(),
    )
}

fn nonnegative_i64_u32_limbs(value: i64, count: usize) -> Option<Vec<F>> {
    u32_limbs(&BigUint::from(u64::try_from(value).ok()?), count)
}

fn bigint_u32_parts(value: &BigInt, count: usize) -> Option<(Vec<F>, F)> {
    Some((
        u32_limbs(&value.abs().to_biguint()?, count)?,
        field_sign(value),
    ))
}

fn bigint_u16_parts(value: &BigInt, count: usize) -> Option<(Vec<F>, F)> {
    Some((
        u16_limbs(&value.abs().to_biguint()?, count)?,
        field_sign(value),
    ))
}

pub fn account_asset_hash(asset: &AccountAsset) -> Option<HashOut<F>> {
    if asset.balance.is_zero() && asset.locked_balance.is_zero() {
        return Some(HashOut::ZERO);
    }
    let mut elements = u32_limbs(&asset.balance, U96_LIMBS)?;
    elements.extend(u32_limbs(&asset.locked_balance, U96_LIMBS)?);
    Some(hash(elements))
}

pub fn bigint_leaf_hash(value: &BigInt) -> Option<HashOut<F>> {
    if value.is_zero() {
        return Some(HashOut::ZERO);
    }
    let (limbs, sign) = bigint_u32_parts(value, U96_LIMBS)?;
    let mut elements = vec![sign];
    elements.extend(limbs);
    Some(hash(elements))
}

fn position_bucket_hash(bucket: &[AccountPosition]) -> Option<(HashOut<F>, HashOut<F>)> {
    let mut public_elements = Vec::new();
    for position in bucket {
        let (limbs, sign) =
            bigint_u16_parts(&position.last_funding_rate_prefix_sum, U16_U64_LIMBS)?;
        public_elements.extend(limbs);
        public_elements.push(sign);
        let (limbs, sign) = bigint_u16_parts(&position.position, U16_U64_LIMBS)?;
        public_elements.extend(limbs);
        public_elements.push(sign);
    }
    let public_hash = hash(public_elements);

    let mut full_elements = public_hash.elements.to_vec();
    for position in bucket {
        let (limbs, sign) = bigint_u32_parts(&position.allocated_margin, U96_LIMBS)?;
        full_elements.extend(limbs);
        full_elements.extend([
            sign,
            F::from_canonical_u8(position.margin_mode),
            field_i64(position.entry_quote),
            F::from_canonical_u16(position.initial_margin_fraction),
            field_i64(position.total_order_count),
            field_i64(position.total_position_tied_order_count),
            F::from_canonical_u8(position.margin_set_flag),
        ]);
    }
    Some((hash(full_elements), public_hash))
}

fn calculated_account_partials(account: &Account<F>) -> Option<(HashOut<F>, HashOut<F>)> {
    let mut positions = account.positions.to_vec();
    positions.push(AccountPosition::default());
    let buckets = positions
        .chunks(POSITION_HASH_BUCKET_SIZE)
        .map(position_bucket_hash)
        .collect::<Option<Vec<_>>>()?;
    if buckets.len() != POSITION_HASH_BUCKET_COUNT {
        return None;
    }

    let mut public_elements = buckets
        .iter()
        .flat_map(|(_, public)| public.elements)
        .collect::<Vec<_>>();
    for share in &account.public_pool_shares {
        public_elements.extend([
            field_i64(share.public_pool_index),
            field_i64(share.share_amount),
        ]);
    }
    public_elements.extend([
        field_i64(account.public_pool_info.total_shares),
        field_i64(account.public_pool_info.operator_shares),
    ]);
    let public_hash = hash(public_elements);

    let mut full_elements = public_hash.elements.to_vec();
    full_elements.extend(buckets.iter().flat_map(|(full, _)| full.elements));
    for share in &account.public_pool_shares {
        full_elements.extend([
            field_i64(share.principal_amount),
            field_i64(share.entry_timestamp),
        ]);
    }
    full_elements.extend([
        F::from_canonical_u8(account.public_pool_info.status),
        field_i64(account.public_pool_info.min_operator_share_rate),
        field_i64(account.public_pool_info.operator_fee),
    ]);

    let mut pending_elements = Vec::new();
    for pending in &account.pending_unlocks {
        pending_elements.extend([
            field_i64(pending.unlock_timestamp),
            field_i64(pending.asset_index),
        ]);
        pending_elements.extend(u32_limbs(&pending.amount, U96_LIMBS)?);
    }
    full_elements.extend(hash(pending_elements).elements);

    let mut integrator_elements = Vec::new();
    for integrator in &account.approved_integrators {
        integrator_elements.extend([
            field_i64(integrator.integrator_account_index),
            F::from_canonical_u32(integrator.max_perps_taker_fee),
            F::from_canonical_u32(integrator.max_perps_maker_fee),
            F::from_canonical_u32(integrator.max_spot_taker_fee),
            F::from_canonical_u32(integrator.max_spot_maker_fee),
            field_i64(integrator.expiry),
        ]);
    }
    full_elements.extend(hash(integrator_elements).elements);
    Some((hash(full_elements), public_hash))
}

fn account_hash_from_partials(
    account: &Account<F>,
    partial: HashOut<F>,
    partial_public: HashOut<F>,
) -> Option<(HashOut<F>, HashOut<F>)> {
    let l1_limbs = u32_limbs(&account.l1_address, U160_LIMBS)?;
    let mut public_elements = partial_public.elements.to_vec();
    public_elements.extend(l1_limbs.iter().copied());
    public_elements.push(F::from_canonical_u8(account.account_type));
    public_elements.extend(account.aggregated_balances_root.elements);

    let mut full_elements = partial.elements.to_vec();
    full_elements.push(field_i64(account.master_account_index));
    full_elements.extend(l1_limbs);
    full_elements.push(F::from_canonical_u8(account.account_type));

    let mut margined_elements = Vec::new();
    for asset in &account.margined_assets {
        let balance = BigInt::from(asset.balance);
        let (limbs, sign) = bigint_u32_parts(&balance, U96_LIMBS)?;
        margined_elements.extend(limbs);
        margined_elements.push(sign);
        margined_elements.push(F::from_canonical_u8(asset.margin_mode));
    }
    full_elements.extend(hash(margined_elements).elements);

    let mut strategy_elements = Vec::new();
    for strategy in &account.public_pool_info.strategies {
        let (limbs, sign) = bigint_u32_parts(strategy, U96_LIMBS)?;
        strategy_elements.extend(limbs);
        strategy_elements.push(sign);
    }
    full_elements.extend(hash(strategy_elements).elements);
    full_elements.extend([
        field_i64(account.total_order_count),
        field_i64(account.total_non_cross_order_count),
        field_i64(account.cancel_all_time),
        F::from_canonical_u8(account.account_trading_mode),
    ]);
    full_elements.extend(account.api_key_root.elements);
    full_elements.extend(account.account_orders_root.elements);
    full_elements.extend(account.asset_root.elements);

    let full = hash(full_elements);
    if full == EMPTY_ACCOUNT_HASH && account.account_index != TREASURY_ACCOUNT_INDEX as i64 {
        Some((HashOut::ZERO, HashOut::ZERO))
    } else {
        Some((full, hash(public_elements)))
    }
}

pub fn account_hash(account: &Account<F>) -> Option<(HashOut<F>, HashOut<F>)> {
    let (partial, partial_public) = calculated_account_partials(account)?;
    account_hash_from_partials(account, partial, partial_public)
}

pub fn fee_account_hash(account: &Account<F>) -> Option<(HashOut<F>, HashOut<F>)> {
    account_hash_from_partials(
        account,
        account.partial_hash,
        account.partial_hash_for_pub_data,
    )
}

fn calculated_delta_partial(delta: &AccountDelta<F>) -> Option<HashOut<F>> {
    let mut elements = Vec::new();
    let mut empty = true;
    for share in &delta.public_pool_shares_delta {
        elements.extend([
            field_i64(share.public_pool_index),
            field_i64(share.shares_delta),
        ]);
        empty &= share.shares_delta == 0;
    }
    elements.extend([
        field_i64(delta.public_pool_info_delta.total_shares_delta),
        field_i64(delta.public_pool_info_delta.operator_shares_delta),
    ]);
    empty &= delta.public_pool_info_delta.total_shares_delta == 0;
    empty &= delta.public_pool_info_delta.operator_shares_delta == 0;

    let l1_limbs = u32_limbs(&delta.l1_address, U160_LIMBS)?;
    empty &= l1_limbs.iter().all(|value| *value == F::ZERO);
    elements.extend(l1_limbs);
    elements.push(F::from_canonical_u8(delta.account_type));
    empty &= delta.account_type == 0;
    elements.extend(delta.position_delta_root.elements);
    empty &= delta.position_delta_root == EMPTY_POSITION_DELTA_TREE_ROOT;
    Some(if empty { HashOut::ZERO } else { hash(elements) })
}

fn delta_hash_from_partial(delta: &AccountDelta<F>, partial: HashOut<F>) -> HashOut<F> {
    if partial == HashOut::ZERO && delta.asset_delta_root == EMPTY_ASSET_TREE_ROOT {
        HashOut::ZERO
    } else {
        let mut elements = partial.elements.to_vec();
        elements.extend(delta.asset_delta_root.elements);
        hash(elements)
    }
}

pub fn account_delta_hash(delta: &AccountDelta<F>) -> Option<HashOut<F>> {
    Some(delta_hash_from_partial(
        delta,
        calculated_delta_partial(delta)?,
    ))
}

pub fn fee_account_delta_hash(delta: &AccountDelta<F>) -> Option<HashOut<F>> {
    u32_limbs(&delta.l1_address, U160_LIMBS)?;
    Some(delta_hash_from_partial(delta, delta.partial_hash))
}

pub fn account_order_hash(order: &AccountOrder) -> HashOut<F> {
    let elements = [
        field_i64(order.order_index),
        field_i64(order.client_order_index),
        field_i64(order.initial_base_amount),
        F::from_canonical_u32(order.price),
        field_i64(order.nonce),
        field_i64(order.remaining_base_amount),
        F::from_canonical_u8(order.is_ask),
        F::from_canonical_u8(order.order_type),
        F::from_canonical_u8(order.time_in_force),
        F::from_canonical_u8(order.reduce_only),
        F::from_canonical_u32(order.trigger_price),
        field_i64(order.expiry),
        F::from_canonical_u8(order.trigger_status),
        field_i64(order.to_trigger_order_index0),
        field_i64(order.to_trigger_order_index1),
        field_i64(order.to_cancel_order_index0),
        field_i64(order.integrator_fee_collector_index),
        field_i64(order.integrator_taker_fee),
        field_i64(order.integrator_maker_fee),
        F::from_canonical_u64(order.order_flags),
    ];
    let added = elements[1..13]
        .iter()
        .chain(elements[16..19].iter())
        .copied()
        .sum::<F>();
    let aliases_empty = added == F::ZERO
        && order.order_index == 0
        && order.to_trigger_order_index0 == 0
        && order.to_trigger_order_index1 == 0
        && order.to_cancel_order_index0 == 0
        && order.order_flags == 0;
    if aliases_empty {
        HashOut::ZERO
    } else {
        hash(elements.to_vec())
    }
}

pub fn market_hash(market: &Market<F>) -> HashOut<F> {
    let aliases_empty = [
        field_i64(market.ask_nonce),
        field_i64(market.bid_nonce),
        F::from_canonical_u32(market.taker_fee),
        F::from_canonical_u32(market.maker_fee),
        F::from_canonical_u32(market.liquidation_fee),
        F::from_canonical_u64(market.min_base_amount),
        F::from_canonical_u64(market.min_quote_amount),
        F::from_canonical_u8(market.status),
        field_i64(market.order_quote_limit),
        field_i64(market.total_order_count),
        F::from_canonical_u8(market.market_type),
        F::from_canonical_u16(market.base_asset_id),
        F::from_canonical_u16(market.quote_asset_id),
        field_i64(market.size_extension_multiplier),
        field_i64(market.quote_extension_multiplier),
    ]
    .into_iter()
    .sum::<F>()
        == F::ZERO;
    if aliases_empty {
        return HashOut::ZERO;
    }

    let mut elements = vec![
        F::from_canonical_u8(market.market_type),
        F::from_canonical_u8(market.status),
        F::from_canonical_u16(market.base_asset_id),
        F::from_canonical_u16(market.quote_asset_id),
        field_i64(market.ask_nonce),
        field_i64(market.bid_nonce),
        F::from_canonical_u32(market.taker_fee),
        F::from_canonical_u32(market.maker_fee),
        F::from_canonical_u32(market.liquidation_fee),
        F::from_canonical_u64(market.min_base_amount),
        F::from_canonical_u64(market.min_quote_amount),
        field_i64(market.order_quote_limit),
        field_i64(market.total_order_count),
        field_i64(market.size_extension_multiplier),
        field_i64(market.quote_extension_multiplier),
    ];
    elements.extend(market.order_book_root.elements);
    hash(elements)
}

pub fn market_details_hash(details: &MarketDetails) -> HashOut<F> {
    let elements = vec![
        field_i64(details.aggregate_premium_sum),
        F::from_canonical_u32(details.interest_rate),
        F::from_canonical_u32(details.impact_ask_price),
        F::from_canonical_u32(details.impact_bid_price),
        F::from_canonical_u32(details.impact_price),
        field_i64(details.open_interest),
        F::from_canonical_u32(details.index_price),
        F::from_canonical_u32(details.funding_clamp_small),
        F::from_canonical_u32(details.funding_clamp_big),
        F::from_canonical_u64(details.open_interest_limit),
        field_i64(details.market_flags),
        F::from_canonical_u16(details.funding_premium_multiplier),
    ];
    if elements.iter().all(|value| *value == F::ZERO) {
        HashOut::ZERO
    } else {
        hash(elements)
    }
}

fn register_fields(register: &BaseRegisterInfo) -> [F; 23] {
    [
        F::from_canonical_u8(register.instruction_type),
        F::from_canonical_u16(register.market_index),
        field_i64(register.account_index),
        field_i64(register.pending_size),
        field_i64(register.pending_order_index),
        field_i64(register.pending_client_order_index),
        field_i64(register.pending_initial_size),
        field_i64(register.pending_price),
        field_i64(register.pending_nonce),
        F::from_canonical_u8(register.pending_is_ask),
        F::from_canonical_u8(register.pending_type),
        F::from_canonical_u8(register.pending_time_in_force),
        F::from_canonical_u8(register.pending_reduce_only),
        field_i64(register.pending_expiry),
        field_i64(register.generic_field_0),
        F::from_canonical_u32(register.pending_trigger_price),
        F::from_canonical_u8(register.pending_trigger_status),
        field_i64(register.pending_to_trigger_order_index0),
        field_i64(register.pending_to_trigger_order_index1),
        field_i64(register.pending_to_cancel_order_index0),
        field_i64(register.generic_field_1),
        field_i64(register.generic_field_2),
        field_i64(register.generic_field_3),
    ]
}

fn register_aliases_empty(register: &BaseRegisterInfo) -> bool {
    let fields = register_fields(register);
    let summed_fields = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 21, 22];
    summed_fields
        .into_iter()
        .fold(F::ZERO, |sum, index| sum + fields[index])
        == F::ZERO
        && register.pending_order_index == 0
        && register.pending_to_trigger_order_index0 == 0
        && register.pending_to_trigger_order_index1 == 0
        && register.pending_to_cancel_order_index0 == 0
        && register.generic_field_0 == 0
        && register.generic_field_1 == 0
}

pub fn register_stack_hash(stack: &RegisterStack) -> HashOut<F> {
    debug_assert_eq!(stack.stack.len(), REGISTER_STACK_SIZE);
    if stack.stack.iter().all(register_aliases_empty) {
        HashOut::ZERO
    } else {
        hash(stack.stack.iter().flat_map(register_fields).collect())
    }
}

pub fn system_config_hash(config: &SystemConfig) -> Option<HashOut<F>> {
    let values = [
        config.liquidity_pool_index,
        config.staking_pool_index,
        config.liquidity_pool_cooldown_period,
        config.staking_pool_lockup_period,
        config.max_integrator_spot_taker_fee,
        config.max_integrator_spot_maker_fee,
        config.max_integrator_perps_taker_fee,
        config.max_integrator_perps_maker_fee,
    ];
    values
        .iter()
        .all(|value| *value >= 0)
        .then(|| hash(values.into_iter().map(field_i64).collect()))
}

pub fn all_assets_hash(assets: &[Asset; ASSET_LIST_SIZE]) -> Option<HashOut<F>> {
    let mut elements = Vec::new();
    for index in MIN_ASSET_INDEX..=MAX_ASSET_INDEX {
        let asset = assets.get(index as usize)?;
        elements.extend([
            F::from_canonical_u8(asset.margin_index),
            F::from_canonical_u8(asset.margin_mode),
        ]);
        elements.extend(nonnegative_i64_u32_limbs(asset.extension_multiplier, 2)?);
        elements.extend(nonnegative_i64_u32_limbs(asset.min_transfer_amount, 2)?);
        elements.extend(nonnegative_i64_u32_limbs(asset.min_withdrawal_amount, 2)?);
    }
    Some(hash(elements))
}

pub fn all_margined_assets_hash(
    assets: &[MarginedAsset; MARGINED_ASSET_LIST_SIZE],
) -> Option<HashOut<F>> {
    let mut elements = Vec::new();
    for asset in assets {
        if asset.asset_index < 0 || asset.index_price < 0 || asset.index_price_divider < 0 {
            return None;
        }
        elements.extend([
            field_i64(i64::from(asset.asset_index)),
            field_i64(asset.index_price),
            F::from_canonical_u16(asset.loan_to_value),
            F::from_canonical_u16(asset.liquidation_threshold),
            F::from_canonical_u32(asset.liquidation_factor),
            F::from_canonical_u32(asset.liquidation_fee),
            field_i64(asset.index_price_divider),
        ]);
        elements.extend(u32_limbs(&asset.global_supply_cap, U96_LIMBS)?);
        elements.extend(u32_limbs(&asset.user_supply_cap, U96_LIMBS)?);
        elements.extend(u32_limbs(&asset.total_supplied_amount, U96_LIMBS)?);
    }
    Some(hash(elements))
}

pub fn all_market_details_hashes(
    details: &[MarketRiskDetails; POSITION_LIST_SIZE],
) -> Option<(HashOut<F>, HashOut<F>)> {
    let mut extended = details.to_vec();
    extended.push(MarketRiskDetails::default());
    let mut risk_buckets = Vec::new();
    let mut public_buckets = Vec::new();
    for bucket in extended.chunks(POSITION_HASH_BUCKET_SIZE) {
        let mut public_elements = Vec::new();
        for market in bucket {
            let (funding_limbs, funding_sign) =
                bigint_u16_parts(&market.funding_rate_prefix_sum, U16_U64_LIMBS)?;
            public_elements.extend(funding_limbs);
            public_elements.extend([
                funding_sign,
                F::from_canonical_u32(market.mark_price),
                F::from_canonical_u32(market.quote_multiplier),
            ]);
        }
        let public_bucket = hash(public_elements);
        let mut risk_elements = public_bucket.elements.to_vec();
        for market in bucket {
            risk_elements.extend([
                F::from_canonical_u8(market.status),
                F::from_canonical_u8(market.strategy_index),
                F::from_canonical_u16(market.default_initial_margin_fraction),
                F::from_canonical_u16(market.min_initial_margin_fraction),
                F::from_canonical_u16(market.maintenance_margin_fraction),
                F::from_canonical_u16(market.close_out_margin_fraction),
            ]);
        }
        risk_buckets.push(hash(risk_elements));
        public_buckets.push(public_bucket);
    }
    if risk_buckets.len() != POSITION_HASH_BUCKET_COUNT
        || public_buckets.len() != POSITION_HASH_BUCKET_COUNT
    {
        return None;
    }
    let risk_hash = hash(
        risk_buckets
            .iter()
            .flat_map(|bucket| bucket.elements)
            .collect(),
    );
    let public_hash = hash(
        public_buckets
            .iter()
            .flat_map(|bucket| bucket.elements)
            .collect(),
    );
    Some((risk_hash, public_hash))
}

fn hash_n_to_one(hashes: &[HashOut<F>]) -> Option<HashOut<F>> {
    let mut hashes = hashes.iter().copied();
    let first = hashes.next()?;
    let second = hashes.next();
    let mut result = match second {
        Some(second) => Poseidon2Hash::two_to_one(first, second),
        None => return Some(first),
    };
    for next in hashes {
        result = Poseidon2Hash::two_to_one(result, next);
    }
    Some(result)
}

#[allow(clippy::too_many_arguments)]
pub fn validium_and_state_root(
    system_config_hash: HashOut<F>,
    assets_hash: HashOut<F>,
    margined_assets_hash: HashOut<F>,
    market_risk_details_hash: HashOut<F>,
    public_market_details_hash: HashOut<F>,
    register_stack_hash: HashOut<F>,
    account_tree_root: HashOut<F>,
    account_pub_data_tree_root: HashOut<F>,
    market_details_tree_root: HashOut<F>,
    market_tree_root: HashOut<F>,
    state_metadata_hash: HashOut<F>,
) -> Option<(HashOut<F>, HashOut<F>)> {
    let validium = hash_n_to_one(&[
        register_stack_hash,
        account_tree_root,
        market_details_tree_root,
        market_tree_root,
        assets_hash,
        margined_assets_hash,
        market_risk_details_hash,
        state_metadata_hash,
        system_config_hash,
    ])?;
    let state = hash_n_to_one(&[
        account_pub_data_tree_root,
        public_market_details_hash,
        validium,
    ])?;
    Some((validium, state))
}

pub fn merkle_root(leaf: HashOut<F>, index: u64, proof: &[HashOut<F>]) -> HashOut<F> {
    proof
        .iter()
        .enumerate()
        .fold(leaf, |current, (level, sibling)| {
            if (index >> level) & 1 == 0 {
                Poseidon2Hash::two_to_one(current, *sibling)
            } else {
                Poseidon2Hash::two_to_one(*sibling, current)
            }
        })
}

fn subtree_at_height(
    leaf: HashOut<F>,
    index: u64,
    proof: &[HashOut<F>],
    height: usize,
) -> HashOut<F> {
    merkle_root(leaf, index, &proof[..height])
}

/// Convert a target leaf's proof from the state after `changed_index` was
/// updated back to the state containing `old_changed_leaf`.
pub fn rewind_one_update(
    changed_index: u64,
    old_changed_leaf: HashOut<F>,
    changed_old_proof: &[HashOut<F>],
    target_index: u64,
    target_after_proof: &[HashOut<F>],
) -> Option<Vec<HashOut<F>>> {
    if changed_index == target_index || changed_old_proof.len() != target_after_proof.len() {
        return None;
    }
    // The proof entry whose sibling subtree contains the changed leaf is the
    // highest bit at which the two indices differ. Lower differing bits are
    // internal to that sibling subtree.
    let xor = changed_index ^ target_index;
    let divergence = (u64::BITS - 1 - xor.leading_zeros()) as usize;
    if divergence >= changed_old_proof.len() {
        return None;
    }
    let mut result = target_after_proof.to_vec();
    result[divergence] = subtree_at_height(
        old_changed_leaf,
        changed_index,
        changed_old_proof,
        divergence,
    );
    Some(result)
}

/// Convert a target leaf's proof from the old state to the state after one
/// other leaf has the supplied new value.
pub fn apply_one_update_to_proof(
    changed_index: u64,
    new_changed_leaf: HashOut<F>,
    changed_before_proof: &[HashOut<F>],
    target_index: u64,
    target_before_proof: &[HashOut<F>],
) -> Option<Vec<HashOut<F>>> {
    rewind_one_update(
        changed_index,
        new_changed_leaf,
        changed_before_proof,
        target_index,
        target_before_proof,
    )
}

#[cfg(test)]
mod tests {
    use circuit::tx_constraints::compute_validium_and_state_root;
    use circuit::types::account::{AccountTarget, AccountTargetWitness};
    use circuit::types::account_asset::{AccountAssetTarget, AccountAssetTargetWitness};
    use circuit::types::account_delta::{AccountDeltaTarget, AccountDeltaTargetWitness};
    use circuit::types::account_order::{AccountOrderTarget, AccountOrderTargetWitness};
    use circuit::types::asset::{
        AssetTarget, AssetTargetWitness, all_assets_hash as circuit_all_assets_hash,
    };
    use circuit::types::config::{Builder, C, CIRCUIT_CONFIG};
    use circuit::types::constants::{
        EMPTY_ACCOUNT_ORDERS_TREE_ROOT, EMPTY_API_KEY_TREE_ROOT, GTT, INSERT_ORDER, LIMIT_ORDER,
        MARKET_TYPE_SPOT, NIL_MARKET_INDEX,
    };
    use circuit::types::margined_asset::{
        MarginedAssetTarget, MarginedAssetTargetWitness,
        all_margined_assets_hash as circuit_all_margined_assets_hash,
    };
    use circuit::types::market::{MarketTarget, MarketTargetWitness};
    use circuit::types::market_details::{
        MarketDetailsTarget, MarketDetailsWitness, MarketRiskDetailsTarget,
        MarketRiskDetailsWitness, all_market_details_hashes as circuit_all_market_details_hashes,
    };
    use circuit::types::register::{RegisterInfoTargetWitness, RegisterStackTarget};
    use circuit::types::system_config::{SystemConfigTarget, SystemConfigTargetWitness};
    use num::{BigInt, BigUint};
    use plonky2::iop::witness::PartialWitness;

    use super::*;

    fn sample_hash(seed: u64) -> HashOut<F> {
        HashOut {
            elements: core::array::from_fn(|offset| F::from_canonical_u64(seed + offset as u64)),
        }
    }

    fn sample_account(index: i64) -> Account<F> {
        let mut account = Account::<F>::default();
        account.account_index = index;
        account.master_account_index = 3;
        account.l1_address = BigUint::from(0x1234_5678_u64);
        account.account_type = 1;
        account.account_trading_mode = 1;
        account.margined_assets[0].balance = -19;
        account.margined_assets[0].margin_mode = 1;
        account.aggregated_balances = [BigInt::from(-7), BigInt::from(13)];
        account.positions[0].last_funding_rate_prefix_sum = BigInt::from(-23);
        account.positions[0].position = BigInt::from(29);
        account.positions[0].entry_quote = 31;
        account.positions[0].initial_margin_fraction = 37;
        account.positions[0].total_order_count = 2;
        account.positions[0].total_position_tied_order_count = 1;
        account.positions[0].margin_mode = 1;
        account.positions[0].margin_set_flag = 1;
        account.positions[0].allocated_margin = BigInt::from(-41);
        account.pending_unlocks[0].unlock_timestamp = 43;
        account.pending_unlocks[0].asset_index = 2;
        account.pending_unlocks[0].amount = BigUint::from(47_u64);
        account.approved_integrators[0].integrator_account_index = 53;
        account.approved_integrators[0].max_perps_taker_fee = 59;
        account.approved_integrators[0].max_perps_maker_fee = 61;
        account.approved_integrators[0].max_spot_taker_fee = 67;
        account.approved_integrators[0].max_spot_maker_fee = 71;
        account.approved_integrators[0].expiry = 73;
        account.public_pool_shares[0].public_pool_index = 79;
        account.public_pool_shares[0].share_amount = 83;
        account.public_pool_shares[0].principal_amount = 89;
        account.public_pool_shares[0].entry_timestamp = 97;
        account.public_pool_info.status = 1;
        account.public_pool_info.operator_fee = 101;
        account.public_pool_info.min_operator_share_rate = 103;
        account.public_pool_info.total_shares = 107;
        account.public_pool_info.operator_shares = 109;
        account.public_pool_info.strategies[0] = BigInt::from(-113);
        account.total_order_count = 5;
        account.total_non_cross_order_count = 3;
        account.cancel_all_time = 127;
        account.api_key_root = EMPTY_API_KEY_TREE_ROOT;
        account.account_orders_root = EMPTY_ACCOUNT_ORDERS_TREE_ROOT;
        account.aggregated_balances_root = sample_hash(131);
        account.asset_root = sample_hash(137);
        account
    }

    fn sample_delta(index: i64) -> AccountDelta<F> {
        let mut delta = AccountDelta::<F>::default();
        delta.account_index = index;
        delta.l1_address = BigUint::from(139_u64);
        delta.account_type = 1;
        delta.aggregated_asset_deltas = [BigInt::from(-149), BigInt::from(151)];
        delta.public_pool_shares_delta[0].public_pool_index = 157;
        delta.public_pool_shares_delta[0].shares_delta = -163;
        delta.public_pool_info_delta.total_shares_delta = 167;
        delta.public_pool_info_delta.operator_shares_delta = -173;
        delta.position_delta_root = sample_hash(179);
        delta.asset_delta_root = sample_hash(181);
        delta
    }

    fn next_hash(public_inputs: &[F], offset: &mut usize) -> HashOut<F> {
        let result = HashOut::from([
            public_inputs[*offset],
            public_inputs[*offset + 1],
            public_inputs[*offset + 2],
            public_inputs[*offset + 3],
        ]);
        *offset += 4;
        result
    }

    #[test]
    fn native_leaf_hashes_match_the_production_circuit_formulas() {
        let asset = AccountAsset {
            index_0: 2,
            balance: BigUint::from(191_u64),
            locked_balance: BigUint::from(193_u64),
        };
        let account = sample_account(17);
        let mut fee_account = sample_account(TREASURY_ACCOUNT_INDEX as i64);
        fee_account.partial_hash = sample_hash(197);
        fee_account.partial_hash_for_pub_data = sample_hash(199);
        let delta = sample_delta(17);
        let mut fee_delta = sample_delta(TREASURY_ACCOUNT_INDEX as i64);
        fee_delta.partial_hash = sample_hash(211);

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let asset_target = AccountAssetTarget::new(&mut builder);
        let asset_hash_target = asset_target.hash(&mut builder);
        builder.register_public_hashout(asset_hash_target);

        let account_target = AccountTarget::new(&mut builder);
        let balance_hash_target = account_target.aggregated_balance_hash(&mut builder, 0);
        builder.register_public_hashout(balance_hash_target);
        let position_hashes = account_target.get_position_bucket_hashes(&mut builder);
        let (account_hash_target, account_public_hash_target, _) =
            account_target.hash(&mut builder, &position_hashes);
        builder.register_public_hashout(account_hash_target);
        builder.register_public_hashout(account_public_hash_target);

        let fee_account_target = AccountTarget::new_fee_account(&mut builder);
        let (fee_hash_target, fee_public_hash_target, _) =
            fee_account_target.fee_account_hash(&mut builder);
        builder.register_public_hashout(fee_hash_target);
        builder.register_public_hashout(fee_public_hash_target);

        let delta_target = AccountDeltaTarget::new(&mut builder);
        let delta_hash_target = delta_target.hash(&mut builder);
        builder.register_public_hashout(delta_hash_target);
        let fee_delta_target = AccountDeltaTarget::new_fee_account(&mut builder);
        let fee_delta_hash_target = fee_delta_target.fee_account_hash(&mut builder);
        builder.register_public_hashout(fee_delta_hash_target);

        let data = builder.build::<C>();
        let mut witness = PartialWitness::<F>::new();
        witness
            .set_account_asset_target(&asset_target, &asset)
            .unwrap();
        witness
            .set_account_target(&account_target, &account)
            .unwrap();
        witness
            .set_fee_account_target(&fee_account_target, &fee_account)
            .unwrap();
        witness
            .set_account_delta_target(&delta_target, &delta)
            .unwrap();
        witness
            .set_fee_account_delta_target(&fee_delta_target, &fee_delta)
            .unwrap();
        let proof = data.prove(witness).expect("differential circuit proves");
        data.verify(proof.clone())
            .expect("differential circuit verifies");

        let mut offset = 0;
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            account_asset_hash(&asset).unwrap()
        );
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            bigint_leaf_hash(&account.aggregated_balances[0]).unwrap()
        );
        let account_hashes = account_hash(&account).unwrap();
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            account_hashes.0
        );
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            account_hashes.1
        );
        let fee_hashes = fee_account_hash(&fee_account).unwrap();
        assert_eq!(next_hash(&proof.public_inputs, &mut offset), fee_hashes.0);
        assert_eq!(next_hash(&proof.public_inputs, &mut offset), fee_hashes.1);
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            account_delta_hash(&delta).unwrap()
        );
        assert_eq!(
            next_hash(&proof.public_inputs, &mut offset),
            fee_account_delta_hash(&fee_delta).unwrap()
        );
        assert_eq!(offset, proof.public_inputs.len());
    }

    #[test]
    fn native_alias_leaf_hashes_match_the_production_circuit_formulas() {
        let order_visible_sum =
            10 + 7 + 11 + 10 + 1 + i64::from(LIMIT_ORDER) + i64::from(GTT) + 2_000;
        let alias_order = AccountOrder {
            index_0: ((i64::from(NIL_MARKET_INDEX) + 1) << 48) + 11,
            owner_account_index: 17,
            initial_base_amount: 10,
            price: 7,
            nonce: 11,
            remaining_base_amount: 10,
            is_ask: 1,
            order_type: LIMIT_ORDER,
            time_in_force: GTT,
            expiry: 2_000,
            integrator_taker_fee: -order_visible_sum,
            ..AccountOrder::default()
        };
        let mut nonempty_order = alias_order.clone();
        nonempty_order.client_order_index = 1;

        let market_visible_sum = MARKET_TYPE_SPOT as i64 + 1 + 2 + 3 + 5;
        let alias_market = Market::<F> {
            market_index: u16::from(NIL_MARKET_INDEX),
            market_type: MARKET_TYPE_SPOT as u8,
            base_asset_id: 1,
            quote_asset_id: 2,
            size_extension_multiplier: 3,
            quote_extension_multiplier: 5,
            order_quote_limit: -market_visible_sum,
            order_book_root: sample_hash(2_000),
            ..Market::default()
        };
        let mut nonempty_market = alias_market.clone();
        nonempty_market.total_order_count = 1;

        let alias_market_details = MarketDetails {
            market_index: u16::from(NIL_MARKET_INDEX),
            ..MarketDetails::default()
        };
        let nonempty_market_details = MarketDetails {
            market_index: u16::from(NIL_MARKET_INDEX),
            aggregate_premium_sum: -157,
            interest_rate: 163,
            impact_bid_price: 167,
            impact_ask_price: 173,
            impact_price: 179,
            open_interest: 181,
            index_price: 191,
            funding_clamp_small: 193,
            funding_clamp_big: 197,
            open_interest_limit: 199,
            market_flags: 211,
            funding_premium_multiplier: 83,
        };

        let alias_register = BaseRegisterInfo {
            instruction_type: INSERT_ORDER,
            market_index: u16::from(NIL_MARKET_INDEX),
            account_index: 17,
            pending_size: 10,
            pending_initial_size: 10,
            pending_price: 7,
            pending_nonce: 11,
            pending_is_ask: 1,
            pending_type: LIMIT_ORDER,
            pending_time_in_force: GTT,
            pending_expiry: -(10 + 10 + 7 + 11 + 1 + i64::from(LIMIT_ORDER) + i64::from(GTT)),
            generic_field_3: -(i64::from(INSERT_ORDER) + i64::from(NIL_MARKET_INDEX) + 17),
            ..BaseRegisterInfo::empty()
        };
        let mut alias_stack = RegisterStack {
            stack: [BaseRegisterInfo::empty(); REGISTER_STACK_SIZE],
            count: 1,
        };
        alias_stack.stack[0] = alias_register;
        let mut nonempty_stack = alias_stack.clone();
        nonempty_stack.stack[0].generic_field_2 = 1;

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let alias_order_target = AccountOrderTarget::new(&mut builder);
        let nonempty_order_target = AccountOrderTarget::new(&mut builder);
        let alias_order_hash_target = alias_order_target.hash(&mut builder);
        let nonempty_order_hash_target = nonempty_order_target.hash(&mut builder);
        builder.register_public_hashout(alias_order_hash_target);
        builder.register_public_hashout(nonempty_order_hash_target);

        let alias_market_target = MarketTarget::new(&mut builder);
        let nonempty_market_target = MarketTarget::new(&mut builder);
        let alias_market_hash_target = alias_market_target.hash(&mut builder);
        let nonempty_market_hash_target = nonempty_market_target.hash(&mut builder);
        builder.register_public_hashout(alias_market_hash_target);
        builder.register_public_hashout(nonempty_market_hash_target);

        let alias_market_details_target = MarketDetailsTarget::new(&mut builder);
        let nonempty_market_details_target = MarketDetailsTarget::new(&mut builder);
        let alias_market_details_hash_target = alias_market_details_target.hash(&mut builder);
        let nonempty_market_details_hash_target = nonempty_market_details_target.hash(&mut builder);
        builder.register_public_hashout(alias_market_details_hash_target);
        builder.register_public_hashout(nonempty_market_details_hash_target);

        let alias_stack_target = RegisterStackTarget::new(&mut builder);
        let nonempty_stack_target = RegisterStackTarget::new(&mut builder);
        let alias_stack_hash_target = alias_stack_target.hash(&mut builder);
        let nonempty_stack_hash_target = nonempty_stack_target.hash(&mut builder);
        builder.register_public_hashout(alias_stack_hash_target);
        builder.register_public_hashout(nonempty_stack_hash_target);
        builder.perform_registered_range_checks();

        let data = builder.build::<C>();
        let mut witness = PartialWitness::<F>::new();
        witness
            .set_account_order_target(&alias_order_target, &alias_order)
            .unwrap();
        witness
            .set_account_order_target(&nonempty_order_target, &nonempty_order)
            .unwrap();
        witness
            .set_market_target(&alias_market_target, &alias_market)
            .unwrap();
        witness
            .set_market_target(&nonempty_market_target, &nonempty_market)
            .unwrap();
        witness
            .set_market_details_target(&alias_market_details_target, &alias_market_details)
            .unwrap();
        witness
            .set_market_details_target(&nonempty_market_details_target, &nonempty_market_details)
            .unwrap();
        witness
            .set_register_info_target(&alias_stack_target, &alias_stack)
            .unwrap();
        witness
            .set_register_info_target(&nonempty_stack_target, &nonempty_stack)
            .unwrap();

        let proof = data.prove(witness).expect("differential circuit proves");
        data.verify(proof.clone())
            .expect("differential circuit verifies");

        let expected = [
            account_order_hash(&alias_order),
            account_order_hash(&nonempty_order),
            market_hash(&alias_market),
            market_hash(&nonempty_market),
            market_details_hash(&alias_market_details),
            market_details_hash(&nonempty_market_details),
            register_stack_hash(&alias_stack),
            register_stack_hash(&nonempty_stack),
        ];
        assert_eq!(expected[0], HashOut::ZERO);
        assert_eq!(expected[2], HashOut::ZERO);
        assert_eq!(expected[4], HashOut::ZERO);
        assert_eq!(expected[6], HashOut::ZERO);
        assert!(
            expected[1..]
                .iter()
                .step_by(2)
                .all(|hash| *hash != HashOut::ZERO)
        );
        let mut offset = 0;
        for expected_hash in expected {
            assert_eq!(next_hash(&proof.public_inputs, &mut offset), expected_hash);
        }
        assert_eq!(offset, proof.public_inputs.len());
    }

    #[test]
    fn native_global_hashes_match_the_production_circuit_formulas() {
        let system_config = SystemConfig {
            liquidity_pool_index: 1,
            staking_pool_index: 2,
            liquidity_pool_cooldown_period: 3,
            staking_pool_lockup_period: 5,
            max_integrator_spot_taker_fee: 7,
            max_integrator_spot_maker_fee: 11,
            max_integrator_perps_taker_fee: 13,
            max_integrator_perps_maker_fee: 17,
        };

        let mut assets: [Asset; ASSET_LIST_SIZE] =
            core::array::from_fn(|index| Asset::empty(index as i16));
        assets[MIN_ASSET_INDEX as usize] = Asset {
            asset_index: MIN_ASSET_INDEX as i16,
            extension_multiplier: 0x1_0000_0013,
            min_transfer_amount: 0x2_0000_0017,
            min_withdrawal_amount: 0x3_0000_001d,
            margin_mode: 1,
            margin_index: 2,
        };
        assets[MAX_ASSET_INDEX as usize] = Asset {
            asset_index: MAX_ASSET_INDEX as i16,
            extension_multiplier: 0x4_0000_001f,
            min_transfer_amount: 0x5_0000_0025,
            min_withdrawal_amount: 0x6_0000_0029,
            margin_mode: 1,
            margin_index: 3,
        };

        let mut margined_assets: [MarginedAsset; MARGINED_ASSET_LIST_SIZE] =
            core::array::from_fn(|index| MarginedAsset::empty(index as u8));
        margined_assets[0] = MarginedAsset {
            margin_index: 0,
            asset_index: 2,
            loan_to_value: 31,
            liquidation_threshold: 37,
            liquidation_factor: 41,
            liquidation_fee: 43,
            index_price: 0x1_0000_002f,
            index_price_divider: 0x2_0000_0035,
            global_supply_cap: (BigUint::from(59_u64) << 64) + BigUint::from(61_u64),
            user_supply_cap: (BigUint::from(67_u64) << 64) + BigUint::from(71_u64),
            total_supplied_amount: (BigUint::from(73_u64) << 64) + BigUint::from(79_u64),
        };

        let mut market_details: [MarketRiskDetails; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| MarketRiskDetails::default());
        market_details[0] = MarketRiskDetails {
            funding_rate_prefix_sum: BigInt::from(-83),
            mark_price: 89,
            quote_multiplier: 97,
            status: 1,
            strategy_index: 2,
            default_initial_margin_fraction: 101,
            min_initial_margin_fraction: 103,
            maintenance_margin_fraction: 107,
            close_out_margin_fraction: 109,
        };
        market_details[POSITION_LIST_SIZE - 1] = MarketRiskDetails {
            funding_rate_prefix_sum: BigInt::from(113),
            mark_price: 127,
            quote_multiplier: 131,
            status: 3,
            strategy_index: 4,
            default_initial_margin_fraction: 137,
            min_initial_margin_fraction: 139,
            maintenance_margin_fraction: 149,
            close_out_margin_fraction: 151,
        };

        let register_stack_hash = sample_hash(1000);
        let account_tree_root = sample_hash(1010);
        let account_pub_data_tree_root = sample_hash(1020);
        let market_details_tree_root = sample_hash(1030);
        let market_tree_root = sample_hash(1040);
        let state_metadata_hash = sample_hash(1050);

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let system_config_target = SystemConfigTarget::new(&mut builder);
        let system_config_hash_target = system_config_target.hash(&mut builder);
        builder.register_public_hashout(system_config_hash_target);

        let asset_targets: [AssetTarget; ASSET_LIST_SIZE] =
            core::array::from_fn(|_| AssetTarget::new(&mut builder));
        let assets_hash_target = circuit_all_assets_hash(&mut builder, &asset_targets);
        builder.register_public_hashout(assets_hash_target);

        let margined_asset_targets: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE] =
            core::array::from_fn(|_| MarginedAssetTarget::new(&mut builder));
        let margined_assets_hash_target =
            circuit_all_margined_assets_hash(&mut builder, &margined_asset_targets);
        builder.register_public_hashout(margined_assets_hash_target);

        let market_details_targets: [MarketRiskDetailsTarget; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| MarketRiskDetailsTarget::new(&mut builder));
        let (market_risk_hash_target, public_market_hash_target, _) =
            circuit_all_market_details_hashes(&mut builder, &market_details_targets);
        builder.register_public_hashout(market_risk_hash_target);
        builder.register_public_hashout(public_market_hash_target);

        let register_stack_hash_target = builder.constant_hash(register_stack_hash);
        let account_tree_root_target = builder.constant_hash(account_tree_root);
        let account_pub_data_tree_root_target = builder.constant_hash(account_pub_data_tree_root);
        let market_details_tree_root_target = builder.constant_hash(market_details_tree_root);
        let market_tree_root_target = builder.constant_hash(market_tree_root);
        let state_metadata_hash_target = builder.constant_hash(state_metadata_hash);
        let (validium_target, state_target) = compute_validium_and_state_root(
            &mut builder,
            system_config_hash_target,
            assets_hash_target,
            margined_assets_hash_target,
            market_risk_hash_target,
            public_market_hash_target,
            register_stack_hash_target,
            account_tree_root_target,
            account_pub_data_tree_root_target,
            market_details_tree_root_target,
            market_tree_root_target,
            state_metadata_hash_target,
        );
        builder.register_public_hashout(validium_target);
        builder.register_public_hashout(state_target);

        let data = builder.build::<C>();
        let mut witness = PartialWitness::<F>::new();
        witness
            .set_system_config_target(&system_config_target, &system_config)
            .unwrap();
        for (target, asset) in asset_targets.iter().zip(&assets) {
            witness.set_asset_target(target, asset).unwrap();
        }
        for (target, asset) in margined_asset_targets.iter().zip(&margined_assets) {
            witness.set_margined_asset_target(target, asset).unwrap();
        }
        for (target, market) in market_details_targets.iter().zip(&market_details) {
            witness
                .set_market_risk_details_target(target, market)
                .unwrap();
        }

        let proof = data.prove(witness).expect("differential circuit proves");
        data.verify(proof.clone())
            .expect("differential circuit verifies");

        let expected_system_config_hash = system_config_hash(&system_config).unwrap();
        let expected_assets_hash = all_assets_hash(&assets).unwrap();
        let expected_margined_assets_hash = all_margined_assets_hash(&margined_assets).unwrap();
        let (expected_market_risk_hash, expected_public_market_hash) =
            all_market_details_hashes(&market_details).unwrap();
        let (expected_validium, expected_state) = validium_and_state_root(
            expected_system_config_hash,
            expected_assets_hash,
            expected_margined_assets_hash,
            expected_market_risk_hash,
            expected_public_market_hash,
            register_stack_hash,
            account_tree_root,
            account_pub_data_tree_root,
            market_details_tree_root,
            market_tree_root,
            state_metadata_hash,
        )
        .unwrap();

        let mut offset = 0;
        for expected in [
            expected_system_config_hash,
            expected_assets_hash,
            expected_margined_assets_hash,
            expected_market_risk_hash,
            expected_public_market_hash,
            expected_validium,
            expected_state,
        ] {
            assert_eq!(next_hash(&proof.public_inputs, &mut offset), expected);
        }
        assert_eq!(offset, proof.public_inputs.len());
    }
}
