use anyhow::Result;
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::order_book_tree_helpers::OrderBookTree;
use circuit::types::account_order::{AccountOrder, AccountOrderTarget};
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::{
    GTT, INSERT_ORDER, IOC, LIMIT_ORDER, ORDER_BOOK_MERKLE_LEVELS, ORDER_NONCE_BITS,
    REGISTER_STACK_SIZE,
};
use circuit::types::market::{Market, MarketTarget};
use circuit::types::order::Order;
use circuit::types::register::{
    BaseRegisterInfo, BaseRegisterInfoTarget, RegisterStack, RegisterStackTarget,
};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::HashOut;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

const MARKET: u16 = 255;
const MAKER: i64 = 17;
const TAKER: i64 = 23;
const SIZE: i64 = 10;
const PRICE: u32 = 7;
const MAKER_NONCE: i64 = 11;
const TAKER_NONCE: i64 = 12;
const BASE_MULTIPLIER: i64 = 3;
const QUOTE_MULTIPLIER: i64 = 5;

fn stack_with(register: BaseRegisterInfo) -> RegisterStack {
    let mut result = RegisterStack {
        stack: [BaseRegisterInfo::empty(); REGISTER_STACK_SIZE],
        count: 1,
    };
    result.stack[0] = register;
    result
}

// The production convenience setters use `from_canonical_i64`, which rejects
// negative Rust values before witness generation. A malicious witness must use
// their canonical field representatives instead; this test-local setter does
// exactly that without changing any production circuit code or its digest.
fn set_base_register_noncanonical(
    pw: &mut PartialWitness<F>,
    target: &BaseRegisterInfoTarget,
    value: &BaseRegisterInfo,
) -> Result<()> {
    let fields = [
        (target.instruction_type, i64::from(value.instruction_type)),
        (target.market_index, i64::from(value.market_index)),
        (target.account_index, value.account_index),
        (target.pending_size, value.pending_size),
        (target.pending_order_index, value.pending_order_index),
        (
            target.pending_client_order_index,
            value.pending_client_order_index,
        ),
        (target.pending_initial_size, value.pending_initial_size),
        (target.pending_price, value.pending_price),
        (target.pending_nonce, value.pending_nonce),
        (
            target.pending_is_ask.target,
            i64::from(value.pending_is_ask),
        ),
        (target.pending_type, i64::from(value.pending_type)),
        (
            target.pending_time_in_force,
            i64::from(value.pending_time_in_force),
        ),
        (
            target.pending_reduce_only,
            i64::from(value.pending_reduce_only),
        ),
        (target.pending_expiry, value.pending_expiry),
        (target.generic_field_0, value.generic_field_0),
        (
            target.pending_trigger_price,
            i64::from(value.pending_trigger_price),
        ),
        (
            target.pending_trigger_status,
            i64::from(value.pending_trigger_status),
        ),
        (
            target.pending_to_trigger_order_index0,
            value.pending_to_trigger_order_index0,
        ),
        (
            target.pending_to_trigger_order_index1,
            value.pending_to_trigger_order_index1,
        ),
        (
            target.pending_to_cancel_order_index0,
            value.pending_to_cancel_order_index0,
        ),
        (target.generic_field_1, value.generic_field_1),
        (target.generic_field_2, value.generic_field_2),
        (target.generic_field_3, value.generic_field_3),
    ];
    for (target, value) in fields {
        pw.set_target(target, F::from_noncanonical_i64(value))?;
    }
    Ok(())
}

fn set_register_stack_noncanonical(
    pw: &mut PartialWitness<F>,
    target: &RegisterStackTarget,
    value: &RegisterStack,
) -> Result<()> {
    for (target, value) in target.stack.iter().zip(value.stack.iter()) {
        set_base_register_noncanonical(pw, target, value)?;
    }
    pw.set_target(target.count, F::from_canonical_usize(value.count))?;
    Ok(())
}

fn set_account_order_noncanonical(
    pw: &mut PartialWitness<F>,
    target: &AccountOrderTarget,
    value: &AccountOrder,
) -> Result<()> {
    let fields = [
        (target.index_0, value.index_0),
        (target.index_1, value.index_1),
        (target.owner_account_index, value.owner_account_index),
        (target.order_index, value.order_index),
        (target.client_order_index, value.client_order_index),
        (target.initial_base_amount, value.initial_base_amount),
        (target.price, i64::from(value.price)),
        (target.nonce, value.nonce),
        (target.remaining_base_amount, value.remaining_base_amount),
        (target.is_ask.target, i64::from(value.is_ask)),
        (target.order_type, i64::from(value.order_type)),
        (target.time_in_force, i64::from(value.time_in_force)),
        (target.reduce_only, i64::from(value.reduce_only)),
        (target.trigger_price, i64::from(value.trigger_price)),
        (target.expiry, value.expiry),
        (target.trigger_status, i64::from(value.trigger_status)),
        (
            target.to_trigger_order_index0,
            value.to_trigger_order_index0,
        ),
        (
            target.to_trigger_order_index1,
            value.to_trigger_order_index1,
        ),
        (target.to_cancel_order_index0, value.to_cancel_order_index0),
        (
            target.integrator_fee_collector_index,
            value.integrator_fee_collector_index,
        ),
        (target.integrator_taker_fee, value.integrator_taker_fee),
        (target.integrator_maker_fee, value.integrator_maker_fee),
        (target.order_flags, value.order_flags as i64),
    ];
    for (target, value) in fields {
        pw.set_target(target, F::from_noncanonical_i64(value))?;
    }
    Ok(())
}

fn set_market_noncanonical(
    pw: &mut PartialWitness<F>,
    target: &MarketTarget,
    value: &Market<F>,
) -> Result<()> {
    let fields = [
        (target.market_index, i64::from(value.market_index)),
        (target.market_type, i64::from(value.market_type)),
        (target.status, i64::from(value.status)),
        (target.base_asset_id, i64::from(value.base_asset_id)),
        (target.quote_asset_id, i64::from(value.quote_asset_id)),
        (target.total_order_count, value.total_order_count),
        (target.ask_nonce, value.ask_nonce),
        (target.bid_nonce, value.bid_nonce),
        (target.taker_fee, i64::from(value.taker_fee)),
        (target.maker_fee, i64::from(value.maker_fee)),
        (target.liquidation_fee, i64::from(value.liquidation_fee)),
        (
            target.size_extension_multiplier,
            value.size_extension_multiplier,
        ),
        (
            target.quote_extension_multiplier,
            value.quote_extension_multiplier,
        ),
        (target.min_base_amount, value.min_base_amount as i64),
        (target.min_quote_amount, value.min_quote_amount as i64),
        (target.order_quote_limit, value.order_quote_limit),
    ];
    for (target, value) in fields {
        pw.set_target(target, F::from_noncanonical_i64(value))?;
    }
    pw.set_hash_target(target.order_book_root, value.order_book_root)?;
    Ok(())
}

fn maker_insert_register() -> BaseRegisterInfo {
    // The fields shared with the generated AccountOrder sum to zero:
    // 10 + 7 + 11 + 10 + 1 + 1 - 40 = 0.
    // generic_field_3 then cancels instruction + market + account in the
    // Register's modular `is_empty` sum. Collector zero disables that field
    // when the register is converted to an AccountOrder.
    BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: MARKET,
        account_index: MAKER,
        pending_size: SIZE,
        pending_initial_size: SIZE,
        pending_price: i64::from(PRICE),
        pending_nonce: MAKER_NONCE,
        pending_is_ask: 1,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: GTT,
        pending_expiry: -40,
        generic_field_3: -(i64::from(INSERT_ORDER) + i64::from(MARKET) + MAKER),
        ..BaseRegisterInfo::empty()
    }
}

fn taker_fill_register() -> BaseRegisterInfo {
    // Again make the order-shaped portion sum to zero, and independently
    // cancel instruction + market + account. IOC makes a full fill pop it.
    BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: MARKET,
        account_index: TAKER,
        pending_size: SIZE,
        pending_initial_size: SIZE,
        pending_price: i64::from(PRICE),
        pending_nonce: TAKER_NONCE,
        pending_is_ask: 0,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: IOC,
        pending_expiry: -(SIZE + SIZE + i64::from(PRICE) + TAKER_NONCE),
        generic_field_3: -(i64::from(INSERT_ORDER) + i64::from(MARKET) + TAKER),
        ..BaseRegisterInfo::empty()
    }
}

fn generated_maker_account_order() -> AccountOrder {
    AccountOrder {
        index_0: 0,
        index_1: 0,
        owner_account_index: MAKER,
        order_index: 0,
        client_order_index: 0,
        initial_base_amount: SIZE,
        price: PRICE,
        nonce: MAKER_NONCE,
        remaining_base_amount: SIZE,
        is_ask: 1,
        order_type: LIMIT_ORDER,
        time_in_force: GTT,
        expiry: -40,
        integrator_fee_collector_index: 0,
        integrator_taker_fee: 0,
        integrator_maker_fee: 0,
        order_flags: 0,
        ..AccountOrder::default()
    }
}

fn supplied_maker_account_order() -> AccountOrder {
    let expiry = 2_000;
    let visible_sum = SIZE
        + i64::from(PRICE)
        + MAKER_NONCE
        + SIZE
        + 1 // is_ask
        + i64::from(GTT)
        + expiry;
    AccountOrder {
        // (market + 1) * 2^48 + nonce. This is path metadata and is not
        // included in AccountOrder::is_empty or its leaf hash.
        index_0: ((i64::from(MARKET) + 1) << ORDER_NONCE_BITS) + MAKER_NONCE,
        index_1: 0,
        owner_account_index: MAKER,
        order_index: 0,
        client_order_index: 0,
        initial_base_amount: SIZE,
        price: PRICE,
        nonce: MAKER_NONCE,
        remaining_base_amount: SIZE,
        is_ask: 1,
        order_type: LIMIT_ORDER,
        time_in_force: GTT,
        expiry,
        integrator_fee_collector_index: 0,
        // Collector zero disables this raw field in matching, while the
        // modular `is_empty` sum still sees it.
        integrator_taker_fee: -visible_sum,
        integrator_maker_fee: 0,
        order_flags: 0,
        ..AccountOrder::default()
    }
}

fn fake_market(total_order_count: i64, order_book_root: HashOut<F>) -> Market<F> {
    // Every scalar except count sums to zero. Count 0 therefore selects the
    // zero leaf hash (hiding order_book_root); count 1 commits the real leaf.
    let scalar_sum_without_limit = 1 // MARKET_TYPE_SPOT
        + 1 // base asset index
        + 2 // quote asset index
        + BASE_MULTIPLIER
        + QUOTE_MULTIPLIER;
    Market {
        market_index: MARKET,
        market_type: 1,
        status: 0,
        base_asset_id: 1,
        quote_asset_id: 2,
        total_order_count,
        size_extension_multiplier: BASE_MULTIPLIER,
        quote_extension_multiplier: QUOTE_MULTIPLIER,
        order_quote_limit: -scalar_sum_without_limit,
        order_book_root,
        ..Market::default()
    }
}

#[test]
fn concrete_mixed_lane_phantom_spot_session_aliases_all_commit_to_zero() -> Result<()> {
    let mut order_book = OrderBookTree::<ORDER_BOOK_MERKLE_LEVELS>::new();
    let empty_order_book_root = order_book.root;
    let inserted_order = Order {
        key_price: i64::from(PRICE),
        key_nonce: MAKER_NONCE,
        ask_base_sum: SIZE,
        ask_quote_sum: SIZE * i64::from(PRICE),
        bid_base_sum: 0,
        bid_quote_sum: 0,
    };
    let order_leaf_index = u128::try_from(MAKER_NONCE)? + (u128::from(PRICE) << ORDER_NONCE_BITS);
    order_book.insert_leaf(order_leaf_index, inserted_order);
    let inserted_order_book_root = order_book.root;

    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let maker_register = RegisterStackTarget::new(&mut builder);
    let taker_register = RegisterStackTarget::new(&mut builder);
    let generated_order = AccountOrderTarget::new(&mut builder);
    let supplied_order = AccountOrderTarget::new(&mut builder);
    let market_before = MarketTarget::new(&mut builder);
    let market_after_insert = MarketTarget::new(&mut builder);
    let market_after_fill = MarketTarget::new(&mut builder);

    let zero = builder.zero_hash_out();
    for hash in [
        maker_register.hash(&mut builder),
        taker_register.hash(&mut builder),
        generated_order.hash(&mut builder),
        supplied_order.hash(&mut builder),
        market_before.hash(&mut builder),
        market_after_fill.hash(&mut builder),
    ] {
        builder.connect_hashes(hash, zero);
    }
    let inserted_market_hash = market_after_insert.hash(&mut builder);
    let inserted_market_is_zero = builder.is_zero_hash_out(&inserted_market_hash);
    builder.assert_false(inserted_market_is_zero);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::<F>::new();
    set_register_stack_noncanonical(
        &mut pw,
        &maker_register,
        &stack_with(maker_insert_register()),
    )?;
    set_register_stack_noncanonical(&mut pw, &taker_register, &stack_with(taker_fill_register()))?;
    set_account_order_noncanonical(&mut pw, &generated_order, &generated_maker_account_order())?;
    set_account_order_noncanonical(&mut pw, &supplied_order, &supplied_maker_account_order())?;
    set_market_noncanonical(
        &mut pw,
        &market_before,
        &fake_market(0, empty_order_book_root),
    )?;
    set_market_noncanonical(
        &mut pw,
        &market_after_insert,
        &fake_market(1, inserted_order_book_root),
    )?;
    set_market_noncanonical(
        &mut pw,
        &market_after_fill,
        &fake_market(0, empty_order_book_root),
    )?;

    // Exact zero-fee endpoint deltas for the consuming (bid-taker) fill.
    let base = SIZE * BASE_MULTIPLIER;
    let quote = SIZE * i64::from(PRICE) * QUOTE_MULTIPLIER;
    assert_eq!((-base) + base, 0); // maker + taker base
    assert_eq!(quote + (-quote), 0); // maker + taker quote
    assert_eq!(base, 30);
    assert_eq!(quote, 350);

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
