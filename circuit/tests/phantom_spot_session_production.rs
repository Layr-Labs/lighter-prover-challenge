use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain::BlockTxChainWitness;
use circuit::block_tx_chain_constraints::{
    BlockTxChainCircuit, Circuit as BlockTxChainCircuitTrait,
};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as BlockTxCircuitTrait};
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::builder::custom::cyclic_base_proof;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::order_book_tree_helpers::OrderBookTree;
use circuit::poseidon2::Poseidon2Hash;
use circuit::tx::Tx;
use circuit::tx_constraints::{TxTarget, TxTargetWitness};
use circuit::types::account::{Account, AccountTarget, AccountTargetWitness};
use circuit::types::account_asset::{AccountAsset, AccountAssetTarget, AccountAssetTargetWitness};
use circuit::types::account_delta::{AccountDelta, AccountDeltaTarget, AccountDeltaTargetWitness};
use circuit::types::account_order::AccountOrder;
use circuit::types::asset::Asset;
use circuit::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::*;
use circuit::types::order::Order;
use circuit::types::register::{BaseRegisterInfo, RegisterStack};
use num::{BigInt, BigUint};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::config::Hasher;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;

const MARKET: u16 = 255;
const MAKER: i64 = 17;
const TAKER: i64 = 23;
const SIZE: i64 = 10;
const PRICE: i64 = 7;
const MAKER_NONCE: i64 = 11;
const TAKER_NONCE: i64 = 12;
const BASE_ASSET: i16 = 1;
const QUOTE_ASSET: i16 = 2;
const BASE_MULTIPLIER: i64 = 3;
const QUOTE_MULTIPLIER: i64 = 5;
const CREATED_AT: i64 = 1_000;

fn template_tx() -> Result<Tx<F>> {
    let value: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../bench/bench_test.json"))?;
    Ok(serde_json::from_value(value["txs"][0].clone())?)
}

fn empty_merkle_proof<const L: usize>() -> [HashOut<F>; L] {
    let mut hash = HashOut::ZERO;
    core::array::from_fn(|_| {
        let result = hash;
        hash = Poseidon2Hash::two_to_one(hash, hash);
        result
    })
}

fn empty_merkle_root<const L: usize>() -> HashOut<F> {
    let mut hash = HashOut::ZERO;
    for _ in 0..L {
        hash = Poseidon2Hash::two_to_one(hash, hash);
    }
    hash
}

fn stack_with(register: BaseRegisterInfo) -> RegisterStack {
    let mut result = RegisterStack {
        stack: [BaseRegisterInfo::empty(); REGISTER_STACK_SIZE],
        count: 1,
    };
    result.stack[0] = register;
    result
}

fn maker_register() -> BaseRegisterInfo {
    BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: MARKET,
        account_index: MAKER,
        pending_size: SIZE,
        pending_initial_size: SIZE,
        pending_price: PRICE,
        pending_nonce: MAKER_NONCE,
        pending_is_ask: 1,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: GTT,
        pending_expiry: -40,
        generic_field_3: -(i64::from(INSERT_ORDER) + i64::from(MARKET) + MAKER),
        ..BaseRegisterInfo::empty()
    }
}

fn taker_register() -> BaseRegisterInfo {
    BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: MARKET,
        account_index: TAKER,
        pending_size: SIZE,
        pending_initial_size: SIZE,
        pending_price: PRICE,
        pending_nonce: TAKER_NONCE,
        pending_is_ask: 0,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: IOC,
        pending_expiry: -(SIZE + SIZE + PRICE + TAKER_NONCE),
        generic_field_3: -(i64::from(INSERT_ORDER) + i64::from(MARKET) + TAKER),
        ..BaseRegisterInfo::empty()
    }
}

fn maker_order_before_insert() -> AccountOrder {
    AccountOrder {
        index_0: 0,
        index_1: 0,
        owner_account_index: MAKER,
        ..AccountOrder::default()
    }
}

fn maker_order_before_fill() -> AccountOrder {
    let expiry = 2_000;
    let visible_sum = SIZE + PRICE + MAKER_NONCE + SIZE + 1 + i64::from(GTT) + expiry;
    AccountOrder {
        index_0: ((i64::from(MARKET) + 1) << ORDER_NONCE_BITS) + MAKER_NONCE,
        index_1: 0,
        owner_account_index: MAKER,
        order_index: 0,
        client_order_index: 0,
        initial_base_amount: SIZE,
        price: PRICE as u32,
        nonce: MAKER_NONCE,
        remaining_base_amount: SIZE,
        is_ask: 1,
        order_type: LIMIT_ORDER,
        time_in_force: GTT,
        expiry,
        integrator_fee_collector_index: 0,
        integrator_taker_fee: -visible_sum,
        integrator_maker_fee: 0,
        order_flags: 0,
        ..AccountOrder::default()
    }
}

fn fake_market(
    total_order_count: i64,
    order_book_root: HashOut<F>,
) -> circuit::types::market::Market<F> {
    circuit::types::market::Market {
        market_index: MARKET,
        market_type: MARKET_TYPE_SPOT as u8,
        base_asset_id: BASE_ASSET as u16,
        quote_asset_id: QUOTE_ASSET as u16,
        total_order_count,
        size_extension_multiplier: BASE_MULTIPLIER,
        quote_extension_multiplier: QUOTE_MULTIPLIER,
        order_quote_limit: -((MARKET_TYPE_SPOT as i64)
            + i64::from(BASE_ASSET)
            + i64::from(QUOTE_ASSET)
            + BASE_MULTIPLIER
            + QUOTE_MULTIPLIER),
        order_book_root,
        ..circuit::types::market::Market::default()
    }
}

fn account(index: i64, balances: [i64; NB_ASSETS_PER_TX]) -> Account<F> {
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

fn account_delta(index: i64, deltas: [i64; NB_ASSETS_PER_TX]) -> AccountDelta<F> {
    AccountDelta {
        account_index: index,
        aggregated_asset_deltas: deltas.map(BigInt::from),
        asset_delta_root: EMPTY_ASSET_TREE_ROOT,
        position_delta_root: EMPTY_POSITION_DELTA_TREE_ROOT,
        ..AccountDelta::default()
    }
}

fn account_assets(
    _index: i64,
    balances: [u64; NB_ASSETS_PER_TX],
    locked: [u64; NB_ASSETS_PER_TX],
) -> [AccountAsset; NB_ASSETS_PER_TX] {
    [
        AccountAsset {
            index_0: i64::from(BASE_ASSET),
            balance: BigUint::from(balances[0]),
            locked_balance: BigUint::from(locked[0]),
        },
        AccountAsset {
            index_0: i64::from(QUOTE_ASSET),
            balance: BigUint::from(balances[1]),
            locked_balance: BigUint::from(locked[1]),
        },
    ]
}

fn set_common_tx_data(
    tx: &mut Tx<F>,
    template: &Tx<F>,
    order_book_path: [circuit::types::order_book_node::OrderBookNode<F>; ORDER_BOOK_MERKLE_LEVELS],
) {
    tx.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;
    tx.internal_claim_order_tx.market_index = MARKET;
    tx.asset_indices = [BASE_ASSET, QUOTE_ASSET];
    tx.all_assets_before = template.all_assets_before.clone();
    tx.all_assets_before[BASE_ASSET as usize] = Asset {
        asset_index: BASE_ASSET,
        extension_multiplier: BASE_MULTIPLIER,
        ..Asset::default()
    };
    tx.all_assets_before[QUOTE_ASSET as usize] = Asset {
        asset_index: QUOTE_ASSET,
        extension_multiplier: QUOTE_MULTIPLIER,
        ..Asset::default()
    };
    tx.order_book_tree_path = order_book_path;

    let empty_ob = OrderBookTree::<ORDER_BOOK_MERKLE_LEVELS>::new();
    tx.impact_ask_order = Order::empty(0, 0);
    tx.impact_bid_order = Order::empty(0, 0);
    tx.impact_ask_order_book_tree_path = empty_ob.proof(0);
    tx.impact_bid_order_book_tree_path = empty_ob.proof(0);

    tx.api_key_tree_merkle_proof = empty_merkle_proof();
    tx.account_orders_tree_merkle_proof = [empty_merkle_proof(); NB_ACCOUNT_ORDERS_PATHS_PER_TX];
    tx.position_delta_merkle_proofs = [empty_merkle_proof(); NB_ACCOUNTS_PER_TX - 1];
    tx.market_details_tree_merkle_proof = empty_merkle_proof();
}

fn build_txs() -> Result<(Tx<F>, Tx<F>, ShadowWitness)> {
    let template = template_tx()?;

    let mut empty_ob = OrderBookTree::<ORDER_BOOK_MERKLE_LEVELS>::new();
    let empty_root = empty_ob.root;
    let order_index = u128::try_from(MAKER_NONCE)? + (u128::try_from(PRICE)? << ORDER_NONCE_BITS);
    let maker_order_leaf = Order {
        key_price: PRICE,
        key_nonce: MAKER_NONCE,
        ask_base_sum: SIZE,
        ask_quote_sum: SIZE * PRICE,
        bid_base_sum: 0,
        bid_quote_sum: 0,
    };
    let empty_path = empty_ob.proof(order_index);
    empty_ob.insert_leaf(order_index, maker_order_leaf.clone());
    let inserted_root = empty_ob.root;
    let inserted_path = empty_ob.proof(order_index);

    let maker_initial = account(MAKER, [1_000, 0]);
    let taker_initial = account(TAKER, [0, 1_000]);
    let treasury_initial = account(TREASURY_ACCOUNT_INDEX as i64, [0, 0]);
    let maker_after_insert = Account {
        total_order_count: 1,
        total_non_cross_order_count: 1,
        ..maker_initial.clone()
    };
    let maker_final = account(MAKER, [990, 70]);
    let taker_final = account(TAKER, [10, 930]);

    let maker_assets_initial = account_assets(MAKER, [3_000, 0], [0, 0]);
    let maker_assets_after_insert = account_assets(MAKER, [3_000, 0], [30, 0]);
    let maker_assets_final = account_assets(MAKER, [2_970, 350], [0, 0]);
    let taker_assets_initial = account_assets(TAKER, [0, 5_000], [0, 0]);
    let taker_assets_final = account_assets(TAKER, [30, 4_650], [0, 0]);
    let treasury_assets = account_assets(TREASURY_ACCOUNT_INDEX as i64, [0, 0], [0, 0]);

    let maker_delta_before = account_delta(MAKER, [0, 0]);
    let taker_delta_before = account_delta(TAKER, [0, 0]);
    let treasury_delta = account_delta(TREASURY_ACCOUNT_INDEX as i64, [0, 0]);
    let maker_delta_final = account_delta(MAKER, [-10, 70]);
    let taker_delta_final = account_delta(TAKER, [10, -70]);

    let mut t1 = template.clone();
    set_common_tx_data(&mut t1, &template, empty_path);
    t1.tx_circuit_type = TX_LIGHT;
    t1.tx_index = 0;
    t1.internal_claim_order_tx.account_index = MAKER;
    t1.register_stack_before = stack_with(maker_register());
    t1.account_order_before = maker_order_before_insert();
    t1.market_before = fake_market(0, empty_root);
    t1.order_before = Order::empty(PRICE, MAKER_NONCE);
    t1.accounts_before[0] = maker_initial.clone();
    t1.accounts_before[1] = taker_initial.clone();
    t1.accounts_before[2] = treasury_initial.clone();
    t1.account_assets_before[0] = maker_assets_initial.clone();
    t1.account_assets_before[1] = taker_assets_initial.clone();
    t1.account_assets_before[2] = treasury_assets.clone();
    t1.accounts_delta_before[0] = maker_delta_before.clone();
    t1.accounts_delta_before[1] = taker_delta_before.clone();
    t1.accounts_delta_before[2] = treasury_delta.clone();

    let mut t2 = template.clone();
    set_common_tx_data(&mut t2, &template, inserted_path);
    t2.tx_circuit_type = TX_HEAVY;
    t2.tx_index = 1;
    t2.internal_claim_order_tx.account_index = TAKER;
    t2.register_stack_before = stack_with(taker_register());
    t2.account_order_before = maker_order_before_fill();
    t2.market_before = fake_market(1, inserted_root);
    t2.order_before = maker_order_leaf;
    t2.accounts_before[0] = taker_initial.clone();
    t2.accounts_before[1] = maker_after_insert;
    t2.accounts_before[2] = treasury_initial;
    t2.account_assets_before[0] = taker_assets_initial.clone();
    t2.account_assets_before[1] = maker_assets_after_insert.clone();
    t2.account_assets_before[2] = treasury_assets;
    t2.accounts_delta_before[0] = taker_delta_before;
    t2.accounts_delta_before[1] = maker_delta_before;
    t2.accounts_delta_before[2] = treasury_delta;

    for tx in [&mut t1, &mut t2] {
        for account_slot in 0..NB_ACCOUNTS_PER_TX {
            tx.account_assets_before[account_slot][0].index_0 = i64::from(BASE_ASSET);
            tx.account_assets_before[account_slot][1].index_0 = i64::from(QUOTE_ASSET);
        }
    }

    Ok((
        t1,
        t2,
        ShadowWitness {
            maker_account: maker_final,
            taker_account: taker_final,
            maker_assets: maker_assets_final,
            taker_assets: taker_assets_final,
            maker_delta: maker_delta_final,
            taker_delta: taker_delta_final,
            maker_asset_after_t1: maker_assets_after_insert[0].clone(),
        },
    ))
}

#[derive(Clone)]
struct ShadowWitness {
    maker_account: Account<F>,
    taker_account: Account<F>,
    maker_assets: [AccountAsset; NB_ASSETS_PER_TX],
    taker_assets: [AccountAsset; NB_ASSETS_PER_TX],
    maker_delta: AccountDelta<F>,
    taker_delta: AccountDelta<F>,
    maker_asset_after_t1: AccountAsset,
}

struct ShadowTargets {
    maker_account: AccountTarget,
    taker_account: AccountTarget,
    maker_assets: [AccountAssetTarget; NB_ASSETS_PER_TX],
    taker_assets: [AccountAssetTarget; NB_ASSETS_PER_TX],
    maker_delta: AccountDeltaTarget,
    taker_delta: AccountDeltaTarget,
    maker_asset_after_t1: AccountAssetTarget,
}

impl ShadowTargets {
    fn new(builder: &mut Builder) -> Self {
        Self {
            maker_account: AccountTarget::new(builder),
            taker_account: AccountTarget::new(builder),
            maker_assets: core::array::from_fn(|_| AccountAssetTarget::new(builder)),
            taker_assets: core::array::from_fn(|_| AccountAssetTarget::new(builder)),
            maker_delta: AccountDeltaTarget::new(builder),
            taker_delta: AccountDeltaTarget::new(builder),
            maker_asset_after_t1: AccountAssetTarget::new(builder),
        }
    }
}

fn sparse_tree_targets<const L: usize>(
    builder: &mut Builder,
    leaves: &[(u64, HashOutTarget)],
) -> (HashOutTarget, Vec<[HashOutTarget; L]>) {
    let mut zero_hashes = Vec::with_capacity(L + 1);
    zero_hashes.push(builder.zero_hash_out());
    for level in 0..L {
        let next = builder.hash_two_to_one(&zero_hashes[level], &zero_hashes[level]);
        zero_hashes.push(next);
    }

    let mut current: HashMap<u64, HashOutTarget> = leaves.iter().copied().collect();
    let mut indices: Vec<u64> = leaves.iter().map(|(index, _)| *index).collect();
    let mut proofs = vec![[builder.zero_hash_out(); L]; leaves.len()];

    for level in 0..L {
        for (leaf_number, index) in indices.iter().enumerate() {
            proofs[leaf_number][level] = current
                .get(&(index ^ 1))
                .copied()
                .unwrap_or(zero_hashes[level]);
        }

        let parent_indices: HashSet<u64> = current.keys().map(|index| index >> 1).collect();
        let mut next = HashMap::new();
        for parent in parent_indices {
            let left = current
                .get(&(parent << 1))
                .copied()
                .unwrap_or(zero_hashes[level]);
            let right = current
                .get(&((parent << 1) | 1))
                .copied()
                .unwrap_or(zero_hashes[level]);
            next.insert(parent, builder.hash_two_to_one(&left, &right));
        }
        current = next;
        for index in &mut indices {
            *index >>= 1;
        }
    }

    (current.get(&0).copied().unwrap_or(zero_hashes[L]), proofs)
}

fn wire_hash(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    destination: HashOutTarget,
    source: HashOutTarget,
) {
    builder.connect_hashes(destination, source);
    witness_owned.push(destination);
}

fn wire_proof<const L: usize>(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    destination: &[HashOutTarget; L],
    source: &[HashOutTarget; L],
) {
    for (&destination, &source) in destination.iter().zip(source) {
        wire_hash(builder, witness_owned, destination, source);
    }
}

fn wire_two_leaf_root<const L: usize>(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    root_target: HashOutTarget,
    hashes: [HashOutTarget; NB_ASSETS_PER_TX],
) {
    let (root, _) = sparse_tree_targets::<L>(
        builder,
        &[
            (BASE_ASSET as u64, hashes[0]),
            (QUOTE_ASSET as u64, hashes[1]),
        ],
    );
    wire_hash(builder, witness_owned, root_target, root);
}

fn wire_two_leaf_update_proofs<const L: usize>(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    root_target: HashOutTarget,
    proof_targets: &[[HashOutTarget; L]; NB_ASSETS_PER_TX],
    old: [HashOutTarget; NB_ASSETS_PER_TX],
    new: [HashOutTarget; NB_ASSETS_PER_TX],
) {
    let (old_root, old_proofs) = sparse_tree_targets::<L>(
        builder,
        &[(BASE_ASSET as u64, old[0]), (QUOTE_ASSET as u64, old[1])],
    );
    wire_hash(builder, witness_owned, root_target, old_root);
    wire_proof(builder, witness_owned, &proof_targets[0], &old_proofs[0]);

    let (_, after_first_proofs) = sparse_tree_targets::<L>(
        builder,
        &[(BASE_ASSET as u64, new[0]), (QUOTE_ASSET as u64, old[1])],
    );
    wire_proof(
        builder,
        witness_owned,
        &proof_targets[1],
        &after_first_proofs[1],
    );
}

fn account_hashes(
    builder: &mut Builder,
    account: &AccountTarget,
) -> (HashOutTarget, HashOutTarget) {
    let buckets = account.get_position_bucket_hashes(builder);
    let (hash, public_hash, _) = account.hash(builder, &buckets);
    (hash, public_hash)
}

fn wire_shadow_inner_roots(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    account: &AccountTarget,
    assets: &[AccountAssetTarget; NB_ASSETS_PER_TX],
    delta: &AccountDeltaTarget,
) {
    let asset_hashes = core::array::from_fn(|i| assets[i].hash(builder));
    wire_two_leaf_root::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        account.asset_root,
        asset_hashes,
    );
    let aggregated_balance_hashes =
        core::array::from_fn(|i| account.aggregated_balance_hash(builder, i));
    wire_two_leaf_root::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        account.aggregated_balances_root,
        aggregated_balance_hashes,
    );
    let aggregated_asset_delta_hashes =
        core::array::from_fn(|i| delta.aggregated_asset_delta_hash(builder, i));
    wire_two_leaf_root::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        delta.asset_delta_root,
        aggregated_asset_delta_hashes,
    );
}

fn wire_tx_account_inner_roots(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    target: &TxTarget,
) {
    for account_slot in 0..NB_ACCOUNTS_PER_TX {
        let asset_hashes =
            core::array::from_fn(|i| target.account_assets_before[account_slot][i].hash(builder));
        wire_two_leaf_root::<ASSET_MERKLE_LEVELS>(
            builder,
            witness_owned,
            target.accounts_before[account_slot].asset_root,
            asset_hashes,
        );
        let aggregated_balance_hashes = core::array::from_fn(|i| {
            target.accounts_before[account_slot].aggregated_balance_hash(builder, i)
        });
        wire_two_leaf_root::<ASSET_MERKLE_LEVELS>(
            builder,
            witness_owned,
            target.accounts_before[account_slot].aggregated_balances_root,
            aggregated_balance_hashes,
        );
    }
}

fn wire_light_state(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    target: &TxTarget,
    shadow: &ShadowTargets,
) {
    wire_tx_account_inner_roots(builder, witness_owned, target);

    let (maker_hash, maker_pub_hash) = account_hashes(builder, &target.accounts_before[0]);
    let (taker_hash, taker_pub_hash) = account_hashes(builder, &target.accounts_before[1]);
    let fee_hashes = target.accounts_before[2].fee_account_hash(builder);
    let leaves = [
        (MAKER as u64, maker_hash),
        (TAKER as u64, taker_hash),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.0),
    ];
    let (root, proofs) = sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &leaves);
    wire_hash(builder, witness_owned, target.old_account_tree_root, root);
    wire_proof(
        builder,
        witness_owned,
        &target.account_tree_merkle_proofs[0],
        &proofs[0],
    );

    let public_leaves = [
        (MAKER as u64, maker_pub_hash),
        (TAKER as u64, taker_pub_hash),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.1),
    ];
    let (public_root, _) = sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &public_leaves);
    wire_hash(
        builder,
        witness_owned,
        target.old_account_pub_data_tree_root,
        public_root,
    );

    let old_asset_hashes =
        core::array::from_fn(|i| target.account_assets_before[0][i].hash(builder));
    let new_asset_hashes = [
        shadow.maker_asset_after_t1.hash(builder),
        old_asset_hashes[1],
    ];
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_before[0].asset_root,
        &target.asset_tree_merkle_proofs[0],
        old_asset_hashes,
        new_asset_hashes,
    );
}

fn wire_heavy_asset_proofs_for_account(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    target: &TxTarget,
    account_slot: usize,
    final_account: &AccountTarget,
    final_assets: &[AccountAssetTarget; NB_ASSETS_PER_TX],
    final_delta: &AccountDeltaTarget,
) {
    let old_asset_hashes =
        core::array::from_fn(|i| target.account_assets_before[account_slot][i].hash(builder));
    let final_asset_hashes = core::array::from_fn(|i| final_assets[i].hash(builder));
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_before[account_slot].asset_root,
        &target.asset_tree_merkle_proofs[account_slot],
        old_asset_hashes,
        final_asset_hashes,
    );
    let old_aggregated_balance_hashes = core::array::from_fn(|i| {
        target.accounts_before[account_slot].aggregated_balance_hash(builder, i)
    });
    let final_aggregated_balance_hashes =
        core::array::from_fn(|i| final_account.aggregated_balance_hash(builder, i));
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_before[account_slot].aggregated_balances_root,
        &target.public_asset_tree_merkle_proofs[account_slot],
        old_aggregated_balance_hashes,
        final_aggregated_balance_hashes,
    );
    let old_aggregated_asset_delta_hashes = core::array::from_fn(|i| {
        target.accounts_delta_before[account_slot].aggregated_asset_delta_hash(builder, i)
    });
    let final_aggregated_asset_delta_hashes =
        core::array::from_fn(|i| final_delta.aggregated_asset_delta_hash(builder, i));
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_delta_before[account_slot].asset_delta_root,
        &target.asset_delta_tree_merkle_proofs[account_slot],
        old_aggregated_asset_delta_hashes,
        final_aggregated_asset_delta_hashes,
    );
}

fn wire_heavy_state(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    target: &TxTarget,
    shadow: &ShadowTargets,
) {
    wire_tx_account_inner_roots(builder, witness_owned, target);
    wire_shadow_inner_roots(
        builder,
        witness_owned,
        &shadow.taker_account,
        &shadow.taker_assets,
        &shadow.taker_delta,
    );
    wire_shadow_inner_roots(
        builder,
        witness_owned,
        &shadow.maker_account,
        &shadow.maker_assets,
        &shadow.maker_delta,
    );

    wire_heavy_asset_proofs_for_account(
        builder,
        witness_owned,
        target,
        0,
        &shadow.taker_account,
        &shadow.taker_assets,
        &shadow.taker_delta,
    );
    wire_heavy_asset_proofs_for_account(
        builder,
        witness_owned,
        target,
        1,
        &shadow.maker_account,
        &shadow.maker_assets,
        &shadow.maker_delta,
    );

    let fee_asset_hashes =
        core::array::from_fn(|i| target.account_assets_before[2][i].hash(builder));
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_before[2].asset_root,
        &target.asset_tree_merkle_proofs[2],
        fee_asset_hashes,
        fee_asset_hashes,
    );
    let fee_balance_hashes =
        core::array::from_fn(|i| target.accounts_before[2].aggregated_balance_hash(builder, i));
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_before[2].aggregated_balances_root,
        &target.public_asset_tree_merkle_proofs[2],
        fee_balance_hashes,
        fee_balance_hashes,
    );
    let fee_delta_hashes = core::array::from_fn(|i| {
        target.accounts_delta_before[2].aggregated_asset_delta_hash(builder, i)
    });
    wire_two_leaf_update_proofs::<ASSET_MERKLE_LEVELS>(
        builder,
        witness_owned,
        target.accounts_delta_before[2].asset_delta_root,
        &target.asset_delta_tree_merkle_proofs[2],
        fee_delta_hashes,
        fee_delta_hashes,
    );

    let (old_taker, old_taker_pub) = account_hashes(builder, &target.accounts_before[0]);
    let (old_maker, old_maker_pub) = account_hashes(builder, &target.accounts_before[1]);
    let (new_taker, new_taker_pub) = account_hashes(builder, &shadow.taker_account);
    let (new_maker, new_maker_pub) = account_hashes(builder, &shadow.maker_account);
    let fee_hashes = target.accounts_before[2].fee_account_hash(builder);

    let initial = [
        (TAKER as u64, old_taker),
        (MAKER as u64, old_maker),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.0),
    ];
    let (initial_root, initial_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &initial);
    wire_hash(
        builder,
        witness_owned,
        target.old_account_tree_root,
        initial_root,
    );
    wire_proof(
        builder,
        witness_owned,
        &target.account_tree_merkle_proofs[0],
        &initial_proofs[0],
    );
    let after_taker = [
        (TAKER as u64, new_taker),
        (MAKER as u64, old_maker),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.0),
    ];
    let (_, after_taker_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_taker);
    wire_proof(
        builder,
        witness_owned,
        &target.account_tree_merkle_proofs[1],
        &after_taker_proofs[1],
    );
    let after_both = [
        (TAKER as u64, new_taker),
        (MAKER as u64, new_maker),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.0),
    ];
    let (_, after_both_proofs) = sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_both);
    wire_proof(
        builder,
        witness_owned,
        &target.account_tree_merkle_proofs[2],
        &after_both_proofs[2],
    );

    let initial_public = [
        (TAKER as u64, old_taker_pub),
        (MAKER as u64, old_maker_pub),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.1),
    ];
    let (initial_public_root, initial_public_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &initial_public);
    wire_hash(
        builder,
        witness_owned,
        target.old_account_pub_data_tree_root,
        initial_public_root,
    );
    wire_proof(
        builder,
        witness_owned,
        &target.account_pub_data_tree_merkle_proofs[0],
        &initial_public_proofs[0],
    );
    let after_taker_public = [
        (TAKER as u64, new_taker_pub),
        (MAKER as u64, old_maker_pub),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.1),
    ];
    let (_, after_taker_public_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_taker_public);
    wire_proof(
        builder,
        witness_owned,
        &target.account_pub_data_tree_merkle_proofs[1],
        &after_taker_public_proofs[1],
    );
    let after_both_public = [
        (TAKER as u64, new_taker_pub),
        (MAKER as u64, new_maker_pub),
        (TREASURY_ACCOUNT_INDEX as u64, fee_hashes.1),
    ];
    let (_, after_both_public_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_both_public);
    wire_proof(
        builder,
        witness_owned,
        &target.account_pub_data_tree_merkle_proofs[2],
        &after_both_public_proofs[2],
    );

    let old_taker_delta = target.accounts_delta_before[0].hash(builder);
    let old_maker_delta = target.accounts_delta_before[1].hash(builder);
    let old_fee_delta = target.accounts_delta_before[2].fee_account_hash(builder);
    let new_taker_delta = shadow.taker_delta.hash(builder);
    let new_maker_delta = shadow.maker_delta.hash(builder);
    let initial_delta = [
        (TAKER as u64, old_taker_delta),
        (MAKER as u64, old_maker_delta),
        (TREASURY_ACCOUNT_INDEX as u64, old_fee_delta),
    ];
    let (initial_delta_root, initial_delta_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &initial_delta);
    wire_hash(
        builder,
        witness_owned,
        target.old_account_delta_tree_root,
        initial_delta_root,
    );
    wire_proof(
        builder,
        witness_owned,
        &target.account_delta_tree_merkle_proofs[0],
        &initial_delta_proofs[0],
    );
    let after_taker_delta = [
        (TAKER as u64, new_taker_delta),
        (MAKER as u64, old_maker_delta),
        (TREASURY_ACCOUNT_INDEX as u64, old_fee_delta),
    ];
    let (_, after_taker_delta_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_taker_delta);
    wire_proof(
        builder,
        witness_owned,
        &target.account_delta_tree_merkle_proofs[1],
        &after_taker_delta_proofs[1],
    );
    let after_both_delta = [
        (TAKER as u64, new_taker_delta),
        (MAKER as u64, new_maker_delta),
        (TREASURY_ACCOUNT_INDEX as u64, old_fee_delta),
    ];
    let (_, after_both_delta_proofs) =
        sparse_tree_targets::<ACCOUNT_MERKLE_LEVELS>(builder, &after_both_delta);
    wire_proof(
        builder,
        witness_owned,
        &target.account_delta_tree_merkle_proofs[2],
        &after_both_delta_proofs[2],
    );
}

fn wire_market_trees(
    builder: &mut Builder,
    witness_owned: &mut Vec<HashOutTarget>,
    target: &TxTarget,
) {
    let market_hash = target.market_before.hash(builder);
    let (market_root, market_proofs) =
        sparse_tree_targets::<MARKET_MERKLE_LEVELS>(builder, &[(u64::from(MARKET), market_hash)]);
    wire_hash(
        builder,
        witness_owned,
        target.old_market_tree_root,
        market_root,
    );
    wire_proof(
        builder,
        witness_owned,
        &target.market_tree_merkle_proof,
        &market_proofs[0],
    );

    let details_hash = target.market_details_before.hash(builder);
    let (details_root, details_proofs) = sparse_tree_targets::<MARKET_DETAILS_TREE_HEIGHT>(
        builder,
        &[(u64::from(NIL_MARKET_INDEX), details_hash)],
    );
    wire_hash(
        builder,
        witness_owned,
        target.old_market_details_tree_root,
        details_root,
    );
    wire_proof(
        builder,
        witness_owned,
        &target.market_details_tree_merkle_proof,
        &details_proofs[0],
    );
}

fn sanitize_for_stock_setter(tx: &Tx<F>) -> Tx<F> {
    let mut result = tx.clone();
    result.register_stack_before.stack[0].pending_expiry = 0;
    result.register_stack_before.stack[0].generic_field_3 = 0;
    result.market_before.order_quote_limit = 0;
    result.account_order_before.integrator_taker_fee = 0;
    result
}

fn overwrite_alias_values(pw: &mut PartialWitness<F>, target: &TxTarget, tx: &Tx<F>) {
    let register = tx.register_stack_before.stack[0];
    pw.target_values.insert(
        target.register_stack_before.stack[0].pending_expiry,
        F::from_noncanonical_i64(register.pending_expiry),
    );
    pw.target_values.insert(
        target.register_stack_before.stack[0].generic_field_3,
        F::from_noncanonical_i64(register.generic_field_3),
    );
    pw.target_values.insert(
        target.market_before.order_quote_limit,
        F::from_noncanonical_i64(tx.market_before.order_quote_limit),
    );
    pw.target_values.insert(
        target.account_order_before.integrator_taker_fee,
        F::from_noncanonical_i64(tx.account_order_before.integrator_taker_fee),
    );
}

fn unset_hashes(pw: &mut PartialWitness<F>, hashes: &[HashOutTarget]) {
    for hash in hashes {
        for target in hash.elements {
            pw.target_values.remove(&target);
        }
    }
}

fn set_shadow_witness(
    pw: &mut PartialWitness<F>,
    targets: &ShadowTargets,
    witness: &ShadowWitness,
) -> Result<()> {
    pw.set_account_target(&targets.maker_account, &witness.maker_account)?;
    pw.set_account_target(&targets.taker_account, &witness.taker_account)?;
    pw.set_account_delta_target(&targets.maker_delta, &witness.maker_delta)?;
    pw.set_account_delta_target(&targets.taker_delta, &witness.taker_delta)?;
    for i in 0..NB_ASSETS_PER_TX {
        pw.set_account_asset_target(&targets.maker_assets[i], &witness.maker_assets[i])?;
        pw.set_account_asset_target(&targets.taker_assets[i], &witness.taker_assets[i])?;
    }
    pw.set_account_asset_target(&targets.maker_asset_after_t1, &witness.maker_asset_after_t1)?;
    Ok(())
}

#[ignore = "large production light+heavy TxTarget proof; run explicitly"]
#[test]
fn production_light_insert_then_heavy_fill_proves_and_chains_all_roots() -> Result<()> {
    let (t1, t2, shadow_witness) = build_txs()?;

    let mut builder = Builder::new(CIRCUIT_CONFIG);
    let mut t1_target = TxTarget::new(&mut builder);
    let mut t2_target = TxTarget::new(&mut builder);
    let shadow_targets = ShadowTargets::new(&mut builder);
    let block_created_at = builder.constant_i64(CREATED_AT);
    let state_metadata_hash = builder.zero_hash_out();

    let t1_outputs =
        t1_target.define_light(0, 0, &mut builder, block_created_at, state_metadata_hash);
    let t2_outputs = t2_target.define(1, 0, &mut builder, block_created_at, state_metadata_hash);

    builder.connect_hashes(t1_target.new_validium_root, t2_target.old_validium_root);
    builder.connect_hashes(t1_target.new_state_root, t2_target.old_state_root);
    builder.connect_hashes(t1_outputs.5, t2_target.old_account_delta_tree_root);
    builder.connect_hashes(t1_outputs.4, t2_outputs.4);
    builder.assert_false(t1_outputs.1);
    builder.assert_false(t1_outputs.3);
    builder.assert_false(t2_outputs.1);
    builder.assert_false(t2_outputs.3);

    let mut witness_owned_hashes = vec![
        t1_target.old_validium_root,
        t1_target.old_state_root,
        t2_target.old_validium_root,
        t2_target.old_state_root,
    ];
    wire_light_state(
        &mut builder,
        &mut witness_owned_hashes,
        &t1_target,
        &shadow_targets,
    );
    wire_heavy_state(
        &mut builder,
        &mut witness_owned_hashes,
        &t2_target,
        &shadow_targets,
    );
    wire_market_trees(&mut builder, &mut witness_owned_hashes, &t1_target);
    wire_market_trees(&mut builder, &mut witness_owned_hashes, &t2_target);

    let empty_account_delta_root =
        builder.constant_hash(empty_merkle_root::<ACCOUNT_MERKLE_LEVELS>());
    wire_hash(
        &mut builder,
        &mut witness_owned_hashes,
        t1_target.old_account_delta_tree_root,
        empty_account_delta_root,
    );

    builder.perform_registered_range_checks();
    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();
    pw.set_tx_target(&t1_target, &sanitize_for_stock_setter(&t1))?;
    pw.set_tx_target(&t2_target, &sanitize_for_stock_setter(&t2))?;
    set_shadow_witness(&mut pw, &shadow_targets, &shadow_witness)?;
    overwrite_alias_values(&mut pw, &t1_target, &t1);
    overwrite_alias_values(&mut pw, &t2_target, &t2);
    unset_hashes(&mut pw, &witness_owned_hashes);

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

struct ProvedBlockTxLane {
    data: CircuitData<F, C, D>,
    proof: ProofWithPublicInputs<F, C, D>,
    witness: BlockTxWitness<F>,
}

fn prove_light_block_tx_lane(
    t1: &Tx<F>,
    shadow_witness: &ShadowWitness,
) -> Result<ProvedBlockTxLane> {
    let mut circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, 1, 0, TX_LIGHT);
    let shadow_targets = ShadowTargets::new(&mut circuit.builder);
    let target = &circuit.target.txs[0];
    let mut witness_owned_hashes = vec![target.old_validium_root, target.old_state_root];

    wire_light_state(
        &mut circuit.builder,
        &mut witness_owned_hashes,
        target,
        &shadow_targets,
    );
    wire_market_trees(&mut circuit.builder, &mut witness_owned_hashes, target);

    let empty_account_delta_root = circuit
        .builder
        .constant_hash(empty_merkle_root::<ACCOUNT_MERKLE_LEVELS>());
    wire_hash(
        &mut circuit.builder,
        &mut witness_owned_hashes,
        target.old_account_delta_tree_root,
        empty_account_delta_root,
    );

    // The light path starts at the block anchor. Leave these witness values
    // generator-owned so the same calculated roots feed the jump state.
    for destination in [
        circuit.target.old_jump.prev_new_state_root,
        circuit.target.old_jump.run_start_old_state_root,
    ] {
        wire_hash(
            &mut circuit.builder,
            &mut witness_owned_hashes,
            destination,
            target.old_state_root,
        );
    }
    for destination in [
        circuit.target.old_jump.prev_new_delta_root,
        circuit.target.old_jump.run_start_old_delta_root,
    ] {
        wire_hash(
            &mut circuit.builder,
            &mut witness_owned_hashes,
            destination,
            target.old_account_delta_tree_root,
        );
    }

    circuit.builder.perform_registered_range_checks();
    let target = circuit.target;
    let data = circuit.builder.build::<C>();

    let block_tx = BlockTx {
        created_at: CREATED_AT,
        state_metadata_hash: HashOut::ZERO,
        old_jump: JumpState::initial(HashOut::ZERO, HashOut::ZERO),
        txs: vec![Arc::new(sanitize_for_stock_setter(t1))],
    };
    let mut pw = BlockTxCircuit::generate_witness(&block_tx, &target)?;
    set_shadow_witness(&mut pw, &shadow_targets, shadow_witness)?;
    overwrite_alias_values(&mut pw, &target.txs[0], t1);
    unset_hashes(&mut pw, &witness_owned_hashes);

    let proof = data.prove(pw)?;
    data.verify(proof.clone())?;
    let witness = BlockTxWitness::from_public_inputs(&proof.public_inputs);
    Ok(ProvedBlockTxLane {
        data,
        proof,
        witness,
    })
}

fn prove_heavy_block_tx_lane(
    t2: &Tx<F>,
    shadow_witness: &ShadowWitness,
    initial_state_root: HashOut<F>,
    initial_delta_root: HashOut<F>,
    expected_old_validium_root: HashOut<F>,
    expected_old_state_root: HashOut<F>,
    expected_old_delta_root: HashOut<F>,
) -> Result<ProvedBlockTxLane> {
    let mut circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, 1, 0, TX_HEAVY);
    let shadow_targets = ShadowTargets::new(&mut circuit.builder);
    let target = &circuit.target.txs[0];
    let mut witness_owned_hashes = vec![target.old_validium_root, target.old_state_root];

    wire_heavy_state(
        &mut circuit.builder,
        &mut witness_owned_hashes,
        target,
        &shadow_targets,
    );
    wire_market_trees(&mut circuit.builder, &mut witness_owned_hashes, target);

    for (destination, expected) in [
        (target.old_validium_root, expected_old_validium_root),
        (target.old_state_root, expected_old_state_root),
        (target.old_account_delta_tree_root, expected_old_delta_root),
    ] {
        let expected = circuit.builder.constant_hash(expected);
        wire_hash(
            &mut circuit.builder,
            &mut witness_owned_hashes,
            destination,
            expected,
        );
    }

    circuit.builder.perform_registered_range_checks();
    let target = circuit.target;
    let data = circuit.builder.build::<C>();

    let block_tx = BlockTx {
        created_at: CREATED_AT,
        state_metadata_hash: HashOut::ZERO,
        old_jump: JumpState::initial(initial_state_root, initial_delta_root),
        txs: vec![Arc::new(sanitize_for_stock_setter(t2))],
    };
    let mut pw = BlockTxCircuit::generate_witness(&block_tx, &target)?;
    set_shadow_witness(&mut pw, &shadow_targets, shadow_witness)?;
    overwrite_alias_values(&mut pw, &target.txs[0], t2);
    unset_hashes(&mut pw, &witness_owned_hashes);

    let proof = data.prove(pw)?;
    data.verify(proof.clone())?;
    let witness = BlockTxWitness::from_public_inputs(&proof.public_inputs);
    Ok(ProvedBlockTxLane {
        data,
        proof,
        witness,
    })
}

fn prove_one_step_chain(
    tx_data: &CircuitData<F, C, D>,
    tx_proof: &ProofWithPublicInputs<F, C, D>,
    initial_state_root: HashOut<F>,
    initial_delta_root: HashOut<F>,
) -> Result<BlockTxChainWitness<F>> {
    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, tx_data, 1);
    let target = chain.target;
    let data = chain.builder.build::<C>();
    let dummy_circuit_data = dummy_circuit(&data.common);
    let dummy_slot_proof = cyclic_base_proof(
        &data.common,
        &data.verifier_only,
        &dummy_circuit_data,
        Vec::<F>::new().iter().copied().enumerate().collect(),
    )?;
    let base = BlockTxChainCircuit::cyclic_base_proof(
        &data,
        &dummy_circuit_data,
        1,
        CREATED_AT,
        initial_state_root,
        HashOut::ZERO,
        initial_delta_root,
    );
    let proof = BlockTxChainCircuit::prove(&target, &data, 0, &base, &dummy_slot_proof, tx_proof)?;
    data.verify(proof.clone())?;
    Ok(BlockTxChainWitness::from_public_inputs(
        &proof.public_inputs,
        1,
        1,
    ))
}

fn finalize_jump_native(
    jump: &JumpState<F>,
    total_tx_count: u64,
    final_state_root: HashOut<F>,
    final_delta_root: HashOut<F>,
) -> (HashOut<F>, HashOut<F>) {
    let total_tx_count = F::from_canonical_u64(total_tx_count);
    let prev_plus_one = jump.last_active_tx_index + F::ONE;

    let coverage = if jump.last_active_tx_index != jump.run_start_prev_index {
        let mut input = vec![jump.run_start_prev_index, prev_plus_one];
        input.extend(jump.run_start_old_state_root.elements);
        input.extend(jump.prev_new_state_root.elements);
        input.extend(jump.run_start_old_delta_root.elements);
        input.extend(jump.prev_new_delta_root.elements);
        Poseidon2Hash::two_to_one(jump.coverage_hash, Poseidon2Hash::hash_no_pad(&input))
    } else {
        jump.coverage_hash
    };

    let claims = if prev_plus_one != total_tx_count {
        let mut input = vec![jump.last_active_tx_index, total_tx_count];
        input.extend(jump.prev_new_state_root.elements);
        input.extend(final_state_root.elements);
        input.extend(jump.prev_new_delta_root.elements);
        input.extend(final_delta_root.elements);
        Poseidon2Hash::two_to_one(jump.claims_hash, Poseidon2Hash::hash_no_pad(&input))
    } else {
        jump.claims_hash
    };

    (coverage, claims)
}

#[ignore = "large production BlockTx plus recursive chain proof; run explicitly"]
#[test]
fn production_cross_lane_light_then_heavy_claims_join() -> Result<()> {
    let (t1, t2, shadow_witness) = build_txs()?;

    let light = prove_light_block_tx_lane(&t1, &shadow_witness)?;
    assert_eq!(light.witness.old_jump.last_active_tx_index, F::NEG_ONE);
    assert_eq!(light.witness.new_jump.last_active_tx_index, F::ZERO);
    assert_eq!(light.witness.new_jump.tx_count, F::ONE);

    let initial_state_root = light.witness.old_jump.prev_new_state_root;
    let initial_delta_root = light.witness.old_jump.prev_new_delta_root;
    let heavy = prove_heavy_block_tx_lane(
        &t2,
        &shadow_witness,
        initial_state_root,
        initial_delta_root,
        light.witness.new_validium_root,
        light.witness.new_state_root,
        light.witness.new_account_delta_tree_root,
    )?;
    assert_eq!(heavy.witness.old_jump.last_active_tx_index, F::NEG_ONE);
    assert_eq!(heavy.witness.new_jump.last_active_tx_index, F::ONE);
    assert_eq!(heavy.witness.new_jump.tx_count, F::ONE);

    let light_chain = prove_one_step_chain(
        &light.data,
        &light.proof,
        initial_state_root,
        initial_delta_root,
    )?;
    let heavy_chain = prove_one_step_chain(
        &heavy.data,
        &heavy.proof,
        initial_state_root,
        initial_delta_root,
    )?;

    let final_state_root = heavy_chain.new_state_root;
    let final_delta_root = heavy_chain.new_account_delta_tree_root;
    let (light_coverage, light_claims) =
        finalize_jump_native(&light_chain.jump, 2, final_state_root, final_delta_root);
    let (heavy_coverage, heavy_claims) =
        finalize_jump_native(&heavy_chain.jump, 2, final_state_root, final_delta_root);

    // These are exactly the two cross-lane equalities enforced by
    // BlockCircuit::define_jump_checks. They demonstrate that the light index
    // 0 run and the heavy index 1 run cover one continuous global execution.
    assert_eq!(heavy_coverage, light_claims);
    assert_eq!(light_coverage, heavy_claims);
    Ok(())
}
