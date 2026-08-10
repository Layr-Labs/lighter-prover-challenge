// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Callback-free materialization of the narrow phantom spot-session primitive.
//!
//! The public constructor starts from a scanner-certified interval and the original left/right
//! boundary witnesses. It normalizes their sequential Merkle paths, synthesizes the otherwise
//! nonexistent after-T1 state, replays both transactions natively, and returns them only when the
//! replay lands exactly on the authenticated original right boundary.

use anyhow::{Context, Result, ensure};
use circuit::block::Block;
use circuit::order_book_tree_helpers::OrderBookTree;
use circuit::poseidon2::Poseidon2Hash;
use circuit::tx::Tx;
use circuit::types::account::Account;
use circuit::types::account_asset::AccountAsset;
use circuit::types::account_delta::{AccountDelta, PositionDelta};
use circuit::types::account_order::AccountOrder;
use circuit::types::api_key::ApiKey;
use circuit::types::config::F;
use circuit::types::constants::*;
use circuit::types::market::Market;
use circuit::types::order::Order;
use circuit::types::register::{BaseRegisterInfo, RegisterStack};
use num::{BigInt, BigUint};
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::config::Hasher;

use crate::phantom_native::{
    account_asset_hash, account_delta_hash, account_hash, account_order_hash, all_assets_hash,
    all_margined_assets_hash, all_market_details_hashes, apply_one_update_to_proof,
    bigint_leaf_hash, fee_account_delta_hash, fee_account_hash, market_details_hash, market_hash,
    merkle_root, register_stack_hash, rewind_one_update, system_config_hash,
    validium_and_state_root,
};
use crate::phantom_spot::{Candidate, lane_proof_saving};

const PHANTOM_MAKER_NONCE: i64 = 11;
const PHANTOM_TAKER_NONCE: i64 = 12;

#[derive(Clone, Debug)]
pub struct PhantomSpotRequest {
    /// Original removed interval is `[start_tx_index, right_tx_index)`.
    start_tx_index: u64,
    right_tx_index: u64,
    created_at: i64,
    /// Hash of the post-pre-execution metadata, not the JSON block's pre-execution metadata.
    state_metadata_hash: HashOut<F>,
    maker_account_index: i64,
    taker_account_index: i64,
    base_asset_index: i16,
    quote_asset_index: i16,
    size: i64,
    price: i64,
    maker_order_expiry: i64,
    total_heavy: usize,
    total_light: usize,
    removed_heavy: usize,
    removed_light: usize,
    expected_proof_saving: usize,
}

impl PhantomSpotRequest {
    fn from_scanned_candidate(
        block: &Block<F>,
        candidate: &Candidate,
        state_metadata_hash: HashOut<F>,
    ) -> Result<Self> {
        ensure!(
            candidate.has_direct_v1_donors(),
            "candidate lacks direct V1 donors"
        );
        let mut active = block
            .tx_chunks
            .iter()
            .flatten()
            .filter(|tx| !tx.is_empty())
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|tx| tx.tx_index);
        ensure!(
            active
                .windows(2)
                .all(|pair| pair[0].tx_index.checked_add(1) == Some(pair[1].tx_index)),
            "active transaction indices are not contiguous"
        );
        ensure!(
            active
                .iter()
                .all(|tx| matches!(tx.tx_circuit_type, TX_HEAVY | TX_LIGHT)),
            "active transaction has an unsupported circuit lane"
        );
        let start = active
            .iter()
            .position(|tx| tx.tx_index == candidate.start_tx_index)
            .context("candidate start is not active")?;
        let end = active
            .iter()
            .position(|tx| tx.tx_index == candidate.end_tx_index)
            .context("candidate right boundary is not active")?;
        ensure!(end > start, "candidate interval is reversed");
        ensure!(
            end - start == candidate.replaced_tx_count && end - start >= 3,
            "candidate replacement count does not match exact interval"
        );
        ensure!(
            active[start..end]
                .iter()
                .all(|tx| is_pubdata_free_order_tx(tx.tx_type)),
            "candidate interval contains a public-data transaction"
        );
        let total_light = active
            .iter()
            .filter(|tx| tx.tx_circuit_type == TX_LIGHT)
            .count();
        let total_heavy = active
            .len()
            .checked_sub(total_light)
            .context("lane count underflow")?;
        let removed_light = active[start..end]
            .iter()
            .filter(|tx| tx.tx_circuit_type == TX_LIGHT)
            .count();
        let removed_heavy = end - start - removed_light;
        ensure!(
            removed_light == candidate.replaced_light_count
                && removed_heavy == candidate.replaced_heavy_count,
            "candidate lane counts do not match exact interval"
        );
        let expected_saving =
            lane_proof_saving(total_heavy, total_light, removed_heavy, removed_light)
                .context("candidate no longer saves a lane proof")?;
        ensure!(
            expected_saving == candidate.saved_lane_proof_count,
            "candidate lane saving changed"
        );
        let maker_order_expiry = block
            .created_at
            .checked_add(MIN_ORDER_EXPIRY_PERIOD)
            .context("phantom expiry overflow")?;
        Ok(Self {
            start_tx_index: candidate.start_tx_index,
            right_tx_index: candidate.end_tx_index,
            created_at: block.created_at,
            state_metadata_hash,
            maker_account_index: candidate.seller_account_index,
            taker_account_index: candidate.buyer_account_index,
            base_asset_index: candidate.base_asset_index,
            quote_asset_index: candidate.quote_asset_index,
            size: candidate.base_amount,
            price: candidate.price,
            maker_order_expiry,
            total_heavy,
            total_light,
            removed_heavy,
            removed_light,
            expected_proof_saving: expected_saving,
        })
    }

    fn validate(&self) -> Result<PhantomAlgebra> {
        let removed_count = self
            .right_tx_index
            .checked_sub(self.start_tx_index)
            .context("reversed replacement interval")?;
        ensure!(
            removed_count >= 3,
            "V1 replacement must remove at least three txs"
        );
        ensure!(
            usize::try_from(removed_count)?
                == self
                    .removed_heavy
                    .checked_add(self.removed_light)
                    .context("removed count overflow")?,
            "removed lane counts do not cover interval"
        );
        let saving = lane_proof_saving(
            self.total_heavy,
            self.total_light,
            self.removed_heavy,
            self.removed_light,
        )
        .context("replacement does not reduce lane proofs")?;
        ensure!(saving == self.expected_proof_saving, "proof saving changed");
        ensure!(
            self.created_at >= 0 && (self.created_at as u128) < (1_u128 << TIMESTAMP_BITS),
            "created_at is out of range"
        );
        ensure!(
            self.maker_order_expiry > self.created_at
                && (self.maker_order_expiry as u128) < (1_u128 << TIMESTAMP_BITS),
            "maker expiry is out of range"
        );
        ensure!(
            self.maker_account_index > 0
                && self.taker_account_index > 0
                && self.maker_account_index != self.taker_account_index,
            "invalid maker/taker indices"
        );
        ensure!(
            self.base_asset_index > 0
                && self.quote_asset_index > 0
                && self.base_asset_index != self.quote_asset_index,
            "invalid asset indices"
        );
        ensure!(self.size > 0 && self.price > 0, "nonpositive size/price");

        let normalized_quote = self
            .size
            .checked_mul(self.price)
            .context("quote overflow")?;
        ensure!(
            self.size as u64 <= MAX_ORDER_BASE_AMOUNT
                && self.price as u64 <= MAX_ORDER_PRICE
                && (normalized_quote as u128) < (1_u128 << QUOTE_SUM_BITS),
            "order amount is out of circuit range"
        );
        let maker_account_order_index = i64::try_from(
            ((u128::from(NIL_MARKET_INDEX) + 1) << ORDER_NONCE_BITS)
                .checked_add(PHANTOM_MAKER_NONCE as u128)
                .context("account-order key overflow")?,
        )?;
        Ok(PhantomAlgebra {
            normalized_quote,
            maker_account_order_index,
            following_tx_index_shift: removed_count - 2,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PhantomSpotDonors {
    left: Tx<F>,
    right: Tx<F>,
    nil_padding: Tx<F>,
}

impl PhantomSpotDonors {
    pub fn from_block(block: &Block<F>, request: &PhantomSpotRequest) -> Result<Self> {
        let mut left = None;
        let mut right = None;
        let mut padding = None;
        for tx in block.tx_chunks.iter().flatten() {
            if tx.is_empty() {
                padding.get_or_insert_with(|| tx.as_ref().clone());
            } else if tx.tx_index == request.start_tx_index {
                ensure!(left.is_none(), "duplicate left boundary index");
                left = Some(tx.as_ref().clone());
            } else if tx.tx_index == request.right_tx_index {
                ensure!(right.is_none(), "duplicate right boundary index");
                right = Some(tx.as_ref().clone());
            }
        }
        Ok(Self {
            left: left.context("missing original left boundary")?,
            right: right.context("missing original right boundary")?,
            nil_padding: padding.context("missing NIL padding donor")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MaterializedPhantomSpotPair {
    pub light_insert: Tx<F>,
    pub heavy_fill: Tx<F>,
    pub first_removed_tx_index: u64,
    pub right_boundary_tx_index: u64,
    /// Subtract from every original index at or after `right_boundary_tx_index`.
    pub following_tx_index_shift: u64,
    pub after_light_validium_root: HashOut<F>,
    pub after_light_state_root: HashOut<F>,
    pub after_light_delta_root: HashOut<F>,
    pub final_validium_root: HashOut<F>,
    pub final_state_root: HashOut<F>,
    pub final_delta_root: HashOut<F>,
}

#[derive(Clone, Copy, Debug)]
struct PhantomAlgebra {
    normalized_quote: i64,
    maker_account_order_index: i64,
    following_tx_index_shift: u64,
}

#[derive(Clone, Copy)]
struct GlobalHashes {
    system: HashOut<F>,
    assets: HashOut<F>,
    margined_assets: HashOut<F>,
    market_risk: HashOut<F>,
    public_market: HashOut<F>,
}

impl GlobalHashes {
    fn from_tx(tx: &Tx<F>) -> Result<Self> {
        let (market_risk, public_market) =
            all_market_details_hashes(&tx.all_market_risk_details_before)
                .context("market-risk snapshot does not fit native hash")?;
        Ok(Self {
            system: system_config_hash(&tx.system_config_before)
                .context("system config does not fit native hash")?,
            assets: all_assets_hash(&tx.all_assets_before)
                .context("asset snapshot does not fit native hash")?,
            margined_assets: all_margined_assets_hash(&tx.all_margined_assets_before)
                .context("margined-asset snapshot does not fit native hash")?,
            market_risk,
            public_market,
        })
    }

    fn roots(
        self,
        register: HashOut<F>,
        account: HashOut<F>,
        account_public: HashOut<F>,
        market_details: HashOut<F>,
        market: HashOut<F>,
        metadata: HashOut<F>,
    ) -> Result<(HashOut<F>, HashOut<F>)> {
        validium_and_state_root(
            self.system,
            self.assets,
            self.margined_assets,
            self.market_risk,
            self.public_market,
            register,
            account,
            account_public,
            market_details,
            market,
            metadata,
        )
        .context("native aggregate root folding failed")
    }
}

fn debug_equal<T: std::fmt::Debug>(left: &T, right: &T) -> bool {
    format!("{left:?}") == format!("{right:?}")
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

fn empty_merkle_proof<const L: usize>() -> [HashOut<F>; L] {
    let mut hash = HashOut::ZERO;
    core::array::from_fn(|_| {
        let sibling = hash;
        hash = Poseidon2Hash::two_to_one(hash, hash);
        sibling
    })
}

fn as_merkle_index(index: i64, levels: usize, label: &str) -> Result<u64> {
    let index = u64::try_from(index).with_context(|| format!("negative {label}"))?;
    ensure!(
        (index as u128) < (1_u128 << levels),
        "{label} does not fit its Merkle path"
    );
    Ok(index)
}

fn unique_account_slot(tx: &Tx<F>, account_index: i64, label: &str) -> Result<usize> {
    let slots = tx
        .accounts_before
        .iter()
        .enumerate()
        .filter(|(_, account)| account.account_index == account_index)
        .map(|(slot, _)| slot)
        .collect::<Vec<_>>();
    ensure!(
        slots.len() == 1,
        "{label} is not unique in boundary witness"
    );
    Ok(slots[0])
}

fn canonical_empty_register_stack(stack: &RegisterStack) -> bool {
    stack.count == 0 && stack.stack.iter().all(BaseRegisterInfo::is_empty)
}

fn canonical_nil_market(tx: &Tx<F>) -> bool {
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

fn canonical_nil_market_details(tx: &Tx<F>) -> bool {
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

#[derive(Clone)]
struct TwoLeafTree<const L: usize> {
    indices: [u64; 2],
    leaves: [HashOut<F>; 2],
    root: HashOut<F>,
    common_proofs: [[HashOut<F>; L]; 2],
}

impl<const L: usize> TwoLeafTree<L> {
    fn from_sequential(
        indices: [u64; 2],
        leaves: [HashOut<F>; 2],
        sequential: [[HashOut<F>; L]; 2],
        root: HashOut<F>,
        label: &str,
    ) -> Result<Self> {
        ensure!(indices[0] != indices[1], "duplicate {label} indices");
        let proof_1 = rewind_one_update(
            indices[0],
            leaves[0],
            &sequential[0],
            indices[1],
            &sequential[1],
        )
        .with_context(|| format!("cannot normalize {label} proof 1"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?;
        let common_proofs = [sequential[0], proof_1];
        for slot in 0..2 {
            ensure!(
                merkle_root(leaves[slot], indices[slot], &common_proofs[slot]) == root,
                "{label} proof {slot} does not open the declared root"
            );
        }
        Ok(Self {
            indices,
            leaves,
            root,
            common_proofs,
        })
    }

    fn transition(
        &self,
        new_leaves: [HashOut<F>; 2],
        label: &str,
    ) -> Result<([[HashOut<F>; L]; 2], Self)> {
        let proof_0 = self.common_proofs[0];
        ensure!(
            merkle_root(self.leaves[0], self.indices[0], &proof_0) == self.root,
            "{label} first old leaf does not open current root"
        );
        let root_after_0 = merkle_root(new_leaves[0], self.indices[0], &proof_0);
        let proof_1: [HashOut<F>; L] = apply_one_update_to_proof(
            self.indices[0],
            new_leaves[0],
            &proof_0,
            self.indices[1],
            &self.common_proofs[1],
        )
        .with_context(|| format!("cannot advance {label} proof 1"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?;
        ensure!(
            merkle_root(self.leaves[1], self.indices[1], &proof_1) == root_after_0,
            "{label} second old leaf does not open root after first update"
        );
        let final_root = merkle_root(new_leaves[1], self.indices[1], &proof_1);
        let final_proof_0: [HashOut<F>; L] = apply_one_update_to_proof(
            self.indices[1],
            new_leaves[1],
            &proof_1,
            self.indices[0],
            &proof_0,
        )
        .with_context(|| format!("cannot normalize final {label} proof 0"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?;
        ensure!(
            merkle_root(new_leaves[0], self.indices[0], &final_proof_0) == final_root,
            "{label} final proof normalization failed"
        );
        Ok((
            [proof_0, proof_1],
            Self {
                indices: self.indices,
                leaves: new_leaves,
                root: final_root,
                common_proofs: [final_proof_0, proof_1],
            },
        ))
    }
}

#[derive(Clone)]
struct ThreeLeafTree<const L: usize> {
    indices: [u64; 3],
    leaves: [HashOut<F>; 3],
    root: HashOut<F>,
    common_proofs: [[HashOut<F>; L]; 3],
}

impl<const L: usize> ThreeLeafTree<L> {
    fn from_sequential(
        indices: [u64; 3],
        leaves: [HashOut<F>; 3],
        sequential: [[HashOut<F>; L]; 3],
        root: HashOut<F>,
        label: &str,
    ) -> Result<Self> {
        ensure!(
            indices[0] != indices[1] && indices[0] != indices[2] && indices[1] != indices[2],
            "duplicate {label} indices"
        );
        let proof_1 = rewind_one_update(
            indices[0],
            leaves[0],
            &sequential[0],
            indices[1],
            &sequential[1],
        )
        .with_context(|| format!("cannot normalize {label} proof 1"))?;
        let proof_2_after_0 = rewind_one_update(
            indices[1],
            leaves[1],
            &sequential[1],
            indices[2],
            &sequential[2],
        )
        .with_context(|| format!("cannot rewind {label} proof 2 through update 1"))?;
        let proof_2 = rewind_one_update(
            indices[0],
            leaves[0],
            &sequential[0],
            indices[2],
            &proof_2_after_0,
        )
        .with_context(|| format!("cannot rewind {label} proof 2 through update 0"))?;
        let common_proofs = [
            sequential[0],
            proof_1
                .try_into()
                .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?,
            proof_2
                .try_into()
                .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?,
        ];
        for slot in 0..3 {
            ensure!(
                merkle_root(leaves[slot], indices[slot], &common_proofs[slot]) == root,
                "{label} proof {slot} does not open the declared root"
            );
        }
        Ok(Self {
            indices,
            leaves,
            root,
            common_proofs,
        })
    }

    /// `order` names slots in this tree; `new_leaves` remains in the tree's original slot order.
    fn transition(
        &self,
        order: [usize; 3],
        new_leaves: [HashOut<F>; 3],
        label: &str,
    ) -> Result<([[HashOut<F>; L]; 3], Self)> {
        let mut seen = [false; 3];
        for slot in order {
            ensure!(slot < 3 && !seen[slot], "invalid {label} update order");
            seen[slot] = true;
        }

        let mut sequential = [[HashOut::ZERO; L]; 3];
        let mut current_root = self.root;
        for (step, slot) in order.into_iter().enumerate() {
            let mut proof = self.common_proofs[slot].to_vec();
            for prior in 0..step {
                let changed = order[prior];
                proof = apply_one_update_to_proof(
                    self.indices[changed],
                    new_leaves[changed],
                    &sequential[prior],
                    self.indices[slot],
                    &proof,
                )
                .with_context(|| format!("cannot advance {label} proof at step {step}"))?;
            }
            let proof: [HashOut<F>; L] = proof
                .try_into()
                .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?;
            ensure!(
                merkle_root(self.leaves[slot], self.indices[slot], &proof) == current_root,
                "{label} old leaf at step {step} does not open current root"
            );
            current_root = merkle_root(new_leaves[slot], self.indices[slot], &proof);
            sequential[step] = proof;
        }

        let mut final_common = [[HashOut::ZERO; L]; 3];
        for target in 0..3 {
            let mut proof = self.common_proofs[target].to_vec();
            for (step, changed) in order.into_iter().enumerate() {
                if changed == target {
                    continue;
                }
                proof = apply_one_update_to_proof(
                    self.indices[changed],
                    new_leaves[changed],
                    &sequential[step],
                    self.indices[target],
                    &proof,
                )
                .with_context(|| format!("cannot normalize final {label} proof {target}"))?;
            }
            final_common[target] = proof
                .try_into()
                .map_err(|_| anyhow::anyhow!("wrong {label} proof length"))?;
            ensure!(
                merkle_root(
                    new_leaves[target],
                    self.indices[target],
                    &final_common[target]
                ) == current_root,
                "final {label} proof {target} does not open final root"
            );
        }

        Ok((
            sequential,
            Self {
                indices: self.indices,
                leaves: new_leaves,
                root: current_root,
                common_proofs: final_common,
            },
        ))
    }
}

#[derive(Clone)]
struct AccountTrees {
    raw: TwoLeafTree<ASSET_MERKLE_LEVELS>,
    public: TwoLeafTree<ASSET_MERKLE_LEVELS>,
    delta: TwoLeafTree<ASSET_MERKLE_LEVELS>,
}

#[derive(Clone)]
struct AccountTreeProofs {
    raw: [[HashOut<F>; ASSET_MERKLE_LEVELS]; 2],
    public: [[HashOut<F>; ASSET_MERKLE_LEVELS]; 2],
    delta: [[HashOut<F>; ASSET_MERKLE_LEVELS]; 2],
}

impl AccountTrees {
    fn from_boundary(tx: &Tx<F>, slot: usize, label: &str) -> Result<Self> {
        let indices = [
            as_merkle_index(
                i64::from(tx.asset_indices[0]),
                ASSET_MERKLE_LEVELS,
                "asset index 0",
            )?,
            as_merkle_index(
                i64::from(tx.asset_indices[1]),
                ASSET_MERKLE_LEVELS,
                "asset index 1",
            )?,
        ];
        let raw_leaves = [
            account_asset_hash(&tx.account_assets_before[slot][0])
                .context("raw asset 0 does not fit native hash")?,
            account_asset_hash(&tx.account_assets_before[slot][1])
                .context("raw asset 1 does not fit native hash")?,
        ];
        let public_leaves = [
            bigint_leaf_hash(&tx.accounts_before[slot].aggregated_balances[0])
                .context("balance 0 does not fit native hash")?,
            bigint_leaf_hash(&tx.accounts_before[slot].aggregated_balances[1])
                .context("balance 1 does not fit native hash")?,
        ];
        let delta_leaves = [
            bigint_leaf_hash(&tx.accounts_delta_before[slot].aggregated_asset_deltas[0])
                .context("delta 0 does not fit native hash")?,
            bigint_leaf_hash(&tx.accounts_delta_before[slot].aggregated_asset_deltas[1])
                .context("delta 1 does not fit native hash")?,
        ];

        Ok(Self {
            raw: TwoLeafTree::from_sequential(
                indices,
                raw_leaves,
                tx.asset_tree_merkle_proofs[slot],
                tx.accounts_before[slot].asset_root,
                &format!("{label} raw asset tree"),
            )?,
            public: TwoLeafTree::from_sequential(
                indices,
                public_leaves,
                tx.public_asset_tree_merkle_proofs[slot],
                tx.accounts_before[slot].aggregated_balances_root,
                &format!("{label} public asset tree"),
            )?,
            delta: TwoLeafTree::from_sequential(
                indices,
                delta_leaves,
                tx.asset_delta_tree_merkle_proofs[slot],
                tx.accounts_delta_before[slot].asset_delta_root,
                &format!("{label} asset-delta tree"),
            )?,
        })
    }

    fn transition(
        &self,
        raw: [HashOut<F>; 2],
        public: [HashOut<F>; 2],
        delta: [HashOut<F>; 2],
        label: &str,
    ) -> Result<(AccountTreeProofs, Self)> {
        let (raw_proofs, raw) = self.raw.transition(raw, &format!("{label} raw"))?;
        let (public_proofs, public) = self.public.transition(public, &format!("{label} public"))?;
        let (delta_proofs, delta) = self.delta.transition(delta, &format!("{label} delta"))?;
        Ok((
            AccountTreeProofs {
                raw: raw_proofs,
                public: public_proofs,
                delta: delta_proofs,
            },
            Self { raw, public, delta },
        ))
    }
}

#[derive(Clone)]
struct OuterTrees {
    full: ThreeLeafTree<ACCOUNT_MERKLE_LEVELS>,
    public: ThreeLeafTree<ACCOUNT_MERKLE_LEVELS>,
    delta: ThreeLeafTree<ACCOUNT_MERKLE_LEVELS>,
}

#[derive(Clone)]
struct OuterTreeProofs {
    full: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 3],
    public: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 3],
    delta: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 3],
}

impl OuterTrees {
    fn from_boundary(tx: &Tx<F>) -> Result<Self> {
        let indices = [
            as_merkle_index(
                tx.accounts_before[0].account_index,
                ACCOUNT_MERKLE_LEVELS,
                "account index 0",
            )?,
            as_merkle_index(
                tx.accounts_before[1].account_index,
                ACCOUNT_MERKLE_LEVELS,
                "account index 1",
            )?,
            as_merkle_index(
                tx.accounts_before[2].account_index,
                ACCOUNT_MERKLE_LEVELS,
                "account index 2",
            )?,
        ];
        let user_0 = account_hash(&tx.accounts_before[0])
            .context("account slot 0 does not fit native hash")?;
        let user_1 = account_hash(&tx.accounts_before[1])
            .context("account slot 1 does not fit native hash")?;
        let fee = fee_account_hash(&tx.accounts_before[FEE_ACCOUNT_ID])
            .context("fee account does not fit native hash")?;
        let delta_0 = account_delta_hash(&tx.accounts_delta_before[0])
            .context("account delta slot 0 does not fit native hash")?;
        let delta_1 = account_delta_hash(&tx.accounts_delta_before[1])
            .context("account delta slot 1 does not fit native hash")?;
        let fee_delta = fee_account_delta_hash(&tx.accounts_delta_before[FEE_ACCOUNT_ID])
            .context("fee-account delta does not fit native hash")?;
        Ok(Self {
            full: ThreeLeafTree::from_sequential(
                indices,
                [user_0.0, user_1.0, fee.0],
                tx.account_tree_merkle_proofs,
                tx.old_account_tree_root,
                "full account tree",
            )?,
            public: ThreeLeafTree::from_sequential(
                indices,
                [user_0.1, user_1.1, fee.1],
                tx.account_pub_data_tree_merkle_proofs,
                tx.old_account_pub_data_tree_root,
                "public account tree",
            )?,
            delta: ThreeLeafTree::from_sequential(
                indices,
                [delta_0, delta_1, fee_delta],
                tx.account_delta_tree_merkle_proofs,
                tx.old_account_delta_tree_root,
                "account-delta tree",
            )?,
        })
    }

    fn transition(
        &self,
        order: [usize; 3],
        full: [HashOut<F>; 3],
        public: [HashOut<F>; 3],
        delta: [HashOut<F>; 3],
        label: &str,
    ) -> Result<(OuterTreeProofs, Self)> {
        let (full_proofs, full) = self
            .full
            .transition(order, full, &format!("{label} full"))?;
        let (public_proofs, public) =
            self.public
                .transition(order, public, &format!("{label} public"))?;
        let (delta_proofs, delta) =
            self.delta
                .transition(order, delta, &format!("{label} delta"))?;
        Ok((
            OuterTreeProofs {
                full: full_proofs,
                public: public_proofs,
                delta: delta_proofs,
            },
            Self {
                full,
                public,
                delta,
            },
        ))
    }
}

fn raw_leaf_hashes(assets: &[AccountAsset; 2]) -> Result<[HashOut<F>; 2]> {
    Ok([
        account_asset_hash(&assets[0]).context("raw asset 0 exceeds native hash domain")?,
        account_asset_hash(&assets[1]).context("raw asset 1 exceeds native hash domain")?,
    ])
}

fn balance_leaf_hashes(account: &Account<F>) -> Result<[HashOut<F>; 2]> {
    Ok([
        bigint_leaf_hash(&account.aggregated_balances[0])
            .context("aggregated balance 0 exceeds native hash domain")?,
        bigint_leaf_hash(&account.aggregated_balances[1])
            .context("aggregated balance 1 exceeds native hash domain")?,
    ])
}

fn delta_leaf_hashes(delta: &AccountDelta<F>) -> Result<[HashOut<F>; 2]> {
    Ok([
        bigint_leaf_hash(&delta.aggregated_asset_deltas[0])
            .context("aggregated delta 0 exceeds native hash domain")?,
        bigint_leaf_hash(&delta.aggregated_asset_deltas[1])
            .context("aggregated delta 1 exceeds native hash domain")?,
    ])
}

fn outer_leaf_hashes(
    accounts: &[Account<F>; 3],
    deltas: &[AccountDelta<F>; 3],
) -> Result<([HashOut<F>; 3], [HashOut<F>; 3], [HashOut<F>; 3])> {
    let user_0 = account_hash(&accounts[0]).context("account slot 0 exceeds native hash domain")?;
    let user_1 = account_hash(&accounts[1]).context("account slot 1 exceeds native hash domain")?;
    let fee = fee_account_hash(&accounts[FEE_ACCOUNT_ID])
        .context("fee account exceeds native hash domain")?;
    let delta_0 = account_delta_hash(&deltas[0])
        .context("account delta slot 0 exceeds native hash domain")?;
    let delta_1 = account_delta_hash(&deltas[1])
        .context("account delta slot 1 exceeds native hash domain")?;
    let fee_delta = fee_account_delta_hash(&deltas[FEE_ACCOUNT_ID])
        .context("fee-account delta exceeds native hash domain")?;
    Ok((
        [user_0.0, user_1.0, fee.0],
        [user_0.1, user_1.1, fee.1],
        [delta_0, delta_1, fee_delta],
    ))
}

fn stack_with(register: BaseRegisterInfo) -> RegisterStack {
    let mut stack = RegisterStack::default();
    stack.count = 1;
    stack.stack[0] = register;
    stack
}

fn maker_register(request: &PhantomSpotRequest) -> Result<BaseRegisterInfo> {
    let expiry_alias = [
        request.size,
        request.size,
        request.price,
        PHANTOM_MAKER_NONCE,
        1,
        i64::from(LIMIT_ORDER),
        i64::from(GTT),
    ]
    .into_iter()
    .try_fold(0_i64, i64::checked_add)
    .and_then(i64::checked_neg)
    .context("maker register alias overflow")?;
    let header_alias = i64::from(INSERT_ORDER)
        .checked_add(i64::from(NIL_MARKET_INDEX))
        .and_then(|sum| sum.checked_add(request.maker_account_index))
        .and_then(i64::checked_neg)
        .context("maker register header alias overflow")?;
    Ok(BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: u16::from(NIL_MARKET_INDEX),
        account_index: request.maker_account_index,
        pending_size: request.size,
        pending_initial_size: request.size,
        pending_price: request.price,
        pending_nonce: PHANTOM_MAKER_NONCE,
        pending_is_ask: 1,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: GTT,
        pending_expiry: expiry_alias,
        generic_field_3: header_alias,
        ..BaseRegisterInfo::empty()
    })
}

fn taker_register(request: &PhantomSpotRequest) -> Result<BaseRegisterInfo> {
    let expiry_alias = [
        request.size,
        request.size,
        request.price,
        PHANTOM_TAKER_NONCE,
        i64::from(LIMIT_ORDER),
        i64::from(IOC),
    ]
    .into_iter()
    .try_fold(0_i64, i64::checked_add)
    .and_then(i64::checked_neg)
    .context("taker register alias overflow")?;
    let header_alias = i64::from(INSERT_ORDER)
        .checked_add(i64::from(NIL_MARKET_INDEX))
        .and_then(|sum| sum.checked_add(request.taker_account_index))
        .and_then(i64::checked_neg)
        .context("taker register header alias overflow")?;
    Ok(BaseRegisterInfo {
        instruction_type: INSERT_ORDER,
        market_index: u16::from(NIL_MARKET_INDEX),
        account_index: request.taker_account_index,
        pending_size: request.size,
        pending_initial_size: request.size,
        pending_price: request.price,
        pending_nonce: PHANTOM_TAKER_NONCE,
        pending_is_ask: 0,
        pending_type: LIMIT_ORDER,
        pending_time_in_force: IOC,
        pending_expiry: expiry_alias,
        generic_field_3: header_alias,
        ..BaseRegisterInfo::empty()
    })
}

fn maker_order_before_insert(request: &PhantomSpotRequest) -> AccountOrder {
    AccountOrder {
        index_0: NIL_ORDER_INDEX,
        index_1: NIL_CLIENT_ORDER_INDEX,
        owner_account_index: request.maker_account_index,
        ..AccountOrder::default()
    }
}

fn maker_order_before_fill(
    request: &PhantomSpotRequest,
    algebra: PhantomAlgebra,
) -> Result<AccountOrder> {
    let alias = [
        request.size,
        request.price,
        PHANTOM_MAKER_NONCE,
        request.size,
        1,
        i64::from(LIMIT_ORDER),
        i64::from(GTT),
        request.maker_order_expiry,
    ]
    .into_iter()
    .try_fold(0_i64, i64::checked_add)
    .and_then(i64::checked_neg)
    .context("maker account-order alias overflow")?;
    Ok(AccountOrder {
        index_0: algebra.maker_account_order_index,
        index_1: NIL_CLIENT_ORDER_INDEX,
        owner_account_index: request.maker_account_index,
        order_index: NIL_ORDER_INDEX,
        client_order_index: NIL_CLIENT_ORDER_INDEX,
        initial_base_amount: request.size,
        price: u32::try_from(request.price)?,
        nonce: PHANTOM_MAKER_NONCE,
        remaining_base_amount: request.size,
        is_ask: 1,
        order_type: LIMIT_ORDER,
        time_in_force: GTT,
        expiry: request.maker_order_expiry,
        integrator_fee_collector_index: 0,
        integrator_taker_fee: alias,
        integrator_maker_fee: 0,
        order_flags: 0,
        ..AccountOrder::default()
    })
}

fn fake_market(
    request: &PhantomSpotRequest,
    base_multiplier: i64,
    quote_multiplier: i64,
    total_order_count: i64,
    order_book_root: HashOut<F>,
) -> Result<Market<F>> {
    let quote_limit = (MARKET_TYPE_SPOT as i64)
        .checked_add(i64::from(request.base_asset_index))
        .and_then(|sum| sum.checked_add(i64::from(request.quote_asset_index)))
        .and_then(|sum| sum.checked_add(base_multiplier))
        .and_then(|sum| sum.checked_add(quote_multiplier))
        .and_then(i64::checked_neg)
        .context("fake market alias overflow")?;
    Ok(Market {
        market_index: u16::from(NIL_MARKET_INDEX),
        market_type: MARKET_TYPE_SPOT as u8,
        base_asset_id: u16::try_from(request.base_asset_index)?,
        quote_asset_id: u16::try_from(request.quote_asset_index)?,
        total_order_count,
        size_extension_multiplier: base_multiplier,
        quote_extension_multiplier: quote_multiplier,
        order_quote_limit: quote_limit,
        order_book_root,
        ..Market::default()
    })
}

fn configure_common_tx(
    tx: &mut Tx<F>,
    left: &Tx<F>,
    nil_padding: &Tx<F>,
    request: &PhantomSpotRequest,
) {
    tx.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;
    tx.nonce = 0;
    tx.expired_at = 0;
    tx.taker_fee = 0;
    tx.maker_fee = 0;
    tx.attributes = Default::default();
    tx.internal_claim_order_tx.market_index = u16::from(NIL_MARKET_INDEX);
    tx.asset_indices = [request.base_asset_index, request.quote_asset_index];
    tx.system_config_before = left.system_config_before.clone();
    tx.all_assets_before = left.all_assets_before.clone();
    tx.all_margined_assets_before = left.all_margined_assets_before.clone();
    tx.all_market_risk_details_before = left.all_market_risk_details_before.clone();
    tx.api_key_before = ApiKey::empty(0);
    tx.api_key_tree_merkle_proof = empty_merkle_proof();
    tx.account_orders_tree_merkle_proof = [empty_merkle_proof(); NB_ACCOUNT_ORDERS_PATHS_PER_TX];
    tx.position_delta_merkle_proofs = [empty_merkle_proof(); NB_ACCOUNTS_PER_TX - 1];
    tx.market_tree_merkle_proof = nil_padding.market_tree_merkle_proof;
    tx.market_details_before = nil_padding.market_details_before.clone();
    tx.market_details_tree_merkle_proof = nil_padding.market_details_tree_merkle_proof;

    let empty_order_book = OrderBookTree::<ORDER_BOOK_MERKLE_LEVELS>::new();
    tx.impact_ask_order = Order::empty(0, 0);
    tx.impact_bid_order = Order::empty(0, 0);
    tx.impact_ask_order_book_tree_path = empty_order_book.proof(0);
    tx.impact_bid_order_book_tree_path = empty_order_book.proof(0);
}

/// Materialize the exact two-transaction phantom session for one already-scanned candidate.
///
/// This function is deliberately read-only and does not trust the scanner's booleans as proof.
/// It rechecks the candidate-local interval, extracts the original boundary witnesses, normalizes
/// their paths, and performs the full native replay before returning either transaction.
pub fn materialize_scanned_candidate(
    block: &Block<F>,
    candidate: &Candidate,
    state_metadata_hash: HashOut<F>,
) -> Result<MaterializedPhantomSpotPair> {
    let request =
        PhantomSpotRequest::from_scanned_candidate(block, candidate, state_metadata_hash)?;
    let donors = PhantomSpotDonors::from_block(block, &request)?;
    materialize_phantom_spot_pair(&request, &donors)
}

fn materialize_phantom_spot_pair(
    request: &PhantomSpotRequest,
    donors: &PhantomSpotDonors,
) -> Result<MaterializedPhantomSpotPair> {
    let algebra = request.validate()?;
    let (left, right, nil_padding) = (&donors.left, &donors.right, &donors.nil_padding);

    ensure!(
        !left.is_empty() && !right.is_empty(),
        "boundary donor is empty"
    );
    ensure!(nil_padding.is_empty(), "padding donor is active");
    ensure!(
        left.tx_index == request.start_tx_index && right.tx_index == request.right_tx_index,
        "boundary donor index changed"
    );
    ensure!(
        left.asset_indices == [request.base_asset_index, request.quote_asset_index]
            && right.asset_indices == left.asset_indices,
        "V1 requires identical [base, quote] asset-slot order at both boundaries"
    );

    let maker_left = unique_account_slot(left, request.maker_account_index, "left maker")?;
    let taker_left = unique_account_slot(left, request.taker_account_index, "left taker")?;
    let maker_right = unique_account_slot(right, request.maker_account_index, "right maker")?;
    let taker_right = unique_account_slot(right, request.taker_account_index, "right taker")?;
    ensure!(
        maker_left < FEE_ACCOUNT_ID
            && taker_left < FEE_ACCOUNT_ID
            && maker_right < FEE_ACCOUNT_ID
            && taker_right < FEE_ACCOUNT_ID
            && maker_left != taker_left
            && maker_right != taker_right,
        "maker/taker do not occupy the two user slots"
    );
    for (name, tx) in [("left", left), ("right", right)] {
        ensure!(
            tx.accounts_before[FEE_ACCOUNT_ID].account_index == TREASURY_ACCOUNT_INDEX as i64
                && tx.accounts_delta_before[FEE_ACCOUNT_ID].account_index
                    == TREASURY_ACCOUNT_INDEX as i64,
            "{name} fee slot is not Treasury"
        );
        for slot in 0..NB_ACCOUNTS_PER_TX {
            ensure!(
                tx.accounts_delta_before[slot].account_index
                    == tx.accounts_before[slot].account_index,
                "{name} account/delta index mismatch in slot {slot}"
            );
            for asset_slot in 0..NB_ASSETS_PER_TX {
                ensure!(
                    tx.account_assets_before[slot][asset_slot].index_0
                        == i64::from(tx.asset_indices[asset_slot]),
                    "{name} account-asset index mismatch in slot {slot}/{asset_slot}"
                );
            }
        }
    }

    let regular_simple = |account: &Account<F>| {
        account.account_index > 0
            && account.account_index != TREASURY_ACCOUNT_INDEX as i64
            && account.account_type != INSURANCE_FUND_ACCOUNT_TYPE
            && matches!(account.account_type, MASTER_ACCOUNT_TYPE | SUB_ACCOUNT_TYPE)
            && account.account_trading_mode == ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE
    };
    ensure!(
        regular_simple(&left.accounts_before[maker_left])
            && regular_simple(&left.accounts_before[taker_left]),
        "V1 supports only distinct regular simple-mode users"
    );
    ensure!(
        left.accounts_before[maker_left].api_key_root == EMPTY_API_KEY_TREE_ROOT
            && left.accounts_before[taker_left].api_key_root == EMPTY_API_KEY_TREE_ROOT,
        "maker/taker API trees are not canonical empty trees"
    );
    ensure!(
        left.accounts_before[maker_left].account_orders_root == EMPTY_ACCOUNT_ORDERS_TREE_ROOT,
        "maker account-order tree is not canonical empty"
    );
    ensure!(
        left.accounts_delta_before[maker_left].position_delta_root
            == EMPTY_POSITION_DELTA_TREE_ROOT
            && left.accounts_delta_before[taker_left].position_delta_root
                == EMPTY_POSITION_DELTA_TREE_ROOT,
        "maker/taker position-delta trees are not canonical empty"
    );
    ensure!(
        debug_equal(
            &left.accounts_delta_before[maker_left].positions_delta,
            &PositionDelta::default(),
        ) && debug_equal(
            &left.accounts_delta_before[taker_left].positions_delta,
            &PositionDelta::default(),
        ),
        "maker/taker selected position-delta leaves are not canonical empty"
    );
    ensure!(
        merkle_root(
            HashOut::ZERO,
            u64::from(NIL_MARKET_INDEX),
            &empty_merkle_proof::<POSITION_MERKLE_LEVELS>(),
        ) == EMPTY_POSITION_DELTA_TREE_ROOT,
        "native empty position proof constant disagrees with circuit constant"
    );

    ensure!(
        canonical_empty_register_stack(&left.register_stack_before)
            && canonical_empty_register_stack(&right.register_stack_before),
        "original boundary register stack is not canonical empty"
    );
    ensure!(
        debug_equal(&left.system_config_before, &right.system_config_before)
            && debug_equal(&left.all_assets_before, &right.all_assets_before)
            && debug_equal(
                &left.all_margined_assets_before,
                &right.all_margined_assets_before,
            )
            && debug_equal(
                &left.all_market_risk_details_before,
                &right.all_market_risk_details_before,
            ),
        "static global snapshots differ across interval"
    );
    let left_globals = GlobalHashes::from_tx(left)?;
    let right_globals = GlobalHashes::from_tx(right)?;
    ensure!(
        left_globals.system == right_globals.system
            && left_globals.assets == right_globals.assets
            && left_globals.margined_assets == right_globals.margined_assets
            && left_globals.market_risk == right_globals.market_risk
            && left_globals.public_market == right_globals.public_market,
        "independently hashed global snapshots differ"
    );

    ensure!(
        left.old_market_tree_root == right.old_market_tree_root
            && left.old_market_details_tree_root == right.old_market_details_tree_root,
        "market roots changed across candidate interval"
    );
    ensure!(
        canonical_nil_market(nil_padding)
            && market_hash(&nil_padding.market_before) == HashOut::ZERO,
        "padding market is not a canonical NIL zero leaf"
    );
    ensure!(
        canonical_nil_market_details(nil_padding)
            && market_details_hash(&nil_padding.market_details_before) == HashOut::ZERO,
        "padding market-details value is not a canonical NIL zero leaf"
    );
    let nil_market_index = u64::from(NIL_MARKET_INDEX);
    ensure!(
        merkle_root(
            HashOut::ZERO,
            nil_market_index,
            &nil_padding.market_tree_merkle_proof,
        ) == left.old_market_tree_root,
        "padding NIL market path does not open the left market root"
    );
    ensure!(
        merkle_root(
            HashOut::ZERO,
            nil_market_index,
            &nil_padding.market_details_tree_merkle_proof,
        ) == left.old_market_details_tree_root,
        "padding NIL market-details path does not open the left details root"
    );

    let (initial_validium, initial_state) = left_globals.roots(
        register_stack_hash(&left.register_stack_before),
        left.old_account_tree_root,
        left.old_account_pub_data_tree_root,
        left.old_market_details_tree_root,
        left.old_market_tree_root,
        request.state_metadata_hash,
    )?;
    ensure!(
        initial_validium == left.old_validium_root && initial_state == left.old_state_root,
        "post-preexecution metadata does not authenticate the left boundary"
    );

    let base_position =
        usize::try_from(request.base_asset_index).context("negative base asset array index")?;
    let quote_position =
        usize::try_from(request.quote_asset_index).context("negative quote asset array index")?;
    let base_asset = left
        .all_assets_before
        .get(base_position)
        .context("base asset is outside global array")?;
    let quote_asset = left
        .all_assets_before
        .get(quote_position)
        .context("quote asset is outside global array")?;
    ensure!(
        base_asset.asset_index == request.base_asset_index
            && quote_asset.asset_index == request.quote_asset_index,
        "asset informational tags disagree with circuit array positions"
    );
    ensure!(
        base_asset.extension_multiplier > 0
            && quote_asset.extension_multiplier > 0
            && (base_asset.extension_multiplier as u64) < (1_u64 << 48)
            && (quote_asset.extension_multiplier as u64) < (1_u64 << 48),
        "selected extension multiplier is outside the circuit's 48-bit range"
    );
    let raw_base = BigUint::from(u64::try_from(request.size)?)
        * BigUint::from(u64::try_from(base_asset.extension_multiplier)?);
    let raw_quote = BigUint::from(u64::try_from(algebra.normalized_quote)?)
        * BigUint::from(u64::try_from(quote_asset.extension_multiplier)?);
    let u96_limit = BigUint::from(1_u8) << 96;
    ensure!(
        raw_base < u96_limit && raw_quote < u96_limit,
        "raw spot amount exceeds 96 bits"
    );
    let maker_base_before = &left.account_assets_before[maker_left][BASE_ASSET_ID];
    let taker_quote_before = &left.account_assets_before[taker_left][QUOTE_ASSET_ID];
    ensure!(
        maker_base_before.balance >= maker_base_before.locked_balance.clone() + &raw_base,
        "maker lacks unlocked base balance"
    );
    ensure!(
        taker_quote_before.balance >= raw_quote,
        "taker lacks quote balance"
    );

    let mut accounts_mid = left.accounts_before.clone();
    let mut assets_mid = left.account_assets_before.clone();
    let deltas_mid = left.accounts_delta_before.clone();
    assets_mid[maker_left][BASE_ASSET_ID].locked_balance += &raw_base;
    accounts_mid[maker_left].total_order_count = accounts_mid[maker_left]
        .total_order_count
        .checked_add(1)
        .context("maker order count overflow")?;
    accounts_mid[maker_left].total_non_cross_order_count = accounts_mid[maker_left]
        .total_non_cross_order_count
        .checked_add(1)
        .context("maker non-cross count overflow")?;

    let initial_account_trees = (0..NB_ACCOUNTS_PER_TX)
        .map(|slot| AccountTrees::from_boundary(left, slot, &format!("left account {slot}")))
        .collect::<Result<Vec<_>>>()?;
    let mut t1_inner = Vec::with_capacity(NB_ACCOUNTS_PER_TX);
    let mut mid_account_trees = Vec::with_capacity(NB_ACCOUNTS_PER_TX);
    for slot in 0..NB_ACCOUNTS_PER_TX {
        let (proofs, trees) = initial_account_trees[slot].transition(
            raw_leaf_hashes(&assets_mid[slot])?,
            balance_leaf_hashes(&accounts_mid[slot])?,
            delta_leaf_hashes(&deltas_mid[slot])?,
            &format!("T1 account {slot}"),
        )?;
        accounts_mid[slot].asset_root = trees.raw.root;
        accounts_mid[slot].aggregated_balances_root = trees.public.root;
        // T1 does not change account deltas, but retain the independently replayed root.
        ensure!(
            deltas_mid[slot].asset_delta_root == trees.delta.root,
            "T1 changed account-delta root in slot {slot}"
        );
        t1_inner.push(proofs);
        mid_account_trees.push(trees);
    }

    let initial_outer = OuterTrees::from_boundary(left)?;
    let (mid_full, mid_public, mid_delta) = outer_leaf_hashes(&accounts_mid, &deltas_mid)?;
    let (t1_outer, mid_outer) = initial_outer.transition(
        [maker_left, taker_left, FEE_ACCOUNT_ID],
        mid_full,
        mid_public,
        mid_delta,
        "T1",
    )?;
    ensure!(
        mid_outer.public.root == left.old_account_pub_data_tree_root
            && mid_outer.delta.root == left.old_account_delta_tree_root,
        "light insertion changed a public or delta root"
    );

    let mut empty_order_book = OrderBookTree::<ORDER_BOOK_MERKLE_LEVELS>::new();
    let empty_order_book_root = empty_order_book.root;
    let order_book_index = (u128::try_from(request.price)? << ORDER_NONCE_BITS)
        .checked_add(u128::try_from(PHANTOM_MAKER_NONCE)?)
        .context("order-book key overflow")?;
    let empty_order_path = empty_order_book.proof(order_book_index);
    let maker_order_leaf = Order {
        key_price: request.price,
        key_nonce: PHANTOM_MAKER_NONCE,
        ask_base_sum: request.size,
        ask_quote_sum: algebra.normalized_quote,
        bid_base_sum: 0,
        bid_quote_sum: 0,
    };
    empty_order_book.insert_leaf(order_book_index, maker_order_leaf.clone());
    let inserted_order_book_root = empty_order_book.root;
    let inserted_order_path = empty_order_book.proof(order_book_index);
    let market_before_insert = fake_market(
        request,
        base_asset.extension_multiplier,
        quote_asset.extension_multiplier,
        0,
        empty_order_book_root,
    )?;
    let market_before_fill = fake_market(
        request,
        base_asset.extension_multiplier,
        quote_asset.extension_multiplier,
        1,
        inserted_order_book_root,
    )?;
    ensure!(
        market_hash(&market_before_insert) == HashOut::ZERO
            && market_hash(&market_before_fill) != HashOut::ZERO,
        "fake market does not cross the intended alias boundary"
    );
    let after_light_market_root = merkle_root(
        market_hash(&market_before_fill),
        nil_market_index,
        &nil_padding.market_tree_merkle_proof,
    );

    let maker_stack = stack_with(maker_register(request)?);
    let taker_stack = stack_with(taker_register(request)?);
    let order_before_insert = maker_order_before_insert(request);
    let order_before_fill = maker_order_before_fill(request, algebra)?;
    ensure!(
        register_stack_hash(&maker_stack) == HashOut::ZERO
            && register_stack_hash(&taker_stack) == HashOut::ZERO,
        "forged register stack does not alias the empty register hash"
    );
    ensure!(
        account_order_hash(&order_before_insert) == HashOut::ZERO
            && account_order_hash(&order_before_fill) == HashOut::ZERO,
        "forged account order does not alias the empty order hash"
    );

    let (after_light_validium, after_light_state) = left_globals.roots(
        HashOut::ZERO,
        mid_outer.full.root,
        mid_outer.public.root,
        left.old_market_details_tree_root,
        after_light_market_root,
        request.state_metadata_hash,
    )?;

    let mut accounts_final = left.accounts_before.clone();
    let mut assets_final = left.account_assets_before.clone();
    let mut deltas_final = left.accounts_delta_before.clone();
    assets_final[maker_left][BASE_ASSET_ID].balance -= &raw_base;
    assets_final[maker_left][QUOTE_ASSET_ID].balance += &raw_quote;
    assets_final[taker_left][BASE_ASSET_ID].balance += &raw_base;
    assets_final[taker_left][QUOTE_ASSET_ID].balance -= &raw_quote;
    let normalized_base = BigInt::from(request.size);
    let normalized_quote = BigInt::from(algebra.normalized_quote);
    accounts_final[maker_left].aggregated_balances[BASE_ASSET_ID] -= &normalized_base;
    accounts_final[maker_left].aggregated_balances[QUOTE_ASSET_ID] += &normalized_quote;
    accounts_final[taker_left].aggregated_balances[BASE_ASSET_ID] += &normalized_base;
    accounts_final[taker_left].aggregated_balances[QUOTE_ASSET_ID] -= &normalized_quote;
    deltas_final[maker_left].aggregated_asset_deltas[BASE_ASSET_ID] -= &normalized_base;
    deltas_final[maker_left].aggregated_asset_deltas[QUOTE_ASSET_ID] += &normalized_quote;
    deltas_final[taker_left].aggregated_asset_deltas[BASE_ASSET_ID] += &normalized_base;
    deltas_final[taker_left].aggregated_asset_deltas[QUOTE_ASSET_ID] -= &normalized_quote;

    let mut t2_inner = Vec::with_capacity(NB_ACCOUNTS_PER_TX);
    let mut final_account_trees = Vec::with_capacity(NB_ACCOUNTS_PER_TX);
    for slot in 0..NB_ACCOUNTS_PER_TX {
        let (proofs, trees) = mid_account_trees[slot].transition(
            raw_leaf_hashes(&assets_final[slot])?,
            balance_leaf_hashes(&accounts_final[slot])?,
            delta_leaf_hashes(&deltas_final[slot])?,
            &format!("T2 account {slot}"),
        )?;
        accounts_final[slot].asset_root = trees.raw.root;
        accounts_final[slot].aggregated_balances_root = trees.public.root;
        deltas_final[slot].asset_delta_root = trees.delta.root;
        t2_inner.push(proofs);
        final_account_trees.push(trees);
    }

    let (final_full, final_public, final_delta) =
        outer_leaf_hashes(&accounts_final, &deltas_final)?;
    let (t2_outer, final_outer) = mid_outer.transition(
        [taker_left, maker_left, FEE_ACCOUNT_ID],
        final_full,
        final_public,
        final_delta,
        "T2",
    )?;
    ensure!(
        final_outer.full.root == right.old_account_tree_root
            && final_outer.public.root == right.old_account_pub_data_tree_root
            && final_outer.delta.root == right.old_account_delta_tree_root,
        "native T2 replay does not land on the right outer roots"
    );
    for (left_slot, right_slot, role) in [
        (maker_left, maker_right, "maker"),
        (taker_left, taker_right, "taker"),
        (FEE_ACCOUNT_ID, FEE_ACCOUNT_ID, "Treasury"),
    ] {
        ensure!(
            debug_equal(
                &accounts_final[left_slot],
                &right.accounts_before[right_slot]
            ),
            "native final {role} account differs from right boundary"
        );
        ensure!(
            debug_equal(
                &assets_final[left_slot],
                &right.account_assets_before[right_slot],
            ),
            "native final {role} raw assets differ from right boundary"
        );
        ensure!(
            debug_equal(
                &deltas_final[left_slot],
                &right.accounts_delta_before[right_slot],
            ),
            "native final {role} delta differs from right boundary"
        );
    }
    let final_market_root = merkle_root(
        HashOut::ZERO,
        nil_market_index,
        &nil_padding.market_tree_merkle_proof,
    );
    ensure!(
        final_market_root == right.old_market_tree_root,
        "market deletion does not restore right market root"
    );
    let (final_validium, final_state) = right_globals.roots(
        register_stack_hash(&right.register_stack_before),
        final_outer.full.root,
        final_outer.public.root,
        right.old_market_details_tree_root,
        final_market_root,
        request.state_metadata_hash,
    )?;
    ensure!(
        final_validium == right.old_validium_root && final_state == right.old_state_root,
        "native final state does not authenticate the right boundary"
    );

    let mut light_insert = nil_padding.clone();
    configure_common_tx(&mut light_insert, left, nil_padding, request);
    light_insert.tx_circuit_type = TX_LIGHT;
    light_insert.tx_index = request.start_tx_index;
    light_insert.internal_claim_order_tx.account_index = request.maker_account_index;
    light_insert.register_stack_before = maker_stack;
    light_insert.account_order_before = order_before_insert;
    light_insert.market_before = market_before_insert;
    light_insert.order_before = Order::empty(request.price, PHANTOM_MAKER_NONCE);
    light_insert.order_book_tree_path = empty_order_path;
    light_insert.accounts_before = [
        left.accounts_before[maker_left].clone(),
        left.accounts_before[taker_left].clone(),
        left.accounts_before[FEE_ACCOUNT_ID].clone(),
    ];
    light_insert.account_assets_before = [
        left.account_assets_before[maker_left].clone(),
        left.account_assets_before[taker_left].clone(),
        left.account_assets_before[FEE_ACCOUNT_ID].clone(),
    ];
    light_insert.accounts_delta_before = [
        left.accounts_delta_before[maker_left].clone(),
        left.accounts_delta_before[taker_left].clone(),
        left.accounts_delta_before[FEE_ACCOUNT_ID].clone(),
    ];
    light_insert.asset_tree_merkle_proofs = [
        t1_inner[maker_left].raw,
        t1_inner[taker_left].raw,
        t1_inner[FEE_ACCOUNT_ID].raw,
    ];
    light_insert.public_asset_tree_merkle_proofs = [
        t1_inner[maker_left].public,
        t1_inner[taker_left].public,
        t1_inner[FEE_ACCOUNT_ID].public,
    ];
    light_insert.asset_delta_tree_merkle_proofs = [
        t1_inner[maker_left].delta,
        t1_inner[taker_left].delta,
        t1_inner[FEE_ACCOUNT_ID].delta,
    ];
    light_insert.account_tree_merkle_proofs = t1_outer.full;
    light_insert.account_pub_data_tree_merkle_proofs = t1_outer.public;
    light_insert.account_delta_tree_merkle_proofs = t1_outer.delta;
    light_insert.old_account_tree_root = left.old_account_tree_root;
    light_insert.old_account_pub_data_tree_root = left.old_account_pub_data_tree_root;
    light_insert.old_account_delta_tree_root = left.old_account_delta_tree_root;
    light_insert.old_market_tree_root = left.old_market_tree_root;
    light_insert.old_market_details_tree_root = left.old_market_details_tree_root;
    light_insert.old_validium_root = left.old_validium_root;
    light_insert.old_state_root = left.old_state_root;

    let mut heavy_fill = nil_padding.clone();
    configure_common_tx(&mut heavy_fill, left, nil_padding, request);
    heavy_fill.tx_circuit_type = TX_HEAVY;
    heavy_fill.tx_index = request
        .start_tx_index
        .checked_add(1)
        .context("T2 index overflow")?;
    heavy_fill.internal_claim_order_tx.account_index = request.taker_account_index;
    heavy_fill.register_stack_before = taker_stack;
    heavy_fill.account_order_before = order_before_fill;
    heavy_fill.market_before = market_before_fill;
    heavy_fill.order_before = maker_order_leaf;
    heavy_fill.order_book_tree_path = inserted_order_path;
    heavy_fill.accounts_before = [
        accounts_mid[taker_left].clone(),
        accounts_mid[maker_left].clone(),
        accounts_mid[FEE_ACCOUNT_ID].clone(),
    ];
    heavy_fill.account_assets_before = [
        assets_mid[taker_left].clone(),
        assets_mid[maker_left].clone(),
        assets_mid[FEE_ACCOUNT_ID].clone(),
    ];
    heavy_fill.accounts_delta_before = [
        deltas_mid[taker_left].clone(),
        deltas_mid[maker_left].clone(),
        deltas_mid[FEE_ACCOUNT_ID].clone(),
    ];
    heavy_fill.asset_tree_merkle_proofs = [
        t2_inner[taker_left].raw,
        t2_inner[maker_left].raw,
        t2_inner[FEE_ACCOUNT_ID].raw,
    ];
    heavy_fill.public_asset_tree_merkle_proofs = [
        t2_inner[taker_left].public,
        t2_inner[maker_left].public,
        t2_inner[FEE_ACCOUNT_ID].public,
    ];
    heavy_fill.asset_delta_tree_merkle_proofs = [
        t2_inner[taker_left].delta,
        t2_inner[maker_left].delta,
        t2_inner[FEE_ACCOUNT_ID].delta,
    ];
    heavy_fill.account_tree_merkle_proofs = t2_outer.full;
    heavy_fill.account_pub_data_tree_merkle_proofs = t2_outer.public;
    heavy_fill.account_delta_tree_merkle_proofs = t2_outer.delta;
    heavy_fill.old_account_tree_root = mid_outer.full.root;
    heavy_fill.old_account_pub_data_tree_root = mid_outer.public.root;
    heavy_fill.old_account_delta_tree_root = mid_outer.delta.root;
    heavy_fill.old_market_tree_root = after_light_market_root;
    heavy_fill.old_market_details_tree_root = left.old_market_details_tree_root;
    heavy_fill.old_validium_root = after_light_validium;
    heavy_fill.old_state_root = after_light_state;

    Ok(MaterializedPhantomSpotPair {
        light_insert,
        heavy_fill,
        first_removed_tx_index: request.start_tx_index,
        right_boundary_tx_index: request.right_tx_index,
        following_tx_index_shift: algebra.following_tx_index_shift,
        after_light_validium_root: after_light_validium,
        after_light_state_root: after_light_state,
        after_light_delta_root: mid_outer.delta.root,
        final_validium_root: final_validium,
        final_state_root: final_state,
        final_delta_root: final_outer.delta.root,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;

    use circuit::types::asset::Asset;
    use num::Zero;
    use plonky2::field::types::Field;

    use super::*;
    use crate::phantom_spot::scan;

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
                        .get(&((index >> level) ^ 1))
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

    fn install_inner_trees(
        left: &mut Tx<F>,
        right: &mut Tx<F>,
        left_slot: usize,
        right_slot: usize,
    ) {
        let indices = [1_u64, 2_u64];

        let old_assets = left.account_assets_before[left_slot]
            .iter()
            .map(|asset| account_asset_hash(asset).unwrap())
            .collect::<Vec<_>>();
        let new_assets = right.account_assets_before[right_slot]
            .iter()
            .map(|asset| account_asset_hash(asset).unwrap())
            .collect::<Vec<_>>();
        let (proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_assets);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_assets);
        left.asset_tree_merkle_proofs[left_slot] = proofs.try_into().unwrap();
        left.accounts_before[left_slot].asset_root = old_root;
        right.accounts_before[right_slot].asset_root = new_root;

        let old_balances = left.accounts_before[left_slot]
            .aggregated_balances
            .iter()
            .map(|balance| bigint_leaf_hash(balance).unwrap())
            .collect::<Vec<_>>();
        let new_balances = right.accounts_before[right_slot]
            .aggregated_balances
            .iter()
            .map(|balance| bigint_leaf_hash(balance).unwrap())
            .collect::<Vec<_>>();
        let (proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_balances);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_balances);
        left.public_asset_tree_merkle_proofs[left_slot] = proofs.try_into().unwrap();
        left.accounts_before[left_slot].aggregated_balances_root = old_root;
        right.accounts_before[right_slot].aggregated_balances_root = new_root;

        let old_deltas = left.accounts_delta_before[left_slot]
            .aggregated_asset_deltas
            .iter()
            .map(|delta| bigint_leaf_hash(delta).unwrap())
            .collect::<Vec<_>>();
        let new_deltas = right.accounts_delta_before[right_slot]
            .aggregated_asset_deltas
            .iter()
            .map(|delta| bigint_leaf_hash(delta).unwrap())
            .collect::<Vec<_>>();
        let (proofs, old_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &old_deltas);
        let (_, new_root) = sparse_proofs::<ASSET_MERKLE_LEVELS>(&indices, &new_deltas);
        left.asset_delta_tree_merkle_proofs[left_slot] = proofs.try_into().unwrap();
        left.accounts_delta_before[left_slot].asset_delta_root = old_root;
        right.accounts_delta_before[right_slot].asset_delta_root = new_root;
    }

    fn install_outer_trees(left: &mut Tx<F>, right: &mut Tx<F>) {
        let left_indices = left
            .accounts_before
            .each_ref()
            .map(|account| u64::try_from(account.account_index).unwrap());
        let right_indices = right
            .accounts_before
            .each_ref()
            .map(|account| u64::try_from(account.account_index).unwrap());

        let old_users = [
            account_hash(&left.accounts_before[0]).unwrap(),
            account_hash(&left.accounts_before[1]).unwrap(),
        ];
        let new_users = [
            account_hash(&right.accounts_before[0]).unwrap(),
            account_hash(&right.accounts_before[1]).unwrap(),
        ];
        let old_fee = fee_account_hash(&left.accounts_before[FEE_ACCOUNT_ID]).unwrap();
        let new_fee = fee_account_hash(&right.accounts_before[FEE_ACCOUNT_ID]).unwrap();

        let old_full = [old_users[0].0, old_users[1].0, old_fee.0];
        let new_full = [new_users[0].0, new_users[1].0, new_fee.0];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&left_indices, &old_full);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&right_indices, &new_full);
        left.account_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_tree_root = root;
        right.old_account_tree_root = new_root;

        let old_public = [old_users[0].1, old_users[1].1, old_fee.1];
        let new_public = [new_users[0].1, new_users[1].1, new_fee.1];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&left_indices, &old_public);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&right_indices, &new_public);
        left.account_pub_data_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_pub_data_tree_root = root;
        right.old_account_pub_data_tree_root = new_root;

        let old_delta = [
            account_delta_hash(&left.accounts_delta_before[0]).unwrap(),
            account_delta_hash(&left.accounts_delta_before[1]).unwrap(),
            fee_account_delta_hash(&left.accounts_delta_before[FEE_ACCOUNT_ID]).unwrap(),
        ];
        let new_delta = [
            account_delta_hash(&right.accounts_delta_before[0]).unwrap(),
            account_delta_hash(&right.accounts_delta_before[1]).unwrap(),
            fee_account_delta_hash(&right.accounts_delta_before[FEE_ACCOUNT_ID]).unwrap(),
        ];
        let (proofs, root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&left_indices, &old_delta);
        let (_, new_root) = sparse_proofs::<ACCOUNT_MERKLE_LEVELS>(&right_indices, &new_delta);
        left.account_delta_tree_merkle_proofs = proofs.try_into().unwrap();
        left.old_account_delta_tree_root = root;
        right.old_account_delta_tree_root = new_root;
    }

    fn authenticate_boundary(tx: &mut Tx<F>, metadata: HashOut<F>) {
        let globals = GlobalHashes::from_tx(tx).unwrap();
        let (validium, state) = globals
            .roots(
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

    fn synthetic_candidate_block(metadata: HashOut<F>, swap_left_users: bool) -> Block<F> {
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

        if swap_left_users {
            left.accounts_before.swap(0, 1);
            left.account_assets_before.swap(0, 1);
            left.accounts_delta_before.swap(0, 1);
        }

        for left_slot in 0..NB_ACCOUNTS_PER_TX {
            let account_index = left.accounts_before[left_slot].account_index;
            let right_slot = right
                .accounts_before
                .iter()
                .position(|account| account.account_index == account_index)
                .unwrap();
            install_inner_trees(&mut left, &mut right, left_slot, right_slot);
        }
        install_outer_trees(&mut left, &mut right);
        authenticate_boundary(&mut left, metadata);
        authenticate_boundary(&mut right, metadata);

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
            .expect("materializer test thread must start")
            .join()
            .expect("materializer test thread must finish");
    }

    #[test]
    fn scanner_candidate_materializes_and_replays_to_authenticated_right_boundary() {
        with_large_stack(|| {
            let metadata = HashOut::ZERO;
            let block = synthetic_candidate_block(metadata, false);
            let candidate = scan(&block)
                .candidates
                .into_iter()
                .find(|candidate| candidate.start_tx_index == 0 && candidate.end_tx_index == 3)
                .expect("synthetic interval must be scanner-certified");

            let pair = materialize_scanned_candidate(&block, &candidate, metadata)
                .expect("certified interval must materialize");
            let right = block.tx_chunks[0][3].as_ref();
            assert_eq!(pair.first_removed_tx_index, 0);
            assert_eq!(pair.right_boundary_tx_index, 3);
            assert_eq!(pair.following_tx_index_shift, 1);
            assert_eq!(pair.light_insert.tx_circuit_type, TX_LIGHT);
            assert_eq!(pair.heavy_fill.tx_circuit_type, TX_HEAVY);
            assert_eq!(pair.light_insert.tx_index, 0);
            assert_eq!(pair.heavy_fill.tx_index, 1);
            assert_eq!(
                pair.heavy_fill.old_validium_root,
                pair.after_light_validium_root
            );
            assert_eq!(pair.heavy_fill.old_state_root, pair.after_light_state_root);
            assert_eq!(
                pair.heavy_fill.old_account_delta_tree_root,
                pair.after_light_delta_root
            );
            assert_eq!(pair.final_validium_root, right.old_validium_root);
            assert_eq!(pair.final_state_root, right.old_state_root);
            assert_eq!(pair.final_delta_root, right.old_account_delta_tree_root);
            assert_eq!(
                register_stack_hash(&pair.light_insert.register_stack_before),
                HashOut::ZERO
            );
            assert_eq!(
                register_stack_hash(&pair.heavy_fill.register_stack_before),
                HashOut::ZERO
            );
            assert_eq!(
                account_order_hash(&pair.light_insert.account_order_before),
                HashOut::ZERO
            );
            assert_eq!(
                account_order_hash(&pair.heavy_fill.account_order_before),
                HashOut::ZERO
            );
        });
    }

    #[test]
    fn swapped_boundary_slots_materialize_and_nonempty_position_delta_is_rejected() {
        with_large_stack(|| {
            let metadata = HashOut::ZERO;
            let block = synthetic_candidate_block(metadata, true);
            assert_eq!(block.tx_chunks[0][0].accounts_before[0].account_index, 23);
            assert_eq!(block.tx_chunks[0][0].accounts_before[1].account_index, 17);
            assert_eq!(block.tx_chunks[0][3].accounts_before[0].account_index, 17);
            assert_eq!(block.tx_chunks[0][3].accounts_before[1].account_index, 23);

            let candidate = scan(&block)
                .candidates
                .into_iter()
                .find(|candidate| candidate.start_tx_index == 0 && candidate.end_tx_index == 3)
                .expect("slot-permuted interval must be scanner-certified");
            let pair = materialize_scanned_candidate(&block, &candidate, metadata)
                .expect("slot-permuted interval must materialize");
            assert_eq!(pair.light_insert.accounts_before[0].account_index, 17);
            assert_eq!(pair.light_insert.accounts_before[1].account_index, 23);
            assert_eq!(pair.heavy_fill.accounts_before[0].account_index, 23);
            assert_eq!(pair.heavy_fill.accounts_before[1].account_index, 17);

            let mut nonempty_position = block.clone();
            Arc::make_mut(&mut nonempty_position.tx_chunks[0][0]).accounts_delta_before[1]
                .positions_delta
                .position_delta = BigInt::from(1);
            let error = materialize_scanned_candidate(&nonempty_position, &candidate, metadata)
                .expect_err("selected nonempty position-delta leaf must fail closed");
            assert!(error.to_string().contains("position-delta"));
        });
    }

    #[test]
    fn metadata_right_root_and_lane_tampering_fail_closed() {
        with_large_stack(|| {
            let metadata = HashOut::ZERO;
            let block = synthetic_candidate_block(metadata, false);
            let candidate = scan(&block)
                .candidates
                .into_iter()
                .find(|candidate| candidate.start_tx_index == 0 && candidate.end_tx_index == 3)
                .unwrap();

            let mut wrong_metadata = metadata;
            wrong_metadata.elements[0] = F::ONE;
            assert!(materialize_scanned_candidate(&block, &candidate, wrong_metadata).is_err());

            let mut wrong_right = block.clone();
            Arc::make_mut(&mut wrong_right.tx_chunks[0][3])
                .old_state_root
                .elements[0] += F::ONE;
            assert!(materialize_scanned_candidate(&wrong_right, &candidate, metadata).is_err());

            let mut wrong_lane = block.clone();
            Arc::make_mut(&mut wrong_lane.tx_chunks[0][4]).tx_circuit_type = 0xff;
            assert!(materialize_scanned_candidate(&wrong_lane, &candidate, metadata).is_err());
        });
    }
}
