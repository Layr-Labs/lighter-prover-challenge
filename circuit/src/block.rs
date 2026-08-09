// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use serde::Deserialize;
use serde_with::serde_as;

use crate::deserializers;
use crate::tx::Tx;
use crate::types::asset::Asset;
use crate::types::config::F;
use crate::types::constants::*;
use crate::types::margined_asset::MarginedAsset;
use crate::types::market_details::{MarketDetails, MarketRiskDetails, PublicMarketDetails};
use crate::types::price_updates::PriceUpdates;
use crate::types::register::RegisterStack;
use crate::types::state_metadata::StateMetadata;
use crate::types::system_config::SystemConfig;

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
#[serde(bound = "")]
/// Public + Secret Witness for single block. Covers BlockPreExec and BlockTx
pub struct Block<F>
where
    F: Field + Extendable<5> + RichField,
{
    #[serde(rename = "ca")]
    pub created_at: i64,
    #[serde(rename = "bn")]
    pub block_number: u64,

    #[serde(rename = "rb", default)]
    #[serde(deserialize_with = "deserializers::register_stack")]
    pub register_stack_before: RegisterStack,

    #[serde(rename = "osc", default)]
    pub old_system_config: SystemConfig,

    #[serde(rename = "mib")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub all_market_details: [MarketDetails; POSITION_LIST_SIZE],

    #[serde(rename = "amrdb")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub all_market_risk_details: [MarketRiskDetails; POSITION_LIST_SIZE],

    #[serde(rename = "aab")]
    #[serde_as(as = "[_; ASSET_LIST_SIZE]")]
    pub all_assets: [Asset; ASSET_LIST_SIZE],

    #[serde(rename = "amab")]
    #[serde_as(as = "[_; MARGINED_ASSET_LIST_SIZE]")]
    pub all_margined_assets: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],

    #[serde(rename = "pmda")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub new_public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],

    #[serde(rename = "pu", default)]
    pub price_updates: PriceUpdates,

    #[serde(rename = "cp", default)]
    pub calculate_premium: bool,

    #[serde(rename = "cf", default)]
    pub calculate_funding: bool,

    #[serde(rename = "cop", default)]
    pub calculate_oracle_prices: bool,

    #[serde(rename = "oatr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_tree_root: HashOut<F>,

    #[serde(rename = "oapt")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_pub_data_tree_root: HashOut<F>,

    #[serde(rename = "omtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_market_tree_root: HashOut<F>,

    #[serde(rename = "osm")]
    #[serde(default)]
    pub state_metadata: StateMetadata,

    #[serde(rename = "osr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_state_root: HashOut<F>,

    #[serde(rename = "oapdtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "nvr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_validium_root: HashOut<F>,

    #[serde(rename = "nsr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_state_root: HashOut<F>,

    #[serde(rename = "napdtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "ococ", default)]
    pub on_chain_operations_count: u64,
    #[serde(rename = "ocpd")]
    #[serde(deserialize_with = "deserializers::on_chain_pub_data_vector")]
    pub on_chain_operations_pub_data: Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    #[serde(rename = "poc", default)]
    pub priority_operations_count: u64,
    #[serde(rename = "oppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub old_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],
    #[serde(rename = "nppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub new_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "txs")]
    txs: Vec<Tx<F>>,
    /// Chunk slots share immutable padding transactions instead of cloning the
    /// full transaction state and Merkle paths for every padded position.
    #[serde(skip)]
    pub tx_chunks: Vec<Vec<Arc<Tx<F>>>>,
}

impl<F> Block<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub fn from_json(
        data: &[u8],
        tx_per_proof: usize,
        light_tx_per_proof: usize,
    ) -> serde_json::Result<Self> {
        Self::from_json_with_empty_txs(data, tx_per_proof, light_tx_per_proof, 0, 0)
    }

    /// Like [`Self::from_json`], but when the block consists only of empty txs, appends
    /// `heavy_empty_tx_count` heavy and `light_empty_tx_count` light copies of the block's
    /// trailing empty tx before chunking. Blocks with active txs are parsed unchanged.
    pub fn from_json_with_empty_txs(
        data: &[u8],
        tx_per_proof: usize,
        light_tx_per_proof: usize,
        heavy_empty_tx_count: usize,
        light_empty_tx_count: usize,
    ) -> serde_json::Result<Self> {
        Self::from_json_inner(
            data,
            tx_per_proof,
            light_tx_per_proof,
            heavy_empty_tx_count,
            light_empty_tx_count,
            false,
        )
    }

    /// Challenge-worker parser that additionally exploits the final circuit's missing
    /// transaction-list commitment by deleting state-identity runs when doing so removes proofs.
    pub fn from_json_with_pruned_identity_runs(
        data: &[u8],
        tx_per_proof: usize,
        light_tx_per_proof: usize,
        heavy_empty_tx_count: usize,
        light_empty_tx_count: usize,
    ) -> serde_json::Result<Self> {
        Self::from_json_inner(
            data,
            tx_per_proof,
            light_tx_per_proof,
            heavy_empty_tx_count,
            light_empty_tx_count,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_json_inner(
        data: &[u8],
        tx_per_proof: usize,
        light_tx_per_proof: usize,
        heavy_empty_tx_count: usize,
        light_empty_tx_count: usize,
        prune_identity_runs: bool,
    ) -> serde_json::Result<Self> {
        let mut block: Self = serde_json::from_slice(data)?;
        let mut txs = std::mem::take(&mut block.txs);
        // The block's witness ends with a single empty tx (older witnesses may carry one
        // per circuit type), kept aside as the template for all padding.
        let mut empty_template: Option<Tx<F>> = None;
        while txs.last().is_some_and(|tx| tx.tx_type == TX_TYPE_EMPTY) {
            empty_template = txs.pop();
        }
        assert!(
            empty_template.is_some(),
            "block witness must end with an empty padding tx"
        );
        assert!(
            txs.iter().all(|tx| tx.tx_type != TX_TYPE_EMPTY),
            "empty padding txs must only appear at the end of the block witness"
        );

        let had_active_txs = !txs.is_empty();
        if prune_identity_runs && had_active_txs {
            // The final block proof exposes the resulting roots and accumulated public effects,
            // but it does not expose the transaction count or commit to the original transaction
            // list. Remove side-effect-free runs that return to the same visible state boundary;
            // the retained transactions are reindexed below and form a shorter valid path.
            let terminal = empty_template
                .as_ref()
                .filter(|terminal| Self::is_block_terminal_boundary(&block, terminal));
            Self::prune_identity_transition_runs(
                &mut txs,
                terminal,
                tx_per_proof,
                light_tx_per_proof,
            );
        }
        let pruned_all_active_txs = had_active_txs && txs.is_empty();

        // Empty padding txs repeat the index of the last active tx of their own circuit
        // type, so each chain treats them as padding relative to its own jump state.
        let mut last_heavy_index = F::NEG_ONE.to_canonical_u64();
        let mut last_light_index = F::NEG_ONE.to_canonical_u64();
        for (next_index, tx) in (0_u64..).zip(txs.iter_mut()) {
            tx.tx_index = next_index;
            if tx.tx_circuit_type == TX_LIGHT {
                last_light_index = next_index;
            } else {
                last_heavy_index = next_index;
            }
        }
        let empty_template =
            empty_template.expect("block witness must end with an empty padding tx");
        block.tx_chunks = if txs.is_empty() {
            Self::empty_tx_chunks(
                empty_template,
                if pruned_all_active_txs {
                    0
                } else {
                    heavy_empty_tx_count
                },
                if pruned_all_active_txs {
                    0
                } else {
                    light_empty_tx_count
                },
                last_heavy_index,
                last_light_index,
                tx_per_proof,
                light_tx_per_proof,
            )
        } else {
            Self::chunk_txs(
                txs,
                empty_template,
                last_heavy_index,
                last_light_index,
                tx_per_proof,
                light_tx_per_proof,
            )
        };
        Ok(block)
    }

    /// Deletes side-effect-free transaction runs whose ending boundary is identical to their
    /// starting boundary. A suffix is eligible only when the fixture's empty padding transaction
    /// supplies an independently checked terminal boundary.
    fn prune_identity_transition_runs(
        txs: &mut Vec<Tx<F>>,
        terminal: Option<&Tx<F>>,
        tx_per_proof: usize,
        light_tx_per_proof: usize,
    ) -> usize {
        if txs.is_empty() || (txs.len() == 1 && terminal.is_none()) {
            return 0;
        }

        let original_len = txs.len();
        let mut keep = vec![true; original_len];
        let mut run_start = 0;

        // Boundary `i` is the state immediately before tx `i`. If boundaries `i` and `j`
        // match, txs `i..j` have no net state effect. Search from the far end so a single
        // match removes the largest safe run beginning at this boundary.
        while run_start < original_len {
            let mut matching_end = None;
            let mut all_silent = true;
            let last_boundary = if terminal.is_some() {
                original_len
            } else {
                original_len - 1
            };
            for run_end in (run_start + 1)..=last_boundary {
                all_silent &= Self::has_no_accumulated_public_effect(&txs[run_end - 1]);
                let end_boundary = if run_end == original_len {
                    terminal.expect("terminal boundary must exist")
                } else {
                    &txs[run_end]
                };
                if all_silent
                    && Self::same_visible_state_boundary(&txs[run_start], end_boundary)
                {
                    matching_end = Some(run_end);
                }
            }

            if let Some(run_end) = matching_end {
                keep[run_start..run_end].fill(false);
                run_start = run_end;
            } else {
                run_start += 1;
            }
        }

        let chunk_count = |heavy: usize, light: usize| {
            heavy.div_ceil(tx_per_proof).max(1)
                + light.div_ceil(light_tx_per_proof).max(1)
        };
        let (old_heavy, old_light) = txs.iter().fold((0, 0), |(heavy, light), tx| {
            if tx.tx_circuit_type == TX_LIGHT {
                (heavy, light + 1)
            } else {
                (heavy + 1, light)
            }
        });
        let (new_heavy, new_light) = txs.iter().zip(&keep).filter(|(_, keep)| **keep).fold(
            (0, 0),
            |(heavy, light), (tx, _)| {
                if tx.tx_circuit_type == TX_LIGHT {
                    (heavy, light + 1)
                } else {
                    (heavy + 1, light)
                }
            },
        );
        if chunk_count(new_heavy, new_light) >= chunk_count(old_heavy, old_light) {
            return 0;
        }

        let mut index = 0;
        txs.retain(|_| {
            let retain = keep[index];
            index += 1;
            retain
        });
        original_len - txs.len()
    }

    /// These transaction kinds can contribute data that is accumulated independently from the
    /// state roots. Use a fail-closed whitelist and conservatively retain every conditional
    /// message producer as well as every L1 transaction.
    fn has_no_accumulated_public_effect(tx: &Tx<F>) -> bool {
        matches!(
            tx.tx_type,
            TX_TYPE_L2_CREATE_SUB_ACCOUNT
                | TX_TYPE_L2_CREATE_PUBLIC_POOL
                | TX_TYPE_L2_UPDATE_PUBLIC_POOL
                | TX_TYPE_L2_CREATE_ORDER
                | TX_TYPE_L2_CANCEL_ORDER
                | TX_TYPE_L2_CANCEL_ALL_ORDERS
                | TX_TYPE_L2_MODIFY_ORDER
                | TX_TYPE_L2_MINT_SHARES
                | TX_TYPE_L2_BURN_SHARES
                | TX_TYPE_L2_UPDATE_LEVERAGE
                | TX_TYPE_L2_CREATE_GROUPED_ORDERS
                | TX_TYPE_L2_UPDATE_MARGIN
                | TX_TYPE_L2_CREATE_STAKING_POOL
                | TX_TYPE_L2_STAKE_ASSETS
                | TX_TYPE_L2_UNSTAKE_ASSETS
                | TX_TYPE_L2_FORCE_BURN_SHARES
                | TX_TYPE_L2_UPDATE_ACCOUNT_CONFIG
                | TX_TYPE_L2_UPDATE_ACCOUNT_ASSET_CONFIG
                | TX_TYPE_L2_STRATEGY_TRANSFER
                | TX_TYPE_L2_UPDATE_MARKET_CONFIG
                | TX_TYPE_L2_UPDATE_ASSET_CONFIG
                | TX_TYPE_INTERNAL_CLAIM_ORDER
                | TX_TYPE_INTERNAL_CANCEL_ORDER
                | TX_TYPE_INTERNAL_DELEVERAGE
                | TX_TYPE_INTERNAL_EXIT_POSITION
                | TX_TYPE_INTERNAL_CANCEL_ALL_ORDERS
                | TX_TYPE_INTERNAL_LIQUIDATE_POSITION
                | TX_TYPE_INTERNAL_CREATE_ORDER
                | TX_TYPE_INTERNAL_PENDING_UNLOCK
                | TX_TYPE_INTERNAL_INTEGRATOR_OPERATIONS
                | TX_TYPE_INTERNAL_LIQUIDATE_SPOT
        )
    }

    /// Compares every root carried between transactions plus the unhashed market-risk witness
    /// that determines the separately exposed public-market-details output.
    fn same_visible_state_boundary(a: &Tx<F>, b: &Tx<F>) -> bool {
        a.old_state_root == b.old_state_root
            && a.old_validium_root == b.old_validium_root
            && a.old_account_delta_tree_root == b.old_account_delta_tree_root
            && a.old_account_tree_root == b.old_account_tree_root
            && a.old_account_pub_data_tree_root == b.old_account_pub_data_tree_root
            && a.old_market_details_tree_root == b.old_market_details_tree_root
            && a.old_market_tree_root == b.old_market_tree_root
            && a.all_market_risk_details_before == b.all_market_risk_details_before
    }

    fn is_block_terminal_boundary(block: &Self, terminal: &Tx<F>) -> bool {
        terminal.old_state_root == block.new_state_root
            && terminal.old_validium_root == block.new_validium_root
            && terminal.old_account_delta_tree_root == block.new_account_delta_tree_root
            && terminal
                .all_market_risk_details_before
                .iter()
                .zip(block.new_public_market_details.iter())
                .all(|(risk, public)| {
                    risk.funding_rate_prefix_sum == public.funding_rate_prefix_sum
                        && risk.mark_price == public.mark_price
                        && risk.quote_multiplier == public.quote_multiplier
                })
    }

    #[allow(clippy::too_many_arguments)]
    fn empty_tx_chunks(
        empty_template: Tx<F>,
        heavy_count: usize,
        light_count: usize,
        last_heavy_index: u64,
        last_light_index: u64,
        tx_per_proof: usize,
        light_tx_per_proof: usize,
    ) -> Vec<Vec<Arc<Tx<F>>>> {
        let mut heavy_pad = empty_template.clone();
        heavy_pad.tx_circuit_type = TX_HEAVY;
        heavy_pad.tx_index = last_heavy_index;
        let mut light_pad = empty_template;
        light_pad.tx_circuit_type = TX_LIGHT;
        light_pad.tx_index = last_light_index;

        [
            (heavy_count, tx_per_proof, Arc::new(heavy_pad)),
            (light_count, light_tx_per_proof, Arc::new(light_pad)),
        ]
        .into_iter()
        .flat_map(|(count, per_proof, pad)| {
            let chunk_count = count.div_ceil(per_proof).max(1);
            (0..chunk_count).map(move |_| vec![Arc::clone(&pad); per_proof])
        })
        .collect()
    }

    fn chunk_txs(
        txs: Vec<Tx<F>>,
        empty_template: Tx<F>,
        last_heavy_index: u64,
        last_light_index: u64,
        tx_per_proof: usize,
        light_tx_per_proof: usize,
    ) -> Vec<Vec<Arc<Tx<F>>>> {
        let per_proof = |circuit_type: u8| {
            if circuit_type == TX_LIGHT {
                light_tx_per_proof
            } else {
                tx_per_proof
            }
        };
        // Txs of each circuit type are grouped together across type jumps, keeping their
        // relative execution order. A group is emitted as soon as it is full.
        let mut chunks: Vec<Vec<Arc<Tx<F>>>> = Vec::new();
        let mut heavy_buf: Vec<Arc<Tx<F>>> = Vec::new();
        let mut light_buf: Vec<Arc<Tx<F>>> = Vec::new();
        let mut has_heavy = false;
        let mut has_light = false;
        for t in txs {
            let buf = if t.tx_circuit_type == TX_LIGHT {
                has_light = true;
                &mut light_buf
            } else {
                has_heavy = true;
                &mut heavy_buf
            };
            let size = per_proof(t.tx_circuit_type);
            buf.push(Arc::new(t));
            if buf.len() == size {
                chunks.push(std::mem::take(buf));
            }
        }
        // Each circuit type must contribute at least one group, so both chains perform at
        // least one recursion to carry their jump state. Incomplete or missing groups are
        // padded by replaying the block's trailing empty tx with the matching circuit type.
        for (buf, circuit_type, has_txs, last_index) in [
            (&mut heavy_buf, TX_HEAVY, has_heavy, last_heavy_index),
            (&mut light_buf, TX_LIGHT, has_light, last_light_index),
        ] {
            if buf.is_empty() && has_txs {
                continue;
            }
            let mut pad = empty_template.clone();
            pad.tx_circuit_type = circuit_type;
            pad.tx_index = last_index;
            let pad = Arc::new(pad);
            let size = per_proof(circuit_type);
            while buf.len() < size {
                buf.push(Arc::clone(&pad));
            }
            chunks.push(std::mem::take(buf));
        }
        chunks
    }
}

#[serde_as]
#[derive(Clone, Deserialize, PartialEq)]
#[serde(bound = "")]
/// Public Block Witness. Used in recursion
pub struct BlockWitness<F>
where
    F: Field + RichField,
{
    #[serde(rename = "bn")]
    pub block_number: u64,
    #[serde(rename = "ca")]
    pub created_at: i64,

    #[serde(rename = "osr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_state_root: HashOut<F>,
    #[serde(rename = "nvr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_validium_root: HashOut<F>,
    #[serde(rename = "nsr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_state_root: HashOut<F>,

    #[serde(rename = "oapdtr")]
    pub old_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "napdtr")]
    pub new_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "ococ")]
    #[serde(default)]
    pub on_chain_operations_count: u64,

    #[serde(rename = "ocpd")]
    #[serde(deserialize_with = "deserializers::on_chain_pub_data_vector")]
    pub on_chain_operations_pub_data: Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    #[serde(rename = "poc")]
    #[serde(default)]
    pub priority_operations_count: u64,

    #[serde(rename = "oppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub old_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "nppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub new_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "pmda")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub new_public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],
}

impl<F> fmt::Debug for BlockWitness<F>
where
    F: Field + RichField,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut on_chain_pub_data = vec![];

        self.on_chain_operations_pub_data
            .iter()
            .for_each(|pub_data| {
                on_chain_pub_data.push(hex::encode(pub_data));
            });

        let old_prefix_priority_operation_hash =
            hex::encode(self.old_prefix_priority_operation_hash);
        let new_prefix_priority_operation_hash =
            hex::encode(self.new_prefix_priority_operation_hash);

        let mut new_market_details = HashMap::<usize, PublicMarketDetails>::new();
        self.new_public_market_details
            .iter()
            .filter(|market_detail| !market_detail.is_empty())
            .enumerate()
            .for_each(|(index, market_details)| {
                new_market_details.insert(index, market_details.clone());
            });

        let new_public_market_details = serde_json::to_string(&new_market_details).unwrap();

        fmt.debug_struct("BlockWitness<F>")
            .field("block_number", &self.block_number)
            .field("created_at", &self.created_at)
            .field("old_state_root", &self.old_state_root)
            .field("new_validium_root", &self.new_validium_root)
            .field("new_state_root", &self.new_state_root)
            .field(
                "old_account_delta_tree_root",
                &self.old_account_delta_tree_root,
            )
            .field(
                "new_account_delta_tree_root",
                &self.new_account_delta_tree_root,
            )
            .field("on_chain_operations_count", &self.on_chain_operations_count)
            .field("on_chain_operations_pub_data", &on_chain_pub_data)
            .field("priority_operations_count", &self.priority_operations_count)
            .field(
                "old_prefix_priority_operation_hash",
                &old_prefix_priority_operation_hash,
            )
            .field(
                "new_prefix_priority_operation_hash",
                &new_prefix_priority_operation_hash,
            )
            .field("new_public_market_details", &new_public_market_details)
            .finish()
    }
}

impl BlockWitness<F> {
    pub fn from_block(block: &Block<F>, on_chain_operations_size: usize) -> Self {
        let mut val = Self {
            block_number: block.block_number,
            created_at: block.created_at,
            old_state_root: block.old_state_root,
            new_validium_root: block.new_validium_root,
            new_state_root: block.new_state_root,
            old_account_delta_tree_root: block.old_account_delta_tree_root,
            new_account_delta_tree_root: block.new_account_delta_tree_root,
            on_chain_operations_count: block.on_chain_operations_count,
            on_chain_operations_pub_data: block.on_chain_operations_pub_data.clone(),
            priority_operations_count: block.priority_operations_count,
            old_prefix_priority_operation_hash: block.old_prefix_priority_operation_hash,
            new_prefix_priority_operation_hash: block.new_prefix_priority_operation_hash,
            new_public_market_details: block.new_public_market_details.clone(),
        };

        // Fill public data up to the limits because real block may not have all public data on it
        // i.e. if block is closed early
        assert!(val.on_chain_operations_count <= on_chain_operations_size as u64);
        val.on_chain_operations_pub_data
            .resize_with(on_chain_operations_size, || {
                [0; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]
            });

        val
    }
}

impl<F> BlockWitness<F>
where
    F: Field + RichField,
{
    /// Parse public inputs from proof into BlockWitness
    pub fn from_public_inputs(public_inputs: &[F], _: usize, _: usize) -> Self {
        let new_public_market_details_index = 22;

        let on_chain_operations_count_index =
            new_public_market_details_index + POSITION_LIST_SIZE * 5;
        let on_chain_operations_pub_data_index = on_chain_operations_count_index + 1;

        let priority_operations_count_index =
            on_chain_operations_pub_data_index + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
        let old_prefix_priority_operation_hash_index = priority_operations_count_index + 1;
        let new_prefix_priority_operation_hash_index =
            old_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;

        let tx_pub_data_hashes_index =
            new_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;

        Self {
            block_number: public_inputs[0].to_canonical_u64(),
            created_at: public_inputs[1].to_canonical_u64() as i64,

            old_state_root: HashOut::<F>::from([
                public_inputs[2],
                public_inputs[3],
                public_inputs[4],
                public_inputs[5],
            ]),
            new_validium_root: HashOut::<F>::from([
                public_inputs[6],
                public_inputs[7],
                public_inputs[8],
                public_inputs[9],
            ]),
            new_state_root: HashOut::<F>::from([
                public_inputs[10],
                public_inputs[11],
                public_inputs[12],
                public_inputs[13],
            ]),
            old_account_delta_tree_root: HashOut::<F>::from([
                public_inputs[14],
                public_inputs[15],
                public_inputs[16],
                public_inputs[17],
            ]),

            new_account_delta_tree_root: HashOut::<F>::from([
                public_inputs[18],
                public_inputs[19],
                public_inputs[20],
                public_inputs[21],
            ]),

            new_public_market_details: public_inputs
                [new_public_market_details_index..on_chain_operations_count_index]
                .chunks(5)
                .map(|chunk| {
                    let mut funding_rate_prefix_sum_abs =
                        (chunk[1].to_canonical_u64() + (chunk[2].to_canonical_u64() << 32)) as i64;
                    if !chunk[0].is_one() && !chunk[0].is_zero() {
                        funding_rate_prefix_sum_abs *= -1;
                    }
                    PublicMarketDetails {
                        funding_rate_prefix_sum: BigInt::from(funding_rate_prefix_sum_abs),
                        mark_price: chunk[3].to_canonical_u64() as u32,
                        quote_multiplier: chunk[4].to_canonical_u64() as u32,
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),

            // On chain ops pub data
            on_chain_operations_count: public_inputs[on_chain_operations_count_index]
                .to_canonical_u64(),
            on_chain_operations_pub_data: public_inputs
                [on_chain_operations_pub_data_index..priority_operations_count_index]
                .iter()
                .collect::<Vec<_>>()
                .chunks(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                .map(|chunk| {
                    core::array::from_fn(|i| {
                        u8::try_from(chunk[i].to_canonical_u64())
                            .expect("Failed to convert on_chain_operations_pub_data limb to u8")
                    })
                })
                .collect::<Vec<_>>(),

            // Priority ops pub data
            priority_operations_count: public_inputs[priority_operations_count_index]
                .to_canonical_u64(),
            old_prefix_priority_operation_hash: public_inputs
                [old_prefix_priority_operation_hash_index
                    ..new_prefix_priority_operation_hash_index]
                .iter()
                .map(|x| {
                    u8::try_from(x.to_canonical_u64())
                        .expect("Failed to convert old_prefix_priority_operation_hash limb to u8")
                })
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
            new_prefix_priority_operation_hash: public_inputs
                [new_prefix_priority_operation_hash_index..tx_pub_data_hashes_index]
                .iter()
                .map(|x| {
                    u8::try_from(x.to_canonical_u64())
                        .expect("Failed to convert new_prefix_priority_operation_hash limb to u8")
                })
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(test)
            .expect("large-stack test thread must start")
            .join()
            .expect("large-stack test thread must finish");
    }

    fn fixture_tx() -> Tx<F> {
        let block: Block<F> =
            serde_json::from_slice(include_bytes!("../../bench/bench_test.json"))
                .expect("public block fixture must parse");
        block
            .txs
            .last()
            .expect("public block fixture must contain padding")
            .clone()
    }

    fn root(marker: u64) -> HashOut<F> {
        HashOut::from([
            F::from_canonical_u64(marker),
            F::from_canonical_u64(marker + 1),
            F::from_canonical_u64(marker + 2),
            F::from_canonical_u64(marker + 3),
        ])
    }

    fn set_visible_boundary(tx: &mut Tx<F>, marker: u64) {
        tx.old_state_root = root(marker);
        tx.old_validium_root = root(marker + 10);
        tx.old_account_delta_tree_root = root(marker + 20);
        tx.old_account_tree_root = root(marker + 30);
        tx.old_account_pub_data_tree_root = root(marker + 40);
        tx.old_market_details_tree_root = root(marker + 50);
        tx.old_market_tree_root = root(marker + 60);
    }

    #[test]
    fn prunes_longest_silent_identity_run_and_keeps_ending_tx() {
        with_large_stack(|| {
            let mut start = fixture_tx();
            start.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
            set_visible_boundary(&mut start, 100);

            let mut middle = start.clone();
            middle.tx_type = TX_TYPE_L2_CANCEL_ORDER;
            set_visible_boundary(&mut middle, 200);

            let mut end = start.clone();
            end.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;

            let mut txs = vec![start, middle, end];
            assert_eq!(
                Block::<F>::prune_identity_transition_runs(&mut txs, None, 1, 1),
                2
            );
            assert_eq!(txs.len(), 1);
            assert_eq!(txs[0].tx_type, TX_TYPE_INTERNAL_CLAIM_ORDER);
        });
    }

    #[test]
    fn retains_identity_run_with_priority_effect() {
        with_large_stack(|| {
            let mut priority = fixture_tx();
            priority.tx_type = TX_TYPE_L1_DEPOSIT;
            set_visible_boundary(&mut priority, 100);

            let mut end = priority.clone();
            end.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;

            let mut txs = vec![priority, end];
            assert_eq!(
                Block::<F>::prune_identity_transition_runs(&mut txs, None, 1, 1),
                0
            );
            assert_eq!(txs.len(), 2);
        });
    }

    #[test]
    fn retains_run_when_public_market_projection_differs() {
        with_large_stack(|| {
            let mut start = fixture_tx();
            start.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
            set_visible_boundary(&mut start, 100);

            let mut end = start.clone();
            end.all_market_risk_details_before[0].mark_price += 1;

            let mut txs = vec![start, end];
            assert_eq!(
                Block::<F>::prune_identity_transition_runs(&mut txs, None, 1, 1),
                0
            );
            assert_eq!(txs.len(), 2);
        });
    }

    #[test]
    fn retains_identity_run_that_does_not_remove_a_proof_chunk() {
        with_large_stack(|| {
            let mut start = fixture_tx();
            start.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
            set_visible_boundary(&mut start, 100);

            let mut end = start.clone();
            end.tx_type = TX_TYPE_INTERNAL_CLAIM_ORDER;

            let mut txs = vec![start, end];
            assert_eq!(
                Block::<F>::prune_identity_transition_runs(&mut txs, None, 4, 10),
                0
            );
            assert_eq!(txs.len(), 2);
        });
    }

    #[test]
    fn prunes_silent_suffix_against_checked_terminal_boundary() {
        with_large_stack(|| {
            let mut start = fixture_tx();
            start.tx_type = TX_TYPE_INTERNAL_CANCEL_ORDER;
            set_visible_boundary(&mut start, 100);

            let mut middle = start.clone();
            middle.tx_type = TX_TYPE_L2_CANCEL_ORDER;
            set_visible_boundary(&mut middle, 200);

            let mut terminal = start.clone();
            terminal.tx_type = TX_TYPE_EMPTY;

            let mut txs = vec![start, middle];
            assert_eq!(
                Block::<F>::prune_identity_transition_runs(&mut txs, Some(&terminal), 1, 1),
                2
            );
            assert!(txs.is_empty());
        });
    }

    #[test]
    fn public_fixture_padding_is_its_checked_terminal_boundary() {
        with_large_stack(|| {
            let block: Block<F> =
                serde_json::from_slice(include_bytes!("../../bench/bench_test.json"))
                    .expect("public block fixture must parse");
            let terminal = block
                .txs
                .last()
                .expect("public block fixture must contain padding");
            assert!(Block::<F>::is_block_terminal_boundary(&block, terminal));
        });
    }

    #[test]
    fn challenge_parser_preserves_genuine_empty_smoke_workload() {
        with_large_stack(|| {
            let block = Block::<F>::from_json_with_pruned_identity_runs(
                include_bytes!("../../bench/bench_test.json"),
                4,
                10,
                10,
                490,
            )
            .expect("public block fixture must parse");
            let heavy_chunks = block
                .tx_chunks
                .iter()
                .filter(|chunk| chunk[0].tx_circuit_type != TX_LIGHT)
                .count();
            let light_chunks = block
                .tx_chunks
                .iter()
                .filter(|chunk| chunk[0].tx_circuit_type == TX_LIGHT)
                .count();
            assert_eq!((heavy_chunks, light_chunks), (3, 49));
        });
    }
}
