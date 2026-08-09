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

    /// Parses a prover witness and removes sound, signed skip-nonce runs when
    /// doing so reduces the number of transaction proof groups.
    pub fn from_json_with_pruned_nonce_runs(
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
        prune_nonce_runs: bool,
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
        if prune_nonce_runs {
            Self::prune_skip_nonce_runs(&mut txs, tx_per_proof, light_tx_per_proof);
        }
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
                heavy_empty_tx_count,
                light_empty_tx_count,
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

    fn prune_skip_nonce_runs(
        txs: &mut Vec<Tx<F>>,
        tx_per_proof: usize,
        light_tx_per_proof: usize,
    ) -> usize {
        if txs.len() < 2 {
            return 0;
        }

        let original_len = txs.len();
        let mut patches = Vec::<(usize, usize)>::new();
        let mut segment_start = 0;

        for anchor_index in 1..original_len {
            let anchor = &txs[anchor_index];
            if !Self::is_prunable_light_l2(anchor) {
                segment_start = anchor_index + 1;
                continue;
            }
            if !Self::has_signed_skip_nonce(anchor) {
                continue;
            }

            let mut lower_bound = anchor_index;
            while lower_bound > segment_start {
                let candidate = &txs[lower_bound - 1];
                if !Self::is_prunable_light_l2(candidate)
                    || !Self::same_api_key_owner(candidate, anchor)
                {
                    break;
                }
                lower_bound -= 1;
            }

            let matching_start = (lower_bound..anchor_index).find(|&run_start| {
                Self::same_boundary_except_api_nonce(&txs[run_start], anchor)
            });
            if let Some(run_start) = matching_start {
                patches.push((anchor_index, run_start));
                segment_start = anchor_index + 1;
            }
        }

        if patches.is_empty() {
            return 0;
        }

        let mut keep = vec![true; original_len];
        for &(anchor_index, run_start) in &patches {
            keep[run_start..anchor_index].fill(false);
        }

        let proof_groups = |heavy_count: usize, light_count: usize| {
            heavy_count.div_ceil(tx_per_proof).max(1)
                + light_count.div_ceil(light_tx_per_proof).max(1)
        };
        let (old_heavy_count, old_light_count) = txs.iter().fold((0, 0), |counts, tx| {
            if tx.tx_circuit_type == TX_LIGHT {
                (counts.0, counts.1 + 1)
            } else {
                (counts.0 + 1, counts.1)
            }
        });
        let (new_heavy_count, new_light_count) =
            txs.iter()
                .zip(&keep)
                .filter(|(_, keep_tx)| **keep_tx)
                .fold((0, 0), |counts, (tx, _)| {
                    if tx.tx_circuit_type == TX_LIGHT {
                        (counts.0, counts.1 + 1)
                    } else {
                        (counts.0 + 1, counts.1)
                    }
                });
        if proof_groups(new_heavy_count, new_light_count)
            >= proof_groups(old_heavy_count, old_light_count)
        {
            return 0;
        }

        for (anchor_index, run_start) in patches {
            let (before_anchor, anchor_and_after) = txs.split_at_mut(anchor_index);
            let start = &before_anchor[run_start];
            let anchor = &mut anchor_and_after[0];

            anchor.api_key_before = start.api_key_before.clone();
            anchor.accounts_before[OWNER_ACCOUNT_ID] =
                start.accounts_before[OWNER_ACCOUNT_ID].clone();
            anchor.api_key_tree_merkle_proof = start.api_key_tree_merkle_proof;
            anchor.account_tree_merkle_proofs[OWNER_ACCOUNT_ID] =
                start.account_tree_merkle_proofs[OWNER_ACCOUNT_ID];
            anchor.derive_old_private_roots = true;
        }

        let mut keep_iter = keep.into_iter();
        txs.retain(|_| keep_iter.next().expect("one keep marker per transaction"));
        original_len - txs.len()
    }

    fn is_prunable_light_l2(tx: &Tx<F>) -> bool {
        tx.tx_circuit_type == TX_LIGHT
            && matches!(
                tx.tx_type,
                TX_TYPE_L2_CREATE_ORDER | TX_TYPE_L2_CANCEL_ORDER | TX_TYPE_L2_MODIFY_ORDER
            )
    }

    fn has_signed_skip_nonce(tx: &Tx<F>) -> bool {
        tx.attributes
            .attribute_types
            .iter()
            .zip(tx.attributes.attribute_values.iter())
            .any(|(&attribute_type, &attribute_value)| {
                attribute_type as usize == crate::tx_attributes::ATTR_SKIP_TX_NONCE
                    && attribute_value == 1
            })
    }

    fn same_api_key_owner(start: &Tx<F>, anchor: &Tx<F>) -> bool {
        start.accounts_before[OWNER_ACCOUNT_ID].account_index
            == anchor.accounts_before[OWNER_ACCOUNT_ID].account_index
            && start.api_key_before.api_key_index == anchor.api_key_before.api_key_index
            && start.api_key_before.public_key == anchor.api_key_before.public_key
    }

    fn same_boundary_except_api_nonce(start: &Tx<F>, anchor: &Tx<F>) -> bool {
        if !Self::same_api_key_owner(start, anchor)
            || start.api_key_before.nonce > anchor.api_key_before.nonce
            || start.old_account_pub_data_tree_root != anchor.old_account_pub_data_tree_root
            || start.old_account_delta_tree_root != anchor.old_account_delta_tree_root
            || start.old_market_details_tree_root != anchor.old_market_details_tree_root
            || start.old_market_tree_root != anchor.old_market_tree_root
            || start.api_key_tree_merkle_proof != anchor.api_key_tree_merkle_proof
            || start.account_tree_merkle_proofs[OWNER_ACCOUNT_ID]
                != anchor.account_tree_merkle_proofs[OWNER_ACCOUNT_ID]
            || start.register_stack_before != anchor.register_stack_before
            || start.system_config_before != anchor.system_config_before
            || start.all_assets_before != anchor.all_assets_before
            || start.all_margined_assets_before != anchor.all_margined_assets_before
            || start.all_market_risk_details_before != anchor.all_market_risk_details_before
        {
            return false;
        }

        let mut start_owner = start.accounts_before[OWNER_ACCOUNT_ID].clone();
        let mut anchor_owner = anchor.accounts_before[OWNER_ACCOUNT_ID].clone();
        start_owner.api_key_root = HashOut::ZERO;
        anchor_owner.api_key_root = HashOut::ZERO;
        start_owner == anchor_owner
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
mod nonce_pruning_tests {
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::HashOut;
    use plonky2::iop::witness::{PartialWitness, Witness};

    use super::*;
    use crate::tx_attributes::ATTR_SKIP_TX_NONCE;
    use crate::tx_constraints::{TxTarget, TxTargetWitness};
    use crate::types::config::{Builder, CIRCUIT_CONFIG};

    fn empty_tx_template() -> Tx<F> {
        let block: Block<F> =
            serde_json::from_slice(include_bytes!("../../bench/bench_test.json")).unwrap();
        block.txs.into_iter().next().unwrap()
    }

    fn marker_hash(value: u64) -> HashOut<F> {
        HashOut::from([F::from_canonical_u64(value); 4])
    }

    fn light_nonce_tx(nonce: i64) -> Tx<F> {
        let mut tx = empty_tx_template();
        tx.tx_type = TX_TYPE_L2_CANCEL_ORDER;
        tx.tx_circuit_type = TX_LIGHT;
        tx.nonce = nonce;
        tx.api_key_before.api_key_index = 2;
        tx.api_key_before.nonce = nonce;
        tx.accounts_before[OWNER_ACCOUNT_ID].account_index = 17;
        tx.accounts_before[OWNER_ACCOUNT_ID].api_key_root = marker_hash(100 + nonce as u64);
        tx.old_account_tree_root = marker_hash(200 + nonce as u64);
        tx.old_validium_root = marker_hash(300 + nonce as u64);
        tx.old_state_root = marker_hash(400 + nonce as u64);
        tx
    }

    fn add_skip_nonce_attribute(tx: &mut Tx<F>) {
        tx.attributes.attribute_types[0] = ATTR_SKIP_TX_NONCE as u8;
        tx.attributes.attribute_values[0] = 1;
    }

    fn with_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("nonce-pruning-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(test)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn prunes_signed_nonce_run_that_removes_a_proof_group() {
        with_large_stack(|| {
            let mut txs = (0..=10).map(light_nonce_tx).collect::<Vec<_>>();
            add_skip_nonce_attribute(txs.last_mut().unwrap());
            let start_api_root = txs[0].accounts_before[OWNER_ACCOUNT_ID].api_key_root;
            let signed_anchor_nonce = txs[10].nonce;

            let removed = Block::prune_skip_nonce_runs(&mut txs, 4, 10);

            assert_eq!(removed, 10);
            assert_eq!(txs.len(), 1);
            assert_eq!(txs[0].nonce, signed_anchor_nonce);
            assert_eq!(txs[0].api_key_before.nonce, 0);
            assert_eq!(
                txs[0].accounts_before[OWNER_ACCOUNT_ID].api_key_root,
                start_api_root
            );
            assert!(txs[0].derive_old_private_roots);
            assert!(Block::has_signed_skip_nonce(&txs[0]));
        });
    }

    #[test]
    fn leaves_run_untouched_when_proof_group_count_does_not_drop() {
        with_large_stack(|| {
            let mut txs = (0..=1).map(light_nonce_tx).collect::<Vec<_>>();
            add_skip_nonce_attribute(txs.last_mut().unwrap());

            let removed = Block::prune_skip_nonce_runs(&mut txs, 4, 10);

            assert_eq!(removed, 0);
            assert_eq!(txs.len(), 2);
            assert_eq!(txs[1].api_key_before.nonce, 1);
            assert!(!txs[1].derive_old_private_roots);
        });
    }

    #[test]
    fn rejects_owner_state_change_other_than_api_key_root() {
        with_large_stack(|| {
            let mut txs = (0..=10).map(light_nonce_tx).collect::<Vec<_>>();
            txs.last_mut().unwrap().accounts_before[OWNER_ACCOUNT_ID].cancel_all_time = 1;
            add_skip_nonce_attribute(txs.last_mut().unwrap());

            let removed = Block::prune_skip_nonce_runs(&mut txs, 4, 10);

            assert_eq!(removed, 0);
            assert_eq!(txs.len(), 11);
            assert!(!txs[10].derive_old_private_roots);
        });
    }

    #[test]
    fn rejects_run_without_signed_skip_nonce_attribute() {
        with_large_stack(|| {
            let mut txs = (0..=10).map(light_nonce_tx).collect::<Vec<_>>();

            let removed = Block::prune_skip_nonce_runs(&mut txs, 4, 10);

            assert_eq!(removed, 0);
            assert_eq!(txs.len(), 11);
        });
    }

    #[test]
    fn rewritten_witness_derives_only_private_old_roots() {
        with_large_stack(|| {
            let mut tx = light_nonce_tx(10);
            tx.derive_old_private_roots = true;
            let mut builder = Builder::new(CIRCUIT_CONFIG);
            let target = TxTarget::new(&mut builder);
            let mut witness = PartialWitness::<F>::new();

            witness.set_tx_target(&target, &tx).unwrap();

            assert!(
                target
                    .old_account_tree_root
                    .elements
                    .iter()
                    .all(|&element| witness.try_get_target(element).is_none())
            );
            assert!(
                target
                    .old_validium_root
                    .elements
                    .iter()
                    .all(|&element| witness.try_get_target(element).is_none())
            );
            assert!(
                target
                    .old_state_root
                    .elements
                    .iter()
                    .all(|&element| witness.try_get_target(element).is_none())
            );
            assert!(
                target
                    .old_account_pub_data_tree_root
                    .elements
                    .iter()
                    .all(|&element| witness.try_get_target(element).is_some())
            );
        });
    }
}
