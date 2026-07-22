// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;
use serde::{Deserialize, Serialize};

pub type Proof = ProofWithPublicInputs<F, C, D>;

#[derive(Deserialize, Serialize)]
pub struct Proofs {
    pub pre: Proof,
    pub chain: Proof,
}

pub struct Circuits {
    pub tx_target: BlockTxTarget,
    pub tx_data: CircuitData<F, C, D>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub chain_target: BlockTxChainTarget,
    pub chain_data: CircuitData<F, C, D>,
    pub chain_witness_size: usize,
    pub dummy_data: CircuitData<F, C, D>,
    pub dummy_proof: Proof,
}

impl Circuits {
    pub fn new(tx_per_proof: usize, chain_id: u32) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, chain_id);
        let tx_target = tx.target;
        let tx_data = tx.builder.build::<C>();

        let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
        let pre_target = pre.target;
        let pre_data = pre.builder.build::<C>();

        let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, tx_per_proof, 1);
        let chain_target = chain.target;
        let chain_witness_size = chain.block_tx_witness_size;
        let chain_data = chain.builder.build::<C>();
        let dummy_data = dummy_circuit(&chain_data.common);
        let dummy_proof = cyclic_base_proof(
            &chain_data.common,
            &chain_data.verifier_only,
            &dummy_data,
            [].into_iter().collect(),
        )
        .expect("cannot construct cyclic base proof");

        Self {
            tx_target,
            tx_data,
            pre_target,
            pre_data,
            chain_target,
            chain_data,
            chain_witness_size,
            dummy_data,
            dummy_proof,
        }
    }
}
