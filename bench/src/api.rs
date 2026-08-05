// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;

pub type Proof = ProofWithPublicInputs<F, C, D>;

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;

pub struct Circuits {
    pub heavy_tx_target: BlockTxTarget,
    pub heavy_tx_data: CircuitData<F, C, D>,
    pub light_tx_target: BlockTxTarget,
    pub light_tx_data: CircuitData<F, C, D>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    pub heavy_chain_data: CircuitData<F, C, D>,
    pub light_chain_target: BlockTxChainTarget,
    pub light_chain_data: CircuitData<F, C, D>,
    pub block_target: BlockTarget,
    pub block_data: CircuitData<F, C, D>,
    pub dummy_heavy_chain_circuit: CircuitData<F, C, D>,
    pub dummy_light_chain_circuit: CircuitData<F, C, D>,
}

impl Circuits {
    pub fn new() -> Self {
        let (heavy, light, pre) = std::thread::scope(|scope| {
            let heavy = scope.spawn(|| {
                let circuit = BlockTxCircuit::define(
                    CIRCUIT_CONFIG,
                    HEAVY_TX_PER_PROOF,
                    CHAIN_ID,
                    HEAVY_TX_MODE,
                );
                (circuit.target, circuit.builder.build::<C>())
            });
            let light = scope.spawn(|| {
                let circuit = BlockTxCircuit::define(
                    CIRCUIT_CONFIG,
                    LIGHT_TX_PER_PROOF,
                    CHAIN_ID,
                    LIGHT_TX_MODE,
                );
                (circuit.target, circuit.builder.build::<C>())
            });
            let pre = scope.spawn(|| {
                let circuit = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (circuit.target, circuit.builder.build::<C>())
            });
            (
                heavy
                    .join()
                    .expect("heavy transaction circuit build failed"),
                light
                    .join()
                    .expect("light transaction circuit build failed"),
                pre.join().expect("pre-execution circuit build failed"),
            )
        });
        let (heavy_tx_target, heavy_tx_data) = heavy;
        let (light_tx_target, light_tx_data) = light;
        let (pre_target, pre_data) = pre;

        let (heavy_chain, light_chain) = std::thread::scope(|scope| {
            let heavy = scope.spawn(|| {
                let circuit = BlockTxChainCircuit::define(
                    CIRCUIT_CONFIG,
                    &heavy_tx_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                (circuit.target, circuit.builder.build::<C>())
            });
            let light = scope.spawn(|| {
                let circuit = BlockTxChainCircuit::define(
                    CIRCUIT_CONFIG,
                    &light_tx_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                (circuit.target, circuit.builder.build::<C>())
            });
            (
                heavy.join().expect("heavy chain circuit build failed"),
                light.join().expect("light chain circuit build failed"),
            )
        });
        let (heavy_chain_target, heavy_chain_data) = heavy_chain;
        let (light_chain_target, light_chain_data) = light_chain;

        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &pre_data,
            &light_chain_data,
            &heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        let block_target = block.target;
        let block_data = block.builder.build::<C>();

        let (dummy_heavy_chain_circuit, dummy_light_chain_circuit) = std::thread::scope(|scope| {
            let heavy = scope.spawn(|| dummy_circuit(&heavy_chain_data.common));
            let light = scope.spawn(|| dummy_circuit(&light_chain_data.common));
            (
                heavy.join().expect("heavy dummy circuit build failed"),
                light.join().expect("light dummy circuit build failed"),
            )
        });

        Self {
            heavy_tx_target,
            heavy_tx_data,
            light_tx_target,
            light_tx_data,
            pre_target,
            pre_data,
            heavy_chain_target,
            heavy_chain_data,
            light_chain_target,
            light_chain_data,
            block_target,
            block_data,
            dummy_heavy_chain_circuit,
            dummy_light_chain_circuit,
        }
    }
}

#[cfg(test)]
mod tests {
    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

    use super::*;

    #[test]
    fn production_mixed_circuit_parameters_are_fixed() {
        assert_eq!(CHAIN_ID, 304);
        assert_eq!(HEAVY_TX_PER_PROOF, 4);
        assert_eq!(HEAVY_TX_MODE, TX_HEAVY);
        assert_eq!(LIGHT_TX_PER_PROOF, 10);
        assert_eq!(LIGHT_TX_MODE, TX_LIGHT);
        assert_eq!(ON_CHAIN_OPERATIONS_LIMIT, 1);
    }
}
