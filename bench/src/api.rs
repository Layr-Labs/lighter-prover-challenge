// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;

pub type Proof = ProofWithPublicInputs<F, C, D>;

/// Stage timing is opt-in via `LIGHTER_STAGE_TIMING`; the ranked harness clears
/// the environment, so official runs never pay for or emit this output.
pub fn stage_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LIGHTER_STAGE_TIMING").is_some())
}

pub fn stage<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if !stage_timing_enabled() {
        return f();
    }
    let start = std::time::Instant::now();
    let value = f();
    eprintln!("[stage] {label}: {:.3}s", start.elapsed().as_secs_f64());
    value
}

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
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

impl Circuits {
    /// Used by the sequential reference path and tests; the shipped binary
    /// builds these circuits overlapped inside `prove_block_pipelined`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new() -> Self {
        let (heavy_tx_target, heavy_tx_data) = stage("build heavy_tx circuit", || {
            let heavy_tx =
                BlockTxCircuit::define(CIRCUIT_CONFIG, HEAVY_TX_PER_PROOF, CHAIN_ID, HEAVY_TX_MODE);
            (heavy_tx.target, heavy_tx.builder.build::<C>())
        });

        let (pre_target, pre_data) = stage("build pre_execution circuit", || {
            let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
            (pre.target, pre.builder.build::<C>())
        });

        let (light_tx_target, light_tx_data) = stage("build light_tx circuit", || {
            let light_tx =
                BlockTxCircuit::define(CIRCUIT_CONFIG, LIGHT_TX_PER_PROOF, CHAIN_ID, LIGHT_TX_MODE);
            (light_tx.target, light_tx.builder.build::<C>())
        });

        let (heavy_chain_target, heavy_chain_data) = stage("build heavy_chain circuit", || {
            let heavy_chain = BlockTxChainCircuit::define(
                CIRCUIT_CONFIG,
                &heavy_tx_data,
                ON_CHAIN_OPERATIONS_LIMIT,
            );
            (heavy_chain.target, heavy_chain.builder.build::<C>())
        });

        let (light_chain_target, light_chain_data) = stage("build light_chain circuit", || {
            let light_chain = BlockTxChainCircuit::define(
                CIRCUIT_CONFIG,
                &light_tx_data,
                ON_CHAIN_OPERATIONS_LIMIT,
            );
            (light_chain.target, light_chain.builder.build::<C>())
        });

        let (block_target, block_data) = stage("build block circuit", || {
            let block = BlockCircuit::define(
                CIRCUIT_CONFIG,
                &pre_data,
                &light_chain_data,
                &heavy_chain_data,
                ON_CHAIN_OPERATIONS_LIMIT,
            );
            (block.target, block.builder.build::<C>())
        });

        let (dummy_heavy_chain_circuit, dummy_light_chain_circuit) =
            stage("build dummy chain circuits", || {
                (
                    dummy_circuit(&heavy_chain_data.common),
                    dummy_circuit(&light_chain_data.common),
                )
            });
        let (dummy_heavy_proof, dummy_light_proof) = stage("build dummy chain proofs", || {
            let dummy_heavy_proof = cyclic_base_proof(
                &heavy_chain_data.common,
                &heavy_chain_data.verifier_only,
                &dummy_heavy_chain_circuit,
                [].into_iter().collect(),
            )
            .expect("cannot construct heavy chain dummy proof");
            let dummy_light_proof = cyclic_base_proof(
                &light_chain_data.common,
                &light_chain_data.verifier_only,
                &dummy_light_chain_circuit,
                [].into_iter().collect(),
            )
            .expect("cannot construct light chain dummy proof");
            (dummy_heavy_proof, dummy_light_proof)
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
            dummy_heavy_proof,
            dummy_light_proof,
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
