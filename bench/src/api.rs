// Redraw marker 64
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
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;

pub type Proof = ProofWithPublicInputs<F, C, D>;

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;
pub const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

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
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

pub(crate) struct PathCircuits {
    pub(crate) tx_target: BlockTxTarget,
    pub(crate) tx_data: CircuitData<F, C, D>,
    pub(crate) chain_target: BlockTxChainTarget,
    pub(crate) chain_data: CircuitData<F, C, D>,
    pub(crate) dummy_proof: Proof,
}

impl PathCircuits {
    pub(crate) fn new(tx_per_proof: usize, tx_mode: u8) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
        let tx_target = tx.target;
        let tx_data = tx.builder.build::<C>();

        let chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_target = chain.target;
        let chain_data = chain.builder.build::<C>();

        let proof_bytes: &[u8] = match tx_mode {
            TX_HEAVY => include_bytes!("../dummy-heavy-chain-proof.bin"),
            TX_LIGHT => include_bytes!("../dummy-light-chain-proof.bin"),
            _ => panic!("unsupported block transaction mode {tx_mode}"),
        };
        let dummy_proof =
            bincode::deserialize(proof_bytes).expect("embedded chain dummy proof is invalid");

        Self {
            tx_target,
            tx_data,
            chain_target,
            chain_data,
            dummy_proof,
        }
    }
}

impl Circuits {
    pub fn new() -> Self {
        let ((pre_target, pre_data), (heavy, light)) = rayon::join(
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, pre.builder.build::<C>())
            },
            || {
                rayon::join(
                    || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
                    || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
                )
            },
        );
        Self {
            heavy_tx_target: heavy.tx_target,
            heavy_tx_data: heavy.tx_data,
            light_tx_target: light.tx_target,
            light_tx_data: light.tx_data,
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: heavy.chain_data,
            light_chain_target: light.chain_target,
            light_chain_data: light.chain_data,
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }

    /// Releases the extended (LDE) constants/sigmas commitment of every circuit
    /// whose proving has already finished when the final block proof starts.
    ///
    /// `ProverOnlyCircuitData::constants_sigmas_commitment` holds a rate-`2^3`
    /// low-degree extension of that circuit's preprocessed columns, built once
    /// at circuit-build time and otherwise kept alive for the whole process. It
    /// is read only by proofs *of that circuit* — the quotient evaluation's
    /// `fill_lde_batch` and the FRI query openings — so once the pre-execution
    /// proof and both transaction chains have produced their proofs, the
    /// pre-execution, transaction and chain extensions are unreachable: the
    /// final block proof reads only `block_data`, those three finished proofs
    /// and the block itself.
    ///
    /// Those five extensions are `2 * 2^19 * 88 + 3 * 2^17 * 86` field elements
    /// = 1.01 GB, and on this host they are resident in CPU-visible Metal
    /// shared buffers whose release returns the pages to the OS immediately.
    /// The final block proof is the process's peak-RSS moment — it stacks its
    /// own `2^21`-row wires, Z and quotient extensions (2.89 GB) on top of
    /// every retained extension — so releasing these first takes 1.01 GB
    /// straight off the high-water mark.
    ///
    /// Nothing else is released here. Generators, representative maps and
    /// witness buffers are CPU-heap objects, and this binary runs jemalloc with
    /// `dirty_decay_ms:-1,muzzy_decay_ms:-1`: freeing them cannot lower RSS,
    /// while their recursive drop is not free.
    ///
    /// Value-exact: no quantity is computed differently, only storage that no
    /// subsequent read can reach is returned early.
    pub fn release_finished_circuit_extensions(&mut self) {
        for data in [
            &mut self.pre_data,
            &mut self.light_tx_data,
            &mut self.heavy_tx_data,
            &mut self.light_chain_data,
            &mut self.heavy_chain_data,
        ] {
            data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
        }
    }

    /// Builds the final block circuit, which depends on the pre-execution and
    /// both chain circuits but is only needed for the final proof. Callers run
    /// this concurrently with transaction/chain proving.
    pub fn build_block_circuit(&self) -> (BlockTarget, CircuitData<F, C, D>) {
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &self.pre_data,
            &self.light_chain_data,
            &self.heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        (block.target, block.builder.build::<C>())
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

    #[test]
    fn embedded_chain_dummy_proofs_deserialize() {
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-heavy-chain-proof.bin"))
            .expect("embedded heavy dummy proof is invalid");
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-light-chain-proof.bin"))
            .expect("embedded light dummy proof is invalid");
    }
}

#[cfg(test)]
mod build_timing {
    use std::time::Instant;

    use super::*;

    /// Manual timing harness for the startup circuit builds, which run once per
    /// worker spawn inside the ranked timed window (five spawns per run). Run:
    /// `cargo test --release -p bench --bin prove -- --ignored build_phase_timing --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn build_phase_timing() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let t = Instant::now();
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                let t_pre_define = t.elapsed();
                let t = Instant::now();
                let pre_data = pre.builder.build::<C>();
                let t_pre_build = t.elapsed();
                drop(pre_data);

                let t = Instant::now();
                let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, LIGHT_TX_PER_PROOF, CHAIN_ID, LIGHT_TX_MODE);
                let t_tx_define = t.elapsed();
                let t = Instant::now();
                let tx_data = tx.builder.build::<C>();
                let t_tx_build = t.elapsed();

                let t = Instant::now();
                let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
                let t_chain_define = t.elapsed();
                let t = Instant::now();
                let chain_data = chain.builder.build::<C>();
                let t_chain_build = t.elapsed();
                drop(chain_data);

                println!("pre:   define {t_pre_define:?} build {t_pre_build:?}");
                println!("tx:    define {t_tx_define:?} build {t_tx_build:?}");
                println!("chain: define {t_chain_define:?} build {t_chain_build:?}");
            })
            .expect("spawn")
            .join()
            .expect("join");
    }
}
