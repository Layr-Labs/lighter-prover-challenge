// Redraw marker 119
// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::BTreeMap;

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::circuit_data::{CircuitData, GeneratorWatchIndex, ProverOnlyCircuitData};
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
    /// The heavy transaction and chain circuits are the only two whose proving
    /// is over long before the pipeline joins — the heavy path has three chunks
    /// against the light path's forty-nine — so their preprocessed extensions
    /// are dead for the great majority of the process lifetime. They sit behind
    /// an `RwLock` purely so that death can be acted on: every reader (the heavy
    /// path for its whole run, the final block circuit's construction for the
    /// duration of `define`) holds a shared guard, and
    /// [`Circuits::release_heavy_circuit_extensions`] takes the exclusive guard
    /// once both are gone. Shared guards never block one another, so no reader
    /// is serialized against another and no work is added; the lock is a proof
    /// obligation discharged by the type system, not a scheduling change.
    pub heavy_tx_data: std::sync::RwLock<CircuitData<F, C, D>>,
    pub light_tx_target: BlockTxTarget,
    pub light_tx_data: CircuitData<F, C, D>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    /// See [`Circuits::heavy_tx_data`].
    pub heavy_chain_data: std::sync::RwLock<CircuitData<F, C, D>>,
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
            heavy_tx_data: std::sync::RwLock::new(heavy.tx_data),
            light_tx_target: light.tx_target,
            light_tx_data: light.tx_data,
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: std::sync::RwLock::new(heavy.chain_data),
            light_chain_target: light.chain_target,
            light_chain_data: light.chain_data,
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }

    /// Releases the full prover-only payload of every circuit whose proving has
    /// already finished when the final block proof starts.
    ///
    /// The tip already returns the rate-`2^3` constants/sigmas LDE commitment
    /// (Metal shared buffers, ~1.01 GB across five circuits). The rest of
    /// [`ProverOnlyCircuitData`] — sigma polynomials, subgroup table,
    /// generators/watch index, representative map, FFT root table, lookup
    /// placement, and the optional quotient-domain cache — is equally dead
    /// after that circuit's last proof: the final block proof reads only
    /// `block_data`, the three finished proofs and the block (`BlockCircuit::define`
    /// → `handle_proofs` uses `verifier_only` and `common` only).
    ///
    /// With five concurrent workers and `dirty_decay_ms:0`, those flat CPU-heap
    /// tails stay in the per-worker resident set and contend with the other
    /// four workers through the light-phase and final-block peak. Emptying them
    /// at the same release points returns the pages earlier without changing
    /// any computed value.
    ///
    /// The heavy pair is normally already empty by the time this runs — see
    /// [`Self::release_heavy_circuit_extensions`]. Re-assigning empty values is
    /// idempotent, so this stays the single backstop covering every circuit.
    pub fn release_finished_circuit_extensions(&mut self) {
        for data in [
            &mut self.pre_data,
            &mut self.light_tx_data,
            &mut self.light_chain_data,
        ] {
            release_prover_only_storage(&mut data.prover_only);
        }
        for lock in [&mut self.heavy_tx_data, &mut self.heavy_chain_data] {
            release_prover_only_storage(
                &mut lock
                    .get_mut()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .prover_only,
            );
        }
    }

    /// Releases the heavy transaction and chain circuits' full prover-only
    /// payload as soon as the heavy path has produced its chain proof.
    ///
    /// Same contract as the tip's LDE-only early release: the exclusive
    /// `RwLock` guard proves no reader remains. Extends that release from the
    /// Metal LDE commitment alone to the flat CPU-heap prover-only tail.
    pub fn release_heavy_circuit_extensions(&self) {
        for lock in [&self.heavy_tx_data, &self.heavy_chain_data] {
            release_prover_only_storage(
                &mut lock
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .prover_only,
            );
        }
    }

    /// Builds the final block circuit, which depends on the pre-execution and
    /// both chain circuits but is only needed for the final proof. Callers run
    /// this concurrently with transaction/chain proving.
    pub fn build_block_circuit(&self) -> (BlockTarget, CircuitData<F, C, D>) {
        // `define` reads only `common` and `verifier_only` of its three inputs
        // (`handle_proofs` calls `constant_verifier_data` and `verify_proof`),
        // so the shared guard is needed only for the construction itself and is
        // dropped before the (much longer) `build` below.
        let heavy_chain_data = self
            .heavy_chain_data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &self.pre_data,
            &self.light_chain_data,
            &heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        drop(heavy_chain_data);
        (block.target, block.builder.build::<C>())
    }
}

/// Empties every prover-only field that is only read by proofs of its own
/// circuit. Called from the tip's existing early-release sites once that
/// circuit's last proof (and any concurrent `define` that held a shared guard)
/// has finished. Value-exact: assigns empty storage; no arithmetic changes.
pub fn release_prover_only_storage(data: &mut ProverOnlyCircuitData<F, C, D>) {
    data.generators = Vec::new();
    data.generator_indices_by_watches = GeneratorWatchIndex::from_map(BTreeMap::new());
    data.generator_watch_counts = Vec::new();
    data.constants_sigmas_commitment = PolynomialBatch::default();
    data.sigmas = Vec::new();
    data.subgroup = Vec::new();
    data.public_inputs = Vec::new();
    data.representative_map = Vec::new();
    data.fft_root_table = None;
    data.circuit_digest = Default::default();
    data.lookup_rows = Vec::new();
    data.lut_to_lookups = Vec::new();
    data.constants_sigmas_quotient_cache = None;
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
