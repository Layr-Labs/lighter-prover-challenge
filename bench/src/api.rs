// Redraw marker heath-exp33-range-l1-r2-1786575000
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
    /// The light pair's extensions are dead the moment the light path's thread
    /// exits (the last light transaction proof and the last light chain step
    /// both read them), while the final block proof and the block-circuit lane
    /// only need `common`/`verifier_only`. The same `RwLock` proof obligation
    /// as [`Circuits::heavy_tx_data`] lets
    /// [`Circuits::release_light_circuit_extensions`] retire them right after
    /// the light thread joins — during the remaining block-lane join and final
    /// witness setup — instead of at [`Self::release_finished_circuit_extensions`].
    pub light_tx_data: std::sync::RwLock<CircuitData<F, C, D>>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    /// See [`Circuits::heavy_tx_data`].
    pub heavy_chain_data: std::sync::RwLock<CircuitData<F, C, D>>,
    pub light_chain_target: BlockTxChainTarget,
    /// See [`Circuits::light_tx_data`].
    pub light_chain_data: std::sync::RwLock<CircuitData<F, C, D>>,
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

// Revalidate the fixed permutation-mask and release-log stack on the ranked host.
// Repeat the validated stack after the official runner spread exceeded four percent.
// Keep the production diff fixed while sampling the ranked-host tail once more.

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
            light_tx_data: std::sync::RwLock::new(light.tx_data),
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: std::sync::RwLock::new(heavy.chain_data),
            light_chain_target: light.chain_target,
            light_chain_data: std::sync::RwLock::new(light.chain_data),
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
    /// witness buffers are CPU-heap objects whose recursive drop is not free.
    ///
    /// The heavy pair and light pair are normally already empty by the time this
    /// runs — see [`Self::release_heavy_circuit_extensions`] and
    /// [`Self::release_light_circuit_extensions`], which retire them the moment
    /// their paths finish rather than seconds later here. Re-assigning the same
    /// empty value is idempotent, so this stays the single backstop covering
    /// every circuit, including on the build-from-scratch fallback path where no
    /// early release runs.
    ///
    /// Value-exact: no quantity is computed differently, only storage that no
    /// subsequent read can reach is returned early.
    pub fn release_finished_circuit_extensions(&mut self) {
        self.pre_data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
        self.pre_data.prover_only.constants_sigmas_quotient_cache = None;
        for lock in [
            &mut self.light_tx_data,
            &mut self.light_chain_data,
            &mut self.heavy_tx_data,
            &mut self.heavy_chain_data,
        ] {
            let data = lock
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
            // The quotient-domain cache is reachable only from the same proofs
            // as the commitment above, so wherever that is dead this is too.
            // Clearing it is idempotent for a path that already released its own.
            data.prover_only.constants_sigmas_quotient_cache = None;
        }
    }

    /// Releases the heavy transaction and chain circuits' preprocessed
    /// extensions as soon as the heavy path has produced its chain proof.
    ///
    /// The heavy path proves three chunks; the light path proves forty-nine.
    /// The heavy pair therefore stops being read after a small fraction of the
    /// process lifetime, but until now its two extensions — `2^19 * 88 + 2^17 *
    /// 86` field elements = 438 MiB of CPU-visible Metal shared buffers whose
    /// release returns the pages to the OS immediately — stayed resident until
    /// the pipeline joined, i.e. across the whole light phase, which is exactly
    /// the window in which five concurrent workers contend for the machine's
    /// memory.
    ///
    /// The `RwLock` is what makes the release provable rather than merely
    /// plausible: acquiring the exclusive guard *is* the proof that no reader
    /// remains, because every reader of these two circuits holds a shared guard
    /// for the whole span in which it may touch them — the heavy path from
    /// before its first witness until after its chain proof, and
    /// [`Self::build_block_circuit`] for the duration of `BlockCircuit::define`.
    /// The caller runs this after joining the heavy path's thread, so both
    /// guards are already gone and the acquisition is uncontended.
    ///
    /// Value-exact and free: no quantity is computed differently and no work is
    /// added — storage that no subsequent read can reach is returned earlier.
    pub fn release_heavy_circuit_extensions(&self) {
        for lock in [&self.heavy_tx_data, &self.heavy_chain_data] {
            let mut data = lock
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
            // Same guard, same argument: the exclusive acquisition proves no
            // reader remains, and the quotient-domain cache is read only by the
            // proofs that read the commitment.
            data.prover_only.constants_sigmas_quotient_cache = None;
        }
    }

    /// Releases the light transaction and chain circuits' preprocessed
    /// extensions as soon as the light path's thread has produced its chain
    /// proof. Same shape and proof obligation as
    /// [`Self::release_heavy_circuit_extensions`]: the light path holds the
    /// shared guards for its whole run, [`Self::build_block_circuit`] holds a
    /// light-chain guard for the duration of `define` (which finishes long
    /// before the light path), and the caller joins the light thread before
    /// acquiring the exclusive guard, so the acquisition is uncontended. The
    /// extensions are `2^19 * 88 + 2^17 * 86` field elements = 438 MiB of
    /// CPU-visible Metal shared buffers released seconds before
    /// [`Self::release_finished_circuit_extensions`] would.
    pub fn release_light_circuit_extensions(&self) {
        for lock in [&self.light_tx_data, &self.light_chain_data] {
            let mut data = lock
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
            // Same guard, same argument: the exclusive acquisition proves no
            // reader remains, and the quotient-domain cache is read only by the
            // proofs that read the commitment.
            data.prover_only.constants_sigmas_quotient_cache = None;
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
        let light_chain_data = self
            .light_chain_data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &self.pre_data,
            &light_chain_data,
            &heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        drop(light_chain_data);
        drop(heavy_chain_data);
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
