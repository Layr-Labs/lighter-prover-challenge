// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

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

/// A value produced by a construction thread that was started ahead of time.
/// The first accessor call blocks until that thread finishes; later calls are
/// a plain load.
struct Deferred<T> {
    ready: OnceLock<T>,
    pending: Mutex<Option<JoinHandle<T>>>,
}

impl<T: Send + 'static> Deferred<T> {
    fn start(name: &str, build: impl FnOnce() -> T + Send + 'static) -> Self {
        let handle = std::thread::Builder::new()
            .name(name.to_owned())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(build)
            .expect("circuit construction thread must start");
        Self {
            ready: OnceLock::new(),
            pending: Mutex::new(Some(handle)),
        }
    }

    fn get(&self) -> &T {
        self.ready.get_or_init(|| {
            self.pending
                .lock()
                .expect("circuit construction handle must not be poisoned")
                .take()
                .expect("circuit construction handle is taken exactly once")
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    }
}

pub struct PreExecutionCircuit {
    pub target: BlockPreExecutionTarget,
    pub data: CircuitData<F, C, D>,
}

pub struct PathCircuits {
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

struct TxPathCircuits {
    heavy: PathCircuits,
    light: PathCircuits,
}

pub struct Circuits {
    pre: Deferred<PreExecutionCircuit>,
    paths: Deferred<TxPathCircuits>,
}

impl Circuits {
    /// Starts circuit construction on background threads and returns without
    /// waiting for it. Each accessor blocks only on the circuits it needs, so
    /// fixture parsing and the pre-execution proof overlap with the
    /// transaction and chain circuit construction they do not depend on.
    pub fn new() -> Self {
        Self {
            pre: Deferred::start("pre-execution-circuit", || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                PreExecutionCircuit {
                    target: pre.target,
                    data: pre.builder.build::<C>(),
                }
            }),
            paths: Deferred::start("transaction-path-circuits", || {
                let (heavy, light) = rayon::join(
                    || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
                    || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
                );
                TxPathCircuits { heavy, light }
            }),
        }
    }

    pub fn pre_target(&self) -> &BlockPreExecutionTarget {
        &self.pre.get().target
    }

    pub fn pre_data(&self) -> &CircuitData<F, C, D> {
        &self.pre.get().data
    }

    pub fn heavy(&self) -> &PathCircuits {
        &self.paths.get().heavy
    }

    pub fn light(&self) -> &PathCircuits {
        &self.paths.get().light
    }

    /// Builds the final block circuit, which depends on the pre-execution and
    /// both chain circuits but is only needed for the final proof. Callers run
    /// this concurrently with transaction/chain proving.
    pub fn build_block_circuit(&self) -> (BlockTarget, CircuitData<F, C, D>) {
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            self.pre_data(),
            &self.light().chain_data,
            &self.heavy().chain_data,
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
