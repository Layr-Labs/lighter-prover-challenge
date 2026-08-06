// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

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

/// Env-gated stage timing for local profiling. The ranked sandbox clears the
/// environment, so this is inert there by construction.
pub struct StageTimer {
    enabled: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl StageTimer {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            enabled: std::env::var_os("LIGHTER_STAGE_TIMING").is_some(),
            start: now,
            last: now,
        }
    }

    pub fn mark(&mut self, label: &str) {
        if self.enabled {
            let now = std::time::Instant::now();
            eprintln!(
                "[stage] {label}: {:+.3}s (t={:.3}s)",
                (now - self.last).as_secs_f64(),
                (now - self.start).as_secs_f64()
            );
            self.last = now;
        }
    }
}

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;
pub const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The leaf (BlockTx) circuits. Together with the native jump plan these are
/// everything leaf proving needs, so they are built eagerly — they gate the
/// start of the dominant phase of the run.
pub struct TxCircuits {
    pub heavy_tx_target: BlockTxTarget,
    pub heavy_tx_data: CircuitData<F, C, D>,
    pub light_tx_target: BlockTxTarget,
    pub light_tx_data: CircuitData<F, C, D>,
}

/// What the pre-execution worker and the chain lanes need; built in the
/// background while the first leaves are already proving. The embedded dummy
/// chain proofs ride along (load-cheap; the lanes use the base proof as the
/// dummy-slot witness, so these are spares).
pub struct PreChainCircuits {
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    pub heavy_chain_data: CircuitData<F, C, D>,
    pub light_chain_target: BlockTxChainTarget,
    pub light_chain_data: CircuitData<F, C, D>,
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

/// The final block circuit; built last in the background — it is needed only
/// after both chain lanes have folded every leaf.
pub struct BlockCircuits {
    pub block_target: BlockTarget,
    pub block_data: CircuitData<F, C, D>,
}

/// A failed background build stores the panic message so waiters re-panic
/// with context instead of hanging on a cell that will never be set.
type StageResult<T> = Result<T, String>;

struct BackgroundStages {
    pre_chains: OnceLock<StageResult<PreChainCircuits>>,
    block: OnceLock<StageResult<BlockCircuits>>,
}

/// Circuit handle with the leaf circuits ready immediately and the rest
/// arriving from a background builder thread. Stage getters block until their
/// stage is delivered; consumers naturally need them strictly later than the
/// leaf circuits (pre proof / base proofs / chain folds are at least one leaf
/// prove away, the block circuit only after both chains finish).
pub struct Circuits {
    pub tx: Arc<TxCircuits>,
    stages: Arc<BackgroundStages>,
}

fn build_tx_circuit(tx_per_proof: usize, tx_mode: u8) -> (BlockTxTarget, CircuitData<F, C, D>) {
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
    (tx.target, tx.builder.build::<C>())
}

struct ChainPath {
    chain_target: BlockTxChainTarget,
    chain_data: CircuitData<F, C, D>,
    dummy_proof: Proof,
}

impl ChainPath {
    fn new(tx_data: &CircuitData<F, C, D>, tx_mode: u8) -> Self {
        let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, tx_data, ON_CHAIN_OPERATIONS_LIMIT);
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
            chain_target,
            chain_data,
            dummy_proof,
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Blocks on a background stage. On a stored failure the original panic was
/// already printed by the builder thread's panic hook; re-panic here so the
/// process still dies with the stage context.
fn wait_stage<'a, T>(cell: &'a OnceLock<StageResult<T>>, stage: &str) -> &'a T {
    match cell.wait() {
        Ok(value) => value,
        Err(message) => panic!("background {stage} circuit build failed: {message}"),
    }
}

/// Runs on the dedicated builder thread: pre-execution + chain circuits first
/// (they unblock the pre worker and the chain lanes), then the block circuit.
/// Every build is caught so a panic poisons the affected stages instead of
/// leaving waiters hanging.
fn build_background_stages(tx: &TxCircuits, stages: &BackgroundStages, mut timer: StageTimer) {
    let pre_chains = catch_unwind(AssertUnwindSafe(|| {
        let ((pre_target, pre_data), (heavy, light)) = rayon::join(
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, pre.builder.build::<C>())
            },
            || {
                rayon::join(
                    || ChainPath::new(&tx.heavy_tx_data, HEAVY_TX_MODE),
                    || ChainPath::new(&tx.light_tx_data, LIGHT_TX_MODE),
                )
            },
        );
        PreChainCircuits {
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: heavy.chain_data,
            light_chain_target: light.chain_target,
            light_chain_data: light.chain_data,
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }));
    timer.mark("pre+chain circuits ready (bg)");
    match pre_chains {
        Ok(circuits) => {
            // Only this thread ever sets the cells; set cannot fail.
            let _ = stages.pre_chains.set(Ok(circuits));
        }
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            let _ = stages.pre_chains.set(Err(message.clone()));
            let _ = stages.block.set(Err(message));
            return;
        }
    }
    let Some(Ok(pre_chains)) = stages.pre_chains.get() else {
        unreachable!("pre+chain stage was stored above");
    };

    let block = catch_unwind(AssertUnwindSafe(|| {
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &pre_chains.pre_data,
            &pre_chains.light_chain_data,
            &pre_chains.heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        timer.mark("block circuit defined (bg)");
        let block_target = block.target;
        let block_data = block.builder.build::<C>();
        BlockCircuits {
            block_target,
            block_data,
        }
    }));
    timer.mark("block circuit ready (bg)");
    let _ = stages
        .block
        .set(block.map_err(|payload| panic_message(payload.as_ref())));
}

impl Circuits {
    /// Builds the two leaf tx circuits (blocking — they gate leaf proving),
    /// then hands the pre-execution, chain, and block circuit builds to a
    /// background thread so they overlap with the leaf proving phase. The
    /// background builds share the global rayon pool with the leaf proves.
    pub fn new() -> Self {
        // Background marks share this origin (≈ Circuits::new entry), so
        // their t= values line up with the caller's "tx circuits ready".
        let bg_timer = StageTimer::new();

        let (heavy, light) = rayon::join(
            || build_tx_circuit(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
            || build_tx_circuit(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
        );
        let tx = Arc::new(TxCircuits {
            heavy_tx_target: heavy.0,
            heavy_tx_data: heavy.1,
            light_tx_target: light.0,
            light_tx_data: light.1,
        });

        let stages = Arc::new(BackgroundStages {
            pre_chains: OnceLock::new(),
            block: OnceLock::new(),
        });
        let builder_tx = Arc::clone(&tx);
        let builder_stages = Arc::clone(&stages);
        std::thread::Builder::new()
            .name("circuit-builder".into())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(move || build_background_stages(&builder_tx, &builder_stages, bg_timer))
            .expect("cannot start background circuit builder");

        Self { tx, stages }
    }

    /// Pre-execution + chain circuits; blocks until the background build
    /// delivers them.
    pub fn pre_chains(&self) -> &PreChainCircuits {
        wait_stage(&self.stages.pre_chains, "pre-execution/chain")
    }

    /// The final block circuit; blocks until the background build delivers it.
    pub fn block(&self) -> &BlockCircuits {
        wait_stage(&self.stages.block, "block")
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

    #[test]
    fn failed_background_stage_repanics_at_every_waiter() {
        let cell: OnceLock<StageResult<()>> = OnceLock::new();
        cell.set(Err("boom".to_string())).unwrap();
        for _ in 0..2 {
            let result = std::panic::catch_unwind(|| wait_stage(&cell, "test"));
            let payload = result.expect_err("wait_stage must re-panic on a failed stage");
            assert!(panic_message(payload.as_ref()).contains("boom"));
        }
    }
}
