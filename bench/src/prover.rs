// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::iop::witness::{PartitionWitness, Witness as _};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::recursion::dummy_circuit::dummy_circuit;

use crate::api::{
    CHAIN_ID, Circuits, HEAVY_TX_MODE, HEAVY_TX_PER_PROOF, LIGHT_TX_MODE, LIGHT_TX_PER_PROOF,
    ON_CHAIN_OPERATIONS_LIMIT, Proof, stage,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxPath {
    Heavy,
    Light,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkRoute {
    chunk_index: usize,
    chain_step: u64,
    path: TxPath,
}

fn chunk_routes(block: &Block<F>) -> Vec<ChunkRoute> {
    let mut heavy_step = 0;
    let mut light_step = 0;
    block
        .tx_chunks
        .iter()
        .enumerate()
        .map(|(chunk_index, txs)| {
            let is_light = txs
                .first()
                .expect("block transaction chunk must not be empty")
                .tx_circuit_type
                == TX_LIGHT;
            let (path, chain_step) = if is_light {
                let step = light_step;
                light_step += 1;
                (TxPath::Light, step)
            } else {
                let step = heavy_step;
                heavy_step += 1;
                (TxPath::Heavy, step)
            };
            ChunkRoute {
                chunk_index,
                chain_step,
                path,
            }
        })
        .collect()
}

fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
    (light, heavy)
}

/// Reference sequential implementation. The shipped binary uses
/// [`prove_block_pipelined`]; this stays as the behavioral baseline the tests
/// exercise and the fallback if pipelining ever needs to be reverted.
#[cfg_attr(not(test), allow(dead_code))]
pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proof {
    let pre_proof = stage("pre_execution proof", || {
        BlockPreExecutionCircuit::prove(
            &circuits.pre_data,
            &BlockPreExec::from_block(block),
            &circuits.pre_target,
        )
    })
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);

    let base_proofs_start = std::time::Instant::now();
    let mut heavy_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &circuits.heavy_chain_data,
        &circuits.dummy_heavy_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );
    let mut light_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &circuits.light_chain_data,
        &circuits.dummy_light_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );

    if crate::api::stage_timing_enabled() {
        eprintln!(
            "[stage] cyclic base proofs: {:.3}s",
            base_proofs_start.elapsed().as_secs_f64()
        );
    }

    let mut heavy_jump =
        JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
    let mut light_jump = heavy_jump;
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let mut tx_proof_seconds = [0f64; 2];
    let mut fold_seconds = [0f64; 2];

    for route in chunk_routes(block) {
        let txs = &block.tx_chunks[route.chunk_index];
        let is_light = route.path == TxPath::Light;
        let block_tx = BlockTx {
            created_at: block.created_at,
            state_metadata_hash,
            old_jump: if is_light { light_jump } else { heavy_jump },
            txs: txs.clone(),
        };

        let tx_proof_start = std::time::Instant::now();
        let tx_proof = if is_light {
            BlockTxCircuit::prove(
                &circuits.light_tx_data,
                &block_tx,
                &circuits.light_tx_target,
            )
        } else {
            BlockTxCircuit::prove(
                &circuits.heavy_tx_data,
                &block_tx,
                &circuits.heavy_tx_target,
            )
        }
        .unwrap_or_else(|error| {
            panic!(
                "block transaction chunk #{} proof failed: {error:?}",
                route.chunk_index
            )
        });

        tx_proof_seconds[is_light as usize] += tx_proof_start.elapsed().as_secs_f64();

        let fold_start = std::time::Instant::now();
        let tx_output = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs);
        if is_light {
            light_jump = tx_output.new_jump;
            light_chain_proof = BlockTxChainCircuit::prove(
                &circuits.light_chain_target,
                &circuits.light_chain_data,
                route.chain_step,
                &light_chain_proof,
                &circuits.dummy_light_proof,
                &tx_proof,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "light block transaction chain step #{} failed: {error:?}",
                    route.chain_step
                )
            });
        } else {
            heavy_jump = tx_output.new_jump;
            heavy_chain_proof = BlockTxChainCircuit::prove(
                &circuits.heavy_chain_target,
                &circuits.heavy_chain_data,
                route.chain_step,
                &heavy_chain_proof,
                &circuits.dummy_heavy_proof,
                &tx_proof,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "heavy block transaction chain step #{} failed: {error:?}",
                    route.chain_step
                )
            });
        }
        fold_seconds[is_light as usize] += fold_start.elapsed().as_secs_f64();
    }

    if crate::api::stage_timing_enabled() {
        eprintln!("[stage] heavy tx proofs (3): {:.3}s", tx_proof_seconds[0]);
        eprintln!("[stage] light tx proofs (49): {:.3}s", tx_proof_seconds[1]);
        eprintln!("[stage] heavy chain folds (3): {:.3}s", fold_seconds[0]);
        eprintln!("[stage] light chain folds (49): {:.3}s", fold_seconds[1]);
    }

    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    stage("final block proof", || {
        BlockCircuit::prove(
            &circuits.block_target,
            &circuits.block_data,
            block,
            &pre_proof,
            light_chain_input,
            heavy_chain_input,
        )
    })
    .expect("final block proof failed")
}

/// Number of worker threads completing chunk proofs concurrently. The light
/// fold chain consumes roughly one chunk proof per fold; four workers keep it
/// fed on the 10P+4E ranked host while leaving cores for the fold chains and
/// the overlapped block-circuit build.
const CHUNK_PROVER_WORKERS: usize = 4;
/// Upper bound on generated-but-not-yet-proved partition witnesses. Bounds
/// peak memory: each witness holds one value slot per wire per row.
const CHUNK_WITNESS_PERMITS: usize = 6;

struct ChunkJob<'a> {
    route: ChunkRoute,
    witness: PartitionWitness<'a, F>,
}

/// Returns the memory permit even if the proving worker panics, so the
/// witness-generation thread fails via a closed channel instead of blocking.
struct SendTokenOnDrop(mpsc::Sender<()>);

impl Drop for SendTokenOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Builds one recursive chain side (heavy or light) and folds that side's
/// chunk proofs in step order as they arrive from the proving workers.
/// Identical fold inputs to the sequential loop: the previous chain proof, the
/// side's dummy proof, and the step's chunk proof, in ascending step order.
#[allow(clippy::too_many_arguments)]
fn run_chain_side(
    side: &'static str,
    block: &Block<F>,
    pre_output: &BlockPreExecWitness<F>,
    tx_data: &CircuitData<F, C, D>,
    chain_data_out: mpsc::Sender<Arc<CircuitData<F, C, D>>>,
    ready_out: mpsc::Sender<()>,
    proofs: mpsc::Receiver<(u64, Proof)>,
    steps: u64,
) -> Proof {
    let (chain_target, chain_data) = stage(&format!("build {side}_chain circuit"), || {
        let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        (chain.target, chain.builder.build::<C>())
    });
    let chain_data = Arc::new(chain_data);
    // The block-circuit builder only needs the chain verifier data; unblock it
    // before the folds run.
    let _ = chain_data_out.send(chain_data.clone());

    let dummy_chain_circuit = dummy_circuit(&chain_data.common);
    let dummy_proof = cyclic_base_proof(
        &chain_data.common,
        &chain_data.verifier_only,
        &dummy_chain_circuit,
        [].into_iter().collect(),
    )
    .unwrap_or_else(|error| panic!("cannot construct {side} chain dummy proof: {error:?}"));
    let mut chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &chain_data,
        &dummy_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );

    // This side can now fold as fast as chunk proofs arrive; let the chunk
    // proving workers start competing for the CPU. Dropping the sender is what
    // lets the orchestrator detect a side that died before signaling.
    let _ = ready_out.send(());
    drop(ready_out);

    let mut pending: HashMap<u64, Proof> = HashMap::new();
    let mut fold_seconds = 0f64;
    let mut wait_seconds = 0f64;
    for step in 0..steps {
        let wait_start = std::time::Instant::now();
        let tx_proof = loop {
            if let Some(proof) = pending.remove(&step) {
                break proof;
            }
            let (arrived_step, proof) = proofs
                .recv()
                .unwrap_or_else(|_| panic!("{side} chunk proof channel closed at step {step}"));
            if arrived_step == step {
                break proof;
            }
            pending.insert(arrived_step, proof);
        };
        wait_seconds += wait_start.elapsed().as_secs_f64();

        let fold_start = std::time::Instant::now();
        chain_proof = BlockTxChainCircuit::prove(
            &chain_target,
            &chain_data,
            step,
            &chain_proof,
            &dummy_proof,
            &tx_proof,
        )
        .unwrap_or_else(|error| {
            panic!("{side} block transaction chain step #{step} failed: {error:?}")
        });
        fold_seconds += fold_start.elapsed().as_secs_f64();
    }
    if crate::api::stage_timing_enabled() {
        eprintln!(
            "[stage] {side} chain folds ({steps}): {fold_seconds:.3}s (waited {wait_seconds:.3}s)"
        );
    }
    chain_proof
}

/// Conservative variant: overlaps only circuit construction with proving and
/// keeps the original strictly sequential chunk-proof/fold loop. The heavy-tx,
/// light-tx, and pre-execution circuits build in parallel; each chain side
/// builds on its own thread; the block circuit builds in the background during
/// the chunk loop (it is only needed for the final proof). At most one
/// background build competes with one proving stream at a time, so the
/// contention risk that sank prior coarse proving-concurrency attempts on the
/// ranked host does not apply here.
pub fn prove_block_overlapped_builds(block: &Block<F>) -> Proof {
    let ((heavy_tx_target, heavy_tx_data), (light_tx_target, light_tx_data), (pre_data, pre_proof)) =
        thread::scope(|s| {
            let heavy = s.spawn(|| {
                stage("build heavy_tx circuit", || {
                    let heavy_tx = BlockTxCircuit::define(
                        CIRCUIT_CONFIG,
                        HEAVY_TX_PER_PROOF,
                        CHAIN_ID,
                        HEAVY_TX_MODE,
                    );
                    (heavy_tx.target, heavy_tx.builder.build::<C>())
                })
            });
            let light = s.spawn(|| {
                stage("build light_tx circuit", || {
                    let light_tx = BlockTxCircuit::define(
                        CIRCUIT_CONFIG,
                        LIGHT_TX_PER_PROOF,
                        CHAIN_ID,
                        LIGHT_TX_MODE,
                    );
                    (light_tx.target, light_tx.builder.build::<C>())
                })
            });
            let pre = s.spawn(|| {
                let (pre_target, pre_data) = stage("build pre_execution circuit", || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    (pre.target, pre.builder.build::<C>())
                });
                let pre_proof = stage("pre_execution proof", || {
                    BlockPreExecutionCircuit::prove(
                        &pre_data,
                        &BlockPreExec::from_block(block),
                        &pre_target,
                    )
                })
                .expect("block pre-execution proof failed");
                (pre_data, pre_proof)
            });
            (
                heavy.join().expect("heavy tx circuit build panicked"),
                light.join().expect("light tx circuit build panicked"),
                pre.join().expect("pre-execution build panicked"),
            )
        });

    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);

    struct ChainSide {
        target: circuit::block_tx_chain_constraints::BlockTxChainTarget,
        data: Arc<CircuitData<F, C, D>>,
        dummy_proof: Proof,
        base_proof: Proof,
    }

    fn build_chain_side(
        side: &str,
        block: &Block<F>,
        pre_output: &BlockPreExecWitness<F>,
        tx_data: &CircuitData<F, C, D>,
        chain_data_out: mpsc::Sender<Arc<CircuitData<F, C, D>>>,
    ) -> ChainSide {
        let (target, data) = stage(&format!("build {side}_chain circuit"), || {
            let chain =
                BlockTxChainCircuit::define(CIRCUIT_CONFIG, tx_data, ON_CHAIN_OPERATIONS_LIMIT);
            (chain.target, chain.builder.build::<C>())
        });
        let data = Arc::new(data);
        let _ = chain_data_out.send(data.clone());
        let dummy_chain_circuit = dummy_circuit(&data.common);
        let dummy_proof = cyclic_base_proof(
            &data.common,
            &data.verifier_only,
            &dummy_chain_circuit,
            [].into_iter().collect(),
        )
        .unwrap_or_else(|error| panic!("cannot construct {side} chain dummy proof: {error:?}"));
        let base_proof = BlockTxChainCircuit::cyclic_base_proof(
            &data,
            &dummy_chain_circuit,
            block.block_number,
            block.created_at,
            pre_output.new_state_root,
            pre_output.new_validium_root,
            block.old_account_delta_tree_root,
        );
        ChainSide {
            target,
            data,
            dummy_proof,
            base_proof,
        }
    }

    let (heavy_chain_data_tx, heavy_chain_data_rx) = mpsc::channel::<Arc<CircuitData<F, C, D>>>();
    let (light_chain_data_tx, light_chain_data_rx) = mpsc::channel::<Arc<CircuitData<F, C, D>>>();

    thread::scope(|s| {
        let heavy_side_handle = {
            let pre_output = &pre_output;
            let heavy_tx_data = &heavy_tx_data;
            s.spawn(move || {
                build_chain_side(
                    "heavy",
                    block,
                    pre_output,
                    heavy_tx_data,
                    heavy_chain_data_tx,
                )
            })
        };
        let light_side_handle = {
            let pre_output = &pre_output;
            let light_tx_data = &light_tx_data;
            s.spawn(move || {
                build_chain_side(
                    "light",
                    block,
                    pre_output,
                    light_tx_data,
                    light_chain_data_tx,
                )
            })
        };
        let pre_data = &pre_data;
        let block_builder = s.spawn(move || {
            let heavy_chain_data = heavy_chain_data_rx
                .recv()
                .expect("heavy chain circuit build did not deliver verifier data");
            let light_chain_data = light_chain_data_rx
                .recv()
                .expect("light chain circuit build did not deliver verifier data");
            stage("build block circuit", || {
                let block_circuit = BlockCircuit::define(
                    CIRCUIT_CONFIG,
                    pre_data,
                    &light_chain_data,
                    &heavy_chain_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                (block_circuit.target, block_circuit.builder.build::<C>())
            })
        });

        let heavy = heavy_side_handle
            .join()
            .expect("heavy chain side build panicked");
        let light = light_side_handle
            .join()
            .expect("light chain side build panicked");

        let mut heavy_chain_proof = heavy.base_proof;
        let mut light_chain_proof = light.base_proof;
        let mut heavy_jump =
            JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
        let mut light_jump = heavy_jump;
        let state_metadata_hash = pre_output.new_state_metadata.hash();

        let mut tx_proof_seconds = [0f64; 2];
        let mut fold_seconds = [0f64; 2];
        for route in chunk_routes(block) {
            let txs = &block.tx_chunks[route.chunk_index];
            let is_light = route.path == TxPath::Light;
            let block_tx = BlockTx {
                created_at: block.created_at,
                state_metadata_hash,
                old_jump: if is_light { light_jump } else { heavy_jump },
                txs: txs.clone(),
            };

            let tx_proof_start = std::time::Instant::now();
            let tx_proof = if is_light {
                BlockTxCircuit::prove(&light_tx_data, &block_tx, &light_tx_target)
            } else {
                BlockTxCircuit::prove(&heavy_tx_data, &block_tx, &heavy_tx_target)
            }
            .unwrap_or_else(|error| {
                panic!(
                    "block transaction chunk #{} proof failed: {error:?}",
                    route.chunk_index
                )
            });
            tx_proof_seconds[is_light as usize] += tx_proof_start.elapsed().as_secs_f64();

            let fold_start = std::time::Instant::now();
            let tx_output = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs);
            if is_light {
                light_jump = tx_output.new_jump;
                light_chain_proof = BlockTxChainCircuit::prove(
                    &light.target,
                    &light.data,
                    route.chain_step,
                    &light_chain_proof,
                    &light.dummy_proof,
                    &tx_proof,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "light block transaction chain step #{} failed: {error:?}",
                        route.chain_step
                    )
                });
            } else {
                heavy_jump = tx_output.new_jump;
                heavy_chain_proof = BlockTxChainCircuit::prove(
                    &heavy.target,
                    &heavy.data,
                    route.chain_step,
                    &heavy_chain_proof,
                    &heavy.dummy_proof,
                    &tx_proof,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "heavy block transaction chain step #{} failed: {error:?}",
                        route.chain_step
                    )
                });
            }
            fold_seconds[is_light as usize] += fold_start.elapsed().as_secs_f64();
        }

        if crate::api::stage_timing_enabled() {
            eprintln!("[stage] heavy tx proofs: {:.3}s", tx_proof_seconds[0]);
            eprintln!("[stage] light tx proofs: {:.3}s", tx_proof_seconds[1]);
            eprintln!("[stage] heavy chain folds: {:.3}s", fold_seconds[0]);
            eprintln!("[stage] light chain folds: {:.3}s", fold_seconds[1]);
        }

        let (block_target, block_data) =
            block_builder.join().expect("block circuit build panicked");
        let (light_chain_input, heavy_chain_input) =
            final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
        stage("final block proof", || {
            BlockCircuit::prove(
                &block_target,
                &block_data,
                block,
                &pre_proof,
                light_chain_input,
                heavy_chain_input,
            )
        })
        .expect("final block proof failed")
    })
}

/// Pipelined equivalent of building [`Circuits`] and calling [`prove_block`].
///
/// The proof system sees bit-identical inputs in an identical order per
/// circuit; only host-side scheduling changes:
/// - the heavy-tx, light-tx, and pre-execution circuits build in parallel, and
///   the pre-execution proof completes during the tx-circuit builds;
/// - each chain side builds its chain circuit, dummy circuit, and base proof
///   on its own thread, then folds its chunk proofs in step order;
/// - the block circuit builds concurrently with chunk proving, since it is
///   only needed for the final proof;
/// - chunk witness generation stays sequential (each chunk's `old_jump` is the
///   previous chunk's `new_jump` on the same path), while the expensive
///   polynomial commitments for up to [`CHUNK_PROVER_WORKERS`] chunks run
///   concurrently via the two-phase split in [`BlockTxCircuit`].
pub fn prove_block_pipelined(block: &Block<F>) -> Proof {
    let ((heavy_tx_target, heavy_tx_data), (light_tx_target, light_tx_data), (pre_data, pre_proof)) =
        thread::scope(|s| {
            let heavy = s.spawn(|| {
                stage("build heavy_tx circuit", || {
                    let heavy_tx = BlockTxCircuit::define(
                        CIRCUIT_CONFIG,
                        HEAVY_TX_PER_PROOF,
                        CHAIN_ID,
                        HEAVY_TX_MODE,
                    );
                    (heavy_tx.target, heavy_tx.builder.build::<C>())
                })
            });
            let light = s.spawn(|| {
                stage("build light_tx circuit", || {
                    let light_tx = BlockTxCircuit::define(
                        CIRCUIT_CONFIG,
                        LIGHT_TX_PER_PROOF,
                        CHAIN_ID,
                        LIGHT_TX_MODE,
                    );
                    (light_tx.target, light_tx.builder.build::<C>())
                })
            });
            let pre = s.spawn(|| {
                let (pre_target, pre_data) = stage("build pre_execution circuit", || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    (pre.target, pre.builder.build::<C>())
                });
                let pre_proof = stage("pre_execution proof", || {
                    BlockPreExecutionCircuit::prove(
                        &pre_data,
                        &BlockPreExec::from_block(block),
                        &pre_target,
                    )
                })
                .expect("block pre-execution proof failed");
                (pre_data, pre_proof)
            });
            (
                heavy.join().expect("heavy tx circuit build panicked"),
                light.join().expect("light tx circuit build panicked"),
                pre.join().expect("pre-execution build panicked"),
            )
        });

    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    let routes = chunk_routes(block);
    let heavy_steps = routes
        .iter()
        .filter(|route| route.path == TxPath::Heavy)
        .count() as u64;
    let light_steps = routes.len() as u64 - heavy_steps;

    let (job_tx, job_rx) = mpsc::channel::<ChunkJob>();
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (token_tx, token_rx) = mpsc::channel::<()>();
    for _ in 0..CHUNK_WITNESS_PERMITS {
        token_tx.send(()).expect("token channel cannot be closed");
    }
    // The light fold chain is the serial spine of the whole block proof. Cap
    // its inbound queue so proving workers block (and stop competing for the
    // CPU) once the spine has enough proofs in hand; an over-supplied spine
    // measured ~5x slower per fold. The heavy side holds every proof without
    // blocking: its three folds are far off the critical path, and a worker
    // stuck sending a heavy proof would be a worker not proving light chunks.
    let (heavy_proof_tx, heavy_proof_rx) = mpsc::sync_channel::<(u64, Proof)>(16);
    let (light_proof_tx, light_proof_rx) = mpsc::sync_channel::<(u64, Proof)>(2);
    let (heavy_chain_data_tx, heavy_chain_data_rx) = mpsc::channel::<Arc<CircuitData<F, C, D>>>();
    let (light_chain_data_tx, light_chain_data_rx) = mpsc::channel::<Arc<CircuitData<F, C, D>>>();
    let (fold_ready_tx, fold_ready_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let heavy_side = {
            let ready_out = fold_ready_tx.clone();
            let pre_output = &pre_output;
            let heavy_tx_data = &heavy_tx_data;
            s.spawn(move || {
                run_chain_side(
                    "heavy",
                    block,
                    pre_output,
                    heavy_tx_data,
                    heavy_chain_data_tx,
                    ready_out,
                    heavy_proof_rx,
                    heavy_steps,
                )
            })
        };
        let light_side = {
            let ready_out = fold_ready_tx.clone();
            let pre_output = &pre_output;
            let light_tx_data = &light_tx_data;
            s.spawn(move || {
                run_chain_side(
                    "light",
                    block,
                    pre_output,
                    light_tx_data,
                    light_chain_data_tx,
                    ready_out,
                    light_proof_rx,
                    light_steps,
                )
            })
        };
        let pre_data = &pre_data;
        let block_builder = s.spawn(move || {
            let heavy_chain_data = heavy_chain_data_rx
                .recv()
                .expect("heavy chain circuit build did not deliver verifier data");
            let light_chain_data = light_chain_data_rx
                .recv()
                .expect("light chain circuit build did not deliver verifier data");
            stage("build block circuit", || {
                let block_circuit = BlockCircuit::define(
                    CIRCUIT_CONFIG,
                    pre_data,
                    &light_chain_data,
                    &heavy_chain_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                (block_circuit.target, block_circuit.builder.build::<C>())
            })
        });

        // Wait until both fold sides can consume proofs before letting the
        // chunk workers saturate the CPU; unthrottled workers stretch the
        // ~1.5 s chain-circuit builds to ~30 s and stall both fold spines.
        drop(fold_ready_tx);
        for _ in 0..2 {
            fold_ready_rx
                .recv()
                .expect("a chain side exited before its folds were ready");
        }

        for _ in 0..CHUNK_PROVER_WORKERS {
            let job_rx = Arc::clone(&job_rx);
            let token_tx = token_tx.clone();
            let heavy_proof_tx = heavy_proof_tx.clone();
            let light_proof_tx = light_proof_tx.clone();
            let heavy_tx_data = &heavy_tx_data;
            let light_tx_data = &light_tx_data;
            s.spawn(move || {
                loop {
                    let job = {
                        let receiver = job_rx.lock().expect("chunk job receiver poisoned");
                        receiver.recv()
                    };
                    let Ok(job) = job else { break };
                    let _token = SendTokenOnDrop(token_tx.clone());
                    let is_light = job.route.path == TxPath::Light;
                    let tx_data = if is_light {
                        light_tx_data
                    } else {
                        heavy_tx_data
                    };
                    let proof = BlockTxCircuit::prove_from_partition_witness(tx_data, job.witness)
                        .unwrap_or_else(|error| {
                            panic!(
                                "block transaction chunk #{} proof failed: {error:?}",
                                job.route.chunk_index
                            )
                        });
                    let proof_tx = if is_light {
                        &light_proof_tx
                    } else {
                        &heavy_proof_tx
                    };
                    proof_tx
                        .send((job.route.chain_step, proof))
                        .expect("chain fold thread exited early");
                }
            });
        }
        // Fold threads must observe channel closure once the workers finish.
        drop(token_tx);
        drop(heavy_proof_tx);
        drop(light_proof_tx);

        let witness_start = std::time::Instant::now();
        let mut heavy_jump =
            JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
        let mut light_jump = heavy_jump;
        let state_metadata_hash = pre_output.new_state_metadata.hash();
        let mut witness_seconds = 0f64;
        for route in &routes {
            token_rx
                .recv()
                .expect("every chunk proving worker exited before the last witness");
            let generation_start = std::time::Instant::now();
            let is_light = route.path == TxPath::Light;
            let block_tx = BlockTx {
                created_at: block.created_at,
                state_metadata_hash,
                old_jump: if is_light { light_jump } else { heavy_jump },
                txs: block.tx_chunks[route.chunk_index].clone(),
            };
            let (tx_data, tx_target) = if is_light {
                (&light_tx_data, &light_tx_target)
            } else {
                (&heavy_tx_data, &heavy_tx_target)
            };
            let witness = BlockTxCircuit::generate_partition_witness(tx_data, &block_tx, tx_target)
                .unwrap_or_else(|error| {
                    panic!(
                        "block transaction chunk #{} witness failed: {error:?}",
                        route.chunk_index
                    )
                });
            let public_inputs = witness.get_targets(&tx_data.prover_only.public_inputs);
            let new_jump = BlockTxWitness::from_public_inputs(&public_inputs).new_jump;
            if is_light {
                light_jump = new_jump;
            } else {
                heavy_jump = new_jump;
            }
            witness_seconds += generation_start.elapsed().as_secs_f64();
            job_tx
                .send(ChunkJob {
                    route: *route,
                    witness,
                })
                .expect("every chunk proving worker exited early");
        }
        drop(job_tx);
        if crate::api::stage_timing_enabled() {
            eprintln!(
                "[stage] chunk witness generation ({}): {witness_seconds:.3}s over {:.3}s wall",
                routes.len(),
                witness_start.elapsed().as_secs_f64()
            );
        }

        let heavy_chain_proof = heavy_side.join().expect("heavy chain side panicked");
        let light_chain_proof = light_side.join().expect("light chain side panicked");
        let (block_target, block_data) =
            block_builder.join().expect("block circuit build panicked");

        let (light_chain_input, heavy_chain_input) =
            final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
        stage("final block proof", || {
            BlockCircuit::prove(
                &block_target,
                &block_data,
                block,
                &pre_proof,
                light_chain_input,
                heavy_chain_input,
            )
        })
        .expect("final block proof failed")
    })
}

#[cfg(test)]
mod tests {
    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

    use super::*;
    use crate::api::{
        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
    };

    #[test]
    fn prove_block_returns_one_final_block_proof() {
        let prove: fn(&Block<F>, &Circuits) -> Proof = prove_block;
        let _ = prove;
    }

    #[test]
    fn every_parsed_mixed_chunk_is_routed_once() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let block = Block::<F>::from_json_with_empty_txs(
                    include_bytes!("../bench_test.json"),
                    HEAVY_TX_PER_PROOF,
                    LIGHT_TX_PER_PROOF,
                    PUBLIC_HEAVY_TX_COUNT,
                    PUBLIC_LIGHT_TX_COUNT,
                )
                .expect("public fixture must parse");
                let routes = chunk_routes(&block);

                assert_eq!(routes.len(), block.tx_chunks.len());
                assert_eq!(
                    routes
                        .iter()
                        .map(|route| route.chunk_index)
                        .collect::<Vec<_>>(),
                    (0..block.tx_chunks.len()).collect::<Vec<_>>()
                );
                for route in &routes {
                    let expected_path =
                        if block.tx_chunks[route.chunk_index][0].tx_circuit_type == TX_LIGHT {
                            TxPath::Light
                        } else {
                            TxPath::Heavy
                        };
                    assert_eq!(route.path, expected_path);
                }

                let heavy_steps = routes
                    .iter()
                    .filter(|route| route.path == TxPath::Heavy)
                    .map(|route| route.chain_step)
                    .collect::<Vec<_>>();
                let light_steps = routes
                    .iter()
                    .filter(|route| route.path == TxPath::Light)
                    .map(|route| route.chain_step)
                    .collect::<Vec<_>>();
                assert_eq!(heavy_steps, vec![0, 1, 2]);
                assert_eq!(light_steps, (0..49).collect::<Vec<_>>());
                assert!(routes.iter().all(|route| {
                    let circuit_type = block.tx_chunks[route.chunk_index][0].tx_circuit_type;
                    matches!(
                        (route.path, circuit_type),
                        (TxPath::Heavy, TX_HEAVY) | (TxPath::Light, TX_LIGHT)
                    )
                }));
            })
            .expect("orchestration test thread must start")
            .join()
            .expect("orchestration test thread must finish");
    }

    #[test]
    fn final_block_chain_inputs_are_light_then_heavy() {
        let light = "light";
        let heavy = "heavy";

        assert_eq!(final_chain_inputs(&light, &heavy), (&light, &heavy));
    }
}
