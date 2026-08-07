// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, JumpState, JumpStateTarget};
use circuit::block_tx_chain_constraints::{
    BlockTxChainCircuit, BlockTxChainTarget, cyclic_base_witness,
};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget};
#[cfg(test)]
use circuit::block_tx_constraints::Circuit as _;
use circuit::tx::Tx;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::generator::{ParallelWitnessGuard, PendingPartitionWitness};
#[cfg(test)]
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;

use crate::api::{
    Circuits, HEAVY_TX_MODE, HEAVY_TX_PER_PROOF, LIGHT_TX_MODE, LIGHT_TX_PER_PROOF,
    ON_CHAIN_OPERATIONS_LIMIT, PROVER_THREAD_STACK_BYTES, PathCircuits, Proof,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxPath {
    Heavy,
    Light,
}

const LIGHT_TX_PROOF_WINDOW: usize = 2;
// Keep the initial light proofs serial while the fixed three-chunk heavy path is active.
const LIGHT_TX_PROOF_OVERLAP_START_STEP: u64 = 3;

fn chunk_is_light(txs: &[Tx<F>]) -> bool {
    txs.first()
        .expect("block transaction chunk must not be empty")
        .tx_circuit_type
        == TX_LIGHT
}

fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
    (light, heavy)
}

/// Marks the calling thread as latency-critical to the macOS scheduler.
///
/// The 49 sequential chain folds are the whole critical path of a block
/// bundle: every serial section of a fold (witness feed, opening
/// evaluation, FRI reduce, transcript work) runs on a chain-step thread
/// while the global worker pool is saturated by transaction proving that
/// hides behind the spine anyway. At default QoS those serial sections
/// compete for cores on equal terms with hideable bulk work and are
/// eligible for efficiency-core placement; per-statement profiling of the
/// fold pipeline shows episodic multi-hundred-millisecond stalls between
/// instrumented spans under exactly this contention. `USER_INTERACTIVE`
/// asks the scheduler to keep the fold thread on a performance core and
/// schedule it ahead of default-QoS pool workers. This changes thread
/// scheduling only: no work is added, moved, or reordered, and proof
/// bytes are untouched. On non-macOS targets this is a no-op.
#[cfg(target_os = "macos")]
fn mark_spine_thread_latency_critical() {
    // `QOS_CLASS_USER_INTERACTIVE` is 0x21 in <sys/qos.h>.
    #[allow(non_camel_case_types)]
    type qos_class_t = u32;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
    }
    // Best-effort: a nonzero return leaves the thread at its previous QoS,
    // which is exactly the pre-change behavior.
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn mark_spine_thread_latency_critical() {}

/// Borrowed references to one transaction path's circuits, so the path
/// pipeline can be driven either from a fully assembled [`Circuits`] or from
/// the pipelined startup's per-path builds.
#[derive(Clone, Copy)]
struct PathPieces<'a> {
    tx_data: &'a CircuitData<F, C, D>,
    tx_target: &'a BlockTxTarget,
    chain_data: &'a CircuitData<F, C, D>,
    chain_target: &'a BlockTxChainTarget,
    dummy_proof: &'a Proof,
}

impl<'a> PathPieces<'a> {
    fn from_circuits(circuits: &'a Circuits, path: TxPath) -> Self {
        match path {
            TxPath::Light => Self {
                tx_data: &circuits.light_tx_data,
                tx_target: &circuits.light_tx_target,
                chain_data: &circuits.light_chain_data,
                chain_target: &circuits.light_chain_target,
                dummy_proof: &circuits.dummy_light_proof,
            },
            TxPath::Heavy => Self {
                tx_data: &circuits.heavy_tx_data,
                tx_target: &circuits.heavy_tx_target,
                chain_data: &circuits.heavy_chain_data,
                chain_target: &circuits.heavy_chain_target,
                dummy_proof: &circuits.dummy_heavy_proof,
            },
        }
    }

    fn from_path_circuits(path_circuits: &'a PathCircuits) -> Self {
        Self {
            tx_data: &path_circuits.tx_data,
            tx_target: &path_circuits.tx_target,
            chain_data: &path_circuits.chain_data,
            chain_target: &path_circuits.chain_target,
            dummy_proof: &path_circuits.dummy_proof,
        }
    }
}

enum ChainState<'scope> {
    Ready(Proof),
    InFlight(std::thread::ScopedJoinHandle<'scope, Proof>),
}

impl ChainState<'_> {
    fn wait(self) -> Proof {
        match self {
            ChainState::Ready(proof) => proof,
            ChainState::InFlight(handle) => handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn chain_step_proof(
    path: TxPath,
    chain_target: &BlockTxChainTarget,
    chain_data: &CircuitData<F, C, D>,
    chain_step: u64,
    previous: Option<ChainState<'_>>,
    base_proof: &Proof,
    dummy_proof: &Proof,
    tx_proof: &Proof,
) -> Proof {
    mark_spine_thread_latency_critical();
    let result = (|| {
        // Phase 1: run every generator that does not depend on the previous chain proof while
        // that proof may still be in flight. Inputs are written directly into
        // the partition's representative slots — no PartialWitness map, no
        // per-path template clone, no replay pass.
        let mut pending = PendingPartitionWitness::start_seeded(
            &chain_data.prover_only,
            &chain_data.common,
            |seeder| {
                BlockTxChainCircuit::witness_inputs_early_into(
                    chain_target,
                    chain_data,
                    chain_step,
                    dummy_proof,
                    tx_proof,
                    seeder,
                )
            },
        )?;

        // Phase 2: wait for the previous chain proof, feed it directly, and prove.
        let previous_proof = previous.map(ChainState::wait);
        pending.feed_seeded(|feeder| {
            BlockTxChainCircuit::witness_inputs_cyclic_into(
                chain_target,
                previous_proof.as_ref().unwrap_or(base_proof),
                feeder,
            )
        })?;
        BlockTxChainCircuit::prove_prepared(pending, chain_data)
    })();
    result.unwrap_or_else(|error| {
        panic!("{path:?} block transaction chain step #{chain_step} failed: {error:?}")
    })
}

fn hash_from_witness(witness: &impl Witness<F>, target: &HashOutTarget) -> HashOut<F> {
    HashOut {
        elements: target.elements.map(|element| witness.get_target(element)),
    }
}

fn jump_from_witness(witness: &impl Witness<F>, target: &JumpStateTarget) -> JumpState<F> {
    JumpState {
        last_active_tx_index: witness.get_target(target.last_active_tx_index),
        prev_new_state_root: hash_from_witness(witness, &target.prev_new_state_root),
        prev_new_delta_root: hash_from_witness(witness, &target.prev_new_delta_root),
        run_start_prev_index: witness.get_target(target.run_start_prev_index),
        run_start_old_state_root: hash_from_witness(witness, &target.run_start_old_state_root),
        run_start_old_delta_root: hash_from_witness(witness, &target.run_start_old_delta_root),
        coverage_hash: hash_from_witness(witness, &target.coverage_hash),
        claims_hash: hash_from_witness(witness, &target.claims_hash),
        tx_count: witness.get_target(target.tx_count),
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_tx_witness<'a>(
    path: TxPath,
    chunk_index: usize,
    txs: Vec<Tx<F>>,
    tx_data: &'a CircuitData<F, C, D>,
    tx_target: &BlockTxTarget,
    created_at: i64,
    state_metadata_hash: HashOut<F>,
    old_jump: JumpState<F>,
) -> (PartitionWitness<'a, F>, JumpState<F>) {
    let block_tx = BlockTx {
        created_at,
        state_metadata_hash,
        old_jump,
        txs,
    };
    // Write witness values directly into the partition's representative
    // slots (array-indexed), bypassing the PartialWitness hash map and its
    // per-target hashing for the ~10^5 inputs of every transaction chunk,
    // while maintaining the same unresolved-watch counters.
    let partition_witness = PendingPartitionWitness::start_seeded(
        &tx_data.prover_only,
        &tx_data.common,
        |seeder| BlockTxCircuit::generate_witness_into(&block_tx, tx_target, seeder),
    )
    .and_then(PendingPartitionWitness::finish)
    .unwrap_or_else(|error| {
        panic!("{path:?} block transaction chunk #{chunk_index} witness generation failed: {error:?}")
    });
    let new_jump = jump_from_witness(&partition_witness, &tx_target.new_jump);
    (partition_witness, new_jump)
}

fn prove_tx_witness(
    path: TxPath,
    chunk_index: usize,
    tx_data: &CircuitData<F, C, D>,
    partition_witness: PartitionWitness<'_, F>,
) -> Proof {
    let proof = prove_with_partition_witness::<F, C, D>(
        &tx_data.prover_only,
        &tx_data.common,
        partition_witness,
        &mut TimingTree::default(),
    )
    .unwrap_or_else(|error| {
        panic!("{path:?} block transaction chunk #{chunk_index} proof failed: {error:?}")
    });
    #[cfg(debug_assertions)]
    tx_data
        .verify(proof.clone())
        .expect("transaction proof self-check failed");
    proof
}

#[allow(clippy::too_many_arguments)]
fn prove_path(
    path: TxPath,
    chunks: Vec<(usize, Vec<Tx<F>>)>,
    pieces: PathPieces<'_>,
    block_number: u64,
    created_at: i64,
    old_account_delta_tree_root: HashOut<F>,
    pre_output: &BlockPreExecWitness<F>,
    state_metadata_hash: HashOut<F>,
) -> Proof {
    assert!(
        !chunks.is_empty(),
        "{path:?} transaction path must contain at least one chunk"
    );
    let PathPieces {
        tx_data,
        tx_target,
        chain_data,
        chain_target,
        dummy_proof,
    } = pieces;

    let base_proof = cyclic_base_witness(
        dummy_proof,
        block_number,
        created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        old_account_delta_tree_root,
    );
    let mut jump = JumpState::initial(pre_output.new_state_root, old_account_delta_tree_root);
    let mut chunks = chunks.into_iter();
    let (mut current_chunk_index, first_txs) =
        chunks.next().expect("transaction path must not be empty");
    let (mut current_witness, next_jump) = generate_tx_witness(
        path,
        current_chunk_index,
        first_txs,
        tx_data,
        tx_target,
        created_at,
        state_metadata_hash,
        jump,
    );
    jump = next_jump;


    std::thread::scope(|scope| {
        let base = &base_proof;
        let mut chain: Option<ChainState<'_>> = None;
        let mut pending_tx: Option<(u64, Proof)> = None;
        let mut in_flight = std::collections::VecDeque::new();
        let mut current_step = 0u64;

        loop {
            if let Some((chain_step, tx_proof)) = pending_tx.take() {
                // The predecessor handle moves into the chain thread, which waits for it only
                // after its tx-proof-side witness generation: the path thread never blocks here.
                let previous = chain.take();
                let handle = std::thread::Builder::new()
                    .name(format!("{path:?}-chain-step-{chain_step}"))
                    .stack_size(PROVER_THREAD_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        chain_step_proof(
                            path,
                            chain_target,
                            chain_data,
                            chain_step,
                            previous,
                            base,
                            dummy_proof,
                            &tx_proof,
                        )
                    })
                    .expect("chain step pipeline thread must start");
                chain = Some(ChainState::InFlight(handle));
            }

            let witness = current_witness;
            let proof_handle = std::thread::Builder::new()
                .name(format!("{path:?}-tx-proof-{current_step}"))
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    prove_tx_witness(path, current_chunk_index, tx_data, witness)
                })
                .expect("transaction proof pipeline thread must start");

            let next_witness = chunks.next().map(|(chunk_index, txs)| {
                let (witness, next_jump) = generate_tx_witness(
                    path,
                    chunk_index,
                    txs,
                    tx_data,
                    tx_target,
                    created_at,
                    state_metadata_hash,
                    jump,
                );
                jump = next_jump;
                (chunk_index, witness)
            });

            in_flight.push_back((current_step, proof_handle));
            let max_in_flight =
                if path == TxPath::Light && current_step >= LIGHT_TX_PROOF_OVERLAP_START_STEP {
                    LIGHT_TX_PROOF_WINDOW
                } else {
                    1
                };
            if in_flight.len() >= max_in_flight {
                let (proof_step, proof_handle) = in_flight
                    .pop_front()
                    .expect("transaction proof window must not be empty");
                let tx_proof = proof_handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                pending_tx = Some((proof_step, tx_proof));
            }
            current_step += 1;

            match next_witness {
                Some((chunk_index, witness)) => {
                    current_chunk_index = chunk_index;
                    current_witness = witness;
                }
                None => break,
            }
        }

        if let Some((chain_step, tx_proof)) = pending_tx.take() {
            let previous = chain.take();
            let handle = std::thread::Builder::new()
                .name(format!("{path:?}-chain-step-{chain_step}"))
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    chain_step_proof(
                        path,
                        chain_target,
                        chain_data,
                        chain_step,
                        previous,
                        base,
                        dummy_proof,
                        &tx_proof,
                    )
                })
                .expect("chain step pipeline thread must start");
            chain = Some(ChainState::InFlight(handle));
        }
        // Past this point the pipeline spawns no new chunk work: the drain
        // below is the strictly sequential chain tail, so its mid-size
        // commitment trees can use the mostly idle GPU exactly like the
        // pre-execution and final block phases.
        plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
        while let Some((chain_step, proof_handle)) = in_flight.pop_front() {
            let tx_proof = proof_handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            let previous = chain.take();
            chain = Some(ChainState::Ready(chain_step_proof(
                path,
                chain_target,
                chain_data,
                chain_step,
                previous,
                base,
                dummy_proof,
                &tx_proof,
            )));
        }
        let chain_proof = chain
            .map(ChainState::wait)
            .expect("transaction path must produce a chain proof");
        plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
        chain_proof
    })
}

/// Reference orchestration: strictly serial pre-execution and final-block
/// phases. The scored binary uses [`prove_block_pipelined`]; this stays as the
/// differential oracle (see `pipelined_matches_reference_orchestration`) and
/// the readable baseline of the schedule.
#[allow(dead_code)]
pub fn prove_block(mut block: Block<F>, circuits: &Circuits) -> Proof {
    // The pre-execution proof runs strictly before any other proving work, so
    // the serialized GPU stream is otherwise idle: route its mid-size column
    // trees to the GPU for just this phase.
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(&block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let mut tx_chunks = std::mem::take(&mut block.tx_chunks);
    let mut heavy_chunks: Vec<(usize, Vec<Tx<F>>)> = Vec::new();
    let mut light_chunks: Vec<(usize, Vec<Tx<F>>)> = Vec::with_capacity(tx_chunks.len());
    for (chunk_index, txs) in tx_chunks.drain(..).enumerate() {
        if chunk_is_light(&txs) {
            light_chunks.push((chunk_index, txs));
        } else {
            heavy_chunks.push((chunk_index, txs));
        }
    }
    block.tx_chunks = tx_chunks;
    block.tx_chunks.push(Vec::new());

    let (light_chain_proof, heavy_chain_proof, block_target, block_data) =
        std::thread::scope(|scope| {
            // The final block circuit depends only on already-built circuit data
            // and is not needed until the final proof, so it builds concurrently
            // with the entire transaction/chain proving pipeline.
            let block_circuit_handle = std::thread::Builder::new()
                .name("block-circuit-build".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, || circuits.build_block_circuit())
                .expect("block circuit build thread must start");
            let heavy_handle = std::thread::Builder::new()
                .name("heavy-tx-chain".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, || {
                    prove_path(
                        TxPath::Heavy,
                        heavy_chunks,
                        PathPieces::from_circuits(circuits, TxPath::Heavy),
                        block.block_number,
                        block.created_at,
                        block.old_account_delta_tree_root,
                        &pre_output,
                        state_metadata_hash,
                    )
                })
                .expect("heavy transaction chain thread must start");
            let light_chain_proof = prove_path(
                TxPath::Light,
                light_chunks,
                PathPieces::from_circuits(circuits, TxPath::Light),
                block.block_number,
                block.created_at,
                block.old_account_delta_tree_root,
                &pre_output,
                state_metadata_hash,
            );
            let heavy_chain_proof = heavy_handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            let (block_target, block_data) = block_circuit_handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            (
                light_chain_proof,
                heavy_chain_proof,
                block_target,
                block_data,
            )
        });

    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    // The final block witness runs on the serial tail with nothing else proving, so it alone
    // opts into parallel worklist rounds; tx-proof and chain witness generation run concurrently
    // with proving and stay sequential.
    let _parallel_block_witness = ParallelWitnessGuard::new();
    // For the same reason the serialized GPU stream is otherwise idle here:
    // route the final block proof's mid-size column trees to the GPU for just
    // this phase.
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
    let final_proof = BlockCircuit::prove(
        &block_target,
        &block_data,
        &block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed");
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
    final_proof
}

/// [`prove_block`] with the serial head and tail pipelined away. Proof bytes
/// are identical — the same witness values flow through the same circuits in
/// the same transcript order — only the wall-clock schedule changes:
///
/// * **Head.** The transaction paths consume only the pre-execution proof's
///   *public inputs*, and those are plain witness values: they exist the
///   moment the pre-execution partition witness is complete, before any
///   commitment or FRI work. The pre-execution circuit is built and its
///   witness generated on a dedicated thread concurrently with the two path
///   circuit builds; the public inputs are published through a channel as
///   soon as the witness finishes, and the (much longer) proving of the
///   pre-execution proof continues on that thread while both transaction
///   pipelines are already running. Today's schedule holds every path build
///   hostage to the finished pre-execution *proof*; this one only holds them
///   to its witness.
///
/// * **Tail.** The final block witness depends on three proofs, two of which
///   (pre-execution and the 3-chunk heavy chain) finish long before the
///   49-fold light chain. Those two are seeded — and their verification
///   generators run — while the light path is still folding, using the same
///   [`PendingPartitionWitness`] split the chain spine already uses. When the
///   light proof lands, only its feed, the worklist resume, and the final
///   proof itself remain on the serial tail.
///
/// The pre-execution proof is proven without the exclusive-GPU marking (it no
/// longer has the stream to itself); its wide trees still route to the GPU
/// unconditionally and its narrow trees take the GPU whenever the stream is
/// free, which during circuit builds it almost always is. Routing is
/// dispatch-only: either route hashes the identical tree.
pub fn prove_block_pipelined(mut block: Block<F>) -> Proof {
    // ---- Head: pre-execution witness first, proof off the critical path ----
    let pre_input = BlockPreExec::from_block(&block);
    let (pre_pis_sender, pre_pis_receiver) = std::sync::mpsc::channel::<Vec<F>>();
    let pre_handle = std::thread::Builder::new()
        .name("pre-execution".into())
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(move || {
            let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
            let pre_target = pre.target;
            let pre_data = pre.builder.build::<C>();
            let witness = PendingPartitionWitness::start_seeded(
                &pre_data.prover_only,
                &pre_data.common,
                |seeder| {
                    BlockPreExecutionCircuit::generate_witness_into(&pre_input, &pre_target, seeder)
                },
            )
            .and_then(PendingPartitionWitness::finish)
            .unwrap_or_else(|error| {
                panic!("block pre-execution witness generation failed: {error:?}")
            });
            // The proof's public inputs are exactly these witness values;
            // publishing them now lets both transaction paths start while this
            // thread is still proving. A dropped receiver only means the
            // orchestrator panicked, and this thread's own panic reporting is
            // not the place to surface that.
            let public_inputs = witness.get_targets(&pre_data.prover_only.public_inputs);
            let _ = pre_pis_sender.send(public_inputs);
            let pre_proof = prove_with_partition_witness::<F, C, D>(
                &pre_data.prover_only,
                &pre_data.common,
                witness,
                &mut TimingTree::default(),
            )
            .unwrap_or_else(|error| panic!("block pre-execution proof failed: {error:?}"));
            (pre_proof, pre_data)
        })
        .expect("pre-execution thread must start");

    // Both path circuit stacks build while the pre-execution witness and proof
    // run; neither side waits for the other beyond the public-input handoff.
    let (heavy_circuits, light_circuits) = rayon::join(
        || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
        || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
    );

    let pre_public_inputs = pre_pis_receiver
        .recv()
        .expect("pre-execution witness must publish public inputs");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_public_inputs);
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let mut tx_chunks = std::mem::take(&mut block.tx_chunks);
    let mut heavy_chunks: Vec<(usize, Vec<Tx<F>>)> = Vec::new();
    let mut light_chunks: Vec<(usize, Vec<Tx<F>>)> = Vec::with_capacity(tx_chunks.len());
    for (chunk_index, txs) in tx_chunks.drain(..).enumerate() {
        if chunk_is_light(&txs) {
            light_chunks.push((chunk_index, txs));
        } else {
            heavy_chunks.push((chunk_index, txs));
        }
    }
    block.tx_chunks = tx_chunks;
    block.tx_chunks.push(Vec::new());

    let light_pieces = PathPieces::from_path_circuits(&light_circuits);
    let heavy_pieces = PathPieces::from_path_circuits(&heavy_circuits);
    let block_number = block.block_number;
    let created_at = block.created_at;
    let old_account_delta_tree_root = block.old_account_delta_tree_root;

    std::thread::scope(|scope| {
        // The final block circuit needs the pre-execution circuit data, so this
        // build first joins the pre-execution thread. That join point is far
        // off the critical path: the pre-execution proof finishes early in the
        // pipeline and the build itself completes long before the light chain
        // drains.
        let block_circuit_handle = std::thread::Builder::new()
            .name("block-circuit-build".into())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn_scoped(scope, || {
                let (pre_proof, pre_data) = pre_handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                let block_circuit = BlockCircuit::define(
                    CIRCUIT_CONFIG,
                    &pre_data,
                    &light_circuits.chain_data,
                    &heavy_circuits.chain_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                let block_target = block_circuit.target;
                let block_data = block_circuit.builder.build::<C>();
                (pre_proof, block_target, block_data)
            })
            .expect("block circuit build thread must start");

        // The light path (49 folds) runs on its own thread so this one — free
        // once the 3-chunk heavy path drains — can assemble the final block
        // witness early.
        let light_handle = std::thread::Builder::new()
            .name("light-tx-chain".into())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn_scoped(scope, || {
                prove_path(
                    TxPath::Light,
                    light_chunks,
                    light_pieces,
                    block_number,
                    created_at,
                    old_account_delta_tree_root,
                    &pre_output,
                    state_metadata_hash,
                )
            })
            .expect("light transaction chain thread must start");

        let heavy_chain_proof = prove_path(
            TxPath::Heavy,
            heavy_chunks,
            heavy_pieces,
            block_number,
            created_at,
            old_account_delta_tree_root,
            &pre_output,
            state_metadata_hash,
        );

        let (pre_proof, block_target, block_data) = block_circuit_handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));

        // Seed everything the final block proof does not owe to the light
        // chain — block inputs, the pre-execution proof, the heavy chain proof
        // — and run their generators now, overlapped with the light path's
        // remaining folds. The worklist stays sequential here: the light
        // pipeline still owns the rayon pool.
        let mut pending = PendingPartitionWitness::start_seeded(
            &block_data.prover_only,
            &block_data.common,
            |seeder| {
                BlockCircuit::witness_inputs_early_into(&block_target, &block, &pre_proof, seeder)?;
                seeder.set_proof_with_pis_target(
                    &block_target.heavy_tx_chain_proof,
                    &heavy_chain_proof,
                )
            },
        )
        .unwrap_or_else(|error| panic!("final block witness seeding failed: {error:?}"));

        let light_chain_proof = light_handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));

        // Only the light-chain feed, the worklist resume, and the proof itself
        // remain on the serial tail; it alone opts into parallel worklist
        // rounds and the exclusive GPU routing, exactly like the tail of
        // [`prove_block`].
        let _parallel_block_witness = ParallelWitnessGuard::new();
        plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
        pending
            .feed_seeded(|feeder| {
                feeder.set_proof_with_pis_target(
                    &block_target.light_tx_chain_proof,
                    &light_chain_proof,
                )
            })
            .unwrap_or_else(|error| panic!("final block light-chain feed failed: {error:?}"));
        let final_proof =
            BlockCircuit::prove_prepared(pending, &block_data).expect("final block proof failed");
        plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
        final_proof
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::api::{
        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
    };

    #[test]
    fn prove_block_returns_one_final_block_proof() {
        let prove: fn(Block<F>, &Circuits) -> Proof = prove_block;
        let _ = prove;
    }

    #[test]
    fn parsed_mixed_chunks_have_expected_paths() {
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
                let paths = block
                    .tx_chunks
                    .iter()
                    .map(|txs| chunk_is_light(txs))
                    .collect::<Vec<_>>();

                assert_eq!(paths.len(), block.tx_chunks.len());
                assert_eq!(paths.iter().filter(|&&is_light| !is_light).count(), 3);
                assert_eq!(paths.iter().filter(|&&is_light| is_light).count(), 49);
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

    /// Differential oracle for the pipelined orchestration: proves the public
    /// fixture through both [`prove_block`] and [`prove_block_pipelined`] and
    /// requires bit-identical serialized proofs. The pipelined schedule moves
    /// work between threads and phases but writes the same witness values into
    /// the same circuits, so any divergence here is a bug. Run with:
    /// `cargo test --release -p bench --bin prove -- --ignored pipelined_matches --nocapture`
    #[test]
    #[ignore = "manual differential oracle; proves the full public fixture twice"]
    fn pipelined_matches_reference_orchestration() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let load_block = || {
                    Block::<F>::from_json_with_empty_txs(
                        include_bytes!("../bench_test.json"),
                        HEAVY_TX_PER_PROOF,
                        LIGHT_TX_PER_PROOF,
                        PUBLIC_HEAVY_TX_COUNT,
                        PUBLIC_LIGHT_TX_COUNT,
                    )
                    .expect("public fixture must parse")
                };
                let reference = prove_block(load_block(), &Circuits::new());
                let pipelined = prove_block_pipelined(load_block());
                // Compare field by field with compact diagnostics: a full-proof
                // assert_eq would dump megabytes of bytes on divergence.
                fn assert_bytes_match(label: &str, reference: &[u8], pipelined: &[u8]) {
                    if reference == pipelined {
                        return;
                    }
                    let diff = reference
                        .iter()
                        .zip(pipelined.iter())
                        .position(|(a, b)| a != b);
                    panic!(
                        "{label} diverges: reference {} bytes, pipelined {} bytes, \
                         first differing byte at {diff:?}",
                        reference.len(),
                        pipelined.len(),
                    );
                }
                fn field_bytes<T: serde::Serialize>(label: &str, value: &T) -> Vec<u8> {
                    bincode::serialize(value)
                        .unwrap_or_else(|error| panic!("{label} must serialize: {error:?}"))
                }
                for (label, reference_bytes, pipelined_bytes) in [
                    (
                        "public inputs",
                        field_bytes("public inputs", &reference.public_inputs),
                        field_bytes("public inputs", &pipelined.public_inputs),
                    ),
                    (
                        "wires cap",
                        field_bytes("wires cap", &reference.proof.wires_cap),
                        field_bytes("wires cap", &pipelined.proof.wires_cap),
                    ),
                    (
                        "zs/partial-products cap",
                        field_bytes("zs cap", &reference.proof.plonk_zs_partial_products_cap),
                        field_bytes("zs cap", &pipelined.proof.plonk_zs_partial_products_cap),
                    ),
                    (
                        "quotient cap",
                        field_bytes("quotient cap", &reference.proof.quotient_polys_cap),
                        field_bytes("quotient cap", &pipelined.proof.quotient_polys_cap),
                    ),
                    (
                        "openings",
                        field_bytes("openings", &reference.proof.openings),
                        field_bytes("openings", &pipelined.proof.openings),
                    ),
                    (
                        "FRI opening proof",
                        field_bytes("FRI proof", &reference.proof.opening_proof),
                        field_bytes("FRI proof", &pipelined.proof.opening_proof),
                    ),
                ] {
                    assert_bytes_match(label, &reference_bytes, &pipelined_bytes);
                }
            })
            .expect("differential oracle thread must start")
            .join()
            .expect("differential oracle thread must finish");
    }

    /// Fast head-isolating differential: the pre-execution witness through the
    /// seeded path must match the map path value-for-value (public inputs) and
    /// proof-byte-for-proof-byte. Run with:
    /// `cargo test --release -p bench --bin prove -- --ignored pre_execution_seeded --nocapture`
    #[test]
    #[ignore = "manual differential oracle; builds and proves the pre-execution circuit twice"]
    fn pre_execution_seeded_matches_map_path() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let block = Block::<F>::from_json_with_empty_txs(
                    include_bytes!("../bench_test.json"),
                    HEAVY_TX_PER_PROOF,
                    LIGHT_TX_PER_PROOF,
                    PUBLIC_HEAVY_TX_COUNT,
                    PUBLIC_LIGHT_TX_COUNT,
                )
                .expect("public fixture must parse");
                let pre_input = BlockPreExec::from_block(&block);
                let pre = BlockPreExecutionCircuit::define(circuit::types::config::CIRCUIT_CONFIG);
                let pre_target = pre.target;
                let pre_data = pre.builder.build::<C>();

                let map_inputs =
                    <BlockPreExecutionCircuit as circuit::block_pre_execution_constraints::Circuit<
                        C,
                        F,
                        D,
                    >>::generate_witness(&pre_input, &pre_target)
                    .expect("map-path witness inputs must generate");
                let map_witness = generate_partial_witness::<F, C, D>(
                    map_inputs,
                    &pre_data.prover_only,
                    &pre_data.common,
                )
                .expect("map-path witness generation must succeed");

                let seeded_witness = PendingPartitionWitness::start_seeded(
                    &pre_data.prover_only,
                    &pre_data.common,
                    |seeder| {
                        BlockPreExecutionCircuit::generate_witness_into(
                            &pre_input,
                            &pre_target,
                            seeder,
                        )
                    },
                )
                .and_then(PendingPartitionWitness::finish)
                .expect("seeded witness generation must succeed");

                assert_eq!(
                    map_witness.get_targets(&pre_data.prover_only.public_inputs),
                    seeded_witness.get_targets(&pre_data.prover_only.public_inputs),
                    "seeded pre-execution public inputs must match the map path"
                );

                let prove_pre = |witness| {
                    prove_with_partition_witness::<F, C, D>(
                        &pre_data.prover_only,
                        &pre_data.common,
                        witness,
                        &mut TimingTree::default(),
                    )
                    .expect("pre-execution proof must succeed")
                };
                let compare_witnesses =
                    |label: &str, a: &PartitionWitness<'_, F>, b: &PartitionWitness<'_, F>| {
                        assert_eq!(a.set_bitmap, b.set_bitmap, "{label}: set bitmaps diverge");
                        let diverging: Vec<usize> = a
                            .values
                            .iter()
                            .zip(b.values.iter())
                            .enumerate()
                            .filter(|(_, (x, y))| x != y)
                            .map(|(i, _)| i)
                            .collect();
                        if !diverging.is_empty() {
                            for &i in diverging.iter().take(8) {
                                println!(
                                    "  slot {i} (row {}, col {}): {} vs {}",
                                    i / a.num_wires,
                                    i % a.num_wires,
                                    a.values[i],
                                    b.values[i],
                                );
                            }
                            panic!(
                                "{label}: {} witness slots diverge; first 20 rep indices: {:?} \
                                 (of {} slots, num_wires {} degree {})",
                                diverging.len(),
                                &diverging[..diverging.len().min(20)],
                                a.values.len(),
                                a.num_wires,
                                a.degree,
                            );
                        }
                        println!("{label}: witness values and bitmaps identical");
                    };
                // Determinism control: the same map-path witness, proven twice,
                // must give identical bytes. If this fails the prover itself is
                // nondeterministic and the seeded path is not the culprit.
                let control_inputs =
                    <BlockPreExecutionCircuit as circuit::block_pre_execution_constraints::Circuit<
                        C,
                        F,
                        D,
                    >>::generate_witness(&pre_input, &pre_target)
                    .expect("control witness inputs must generate");
                let control_witness = generate_partial_witness::<F, C, D>(
                    control_inputs,
                    &pre_data.prover_only,
                    &pre_data.common,
                )
                .expect("control witness generation must succeed");
                compare_witnesses("map vs control", &map_witness, &control_witness);
                compare_witnesses("map vs seeded", &map_witness, &seeded_witness);
                let map_proof = prove_pre(map_witness);
                let control_proof = prove_pre(control_witness);
                let seeded_proof = prove_pre(seeded_witness);
                pre_data
                    .verify(map_proof.clone())
                    .expect("map-path proof must verify");
                pre_data
                    .verify(control_proof.clone())
                    .expect("control proof must verify");
                pre_data
                    .verify(seeded_proof.clone())
                    .expect("seeded proof must verify");
                println!("all three pre-execution proofs verify");

                fn compare(label: &str, reference: &Proof, candidate: &Proof) {
                    let fields: [(&str, Vec<u8>, Vec<u8>); 9] = [
                        (
                            "public inputs",
                            bincode::serialize(&reference.public_inputs).unwrap(),
                            bincode::serialize(&candidate.public_inputs).unwrap(),
                        ),
                        (
                            "wires cap",
                            bincode::serialize(&reference.proof.wires_cap).unwrap(),
                            bincode::serialize(&candidate.proof.wires_cap).unwrap(),
                        ),
                        (
                            "zs/partial-products cap",
                            bincode::serialize(&reference.proof.plonk_zs_partial_products_cap)
                                .unwrap(),
                            bincode::serialize(&candidate.proof.plonk_zs_partial_products_cap)
                                .unwrap(),
                        ),
                        (
                            "quotient cap",
                            bincode::serialize(&reference.proof.quotient_polys_cap).unwrap(),
                            bincode::serialize(&candidate.proof.quotient_polys_cap).unwrap(),
                        ),
                        (
                            "openings",
                            bincode::serialize(&reference.proof.openings).unwrap(),
                            bincode::serialize(&candidate.proof.openings).unwrap(),
                        ),
                        (
                            "FRI commit phase caps",
                            bincode::serialize(&reference.proof.opening_proof.commit_phase_merkle_caps)
                                .unwrap(),
                            bincode::serialize(&candidate.proof.opening_proof.commit_phase_merkle_caps)
                                .unwrap(),
                        ),
                        (
                            "FRI PoW witness",
                            bincode::serialize(&reference.proof.opening_proof.pow_witness).unwrap(),
                            bincode::serialize(&candidate.proof.opening_proof.pow_witness).unwrap(),
                        ),
                        (
                            "FRI final poly",
                            bincode::serialize(&reference.proof.opening_proof.final_poly).unwrap(),
                            bincode::serialize(&candidate.proof.opening_proof.final_poly).unwrap(),
                        ),
                        (
                            "FRI query round proofs",
                            bincode::serialize(&reference.proof.opening_proof.query_round_proofs)
                                .unwrap(),
                            bincode::serialize(&candidate.proof.opening_proof.query_round_proofs)
                                .unwrap(),
                        ),
                    ];
                    for (field, reference_bytes, candidate_bytes) in &fields {
                        if reference_bytes == candidate_bytes {
                            continue;
                        }
                        let diff = reference_bytes
                            .iter()
                            .zip(candidate_bytes.iter())
                            .position(|(a, b)| a != b);
                        panic!(
                            "{label}: {field} diverges ({} vs {} bytes, first differing byte at {diff:?})",
                            reference_bytes.len(),
                            candidate_bytes.len(),
                        );
                    }
                    println!("{label}: all proof fields byte-identical");
                }
                compare("control (map witness proven twice)", &map_proof, &control_proof);
                compare("seeded vs map", &map_proof, &seeded_proof);
            })
            .expect("pre-execution differential thread must start")
            .join()
            .expect("pre-execution differential thread must finish");
    }

    /// Manual timing harness for the two-phase chain-step witness split. Run with:
    /// `RAYON_NUM_THREADS=8 cargo test --release -p bench --bin prove -- --ignored chain_step`
    #[test]
    #[ignore = "manual timing harness; run explicitly with --release"]
    fn chain_step_two_phase_timing() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(chain_step_two_phase_timing_impl)
            .expect("timing harness thread must start")
            .join()
            .expect("timing harness thread must finish");
    }

    fn chain_step_two_phase_timing_impl() {
        use std::time::Instant;

        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .is_test(false)
            .try_init();

        use circuit::block_tx_chain_constraints::Circuit as _;
        use circuit::types::constants::TX_TYPE_EMPTY;
        use plonky2::field::types::{Field, PrimeField64};

        use crate::api::{LIGHT_TX_MODE, PathCircuits};

        let build_start = Instant::now();
        let circuits = PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE);
        println!("light path circuits built in {:?}", build_start.elapsed());

        let block = Block::<F>::from_json_with_empty_txs(
            include_bytes!("../bench_test.json"),
            HEAVY_TX_PER_PROOF,
            LIGHT_TX_PER_PROOF,
            PUBLIC_HEAVY_TX_COUNT,
            PUBLIC_LIGHT_TX_COUNT,
        )
        .expect("public fixture must parse");

        // An all-empty (padding) chunk carries no state transition, so its embedded roots and
        // metadata hash are the only values the tx and chain constraints must agree on.
        // Chain-step cost is independent of tx contents: the chain circuit is fixed-size.
        let mut empty_tx = block
            .tx_chunks
            .iter()
            .flatten()
            .find(|tx| tx.tx_type == TX_TYPE_EMPTY)
            .expect("fixture must contain an empty padding tx")
            .clone();
        empty_tx.tx_circuit_type = TX_LIGHT;
        empty_tx.tx_index = F::NEG_ONE.to_canonical_u64();

        let new_state_root = empty_tx.old_state_root;
        let old_delta_root = empty_tx.old_account_delta_tree_root;
        // The post-pre-execution metadata replayed natively: pre-execution only refreshes the
        // timestamps of the enabled recalculations.
        let mut new_state_metadata = block.state_metadata.clone();
        if block.calculate_funding {
            new_state_metadata.last_funding_round_timestamp = block.created_at;
        }
        if block.calculate_oracle_prices {
            new_state_metadata.last_oracle_price_timestamp = block.created_at;
        }
        if block.calculate_premium {
            new_state_metadata.last_premium_timestamp = block.created_at;
        }
        let state_metadata_hash = new_state_metadata.hash();
        let jump = JumpState::initial(new_state_root, old_delta_root);

        let light_chunk = vec![empty_tx; LIGHT_TX_PER_PROOF];
        let (witness, _) = generate_tx_witness(
            TxPath::Light,
            0,
            light_chunk,
            &circuits.tx_data,
            &circuits.tx_target,
            block.created_at,
            state_metadata_hash,
            jump,
        );
        let tx_prove_start = Instant::now();
        let mut tx_timing = TimingTree::new("tx-chunk-prove", log::Level::Debug);
        let tx_proof = plonky2::plonk::prover::prove_with_partition_witness::<F, C, D>(
            &circuits.tx_data.prover_only,
            &circuits.tx_data.common,
            witness,
            &mut tx_timing,
        )
        .expect("tx proof failed");
        println!("tx chunk prove total {:?}", tx_prove_start.elapsed());
        tx_timing.print();

        let base_proof = cyclic_base_witness(
            &circuits.dummy_proof,
            block.block_number,
            block.created_at,
            new_state_root,
            new_state_root,
            old_delta_root,
        );

        let mut previous: Option<Proof> = None;
        for chain_step in 0..3u64 {
            let cyclic_proof = previous.as_ref().unwrap_or(&base_proof);

            let single_shot_start = Instant::now();
            let inputs = BlockTxChainCircuit::generate_witness(
                &circuits.chain_target,
                &circuits.chain_data,
                chain_step,
                cyclic_proof,
                &circuits.dummy_proof,
                &tx_proof,
            )
            .expect("single-shot witness inputs failed");
            let single_shot = generate_partial_witness::<F, C, D>(
                inputs,
                &circuits.chain_data.prover_only,
                &circuits.chain_data.common,
            )
            .expect("single-shot witness generation failed");
            let single_shot_elapsed = single_shot_start.elapsed();
            drop(single_shot);

            let phase1_start = Instant::now();
            let early_inputs = BlockTxChainCircuit::witness_inputs_early(
                &circuits.chain_target,
                &circuits.chain_data,
                chain_step,
                &circuits.dummy_proof,
                &tx_proof,
            )
            .expect("early witness inputs failed");
            let mut pending = PendingPartitionWitness::start(
                early_inputs,
                &circuits.chain_data.prover_only,
                &circuits.chain_data.common,
            )
            .expect("early witness generation failed");
            let phase1_elapsed = phase1_start.elapsed();

            let phase2_start = Instant::now();
            pending
                .feed(
                    BlockTxChainCircuit::witness_inputs_cyclic(
                        &circuits.chain_target,
                        cyclic_proof,
                    )
                    .expect("cyclic witness inputs failed"),
                )
                .expect("cyclic witness generation failed");
            let witness = pending
                .finish()
                .expect("chain step witness must be complete");
            let phase2_elapsed = phase2_start.elapsed();

            let prove_start = Instant::now();
            let mut timing = TimingTree::new("chain-step-prove", log::Level::Debug);
            let proof = prove_with_partition_witness::<F, C, D>(
                &circuits.chain_data.prover_only,
                &circuits.chain_data.common,
                witness,
                &mut timing,
            )
            .expect("chain step proof failed");
            let prove_elapsed = prove_start.elapsed();
            timing.print();

            println!(
                "chain step {chain_step}: single-shot witness {single_shot_elapsed:?}, \
                 phase1 {phase1_elapsed:?}, phase2 {phase2_elapsed:?}, prove {prove_elapsed:?}",
            );
            previous = Some(proof);
        }

        circuits
            .chain_data
            .verify(previous.expect("chain must produce proofs"))
            .expect("final chain step proof must verify");
    }
}
