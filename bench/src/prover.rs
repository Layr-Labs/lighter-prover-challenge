// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx::{BlockTx, JumpState, JumpStateTarget};
use circuit::block_tx_chain_constraints::{
    cyclic_base_witness, BlockTxChainCircuit, BlockTxChainTarget,
};
#[cfg(test)]
use circuit::block_tx_constraints::Circuit as _;
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
#[cfg(test)]
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::generator::{ParallelWitnessGuard, PendingPartitionWitness};
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;

use crate::api::{Circuits, Proof, PROVER_THREAD_STACK_BYTES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxPath {
    Heavy,
    Light,
}

#[cfg(feature = "diagnostic_profile")]
fn profile_path_context(path: TxPath, stage: &str) -> &'static str {
    match (path, stage) {
        (TxPath::Heavy, "witness") => "heavy_tx_witness",
        (TxPath::Light, "witness") => "light_tx_witness",
        (TxPath::Heavy, "proof") => "heavy_tx_proof",
        (TxPath::Light, "proof") => "light_tx_proof",
        (TxPath::Heavy, "chain") => "heavy_chain",
        (TxPath::Light, "chain") => "light_chain",
        _ => "unknown_path_stage",
    }
}

const LIGHT_TX_PROOF_WINDOW: usize = 4;
// Keep the initial light proofs serial while the fixed three-chunk heavy path is active.
const LIGHT_TX_PROOF_OVERLAP_START_STEP: u64 = 3;
const REUSABLE_CHAIN_EXECUTOR_ENV: &str = "LIGHTER_REUSABLE_CHAIN_EXECUTOR";
const REUSABLE_CHAIN_EXECUTOR_DIAGNOSTIC_ENV: &str = "LIGHTER_REUSABLE_CHAIN_EXECUTOR_DIAGNOSTIC";
// Two workers are the minimum that retains the two-phase chain schedule: while one
// worker proves step N, the other can seed step N + 1 before waiting for N's proof.
const REUSABLE_CHAIN_EXECUTOR_WORKERS: usize = 2;
// Keep queued ownership bounded without throttling the existing four-proof light
// transaction window in the usual case. Each worker has its own queue, and adjacent
// steps are deliberately routed to different workers.
const REUSABLE_CHAIN_EXECUTOR_QUEUE_DEPTH: usize = LIGHT_TX_PROOF_WINDOW;

fn reusable_chain_executor_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // The ranked worker intentionally starts with an empty environment, so a
    // candidate that is enabled only by an opt-in variable is indistinguishable
    // from the legacy control to the trusted verifier. Make the bounded
    // executor the production default; `=0` remains an explicit local control
    // for differential testing.
    *ENABLED.get_or_init(|| {
        !std::env::var_os(REUSABLE_CHAIN_EXECUTOR_ENV).is_some_and(|value| value == "0")
    })
}

fn reusable_chain_executor_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(REUSABLE_CHAIN_EXECUTOR_DIAGNOSTIC_ENV).is_some_and(|value| value == "1")
    })
}

fn chunk_is_light(txs: &[Arc<Tx<F>>]) -> bool {
    txs.first()
        .expect("block transaction chunk must not be empty")
        .tx_circuit_type
        == TX_LIGHT
}

fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
    (light, heavy)
}

/// Whether the calling transaction path may claim the process-global exclusive
/// GPU phase for its chain tail.
///
/// `set_exclusive_gpu_phase` lowers the CPU/GPU Merkle routing cutoff and makes
/// the 2^17-leaf narrow commitment trees (the chain steps' Z/partial-product and
/// quotient trees) bypass the GPU occupancy check entirely. Its documented
/// contract is that no other proof runs concurrently while it is enabled, because
/// Metal command buffers execute FIFO on one queue: a fold's ~8 ms tree enqueued
/// behind a pipelined 2^19-leaf chunk tree waits hundreds of milliseconds instead
/// of ~15 ms on the CPU.
///
/// The tail-drain condition each path can test locally — "this path spawns no
/// further chunk work" — is *not* that contract. The heavy path has three chunks
/// and the light path forty-nine, so the heavy path reaches its drain while the
/// light pipeline is at full saturation. Claiming the exclusive phase there
/// disables occupancy-conditional routing process-wide for the light pipeline and
/// simultaneously force-routes this path's own fold trees behind the light
/// pipeline's chunk trees — it hurts both sides. The claim is legitimate only for
/// the path that is the last one still proving, which this counter identifies.
///
/// Routing is a scheduling heuristic: either outcome hashes the identical tree,
/// so a stale read here is benign and no proof byte depends on the answer.
fn claims_exclusive_gpu_phase(active_paths: &AtomicUsize) -> bool {
    active_paths.load(Ordering::Acquire) == 1
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

enum ChainState<'scope> {
    Ready(Proof),
    InFlight(std::thread::ScopedJoinHandle<'scope, Proof>),
    Executor(std::sync::mpsc::Receiver<std::thread::Result<Proof>>),
}

impl ChainState<'_> {
    fn wait(self) -> Proof {
        #[cfg(feature = "diagnostic_profile")]
        let _wait = plonky2::util::profile::span("wait", "chain_predecessor_join");
        match self {
            ChainState::Ready(proof) => proof,
            ChainState::InFlight(handle) => handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            ChainState::Executor(receiver) => receiver
                .recv()
                .expect("reusable chain executor dropped a predecessor result")
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        }
    }
}

struct OrderedExecutorJob<Input, Output> {
    step: u64,
    input: Input,
    previous: Option<std::sync::mpsc::Receiver<std::thread::Result<Output>>>,
    result: std::sync::mpsc::SyncSender<std::thread::Result<Output>>,
}

/// A bounded two-lane executor for an ordered dependency chain.
///
/// Adjacent steps always use different workers. A worker executes the step's
/// dependency-independent prefix and then receives the preceding result inside
/// the supplied job function. This retains the existing early-seed overlap while
/// replacing one OS-thread lifecycle per submitted step with two reusable scoped
/// workers. Panics are caught only for transport across the one-shot dependency
/// channel and resumed unchanged at the chain tail.
struct BoundedOrderedExecutor<'scope, Input, Output> {
    senders: Vec<std::sync::mpsc::SyncSender<OrderedExecutorJob<Input, Output>>>,
    workers: Vec<std::thread::ScopedJoinHandle<'scope, ()>>,
    tail: Option<std::sync::mpsc::Receiver<std::thread::Result<Output>>>,
    submitted: u64,
}

impl<'scope, Input, Output> BoundedOrderedExecutor<'scope, Input, Output>
where
    Input: Send + 'scope,
    Output: Send + 'scope,
{
    fn new<'env, Worker>(
        scope: &'scope std::thread::Scope<'scope, 'env>,
        thread_prefix: &str,
        stack_size: usize,
        worker: Worker,
    ) -> Self
    where
        Worker: Fn(u64, Input, Option<std::sync::mpsc::Receiver<std::thread::Result<Output>>>) -> Output
            + Send
            + Sync
            + 'scope,
    {
        let worker = Arc::new(worker);
        let mut senders = Vec::with_capacity(REUSABLE_CHAIN_EXECUTOR_WORKERS);
        let mut workers = Vec::with_capacity(REUSABLE_CHAIN_EXECUTOR_WORKERS);
        for worker_index in 0..REUSABLE_CHAIN_EXECUTOR_WORKERS {
            let (sender, receiver): (
                std::sync::mpsc::SyncSender<OrderedExecutorJob<Input, Output>>,
                std::sync::mpsc::Receiver<OrderedExecutorJob<Input, Output>>,
            ) = std::sync::mpsc::sync_channel(REUSABLE_CHAIN_EXECUTOR_QUEUE_DEPTH);
            senders.push(sender);
            let worker = Arc::clone(&worker);
            let handle = std::thread::Builder::new()
                .name(format!("{thread_prefix}-{worker_index}"))
                .stack_size(stack_size)
                .spawn_scoped(scope, move || {
                    while let Ok(job) = receiver.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            worker(job.step, job.input, job.previous)
                        }));
                        if job.result.send(result).is_err() {
                            break;
                        }
                    }
                })
                .expect("reusable chain executor worker must start");
            workers.push(handle);
        }
        Self {
            senders,
            workers,
            tail: None,
            submitted: 0,
        }
    }

    fn submit(&mut self, step: u64, input: Input) {
        assert_eq!(
            step, self.submitted,
            "reusable chain executor steps must be submitted in order"
        );
        let (result, receiver) = std::sync::mpsc::sync_channel(1);
        let job = OrderedExecutorJob {
            step,
            input,
            previous: self.tail.take(),
            result,
        };
        let worker_index = step as usize % self.senders.len();
        self.senders[worker_index]
            .send(job)
            .expect("reusable chain executor worker stopped before submission");
        self.tail = Some(receiver);
        self.submitted += 1;
    }

    fn submitted(&self) -> u64 {
        self.submitted
    }

    fn finish(mut self) -> Output {
        let tail = self
            .tail
            .take()
            .expect("reusable chain executor must receive at least one step");
        // Closing both job queues lets the workers retire after draining all
        // submitted work. This must happen before joining them.
        self.senders.clear();
        let result = tail.recv();
        let mut worker_panic = None;
        for worker in self.workers.drain(..) {
            if let Err(panic) = worker.join() {
                if worker_panic.is_none() {
                    worker_panic = Some(panic);
                }
            }
        }
        match result {
            Ok(Ok(output)) => {
                if let Some(panic) = worker_panic {
                    std::panic::resume_unwind(panic);
                }
                output
            }
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => match worker_panic {
                Some(panic) => std::panic::resume_unwind(panic),
                None => panic!("reusable chain executor dropped its final result"),
            },
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
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context = plonky2::util::profile::enter_context(
        profile_path_context(path, "chain"),
        chain_step,
        &[("chain_step", chain_step), ("path", path as u64)],
    );
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "chain_step");
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
    txs: Vec<Arc<Tx<F>>>,
    tx_data: &'a CircuitData<F, C, D>,
    tx_target: &BlockTxTarget,
    created_at: i64,
    state_metadata_hash: HashOut<F>,
    old_jump: JumpState<F>,
) -> (PartitionWitness<'a, F>, JumpState<F>) {
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context = plonky2::util::profile::enter_context(
        profile_path_context(path, "witness"),
        chunk_index as u64,
        &[("chunk_index", chunk_index as u64), ("path", path as u64)],
    );
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "generate_tx_witness");
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
    let partition_witness =
        PendingPartitionWitness::start_seeded(&tx_data.prover_only, &tx_data.common, |seeder| {
            BlockTxCircuit::generate_witness_into(&block_tx, tx_target, seeder)
        })
        .and_then(PendingPartitionWitness::finish)
        .unwrap_or_else(|error| {
            panic!(
            "{path:?} block transaction chunk #{chunk_index} witness generation failed: {error:?}"
        )
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
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context = plonky2::util::profile::enter_context(
        profile_path_context(path, "proof"),
        chunk_index as u64,
        &[("chunk_index", chunk_index as u64), ("path", path as u64)],
    );
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "prove_tx_witness");
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
    chunks: Vec<(usize, Vec<Arc<Tx<F>>>)>,
    circuits: &Circuits,
    block_number: u64,
    created_at: i64,
    old_account_delta_tree_root: HashOut<F>,
    pre_output: &BlockPreExecWitness<F>,
    state_metadata_hash: HashOut<F>,
    active_paths: &AtomicUsize,
) -> Proof {
    assert!(
        !chunks.is_empty(),
        "{path:?} transaction path must contain at least one chunk"
    );
    let chain_step_count = chunks.len();
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context = plonky2::util::profile::enter_context(
        match path {
            TxPath::Heavy => "heavy_path",
            TxPath::Light => "light_path",
        },
        0,
        &[("chunks", chunks.len() as u64), ("path", path as u64)],
    );
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "prove_path");
    // The heavy pair's shared guards are held for exactly as long as this path
    // may read them — from here to the `return`, which is after its chain proof
    // exists — so the exclusive acquisition in
    // `Circuits::release_heavy_circuit_extensions` is a proof that the heavy
    // path is finished with them. Shared guards never block one another, so
    // this neither serializes the two paths nor delays the concurrent block
    // circuit construction, which takes its own shared guard.
    let heavy_tx_guard;
    let heavy_chain_guard;
    let light_tx_guard;
    let light_chain_guard;
    let (tx_data, tx_target, chain_data, chain_target, dummy_proof) = match path {
        TxPath::Light => {
            light_tx_guard = circuits
                .light_tx_data
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            light_chain_guard = circuits
                .light_chain_data
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                &*light_tx_guard,
                &circuits.light_tx_target,
                &*light_chain_guard,
                &circuits.light_chain_target,
                &circuits.dummy_light_proof,
            )
        }
        TxPath::Heavy => {
            heavy_tx_guard = circuits
                .heavy_tx_data
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            heavy_chain_guard = circuits
                .heavy_chain_data
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                &*heavy_tx_guard,
                &circuits.heavy_tx_target,
                &*heavy_chain_guard,
                &circuits.heavy_chain_target,
                &circuits.dummy_heavy_proof,
            )
        }
    };

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

    let reusable_chain_executor = reusable_chain_executor_enabled();
    let (chain_proof, chain_os_threads_created) = if reusable_chain_executor {
        std::thread::scope(|scope| {
            let base = &base_proof;
            let mut executor = BoundedOrderedExecutor::new(
                scope,
                &format!("{path:?}-chain-worker"),
                PROVER_THREAD_STACK_BYTES,
                move |chain_step, tx_proof, previous| {
                    chain_step_proof(
                        path,
                        chain_target,
                        chain_data,
                        chain_step,
                        previous.map(ChainState::Executor),
                        base,
                        dummy_proof,
                        &tx_proof,
                    )
                },
            );
            let mut pending_tx: Option<(u64, Proof)> = None;
            let mut in_flight = std::collections::VecDeque::new();
            let mut current_step = 0u64;

            loop {
                if let Some((chain_step, tx_proof)) = pending_tx.take() {
                    executor.submit(chain_step, tx_proof);
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
                #[cfg(feature = "diagnostic_profile")]
                plonky2::util::profile::counter(
                    "scheduler",
                    "tx_in_flight",
                    in_flight.len() as u64,
                );
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
                    #[cfg(feature = "diagnostic_profile")]
                    let _join_wait = plonky2::util::profile::span("wait", "tx_proof_window_join");
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
                executor.submit(chain_step, tx_proof);
            }
            // The same exclusive-tail condition applies to the reusable workers:
            // every remaining job belongs to this path and no new chunk work is
            // spawned below. The sibling-path counter remains the authority for
            // the process-global GPU phase.
            let exclusive_drain = claims_exclusive_gpu_phase(active_paths);
            #[cfg(feature = "diagnostic_profile")]
            {
                plonky2::util::profile::counter(
                    "scheduler",
                    "drain_tx_in_flight",
                    in_flight.len() as u64,
                );
                plonky2::util::profile::counter(
                    "scheduler",
                    "exclusive_drain_claimed",
                    exclusive_drain as u64,
                );
            }
            if exclusive_drain {
                plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
            }
            while let Some((chain_step, proof_handle)) = in_flight.pop_front() {
                #[cfg(feature = "diagnostic_profile")]
                let _join_wait = plonky2::util::profile::span("wait", "tx_proof_drain_join");
                let tx_proof = proof_handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                executor.submit(chain_step, tx_proof);
            }
            assert_eq!(
                executor.submitted(),
                chain_step_count as u64,
                "reusable chain executor must receive every chain step"
            );
            let chain_proof = executor.finish();
            if exclusive_drain {
                plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
            }
            (chain_proof, REUSABLE_CHAIN_EXECUTOR_WORKERS)
        })
    } else {
        let mut legacy_chain_step_threads = 0usize;
        let chain_proof = std::thread::scope(|scope| {
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
                    legacy_chain_step_threads += 1;
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
                #[cfg(feature = "diagnostic_profile")]
                plonky2::util::profile::counter(
                    "scheduler",
                    "tx_in_flight",
                    in_flight.len() as u64,
                );
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
                    #[cfg(feature = "diagnostic_profile")]
                    let _join_wait = plonky2::util::profile::span("wait", "tx_proof_window_join");
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
                legacy_chain_step_threads += 1;
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
            // pre-execution and final block phases — but only once this path is the
            // last one proving, since the switch is process-global (see
            // [`claims_exclusive_gpu_phase`]).
            let exclusive_drain = claims_exclusive_gpu_phase(active_paths);
            #[cfg(feature = "diagnostic_profile")]
            {
                plonky2::util::profile::counter(
                    "scheduler",
                    "drain_tx_in_flight",
                    in_flight.len() as u64,
                );
                plonky2::util::profile::counter(
                    "scheduler",
                    "exclusive_drain_claimed",
                    exclusive_drain as u64,
                );
            }
            if exclusive_drain {
                plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
            }
            while let Some((chain_step, proof_handle)) = in_flight.pop_front() {
                #[cfg(feature = "diagnostic_profile")]
                let _join_wait = plonky2::util::profile::span("wait", "tx_proof_drain_join");
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
            if exclusive_drain {
                plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
            }
            chain_proof
        });
        (chain_proof, legacy_chain_step_threads)
    };
    if reusable_chain_executor_diagnostics_enabled() {
        eprintln!(
            "reusable_chain_executor path={path:?} mode={} chain_steps={} chain_os_threads_created={} worker_limit={}",
            if reusable_chain_executor {
                "candidate"
            } else {
                "legacy-control"
            },
            chain_step_count,
            chain_os_threads_created,
            if reusable_chain_executor {
                REUSABLE_CHAIN_EXECUTOR_WORKERS
            } else {
                0
            },
        );
    }
    // This path has produced its last proof. Retiring it here — after the scope,
    // so every thread it spawned has joined — is what lets the sibling path's
    // drain observe that it is alone and claim the exclusive GPU phase.
    active_paths.fetch_sub(1, Ordering::Release);
    chain_proof
}

/// Proves the block pre-execution circuit. The startup-overlap path must NOT
/// set the exclusive GPU phase, because the remaining circuit loads are still
/// using the GPU normally; the serial path sets it around the call.
///
/// Measured, so nobody re-mines this (10 interleaved runs, one binary, the
/// switch runtime-gated, census taken in `gpu_worthwhile`):
/// `Circuits::load_remaining_embedded` recomputes each blob's
/// `constants_sigmas_commitment` through `PolynomialBatch::from_values`, so
/// four commitment trees — 2 x (2^19 leaves x 88 cols) for the transaction
/// circuits and 2 x (2^17 x 86) for the chain circuits, all either above the
/// routing cutoff or wider than 64 and therefore GPU-bound unconditionally —
/// are hashing on the GPU *inside* the pre-execution window (8 of the window's
/// routing decisions). With `MAX_BUFFER_SETS == 1` those builds serialize, and
/// this proof's own narrow trees observe `GPU_JOBS_IN_FLIGHT` at 1-2 for 5-7 of
/// their 9 routing decisions. So "no other proof runs concurrently" holds here
/// while "the GPU stream is idle" does not, and only the latter is the switch's
/// real contract. Enabling it does change routing — the 2^17 width-20
/// Zs/partial-products tree goes 1/10 -> 10/10 GPU and the width-16 quotient
/// tree 6/10 -> 10/10 — but each flipped tree then queues FIFO behind a
/// 2^19-leaf load build, and the phase inflated from a 325 ms median to 425 ms.
/// It buys nothing even when it wins: this proof finishes a median 187 ms before
/// the loads it hides behind, so the join waits on the loads, not on it.
/// Enabling the switch only spends that slack (median 187 ms -> 126 ms) and put
/// the proof on the critical path in 1 of the 5 runs that had it enabled.
pub(crate) fn prove_pre_execution_parallel(
    pre_data: &CircuitData<F, C, D>,
    pre_target: &BlockPreExecutionTarget,
    pre_exec: &BlockPreExec<F>,
) -> Proof {
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context =
        plonky2::util::profile::enter_context("pre_execution", 0, &[("proof_kind", 0)]);
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "pre_execution_proof");
    BlockPreExecutionCircuit::prove(pre_data, pre_exec, pre_target)
        .expect("block pre-execution proof failed")
}

/// The fully serial entry point: pre-execution proof first, then the pipeline.
///
/// Test-only. The `prove` binary starts the pre-execution proof on a startup
/// thread that overlaps the remaining circuit loads and then calls
/// [`prove_block_after_pre`] directly, so nothing on the scored path routes
/// through here. It is retained as the reference for what the serial ordering
/// looked like — in particular that the exclusive-GPU switch below is legitimate
/// only under that ordering, which no longer exists (see
/// [`prove_pre_execution_parallel`]). `#[cfg(test)]` because the release build
/// would otherwise warn it dead.
#[cfg(test)]
pub fn prove_block(block: Block<F>, circuits: Circuits) -> Proof {
    // The pre-execution proof runs strictly before any other proving work, so
    // the serialized GPU stream is otherwise idle: route its mid-size column
    // trees to the GPU for just this phase.
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
    let pre_proof = prove_pre_execution_parallel(
        &circuits.pre_data,
        &circuits.pre_target,
        &BlockPreExec::from_block(&block),
    );
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
    prove_block_after_pre(block, circuits, pre_proof)
}

/// The pipeline after the pre-execution proof. The startup-overlap path calls
/// this once both the pre-execution proof and the remaining circuit loads have
/// completed.
pub(crate) fn prove_block_after_pre(
    mut block: Block<F>,
    mut circuits: Circuits,
    pre_proof: Proof,
) -> Proof {
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context =
        plonky2::util::profile::enter_context("block_pipeline", block.block_number, &[]);
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "prove_block_after_pre");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let mut tx_chunks = std::mem::take(&mut block.tx_chunks);
    let mut heavy_chunks: Vec<(usize, Vec<Arc<Tx<F>>>)> = Vec::new();
    let mut light_chunks: Vec<(usize, Vec<Arc<Tx<F>>>)> = Vec::with_capacity(tx_chunks.len());
    for (chunk_index, txs) in tx_chunks.drain(..).enumerate() {
        if chunk_is_light(&txs) {
            light_chunks.push((chunk_index, txs));
        } else {
            heavy_chunks.push((chunk_index, txs));
        }
    }
    block.tx_chunks = tx_chunks;
    block.tx_chunks.push(Vec::new());

    // Both transaction paths prove concurrently and each ends in a strictly
    // sequential chain tail, but the exclusive-GPU switch that tail wants is
    // process-global. This counter lets a path tell "my own pipeline is done"
    // apart from "no other proof is running": each path retires itself when its
    // chain proof is finished, so only the last one standing claims the phase.
    let active_paths = AtomicUsize::new(2);
    let (light_chain_proof, heavy_chain_proof, block_target, block_data, block_pending) = {
        // The pipeline only ever reads the circuits; the borrow ends with this
        // block so the finished extensions can be released below.
        let circuits = &circuits;
        let active_paths = &active_paths;
        std::thread::scope(|scope| {
            // The final block circuit depends only on already-built circuit data
            // and is not needed until the final proof, so it builds concurrently
            // with the entire transaction/chain proving pipeline.
            // Two-phase final-block witness (H13): this lane also runs the
            // EARLY witness phase (block data + pre-proof generators) after the
            // build, then joins the heavy path — which finishes ~30 s before
            // the light path — and feeds its verify subtree here, mid-pipeline.
            // Measured feed split: light 0.018 s vs heavy 0.575 s; the heavy
            // verify subtree (ECDSA/keccak) owns the late witness cost, and
            // moving it here deletes it from the serial tail. Both phases run
            // WITHOUT `ParallelWitnessGuard` (thread-local; parallel rounds
            // here would contend with the pipeline's pool). This is witness
            // WORK MOVED OFF THE TAIL onto an otherwise-idle lane, not new
            // parallelism: the lane sleeps in `join` until the heavy proof
            // arrives, then does 0.6 s of serial work while ~the light spine
            // alone is running. The circuit data is leaked to hand the pending
            // witness a 'static borrow across the thread boundary — free, the
            // worker exits via `process::exit`.
            let heavy_handle_outer = std::thread::Builder::new()
                .name("heavy-tx-chain".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, || {
                    prove_path(
                        TxPath::Heavy,
                        heavy_chunks,
                        circuits,
                        block.block_number,
                        block.created_at,
                        block.old_account_delta_tree_root,
                        &pre_output,
                        state_metadata_hash,
                        active_paths,
                    )
                })
                .expect("heavy transaction chain thread must start");
            let block_ref = &block;
            let pre_proof_ref = &pre_proof;
            let block_circuit_handle = std::thread::Builder::new()
                .name("block-circuit-build".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    #[cfg(feature = "diagnostic_profile")]
                    let _profile_context = plonky2::util::profile::enter_context(
                        "final_block_build",
                        block_ref.block_number,
                        &[],
                    );
                    #[cfg(feature = "diagnostic_profile")]
                    let _profile_span =
                        plonky2::util::profile::span("orchestration", "final_block_build_lane");
                    let (block_target, block_data) = {
                        #[cfg(feature = "diagnostic_profile")]
                        let _span =
                            plonky2::util::profile::span("orchestration", "build_block_circuit");
                        circuits.build_block_circuit()
                    };
                    let block_data: &'static CircuitData<F, C, D> = Box::leak(Box::new(block_data));
                    let early =
                        BlockCircuit::witness_inputs_early(&block_target, block_ref, pre_proof_ref)
                            .expect("final block early witness inputs failed");
                    let mut pending = PendingPartitionWitness::start(
                        early,
                        &block_data.prover_only,
                        &block_data.common,
                    )
                    .expect("final block early witness phase failed");
                    #[cfg(feature = "diagnostic_profile")]
                    let _heavy_wait =
                        plonky2::util::profile::span("wait", "heavy_path_join_for_final");
                    let heavy_chain_proof = heavy_handle_outer
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                    // The heavy path's thread has exited, so its shared guards
                    // on the heavy transaction and chain circuits are gone, and
                    // this lane dropped its own guard when `build_block_circuit`
                    // returned above. Nothing reads those two circuits again:
                    // the light pipeline uses the light pair, and the final
                    // block proof uses only `block_data`, the three finished
                    // proofs and the block. Retire their preprocessed
                    // extensions here — 438 MiB of Metal shared buffers whose
                    // release returns the pages to the OS immediately — instead
                    // of holding them across the whole light phase.
                    circuits.release_heavy_circuit_extensions();
                    pending
                        .feed(
                            BlockCircuit::witness_inputs_heavy_chain(
                                &block_target,
                                &heavy_chain_proof,
                            )
                            .expect("final block heavy-chain witness inputs failed"),
                        )
                        .expect("final block heavy-chain witness feed failed");
                    (block_target, block_data, pending, heavy_chain_proof)
                })
                .expect("block circuit build thread must start");
            let light_chunks = std::mem::take(&mut light_chunks);
            let light_handle = std::thread::Builder::new()
                .name("light-tx-chain".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, || {
                    mark_spine_thread_latency_critical();
                    prove_path(
                        TxPath::Light,
                        light_chunks,
                        circuits,
                        block.block_number,
                        block.created_at,
                        block.old_account_delta_tree_root,
                        &pre_output,
                        state_metadata_hash,
                        active_paths,
                    )
                })
                .expect("light transaction chain thread must start");
            #[cfg(feature = "diagnostic_profile")]
            let _block_lane_wait =
                plonky2::util::profile::span("wait", "final_block_build_lane_join");
            let (block_target, block_data, block_pending, heavy_chain_proof) = block_circuit_handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            #[cfg(feature = "diagnostic_profile")]
            drop(_block_lane_wait);
            #[cfg(feature = "diagnostic_profile")]
            let _light_wait = plonky2::util::profile::span("wait", "light_path_join");
            let light_chain_proof = light_handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            // The light path's thread has exited, so its shared guards on the
            // light transaction and chain circuits are gone, and the block lane
            // dropped its own light-chain guard when `build_block_circuit`
            // returned long ago. Nothing reads the light pair again: the final
            // block proof uses only `block_data`, the three finished proofs and
            // the block. Retire their preprocessed extensions here — 438 MiB of
            // Metal shared buffers whose release returns the pages to the OS
            // immediately — instead of holding them through the final witness
            // setup until the backstop below.
            circuits.release_light_circuit_extensions();
            (
                light_chain_proof,
                heavy_chain_proof,
                block_target,
                block_data,
                block_pending,
            )
        })
    };

    // Every circuit but the block circuit has now produced its last proof, so
    // their preprocessed low-degree extensions are unreachable. Release them
    // before the final block proof — the process's peak-RSS moment — stacks its
    // own extensions on top of them.
    circuits.release_finished_circuit_extensions();

    #[cfg(feature = "diagnostic_profile")]
    let _profile_context =
        plonky2::util::profile::enter_context("final_block", block.block_number, &[]);
    #[cfg(feature = "diagnostic_profile")]
    let _profile_span = plonky2::util::profile::span("orchestration", "final_block_tail");
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
    let mut block_pending = block_pending;
    {
        #[cfg(feature = "diagnostic_profile")]
        let _span = plonky2::util::profile::span("witness", "final_light_feed");
        block_pending
            .feed(
                BlockCircuit::witness_inputs_light_chain(&block_target, light_chain_input)
                    .expect("final block light-chain witness inputs failed"),
            )
            .expect("final block light-chain witness feed failed");
    }
    let _ = heavy_chain_input;
    let final_proof = {
        #[cfg(feature = "diagnostic_profile")]
        let _span = plonky2::util::profile::span("orchestration", "final_block_proof");
        BlockCircuit::prove_prepared(block_pending, block_data).expect("final block proof failed")
    };
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
    final_proof
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::api::{
        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
    };

    #[derive(Debug, Eq, PartialEq)]
    enum OrderedExecutorEvent {
        Seeded(u64),
        Finished(u64),
    }

    #[test]
    fn reusable_chain_executor_orders_steps_and_seeds_before_predecessor_is_available() {
        use std::sync::{Condvar, Mutex};
        use std::time::Duration;

        std::thread::scope(|scope| {
            let (event_sender, event_receiver) = std::sync::mpsc::channel();
            let worker_events = event_sender.clone();
            let first_step_gate = Arc::new((Mutex::new(false), Condvar::new()));
            let worker_gate = Arc::clone(&first_step_gate);
            let mut executor: BoundedOrderedExecutor<'_, u64, u64> = BoundedOrderedExecutor::new(
                scope,
                "ordered-chain-test-worker",
                2 * 1024 * 1024,
                move |step, input, previous| {
                    worker_events
                        .send(OrderedExecutorEvent::Seeded(step))
                        .expect("test event receiver must remain available");
                    if step == 0 {
                        let (gate, condition) = &*worker_gate;
                        let open = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        drop(
                            condition
                                .wait_while(open, |open| !*open)
                                .unwrap_or_else(|poisoned| poisoned.into_inner()),
                        );
                    }
                    let previous = previous
                        .map(|receiver| {
                            receiver
                                .recv()
                                .expect("test predecessor result must arrive")
                                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                        })
                        .unwrap_or(0);
                    worker_events
                        .send(OrderedExecutorEvent::Finished(step))
                        .expect("test event receiver must remain available");
                    previous + input
                },
            );
            executor.submit(0, 1);
            executor.submit(1, 2);
            executor.submit(2, 3);
            drop(event_sender);

            // Step 0 is deliberately unable to publish its result. Step 1 must
            // nevertheless complete its dependency-independent seed phase on the
            // other reusable worker before it waits for that unavailable proof.
            let mut observed = Vec::new();
            while observed.len() < 2 {
                let event = event_receiver
                    .recv_timeout(Duration::from_secs(5))
                    .expect("the first two reusable workers must start");
                assert!(
                    matches!(event, OrderedExecutorEvent::Seeded(_)),
                    "no chain step can finish while step 0 is gated"
                );
                observed.push(event);
            }
            assert!(observed.contains(&OrderedExecutorEvent::Seeded(0)));
            assert!(observed.contains(&OrderedExecutorEvent::Seeded(1)));

            let (gate, condition) = &*first_step_gate;
            *gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
            condition.notify_all();

            assert_eq!(executor.finish(), 6);
            observed.extend(event_receiver);
            let mut seeded = observed
                .iter()
                .filter_map(|event| match event {
                    OrderedExecutorEvent::Seeded(step) => Some(*step),
                    OrderedExecutorEvent::Finished(_) => None,
                })
                .collect::<Vec<_>>();
            seeded.sort_unstable();
            assert_eq!(seeded, vec![0, 1, 2]);
            assert_eq!(
                observed
                    .iter()
                    .filter_map(|event| match event {
                        OrderedExecutorEvent::Seeded(_) => None,
                        OrderedExecutorEvent::Finished(step) => Some(*step),
                    })
                    .collect::<Vec<_>>(),
                vec![0, 1, 2],
            );
        });
    }

    #[cfg(feature = "diagnostic_profile")]
    #[test]
    fn profile_path_context_names_are_stable() {
        assert_eq!(
            profile_path_context(TxPath::Heavy, "witness"),
            "heavy_tx_witness"
        );
        assert_eq!(
            profile_path_context(TxPath::Light, "proof"),
            "light_tx_proof"
        );
        assert_eq!(profile_path_context(TxPath::Heavy, "chain"), "heavy_chain");
        assert_eq!(profile_path_context(TxPath::Light, "chain"), "light_chain");
    }

    #[test]
    fn prove_block_returns_one_final_block_proof() {
        let prove: fn(Block<F>, Circuits) -> Proof = prove_block;
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
    fn empty_padding_transactions_share_storage_per_path() {
        use std::sync::Arc;

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
                let heavy = block
                    .tx_chunks
                    .iter()
                    .flatten()
                    .find(|tx| tx.tx_circuit_type != TX_LIGHT)
                    .expect("heavy padding must exist");
                let light = block
                    .tx_chunks
                    .iter()
                    .flatten()
                    .find(|tx| tx.tx_circuit_type == TX_LIGHT)
                    .expect("light padding must exist");
                assert!(block
                    .tx_chunks
                    .iter()
                    .flatten()
                    .filter(|tx| tx.tx_circuit_type != TX_LIGHT)
                    .all(|tx| Arc::ptr_eq(tx, heavy)));
                assert!(block
                    .tx_chunks
                    .iter()
                    .flatten()
                    .filter(|tx| tx.tx_circuit_type == TX_LIGHT)
                    .all(|tx| Arc::ptr_eq(tx, light)));
                assert!(!Arc::ptr_eq(heavy, light));
            })
            .expect("padding sharing test thread must start")
            .join()
            .expect("padding sharing test thread must finish");
    }

    #[test]
    fn exclusive_gpu_phase_is_claimed_only_by_the_last_running_path() {
        // Two paths proving: the one that reaches its drain first (the three-chunk
        // heavy path) must not claim the process-global exclusive phase while the
        // forty-nine-chunk light pipeline is still running.
        let active_paths = AtomicUsize::new(2);
        assert!(!claims_exclusive_gpu_phase(&active_paths));

        // The heavy path retires; the light path's drain is now genuinely alone.
        active_paths.fetch_sub(1, Ordering::Release);
        assert!(claims_exclusive_gpu_phase(&active_paths));

        // Both retired: nothing is proving, so nothing claims the phase either.
        active_paths.fetch_sub(1, Ordering::Release);
        assert!(!claims_exclusive_gpu_phase(&active_paths));
    }

    #[test]
    fn final_block_chain_inputs_are_light_then_heavy() {
        let light = "light";
        let heavy = "heavy";

        assert_eq!(final_chain_inputs(&light, &heavy), (&light, &heavy));
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

        const CHAIN_STEPS: u64 = 10;

        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .is_test(false)
            .try_init();

        use circuit::block_tx_chain_constraints::Circuit as _;
        use circuit::types::constants::TX_TYPE_EMPTY;
        use plonky2::field::types::{Field, PrimeField64};

        use crate::api::{PathCircuits, LIGHT_TX_MODE};

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
        let mut empty_tx = (**block
            .tx_chunks
            .iter()
            .flatten()
            .find(|tx| tx.tx_type == TX_TYPE_EMPTY)
            .expect("fixture must contain an empty padding tx"))
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

        let light_chunk = vec![Arc::new(empty_tx); LIGHT_TX_PER_PROOF];
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
        for chain_step in 0..CHAIN_STEPS {
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

            // Differential integration check for the production direct-seeding
            // path. The reference above keeps the old PartialWitness map
            // path solely for this manual timing harness.
            let direct_start = Instant::now();
            let direct_proof = chain_step_proof(
                TxPath::Light,
                &circuits.chain_target,
                &circuits.chain_data,
                chain_step,
                previous.clone().map(ChainState::Ready),
                &base_proof,
                &circuits.dummy_proof,
                &tx_proof,
            );
            let direct_elapsed = direct_start.elapsed();
            assert_eq!(proof.public_inputs, direct_proof.public_inputs);

            println!(
                "chain step {chain_step}: single-shot witness {single_shot_elapsed:?}, \
                 map phase1 {phase1_elapsed:?}, map phase2 {phase2_elapsed:?}, \
                 map prove {prove_elapsed:?}, direct total {direct_elapsed:?}",
            );
            previous = Some(direct_proof);
        }

        circuits
            .chain_data
            .verify(previous.expect("chain must produce proofs"))
            .expect("final chain step proof must verify");
    }
}
