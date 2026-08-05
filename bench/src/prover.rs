// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, mpsc};

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::block_tx_native::{NativePreOutput, chunk_end_roots, native_pre_output};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::recursion::dummy_circuit::dummy_circuit;

use crate::api::{Circuits, Proof, StageTimer};

const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Upper bound on concurrently proving leaf (BlockTx) proofs. Each in-flight
/// 2^16 leaf prove costs a few GB of working set on top of the resident
/// circuit data, so 4 keeps the peak comfortably below ~30GB on the 48GB
/// hosts while soaking up the cores a single prove leaves idle; smaller hosts
/// scale down with their core count. The pool only ever blocks on taking new
/// work and the bound never drops below 1, so it cannot deadlock.
fn leaf_concurrency() -> usize {
    // Local profiling override; the ranked sandbox clears the environment.
    if let Some(value) = std::env::var_os("LIGHTER_LEAF_CONCURRENCY") {
        if let Some(n) = value.to_str().and_then(|v| v.parse::<usize>().ok()) {
            return n.max(1);
        }
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    (cores / 4).clamp(1, 4)
}

fn chunk_is_light(txs: &[Tx<F>]) -> bool {
    txs.first()
        .expect("block transaction chunk must not be empty")
        .tx_circuit_type
        == TX_LIGHT
}

fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
    (light, heavy)
}

/// One leaf tx proof, fully scheduled upfront: both jump states are computed
/// natively (`JumpState::advance`), so no leaf waits on any other proof. The
/// chunk's txs live in the shared [`TxStore`] so a chain lane can re-prove the
/// step if the native prediction turns out wrong.
struct LeafJob {
    chunk_index: usize,
    is_light: bool,
    chain_step: u64,
    old_jump: JumpState<F>,
    new_jump: JumpState<F>,
}

struct PathPlan {
    steps: u64,
    /// chain_step -> chunk_index, for fallback re-proving.
    step_chunks: Vec<usize>,
}

struct LeafPlan {
    jobs: VecDeque<LeafJob>,
    chunks: Vec<Vec<Tx<F>>>,
    heavy: PathPlan,
    light: PathPlan,
}

/// Routes every chunk to its path, assigns chain steps in fixture order, and
/// chains the native jump states through each path. Moves the chunks out of
/// the block (into the shared store); the caller restores a sentinel chunk
/// before the final block witness.
fn plan_leaves(block: &mut Block<F>, initial_jump: JumpState<F>) -> LeafPlan {
    let end_roots = chunk_end_roots(block);
    let chunks = std::mem::take(&mut block.tx_chunks);
    let mut heavy_jump = initial_jump;
    let mut light_jump = initial_jump;
    let mut heavy = PathPlan {
        steps: 0,
        step_chunks: Vec::new(),
    };
    let mut light = PathPlan {
        steps: 0,
        step_chunks: Vec::new(),
    };
    let jobs = chunks
        .iter()
        .zip(&end_roots)
        .enumerate()
        .map(|(chunk_index, (txs, chunk_end_roots))| {
            let is_light = chunk_is_light(txs);
            let (jump, path) = if is_light {
                (&mut light_jump, &mut light)
            } else {
                (&mut heavy_jump, &mut heavy)
            };
            let old_jump = *jump;
            let new_jump = old_jump.advance(txs, chunk_end_roots);
            *jump = new_jump;
            let chain_step = path.steps;
            path.steps += 1;
            path.step_chunks.push(chunk_index);
            LeafJob {
                chunk_index,
                is_light,
                chain_step,
                old_jump,
                new_jump,
            }
        })
        .collect();
    LeafPlan {
        jobs,
        chunks,
        heavy,
        light,
    }
}

/// Every chunk's txs, shared between the leaf pool and the chain lanes: a leaf
/// worker borrows a chunk to prove it and always puts it back; a chain lane
/// takes it again only when that step must be re-proved. Keeping the txs for
/// the whole run costs only the parsed fixture's footprint (the chunks used to
/// be dropped after their leaf prove; nothing else grows).
struct TxStore {
    chunks: Mutex<Vec<Option<Vec<Tx<F>>>>>,
    returned: Condvar,
}

impl TxStore {
    fn new(chunks: Vec<Vec<Tx<F>>>) -> Self {
        Self {
            chunks: Mutex::new(chunks.into_iter().map(Some).collect()),
            returned: Condvar::new(),
        }
    }

    /// Non-blocking: `None` means a chain lane already took this chunk to
    /// re-prove the step itself, so the caller must skip the job.
    fn try_take(&self, chunk_index: usize) -> Option<Vec<Tx<F>>> {
        self.chunks.lock().expect("tx store poisoned")[chunk_index].take()
    }

    /// Blocking: waits for the leaf worker currently proving this chunk to
    /// return it (workers always put chunks back right after proving).
    fn take_blocking(&self, chunk_index: usize) -> Vec<Tx<F>> {
        let mut chunks = self.chunks.lock().expect("tx store poisoned");
        loop {
            if let Some(txs) = chunks[chunk_index].take() {
                return txs;
            }
            chunks = self.returned.wait(chunks).expect("tx store poisoned");
        }
    }

    fn put_back(&self, chunk_index: usize, txs: Vec<Tx<F>>) {
        let mut chunks = self.chunks.lock().expect("tx store poisoned");
        debug_assert!(
            chunks[chunk_index].is_none(),
            "chunk #{chunk_index} returned twice"
        );
        chunks[chunk_index] = Some(txs);
        drop(chunks);
        self.returned.notify_all();
    }
}

/// Once a path's native jump predictions are known-bad from some step on, the
/// leaf workers must stop spending proves on them (the lane re-proves those
/// steps sequentially from proof-derived jumps). Monotone-min step marker.
struct PathPoison(AtomicU64);

impl Default for PathPoison {
    fn default() -> Self {
        Self(AtomicU64::new(u64::MAX))
    }
}

impl PathPoison {
    fn mark_from(&self, step: u64) {
        self.0.fetch_min(step, Ordering::Relaxed);
    }

    fn covers(&self, step: u64) -> bool {
        self.0.load(Ordering::Relaxed) <= step
    }
}

/// Counters the fallback path bumps; a completed run with zeros means the
/// native plan was exact (the expected case).
#[derive(Debug, Default)]
pub struct FallbackStats {
    pub reproved_leaves: AtomicUsize,
}

/// Field-by-field diff of two jump states; empty means equal.
fn jump_state_diff(expected: &JumpState<F>, actual: &JumpState<F>) -> String {
    let mut diffs = String::new();
    macro_rules! check {
        ($field:ident) => {
            if expected.$field != actual.$field {
                diffs.push_str(&format!(
                    "  {}: expected {:?} != actual {:?}\n",
                    stringify!($field),
                    expected.$field,
                    actual.$field,
                ));
            }
        };
    }
    check!(last_active_tx_index);
    check!(prev_new_state_root);
    check!(prev_new_delta_root);
    check!(run_start_prev_index);
    check!(run_start_old_state_root);
    check!(run_start_old_delta_root);
    check!(coverage_hash);
    check!(claims_hash);
    check!(tx_count);
    diffs
}

/// The leaves start from natively derived pre-execution outputs; the real pre
/// proof's outputs are authoritative. A mismatch no longer kills the run: the
/// chain lanes are seeded from the PROOF's outputs and reject+re-prove every
/// leaf built on wrong values, so a native pre divergence degrades to a slow
/// pass instead of a zero.
fn native_pre_divergence(
    native: &NativePreOutput,
    proved: &BlockPreExecWitness<F>,
) -> Option<String> {
    let mut diffs = String::new();
    if native.new_state_metadata.to_public_inputs() != proved.new_state_metadata.to_public_inputs()
    {
        diffs.push_str(&format!(
            "  new_state_metadata: native {:?} != proof {:?}\n",
            native.new_state_metadata, proved.new_state_metadata
        ));
    }
    if native.new_state_root != proved.new_state_root {
        diffs.push_str(&format!(
            "  new_state_root: native {:?} != proof {:?}\n",
            native.new_state_root, proved.new_state_root
        ));
    }
    (!diffs.is_empty()).then_some(diffs)
}

/// A finished (or failed) leaf prove, tagged with its chain step.
enum LeafMsg {
    Proved {
        proof: Proof,
        /// The proof itself is fine, but its `new_jump` disagreed with the
        /// native prediction — every LATER step of this path was planned from
        /// a wrong jump and must be re-proved by the lane.
        prediction_diverged: bool,
    },
    Failed(String),
}

/// What a chain lane needs from the pre-execution proof: its per-block base
/// proof plus the PROOF-derived values every leaf witness must have used. The
/// lane validates candidates against these and re-proves with them on any
/// mismatch — they are correct by construction, unlike the native derivations.
struct LaneSeed {
    base: Proof,
    initial_jump: JumpState<F>,
    state_metadata_hash: HashOut<F>,
}

/// One chain path (heavy or light): waits for its per-block base proof, then
/// folds its leaf proofs in chain-step order. The base proof doubles as the
/// dummy-slot argument of every chain prove. Leaves complete in arbitrary
/// order, so out-of-order arrivals wait in a reorder buffer (completed proofs
/// are small; the heavy memory lives in the bounded leaf pool).
///
/// The lane is the correctness authority for its path: it tracks the true
/// jump state (seed for step 0, then each folded proof's `new_jump`) and folds
/// a candidate only if the candidate's `old_jump` extends it — the same
/// equation the chain circuit enforces (`JumpStateTarget::connect(block.jump,
/// current_tx.old_jump)`). Any rejected/failed/flagged candidate flips the
/// lane into sequential re-proving of the remaining steps from proof-derived
/// jumps (the pre-A3 correct-by-construction schedule), so a native-transition
/// bug costs time, not the fixture.
struct ChainLane<'a> {
    label: &'static str,
    target: &'a BlockTxChainTarget,
    data: &'a CircuitData<F, C, D>,
    tx_target: &'a BlockTxTarget,
    tx_data: &'a CircuitData<F, C, D>,
    steps: u64,
    step_chunks: &'a [usize],
    created_at: i64,
    base: mpsc::Receiver<LaneSeed>,
    leaves: mpsc::Receiver<(u64, LeafMsg)>,
    store: &'a TxStore,
    poison: &'a PathPoison,
    stats: &'a FallbackStats,
}

impl ChainLane<'_> {
    /// Sequential fallback: re-prove this step's leaf with the lane's
    /// proof-derived jump and metadata hash. If even this fails, the fixture
    /// cannot be proved by any schedule — die like the sequential prover
    /// would.
    fn reprove_leaf(&self, chain_step: u64, expected_jump: &JumpState<F>, seed: &LaneSeed) -> Proof {
        self.stats.reproved_leaves.fetch_add(1, Ordering::Relaxed);
        let chunk_index = self.step_chunks[chain_step as usize];
        let txs = self.store.take_blocking(chunk_index);
        let block_tx = BlockTx {
            created_at: self.created_at,
            state_metadata_hash: seed.state_metadata_hash,
            old_jump: *expected_jump,
            txs,
        };
        let result = BlockTxCircuit::prove(self.tx_data, &block_tx, self.tx_target);
        self.store.put_back(chunk_index, block_tx.txs);
        result.unwrap_or_else(|error| {
            panic!(
                "{} chain step #{chain_step}: leaf re-prove with proof-derived inputs failed — \
                 fixture unprovable: {error:?}",
                self.label
            )
        })
    }

    fn next_candidate(&self, pending: &mut BTreeMap<u64, LeafMsg>, chain_step: u64) -> LeafMsg {
        loop {
            if let Some(msg) = pending.remove(&chain_step) {
                return msg;
            }
            let (step, msg) = self.leaves.recv().unwrap_or_else(|_| {
                panic!(
                    "{} leaf proofs ended before chain step #{chain_step}",
                    self.label
                )
            });
            pending.insert(step, msg);
        }
    }
}

fn run_chain_lane(lane: ChainLane<'_>) -> Proof {
    let seed = lane.base.recv().unwrap_or_else(|_| {
        panic!(
            "pre-execution stage failed before the {} chain base proof arrived",
            lane.label
        )
    });
    let mut expected_jump = seed.initial_jump;
    let mut chain_proof: Option<Proof> = None;
    let mut pending: BTreeMap<u64, LeafMsg> = BTreeMap::new();
    // Once true, the native predictions past this point are wrong: stop
    // consuming the channel and re-prove the remaining steps sequentially.
    let mut reprove_tail = false;
    for chain_step in 0..lane.steps {
        let (tx_proof, prediction_diverged, candidate_reproved) = if reprove_tail {
            (
                lane.reprove_leaf(chain_step, &expected_jump, &seed),
                false,
                true,
            )
        } else {
            match lane.next_candidate(&mut pending, chain_step) {
                LeafMsg::Proved {
                    proof,
                    prediction_diverged,
                } => {
                    let proved_old =
                        BlockTxWitness::from_public_inputs(&proof.public_inputs).old_jump;
                    let diff = jump_state_diff(&expected_jump, &proved_old);
                    if diff.is_empty() {
                        (proof, prediction_diverged, false)
                    } else {
                        eprintln!(
                            "[fallback] {} chain step #{chain_step}: leaf proof's old_jump does \
                             not extend the chain; re-proving this path sequentially from \
                             here\n{diff}",
                            lane.label
                        );
                        lane.poison.mark_from(chain_step);
                        reprove_tail = true;
                        (
                            lane.reprove_leaf(chain_step, &expected_jump, &seed),
                            false,
                            true,
                        )
                    }
                }
                LeafMsg::Failed(error) => {
                    eprintln!(
                        "[fallback] {} chain step #{chain_step}: leaf prove failed ({error}); \
                         re-proving this path sequentially from here",
                        lane.label
                    );
                    lane.poison.mark_from(chain_step);
                    reprove_tail = true;
                    (
                        lane.reprove_leaf(chain_step, &expected_jump, &seed),
                        false,
                        true,
                    )
                }
            }
        };

        // Fold. If the chain circuit still rejects a channel candidate (its
        // witness generation fails on anything the old_jump check does not
        // cover), re-prove the leaf from the lane's authoritative inputs and
        // retry once; a rejection of a re-proved leaf is a real bug.
        let mut folded_leaf = tx_proof;
        let next_chain_proof = {
            let fold = |leaf: &Proof, previous: &Option<Proof>| {
                BlockTxChainCircuit::prove(
                    lane.target,
                    lane.data,
                    chain_step,
                    previous.as_ref().unwrap_or(&seed.base),
                    &seed.base,
                    leaf,
                )
            };
            match fold(&folded_leaf, &chain_proof) {
                Ok(next) => next,
                Err(error) if !candidate_reproved => {
                    eprintln!(
                        "[fallback] {} chain step #{chain_step} rejected its leaf proof \
                         ({error:?}); re-proving the leaf and retrying once",
                        lane.label
                    );
                    lane.poison.mark_from(chain_step);
                    reprove_tail = true;
                    folded_leaf = lane.reprove_leaf(chain_step, &expected_jump, &seed);
                    fold(&folded_leaf, &chain_proof).unwrap_or_else(|error| {
                        panic!(
                            "{} block transaction chain step #{chain_step} failed after leaf \
                             re-prove: {error:?}",
                            lane.label
                        )
                    })
                }
                Err(error) => panic!(
                    "{} block transaction chain step #{chain_step} failed on a re-proved leaf: \
                     {error:?}",
                    lane.label
                ),
            }
        };

        // The folded proof's new_jump is the true chain state — by circuit
        // construction, not native derivation.
        expected_jump = BlockTxWitness::from_public_inputs(&folded_leaf.public_inputs).new_jump;
        if prediction_diverged && !reprove_tail {
            eprintln!(
                "[fallback] {} chain step #{chain_step}: leaf proof folded, but its new_jump \
                 diverged from the native prediction; re-proving the remaining {} step(s) \
                 sequentially",
                lane.label,
                lane.steps - chain_step - 1
            );
            lane.poison.mark_from(chain_step + 1);
            reprove_tail = true;
        }
        chain_proof = Some(next_chain_proof);
    }
    chain_proof.unwrap_or(seed.base)
}

/// Parallel proof DAG. The only cross-proof data dependencies the sequential
/// orchestration waited on — leaf k+1's `old_jump` from leaf k's public
/// inputs, and the pre-execution outputs — are computed natively upfront, so:
///
/// ```text
///   native plan (jumps, metadata, pre state root)      [~0s]
///     ├─ pre proof ─→ seed lanes (proof outputs) ─→ base proofs ─┐
///     ├─ leaf pool (bounded): tx proofs, any order ──────────────┼─→ heavy chain ─┐
///     │    txs borrowed from the shared TxStore    ──────────────┼─→ light chain ─┼─→ block proof
///     └───────────────────────────────────────────────────────────┘  (per-path order,
///                                                                     reorder buffers)
/// ```
///
/// Same circuits, same witnesses, same public inputs, same proof encoding as
/// the sequential order — only the schedule changed. Every leaf proof is
/// validated by its chain lane against proof-derived values before folding
/// (see [`ChainLane`]); a native divergence triggers sequential re-proving of
/// the affected path's tail instead of failing the fixture.
pub fn prove_block(block: Block<F>, circuits: &Circuits) -> Proof {
    prove_block_with(block, circuits, |_| {}).0
}

/// `plan_tweak` is a test seam: it may corrupt the natively planned jump
/// states to exercise the divergence fallback. Production passes a no-op.
fn prove_block_with(
    mut block: Block<F>,
    circuits: &Circuits,
    plan_tweak: impl FnOnce(&mut VecDeque<LeafJob>),
) -> (Proof, FallbackStats) {
    let mut timer = StageTimer::new();

    let native_pre = native_pre_output(&block);
    let initial_jump = JumpState::initial(
        native_pre.new_state_root,
        block.old_account_delta_tree_root,
    );
    let mut plan = plan_leaves(&mut block, initial_jump);
    plan_tweak(&mut plan.jobs);
    timer.mark("native plan");

    let LeafPlan {
        jobs,
        chunks,
        heavy: heavy_plan,
        light: light_plan,
    } = plan;
    let stats = FallbackStats::default();
    let store = TxStore::new(chunks);
    let heavy_poison = PathPoison::default();
    let light_poison = PathPoison::default();
    let created_at = block.created_at;

    let (heavy_send, heavy_recv) = mpsc::channel::<(u64, LeafMsg)>();
    let (light_send, light_recv) = mpsc::channel::<(u64, LeafMsg)>();
    let (heavy_base_send, heavy_base_recv) = mpsc::channel::<LaneSeed>();
    let (light_base_send, light_base_recv) = mpsc::channel::<LaneSeed>();
    let job_queue = Mutex::new(jobs);

    let block_ref = &block;
    let native_pre_ref = &native_pre;
    let job_queue_ref = &job_queue;
    let store_ref = &store;
    let stats_ref = &stats;
    let heavy_poison_ref = &heavy_poison;
    let light_poison_ref = &light_poison;

    let (pre_proof, light_chain_proof, heavy_chain_proof) = std::thread::scope(|scope| {
        // Pre-execution proof, then (from its actual outputs, exactly as the
        // sequential order did) both per-block cyclic base proofs. The chains
        // only need these at step 0, which is at least one leaf prove away.
        let pre_worker = std::thread::Builder::new()
            .name("pre-exec".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let pre_proof = BlockPreExecutionCircuit::prove(
                    &circuits.pre_data,
                    &BlockPreExec::from_block(block_ref),
                    &circuits.pre_target,
                )
                .expect("block pre-execution proof failed");
                let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
                if let Some(diff) = native_pre_divergence(native_pre_ref, &pre_output) {
                    eprintln!(
                        "[fallback] native pre-execution outputs diverged from the pre proof; \
                         leaves built on them will be rejected and re-proved by their chain \
                         lanes\n{diff}"
                    );
                }
                // Lane seeds carry PROOF-derived values: the lanes validate
                // candidates against these and re-prove from them if needed.
                let initial_jump = JumpState::initial(
                    pre_output.new_state_root,
                    block_ref.old_account_delta_tree_root,
                );
                let state_metadata_hash = pre_output.new_state_metadata.hash();

                // The upstream api no longer carries the dummy chain circuits;
                // build them here, off the leaf critical path (leaves are
                // already proving from natively computed jumps by now).
                let (heavy_base, light_base) = rayon::join(
                    || {
                        let dummy = dummy_circuit(&circuits.heavy_chain_data.common);
                        BlockTxChainCircuit::cyclic_base_proof(
                            &circuits.heavy_chain_data,
                            &dummy,
                            block_ref.block_number,
                            block_ref.created_at,
                            pre_output.new_state_root,
                            pre_output.new_validium_root,
                            block_ref.old_account_delta_tree_root,
                        )
                    },
                    || {
                        let dummy = dummy_circuit(&circuits.light_chain_data.common);
                        BlockTxChainCircuit::cyclic_base_proof(
                            &circuits.light_chain_data,
                            &dummy,
                            block_ref.block_number,
                            block_ref.created_at,
                            pre_output.new_state_root,
                            pre_output.new_validium_root,
                            block_ref.old_account_delta_tree_root,
                        )
                    },
                );
                // A closed receiver means a chain worker already panicked; that
                // panic surfaces at its join, so don't double-panic here.
                let _ = heavy_base_send.send(LaneSeed {
                    base: heavy_base,
                    initial_jump,
                    state_metadata_hash,
                });
                let _ = light_base_send.send(LaneSeed {
                    base: light_base,
                    initial_jump,
                    state_metadata_hash,
                });
                pre_proof
            })
            .expect("cannot start pre-execution worker");

        let heavy_chain_worker = std::thread::Builder::new()
            .name("heavy-chain".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, {
                let lane = ChainLane {
                    label: "heavy",
                    target: &circuits.heavy_chain_target,
                    data: &circuits.heavy_chain_data,
                    tx_target: &circuits.heavy_tx_target,
                    tx_data: &circuits.heavy_tx_data,
                    steps: heavy_plan.steps,
                    step_chunks: &heavy_plan.step_chunks,
                    created_at,
                    base: heavy_base_recv,
                    leaves: heavy_recv,
                    store: store_ref,
                    poison: heavy_poison_ref,
                    stats: stats_ref,
                };
                move || run_chain_lane(lane)
            })
            .expect("cannot start heavy chain recursion worker");
        let light_chain_worker = std::thread::Builder::new()
            .name("light-chain".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, {
                let lane = ChainLane {
                    label: "light",
                    target: &circuits.light_chain_target,
                    data: &circuits.light_chain_data,
                    tx_target: &circuits.light_tx_target,
                    tx_data: &circuits.light_tx_data,
                    steps: light_plan.steps,
                    step_chunks: &light_plan.step_chunks,
                    created_at,
                    base: light_base_recv,
                    leaves: light_recv,
                    store: store_ref,
                    poison: light_poison_ref,
                    stats: stats_ref,
                };
                move || run_chain_lane(lane)
            })
            .expect("cannot start light chain recursion worker");

        // Bounded leaf pool: workers pull jobs in fixture order (so each path's
        // early chain steps unblock first) and hand finished proofs to the
        // chain lanes; witness generation overlaps across workers, the inner
        // FFT/FRI stages share the global rayon pool.
        let leaf_workers = (0..leaf_concurrency())
            .map(|worker_index| {
                let heavy_send = heavy_send.clone();
                let light_send = light_send.clone();
                std::thread::Builder::new()
                    .name(format!("leaf-prover-{worker_index}"))
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        loop {
                            let job = job_queue_ref
                                .lock()
                                .expect("leaf job queue poisoned")
                                .pop_front();
                            let Some(job) = job else {
                                break;
                            };
                            let poison = if job.is_light {
                                light_poison_ref
                            } else {
                                heavy_poison_ref
                            };
                            // The lane re-proves poisoned steps itself, from
                            // correct jumps; proving them here would be wasted
                            // work on wrong predictions.
                            if poison.covers(job.chain_step) {
                                continue;
                            }
                            // A missing chunk means the lane took it for its
                            // own re-prove of this very step.
                            let Some(txs) = store_ref.try_take(job.chunk_index) else {
                                continue;
                            };
                            let block_tx = BlockTx {
                                created_at: block_ref.created_at,
                                state_metadata_hash: native_pre_ref.state_metadata_hash,
                                old_jump: job.old_jump,
                                txs,
                            };
                            let (data, target) = if job.is_light {
                                (&circuits.light_tx_data, &circuits.light_tx_target)
                            } else {
                                (&circuits.heavy_tx_data, &circuits.heavy_tx_target)
                            };
                            let result = BlockTxCircuit::prove(data, &block_tx, target);
                            store_ref.put_back(job.chunk_index, block_tx.txs);
                            let msg = match result {
                                Ok(proof) => {
                                    let proved =
                                        BlockTxWitness::from_public_inputs(&proof.public_inputs);
                                    let diff = jump_state_diff(&job.new_jump, &proved.new_jump);
                                    let prediction_diverged = !diff.is_empty();
                                    if prediction_diverged {
                                        eprintln!(
                                            "[fallback] native jump prediction diverged from \
                                             chunk #{} proof (chain step #{}); later steps of \
                                             this path will be re-proved sequentially\n{diff}",
                                            job.chunk_index, job.chain_step
                                        );
                                        poison.mark_from(job.chain_step + 1);
                                    }
                                    LeafMsg::Proved {
                                        proof,
                                        prediction_diverged,
                                    }
                                }
                                Err(error) => {
                                    eprintln!(
                                        "[fallback] chunk #{} leaf prove failed ({error:?}); its \
                                         chain lane will re-prove it",
                                        job.chunk_index
                                    );
                                    poison.mark_from(job.chain_step);
                                    LeafMsg::Failed(format!("{error:?}"))
                                }
                            };
                            let send = if job.is_light { &light_send } else { &heavy_send };
                            // A lane that switched to sequential re-proving may
                            // finish (and drop its receiver) while stale sends
                            // are still in flight — that is expected, never a
                            // panic. A genuinely dead lane surfaces at its
                            // join.
                            let _ = send.send((job.chain_step, msg));
                        }
                    })
                    .expect("cannot start leaf prover worker")
            })
            .collect::<Vec<_>>();
        // Only the leaf workers may hold live leaf senders: a chain lane must
        // see its channel close (not hang) if the pool dies early.
        drop(heavy_send);
        drop(light_send);

        for worker in leaf_workers {
            worker.join().expect("leaf prover worker panicked");
        }
        timer.mark("leaves done");
        let heavy_chain_proof = heavy_chain_worker
            .join()
            .expect("heavy chain recursion worker panicked");
        let light_chain_proof = light_chain_worker
            .join()
            .expect("light chain recursion worker panicked");
        timer.mark("chains done");
        let pre_proof = pre_worker.join().expect("pre-execution worker panicked");
        (pre_proof, light_chain_proof, heavy_chain_proof)
    });

    // The chunk Vecs live in the tx store (the leaf proves only borrowed
    // them); the final block witness only asserts the chunk list is non-empty
    // — the tx data itself is already folded into the chain proofs — so
    // restore a sentinel.
    block.tx_chunks.push(Vec::new());

    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    let proof = BlockCircuit::prove(
        &circuits.block_target,
        &circuits.block_data,
        &block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed");
    timer.mark("final block proof");
    (proof, stats)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::api::{
        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
    };

    fn smoke_block() -> Block<F> {
        Block::<F>::from_json_with_empty_txs(
            include_bytes!("../bench_test.json"),
            HEAVY_TX_PER_PROOF,
            LIGHT_TX_PER_PROOF,
            PUBLIC_HEAVY_TX_COUNT,
            PUBLIC_LIGHT_TX_COUNT,
        )
        .expect("public fixture must parse")
    }

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
                let block = smoke_block();
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
    fn native_plan_on_all_empty_fixture_keeps_jumps_at_initial() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let mut block = smoke_block();
                // All-empty block: the pre-execution output state root is the
                // block's (unchanged) new state root, and every chunk is pure
                // padding, so every jump transition must be the identity.
                let native_pre = native_pre_output(&block);
                assert_eq!(native_pre.new_state_root, block.new_state_root);
                assert_eq!(
                    native_pre.state_metadata_hash,
                    native_pre.new_state_metadata.hash()
                );

                let initial_jump = JumpState::initial(
                    native_pre.new_state_root,
                    block.old_account_delta_tree_root,
                );
                let chunk_count = block.tx_chunks.len();
                let plan = plan_leaves(&mut block, initial_jump);
                assert_eq!(plan.jobs.len(), chunk_count);
                assert_eq!(plan.chunks.len(), chunk_count, "txs move into the store");
                assert!(block.tx_chunks.is_empty(), "chunks move out of the block");
                assert_eq!(plan.heavy.steps, 3);
                assert_eq!(plan.light.steps, 49);
                assert_eq!(plan.heavy.step_chunks.len(), 3);
                assert_eq!(plan.light.step_chunks.len(), 49);
                for job in &plan.jobs {
                    assert_eq!(job.old_jump.to_vec(), initial_jump.to_vec());
                    assert_eq!(job.new_jump.to_vec(), initial_jump.to_vec());
                    let path = if job.is_light {
                        &plan.light
                    } else {
                        &plan.heavy
                    };
                    assert_eq!(
                        path.step_chunks[job.chain_step as usize], job.chunk_index,
                        "step-to-chunk map must invert the job assignment"
                    );
                }
                let heavy_step_order = plan
                    .jobs
                    .iter()
                    .filter(|job| !job.is_light)
                    .map(|job| job.chain_step)
                    .collect::<Vec<_>>();
                let light_step_order = plan
                    .jobs
                    .iter()
                    .filter(|job| job.is_light)
                    .map(|job| job.chain_step)
                    .collect::<Vec<_>>();
                assert_eq!(heavy_step_order, (0..3u64).collect::<Vec<_>>());
                assert_eq!(light_step_order, (0..49u64).collect::<Vec<_>>());
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

    /// End-to-end fallback drill on a shrunken (2 heavy + 2 light chunk)
    /// all-empty block: corrupt one path's native `new_jump` prediction (valid
    /// proof, wrong plan — the flagged-divergence route) and the other path's
    /// `old_jump` (valid but unusable proof — the lane-rejection route), then
    /// require the run to complete with exactly the two expected sequential
    /// re-proves and a block proof that still verifies. Debug builds skip it
    /// (full 2^16 proves are minutes-slow unoptimized); run with
    /// `cargo test -p bench --release`.
    #[cfg(not(debug_assertions))]
    #[test]
    fn divergence_fallback_reproves_tail_and_completes() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let block = Block::<F>::from_json_with_empty_txs(
                    include_bytes!("../bench_test.json"),
                    HEAVY_TX_PER_PROOF,
                    LIGHT_TX_PER_PROOF,
                    2 * HEAVY_TX_PER_PROOF,
                    2 * LIGHT_TX_PER_PROOF,
                )
                .expect("public fixture must parse");
                let circuits = Circuits::new();
                let (proof, stats) = prove_block_with(block, &circuits, |jobs| {
                    use plonky2::field::types::Field;
                    for job in jobs.iter_mut() {
                        // Light step 0: proof stays valid, but the planned
                        // new_jump is wrong — the worker must flag it and the
                        // lane must re-prove the tail (step 1).
                        if job.is_light && job.chain_step == 0 {
                            job.new_jump.tx_count += F::ONE;
                        }
                        // Heavy step 1: wrong old_jump on an all-padding chunk
                        // still proves (padding checks nothing about roots),
                        // but its public inputs cannot extend the chain — the
                        // lane must reject and re-prove it.
                        if !job.is_light && job.chain_step == 1 {
                            job.old_jump.prev_new_state_root.elements[0] += F::ONE;
                        }
                    }
                });
                assert_eq!(
                    stats.reproved_leaves.load(Ordering::Relaxed),
                    2,
                    "exactly light step 1 (poisoned tail) and heavy step 1 (rejected) re-prove"
                );
                circuits
                    .block_data
                    .verify(proof)
                    .expect("fallback run must still produce a verifying block proof");
            })
            .expect("fallback test thread must start")
            .join()
            .expect("fallback test thread must finish");
    }
}
