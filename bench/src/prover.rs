// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState, JumpStateTarget};
use circuit::block_tx_chain_constraints::{
    BlockTxChainCircuit, BlockTxChainTarget, Circuit as _, cyclic_base_witness,
};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::block_tx_native::{NativePreOutput, chunk_end_roots, native_pre_output};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;

use crate::api::{Circuits, PreChainCircuits, Proof, StageTimer};

const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The ranked-validated concurrency envelope (from the promoted windowed
/// design): each path pipelines witness generation against at most one
/// in-flight leaf proof, and the light path widens to this window once the
/// heavy path has finished. Deliberately NOT widened further — the envelope
/// itself is the ranked-proven asset.
const LIGHT_TX_PROOF_WINDOW: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxPath {
    Heavy,
    Light,
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

/// Waits for a chain fold running on its trailing thread; the next fold (or
/// the path's return) calls this on the previous one.
fn wait_fold(handle: std::thread::ScopedJoinHandle<'_, Proof>) -> Proof {
    handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
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

/// One planned leaf: its chunk and the natively predicted jump transition
/// (`JumpState::advance`). The pipeline threads the WITNESS-extracted jumps
/// (ground truth by construction); the native prediction is a zero-cost
/// tripwire — a divergence is logged and costs nothing, because nothing
/// downstream depends on it.
struct PlannedLeaf {
    chunk_index: usize,
    old_jump: JumpState<F>,
    new_jump: JumpState<F>,
}

/// Natively derived plan for the whole block, available before any circuit or
/// proof exists: the pre-execution outputs (which let both paths start
/// immediately, with no pre proof and no built pre circuit) and both paths'
/// leaf schedules.
struct NativePlan {
    pre: NativePreOutput,
    initial_jump: JumpState<F>,
    heavy: Vec<PlannedLeaf>,
    light: Vec<PlannedLeaf>,
}

/// Routes every chunk to its path in fixture order and chains the native jump
/// prediction through each path. Moves the chunks out of the block (into the
/// shared store); the caller restores a sentinel chunk before the final block
/// witness.
fn plan_paths(block: &mut Block<F>) -> (NativePlan, Vec<Vec<Tx<F>>>) {
    let pre = native_pre_output(block);
    let initial_jump = JumpState::initial(pre.new_state_root, block.old_account_delta_tree_root);
    let end_roots = chunk_end_roots(block);
    let chunks = std::mem::take(&mut block.tx_chunks);
    let mut heavy_jump = initial_jump;
    let mut light_jump = initial_jump;
    let mut heavy = Vec::new();
    let mut light = Vec::new();
    for (chunk_index, (txs, chunk_end_roots)) in chunks.iter().zip(&end_roots).enumerate() {
        let is_light = chunk_is_light(txs);
        let jump = if is_light {
            &mut light_jump
        } else {
            &mut heavy_jump
        };
        let old_jump = *jump;
        let new_jump = old_jump.advance(txs, chunk_end_roots);
        *jump = new_jump;
        let leaf = PlannedLeaf {
            chunk_index,
            old_jump,
            new_jump,
        };
        if is_light {
            light.push(leaf);
        } else {
            heavy.push(leaf);
        }
    }
    (
        NativePlan {
            pre,
            initial_jump,
            heavy,
            light,
        },
        chunks,
    )
}

/// Every chunk's txs, shared between the path pipelines and the fallback
/// re-provers: a pipeline borrows a chunk around witness generation and
/// always puts it back; fallback paths take it again to re-prove a step.
/// Keeping the txs for the whole run costs only the parsed fixture's
/// footprint.
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

    /// Blocking: waits for whoever holds this chunk to return it (holders
    /// always put chunks back right after generating/proving).
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

/// Counters the fallback path bumps; a completed run with zeros means the
/// native pre-execution values were exact and no prove failed (the expected
/// case).
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
/// proof's outputs are authoritative. A mismatch does not kill the run: each
/// path validates its seed (and every fold) against the PROOF's outputs and
/// re-proves its leaves from them, so a native pre divergence degrades to a
/// slow pass instead of a zero.
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

/// What a path needs from the pre-execution proof: its base-proof witness
/// (a cheap construction from the embedded dummy proof — no proving) plus the
/// PROOF-derived values every leaf witness must have used. The path validates
/// its native values against these and re-proves from them on any mismatch —
/// they are correct by construction, unlike the native derivations.
struct LaneSeed {
    base: Proof,
    initial_jump: JumpState<F>,
    state_metadata_hash: HashOut<F>,
}

/// One transaction path's pipeline context. The path worker owns its
/// schedule: witness generation runs on the path thread double-buffered
/// against the in-flight leaf proofs (their windowed shape), chain steps fold
/// on trailing threads, and the witness-extracted jump states — not proof
/// public inputs — thread the cross-leaf dependency, so no leaf ever waits on
/// another leaf's proof.
struct PathCtx<'a> {
    path: TxPath,
    steps: &'a [PlannedLeaf],
    tx_target: &'a BlockTxTarget,
    tx_data: &'a CircuitData<F, C, D>,
    circuits: &'a Circuits,
    created_at: i64,
    native_metadata_hash: HashOut<F>,
    native_initial_jump: JumpState<F>,
    seed: mpsc::Receiver<LaneSeed>,
    store: &'a TxStore,
    stats: &'a FallbackStats,
    /// Light only: the leaf-proof window stays at 1 until this many steps
    /// have been scheduled (the heavy path is still proving); `u64::MAX` for
    /// heavy. Their hardcoded 3 generalized to the parsed heavy-chunk count.
    overlap_start_step: u64,
}

impl<'a> PathCtx<'a> {
    fn max_in_flight(&self, current_step: u64) -> usize {
        if self.path == TxPath::Light && current_step >= self.overlap_start_step {
            LIGHT_TX_PROOF_WINDOW
        } else {
            1
        }
    }

    fn chain_parts(
        &self,
        stage: &'a PreChainCircuits,
    ) -> (&'a BlockTxChainTarget, &'a CircuitData<F, C, D>, &'a Proof) {
        match self.path {
            TxPath::Heavy => (
                &stage.heavy_chain_target,
                &stage.heavy_chain_data,
                &stage.dummy_heavy_proof,
            ),
            TxPath::Light => (
                &stage.light_chain_target,
                &stage.light_chain_data,
                &stage.dummy_light_proof,
            ),
        }
    }

    /// Generates `step`'s partition witness from the authoritative `old_jump`
    /// and extracts the ground-truth `new_jump` the next witness chains on.
    /// Also cross-checks the native prediction (diagnostic only). `Err` means
    /// a generator failure — the caller re-proves the tail from proof-derived
    /// inputs.
    fn generate(
        &self,
        step: usize,
        old_jump: JumpState<F>,
    ) -> Result<(PartitionWitness<'a, F>, JumpState<F>), String> {
        let planned = &self.steps[step];
        let txs = self.store.take_blocking(planned.chunk_index);
        let block_tx = BlockTx {
            created_at: self.created_at,
            state_metadata_hash: self.native_metadata_hash,
            old_jump,
            txs,
        };
        let result = BlockTxCircuit::generate_witness(&block_tx, self.tx_target)
            .map_err(|error| format!("witness generation failed ({error:?})"))
            .and_then(|witness| {
                generate_partial_witness::<F, C, D>(
                    witness,
                    &self.tx_data.prover_only,
                    &self.tx_data.common,
                )
                .map_err(|error| format!("witness generators failed ({error:?})"))
            });
        self.store.put_back(planned.chunk_index, block_tx.txs);
        let witness = result?;
        let new_jump = jump_from_witness(&witness, &self.tx_target.new_jump);
        // Tripwire: log a native prediction divergence at its first step (once
        // the chains disagree, every later planned value differs too — only
        // the first divergence is news). Nothing is re-proved: the extracted
        // state is what the chain threads on.
        if jump_state_diff(&planned.old_jump, &old_jump).is_empty() {
            let diff = jump_state_diff(&planned.new_jump, &new_jump);
            if !diff.is_empty() {
                eprintln!(
                    "[fallback] native jump prediction diverged from chunk #{} witness ({:?} \
                     step #{step}); the witness-extracted state threads the chain — no \
                     re-proving needed\n{diff}",
                    planned.chunk_index, self.path
                );
            }
        }
        Ok((witness, new_jump))
    }

    /// Sequential fallback: full leaf prove with proof-derived inputs. If
    /// even this fails, the fixture cannot be proved by any schedule — die
    /// like the sequential prover would.
    fn reprove_leaf(&self, step: usize, expected_jump: &JumpState<F>, seed: &LaneSeed) -> Proof {
        self.stats.reproved_leaves.fetch_add(1, Ordering::Relaxed);
        let chunk_index = self.steps[step].chunk_index;
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
                "{:?} chain step #{step}: leaf re-prove with proof-derived inputs failed — \
                 fixture unprovable: {error:?}",
                self.path
            )
        })
    }
}

/// Proves a fully generated partition witness. `Err` feeds the fallback (the
/// path re-proves the step from proof-derived inputs).
fn prove_tx_witness(
    path: TxPath,
    chunk_index: usize,
    tx_data: &CircuitData<F, C, D>,
    witness: PartitionWitness<'_, F>,
) -> Result<Proof, String> {
    let proof = prove_with_partition_witness::<F, C, D>(
        &tx_data.prover_only,
        &tx_data.common,
        witness,
        &mut TimingTree::default(),
    )
    .map_err(|error| format!("chunk #{chunk_index} proof failed on {path:?} path: {error:?}"))?;
    #[cfg(debug_assertions)]
    tx_data
        .verify(proof.clone())
        .expect("transaction proof self-check failed");
    Ok(proof)
}

/// The fold pipeline's state once the proof-derived seed has arrived. The
/// path is the correctness authority for its chain: `expected_jump` tracks
/// the true jump state (seed for step 0, then each folded proof's `new_jump`)
/// and a candidate folds only if its `old_jump` extends it — the same
/// equation the chain circuit enforces (`JumpStateTarget::connect(block.jump,
/// current_tx.old_jump)`).
struct FoldState<'scope, 'env> {
    chain: Option<std::thread::ScopedJoinHandle<'scope, Proof>>,
    expected_jump: JumpState<F>,
    next_fold_step: usize,
    seed: Arc<LaneSeed>,
    chain_target: &'env BlockTxChainTarget,
    chain_data: &'env CircuitData<F, C, D>,
    dummy: &'env Proof,
}

/// Validates the proof-derived seed against the native values the leaves were
/// built from. A divergence poisons the whole path (every optimistic witness
/// used wrong inputs): sequential re-prove from step 0.
fn attach_seed<'scope, 'env>(
    ctx: &PathCtx<'env>,
    seed: LaneSeed,
    fallback_from: &mut Option<usize>,
) -> FoldState<'scope, 'env> {
    let stage = ctx.circuits.pre_chains();
    let (chain_target, chain_data, dummy) = ctx.chain_parts(stage);
    let mut diffs = jump_state_diff(&ctx.native_initial_jump, &seed.initial_jump);
    if ctx.native_metadata_hash != seed.state_metadata_hash {
        diffs.push_str(&format!(
            "  state_metadata_hash: native {:?} != proof {:?}\n",
            ctx.native_metadata_hash, seed.state_metadata_hash
        ));
    }
    if !diffs.is_empty() {
        eprintln!(
            "[fallback] {:?} path: native pre-execution values diverged from the pre proof; \
             re-proving every leaf of this path from proof-derived inputs\n{diffs}",
            ctx.path
        );
        *fallback_from = Some(0);
    }
    FoldState {
        chain: None,
        expected_jump: seed.initial_jump,
        next_fold_step: 0,
        seed: Arc::new(seed),
        chain_target,
        chain_data,
        dummy,
    }
}

/// One chain fold, run on a trailing thread. If the chain circuit still
/// rejects a channel candidate (its witness generation fails on anything the
/// old_jump check does not cover), re-prove the leaf from the authoritative
/// inputs and retry once; a rejection of a re-proved leaf is a real bug.
struct FoldJob<'env> {
    path: TxPath,
    chain_target: &'env BlockTxChainTarget,
    chain_data: &'env CircuitData<F, C, D>,
    dummy: &'env Proof,
    chain_step: u64,
    previous: Option<Proof>,
    seed: Arc<LaneSeed>,
    tx_proof: Proof,
    // Retry context (sequential re-prove of this step's leaf).
    tx_target: &'env BlockTxTarget,
    tx_data: &'env CircuitData<F, C, D>,
    store: &'env TxStore,
    stats: &'env FallbackStats,
    chunk_index: usize,
    created_at: i64,
    expected_leaf_jump: JumpState<F>,
}

fn run_fold(job: FoldJob<'_>) -> Proof {
    let fold = |leaf: &Proof| {
        BlockTxChainCircuit::prove(
            job.chain_target,
            job.chain_data,
            job.chain_step,
            job.previous.as_ref().unwrap_or(&job.seed.base),
            job.dummy,
            leaf,
        )
    };
    match fold(&job.tx_proof) {
        Ok(next) => next,
        Err(error) => {
            eprintln!(
                "[fallback] {:?} chain step #{} rejected its leaf proof ({error:?}); \
                 re-proving the leaf and retrying once",
                job.path, job.chain_step
            );
            job.stats.reproved_leaves.fetch_add(1, Ordering::Relaxed);
            let txs = job.store.take_blocking(job.chunk_index);
            let block_tx = BlockTx {
                created_at: job.created_at,
                state_metadata_hash: job.seed.state_metadata_hash,
                old_jump: job.expected_leaf_jump,
                txs,
            };
            let result = BlockTxCircuit::prove(job.tx_data, &block_tx, job.tx_target);
            job.store.put_back(job.chunk_index, block_tx.txs);
            let reproved = result.unwrap_or_else(|error| {
                panic!(
                    "{:?} chain step #{}: leaf re-prove with proof-derived inputs failed — \
                     fixture unprovable: {error:?}",
                    job.path, job.chain_step
                )
            });
            fold(&reproved).unwrap_or_else(|error| {
                panic!(
                    "{:?} block transaction chain step #{} failed after leaf re-prove: {error:?}",
                    job.path, job.chain_step
                )
            })
        }
    }
}

/// Folds one completed leaf candidate: check that its `old_jump` extends the
/// proof-derived chain, wait for the previous fold, then fold on a trailing
/// thread. A failed prove or a non-extending candidate poisons the path from
/// that step (`fallback_from`): the remaining steps are re-proved
/// sequentially and any later optimistic proofs are discarded.
fn fold_candidate<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    ctx: &PathCtx<'env>,
    fs: &mut FoldState<'scope, 'env>,
    step: usize,
    result: Result<Proof, String>,
    fallback_from: &mut Option<usize>,
) {
    debug_assert_eq!(step, fs.next_fold_step, "chain folds must run in order");
    let tx_proof = match result {
        Ok(proof) => proof,
        Err(error) => {
            eprintln!(
                "[fallback] {:?} chain step #{step}: leaf prove failed ({error}); re-proving \
                 this path sequentially from here",
                ctx.path
            );
            *fallback_from = Some(step);
            return;
        }
    };
    let proved = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs);
    let diff = jump_state_diff(&fs.expected_jump, &proved.old_jump);
    if !diff.is_empty() {
        eprintln!(
            "[fallback] {:?} chain step #{step}: leaf proof's old_jump does not extend the \
             chain; re-proving this path sequentially from here\n{diff}",
            ctx.path
        );
        *fallback_from = Some(step);
        return;
    }
    let previous = fs.chain.take().map(wait_fold);
    let job = FoldJob {
        path: ctx.path,
        chain_target: fs.chain_target,
        chain_data: fs.chain_data,
        dummy: fs.dummy,
        chain_step: step as u64,
        previous,
        seed: Arc::clone(&fs.seed),
        tx_proof,
        tx_target: ctx.tx_target,
        tx_data: ctx.tx_data,
        store: ctx.store,
        stats: ctx.stats,
        chunk_index: ctx.steps[step].chunk_index,
        created_at: ctx.created_at,
        expected_leaf_jump: fs.expected_jump,
    };
    // The folded proof's new_jump is the true chain state — by circuit
    // construction, not native derivation.
    fs.expected_jump = proved.new_jump;
    fs.next_fold_step = step + 1;
    let handle = std::thread::Builder::new()
        .name(format!("{:?}-chain-step-{step}", ctx.path))
        .stack_size(WORKER_STACK_BYTES)
        .spawn_scoped(scope, move || run_fold(job))
        .expect("chain step pipeline thread must start");
    fs.chain = Some(handle);
}

fn drain_ready<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    ctx: &PathCtx<'env>,
    fs: &mut FoldState<'scope, 'env>,
    ready: &mut VecDeque<(usize, Result<Proof, String>)>,
    fallback_from: &mut Option<usize>,
) {
    while fallback_from.is_none() {
        let Some((step, result)) = ready.pop_front() else {
            return;
        };
        fold_candidate(scope, ctx, fs, step, result, fallback_from);
    }
}

/// The pre-A3 correct-by-construction schedule: sequentially re-prove every
/// unfolded step from proof-derived inputs, folding inline. Worst case (a
/// native pre divergence) this is the whole path — a slow pass, never a zero.
fn reprove_tail(ctx: &PathCtx<'_>, fs: FoldState<'_, '_>) -> Proof {
    let FoldState {
        chain,
        mut expected_jump,
        next_fold_step,
        seed,
        chain_target,
        chain_data,
        dummy,
    } = fs;
    let mut chain = chain.map(wait_fold);
    for step in next_fold_step..ctx.steps.len() {
        let leaf = ctx.reprove_leaf(step, &expected_jump, &seed);
        let folded = BlockTxChainCircuit::prove(
            chain_target,
            chain_data,
            step as u64,
            chain.as_ref().unwrap_or(&seed.base),
            dummy,
            &leaf,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{:?} block transaction chain step #{step} failed on a re-proved leaf: {error:?}",
                ctx.path
            )
        });
        expected_jump = BlockTxWitness::from_public_inputs(&leaf.public_inputs).new_jump;
        chain = Some(folded);
    }
    chain.unwrap_or_else(|| (*seed).base.clone())
}

/// One transaction path: the promoted windowed schedule with the fallback net
/// grafted on.
///
/// ```text
///   witness(0) from native pre outputs           [t≈0 — no pre proof needed]
///   loop: fold any seeded candidates → spawn prove(s) → generate
///         witness(s+1) (extraction-chained jumps) → window-join oldest prove
///   drain: seed (blocking) → fold survivors → happy: wait trailing fold
///                                           → poisoned: sequential re-prove
/// ```
fn run_path(ctx: PathCtx<'_>) -> Proof {
    // An absent path folds nothing: its chain proof is its base proof.
    if ctx.steps.is_empty() {
        let seed = ctx.seed.recv().unwrap_or_else(|_| {
            panic!(
                "pre-execution stage failed before the {:?} chain base proof arrived",
                ctx.path
            )
        });
        return seed.base;
    }

    let mut fallback_from: Option<usize> = None;
    let mut authority = ctx.native_initial_jump;
    let mut current = match ctx.generate(0, authority) {
        Ok((witness, new_jump)) => {
            authority = new_jump;
            Some((0usize, witness))
        }
        Err(error) => {
            eprintln!(
                "[fallback] {:?} step #0 {error}; re-proving this path sequentially",
                ctx.path
            );
            fallback_from = Some(0);
            None
        }
    };

    std::thread::scope(|scope| {
        let mut fold_state: Option<FoldState<'_, '_>> = None;
        let mut ready: VecDeque<(usize, Result<Proof, String>)> = VecDeque::new();
        let mut in_flight: VecDeque<(
            usize,
            std::thread::ScopedJoinHandle<'_, Result<Proof, String>>,
        )> = VecDeque::new();

        // Pipeline phase (their windowed schedule, verbatim cadence).
        while fallback_from.is_none() {
            let Some((step, witness)) = current.take() else {
                break;
            };

            // Fold completed leaves as soon as the proof-derived seed exists;
            // until then they queue (completed proofs are small).
            if fold_state.is_none() {
                if let Ok(seed) = ctx.seed.try_recv() {
                    fold_state = Some(attach_seed(&ctx, seed, &mut fallback_from));
                }
            }
            if let Some(fs) = fold_state.as_mut() {
                drain_ready(scope, &ctx, fs, &mut ready, &mut fallback_from);
            }
            if fallback_from.is_some() {
                // The path is poisoned at/before this witness's step: stop
                // scheduling; the drain below re-proves the tail.
                break;
            }

            let proof_handle = std::thread::Builder::new()
                .name(format!("{:?}-tx-proof-{step}", ctx.path))
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, {
                    let path = ctx.path;
                    let tx_data = ctx.tx_data;
                    let chunk_index = ctx.steps[step].chunk_index;
                    move || prove_tx_witness(path, chunk_index, tx_data, witness)
                })
                .expect("transaction proof pipeline thread must start");
            in_flight.push_back((step, proof_handle));

            // Double buffer: generate the next witness while the proof runs.
            let next_step = step + 1;
            if next_step < ctx.steps.len() {
                match ctx.generate(next_step, authority) {
                    Ok((witness, new_jump)) => {
                        authority = new_jump;
                        current = Some((next_step, witness));
                    }
                    Err(error) => {
                        eprintln!(
                            "[fallback] {:?} step #{next_step} {error}; re-proving this path \
                             sequentially from that step",
                            ctx.path
                        );
                        // Earlier proofs still fold in the drain below.
                        fallback_from = Some(next_step);
                    }
                }
            }

            if in_flight.len() >= ctx.max_in_flight(step as u64) {
                let (proof_step, handle) = in_flight
                    .pop_front()
                    .expect("transaction proof window must not be empty");
                let result = handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                ready.push_back((proof_step, result));
            }
        }

        // Drain phase: the seed is mandatory now (the pre worker always sends
        // it, or its panic surfaces at join).
        if fold_state.is_none() {
            match ctx.seed.recv() {
                Ok(seed) => fold_state = Some(attach_seed(&ctx, seed, &mut fallback_from)),
                Err(_) => panic!(
                    "pre-execution stage failed before the {:?} chain base proof arrived",
                    ctx.path
                ),
            }
        }
        for (step, handle) in in_flight.drain(..) {
            let result = handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            ready.push_back((step, result));
        }
        let mut fs = fold_state.take().expect("fold state exists after seeding");
        drain_ready(scope, &ctx, &mut fs, &mut ready, &mut fallback_from);
        // Anything left in `ready` is beyond a poison point: wasted
        // optimistic work, discarded.
        ready.clear();

        if fallback_from.is_some() {
            reprove_tail(&ctx, fs)
        } else {
            fs.chain
                .map(wait_fold)
                .expect("transaction path must produce a chain proof")
        }
    })
}

/// Hybrid proof pipeline: the promoted windowed per-path schedule, started
/// from natively derived pre-execution outputs so nothing waits on the pre
/// proof or on circuit building:
///
/// ```text
///   native plan (pre outputs, jump predictions)                  [~0s]
///     ├─ heavy path: witness↔prove pipeline + trailing chain folds ─┐
///     ├─ light path: same, window 2 once the heavy path is done ────┼─→ block proof
///     └─ pre worker: bg circuits → pre proof → seeds (proof-derived │   (bg-built
///        values + no-prove base witnesses) → paths validate on fold ┘    circuit)
/// ```
///
/// Same circuits, same witnesses, same public inputs, same proof encoding as
/// the sequential order — only the schedule changed. Every leaf proof is
/// validated against proof-derived values before folding; any divergence or
/// failure triggers sequential re-proving of the affected path's tail instead
/// of failing the fixture.
pub fn prove_block(block: Block<F>, circuits: &Circuits) -> Proof {
    prove_block_with(block, circuits, |_| {}).0
}

/// `plan_tweak` is a test seam: it may corrupt the native plan to exercise
/// the divergence fallbacks. Production passes a no-op.
fn prove_block_with(
    mut block: Block<F>,
    circuits: &Circuits,
    plan_tweak: impl FnOnce(&mut NativePlan),
) -> (Proof, FallbackStats) {
    let mut timer = StageTimer::new();

    let (mut plan, chunks) = plan_paths(&mut block);
    plan_tweak(&mut plan);
    timer.mark("native plan");

    let stats = FallbackStats::default();
    let store = TxStore::new(chunks);
    let (heavy_seed_send, heavy_seed_recv) = mpsc::channel::<LaneSeed>();
    let (light_seed_send, light_seed_recv) = mpsc::channel::<LaneSeed>();

    let block_ref = &block;
    let plan_ref = &plan;
    let store_ref = &store;
    let stats_ref = &stats;

    let (pre_proof, light_chain_proof, heavy_chain_proof) = std::thread::scope(|scope| {
        // Pre-execution worker: waits for the background pre+chain circuit
        // build, proves the pre proof, then seeds both paths with
        // PROOF-derived values and their base-proof witnesses (cheap
        // constructions from the embedded dummy proofs — no proving). The
        // paths need the seed only at fold time, at least one leaf prove
        // away.
        let pre_worker = std::thread::Builder::new()
            .name("pre-exec".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let stage = circuits.pre_chains();
                let pre_proof = BlockPreExecutionCircuit::prove(
                    &stage.pre_data,
                    &BlockPreExec::from_block(block_ref),
                    &stage.pre_target,
                )
                .expect("block pre-execution proof failed");
                let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
                if let Some(diff) = native_pre_divergence(&plan_ref.pre, &pre_output) {
                    eprintln!(
                        "[fallback] native pre-execution outputs diverged from the pre proof; \
                         each path re-proves its leaves from proof-derived inputs\n{diff}"
                    );
                }
                let initial_jump = JumpState::initial(
                    pre_output.new_state_root,
                    block_ref.old_account_delta_tree_root,
                );
                let state_metadata_hash = pre_output.new_state_metadata.hash();
                let heavy_base = cyclic_base_witness(
                    &stage.dummy_heavy_proof,
                    block_ref.block_number,
                    block_ref.created_at,
                    pre_output.new_state_root,
                    pre_output.new_validium_root,
                    block_ref.old_account_delta_tree_root,
                );
                let light_base = cyclic_base_witness(
                    &stage.dummy_light_proof,
                    block_ref.block_number,
                    block_ref.created_at,
                    pre_output.new_state_root,
                    pre_output.new_validium_root,
                    block_ref.old_account_delta_tree_root,
                );
                // A closed receiver means a path worker already panicked; that
                // panic surfaces at its join, so don't double-panic here.
                let _ = heavy_seed_send.send(LaneSeed {
                    base: heavy_base,
                    initial_jump,
                    state_metadata_hash,
                });
                let _ = light_seed_send.send(LaneSeed {
                    base: light_base,
                    initial_jump,
                    state_metadata_hash,
                });
                pre_proof
            })
            .expect("cannot start pre-execution worker");

        let heavy_worker = std::thread::Builder::new()
            .name("heavy-path".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, move || {
                run_path(PathCtx {
                    path: TxPath::Heavy,
                    steps: &plan_ref.heavy,
                    tx_target: &circuits.tx.heavy_tx_target,
                    tx_data: &circuits.tx.heavy_tx_data,
                    circuits,
                    created_at: block_ref.created_at,
                    native_metadata_hash: plan_ref.pre.state_metadata_hash,
                    native_initial_jump: plan_ref.initial_jump,
                    seed: heavy_seed_recv,
                    store: store_ref,
                    stats: stats_ref,
                    overlap_start_step: u64::MAX,
                })
            })
            .expect("cannot start heavy path worker");
        let light_worker = std::thread::Builder::new()
            .name("light-path".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn_scoped(scope, move || {
                run_path(PathCtx {
                    path: TxPath::Light,
                    steps: &plan_ref.light,
                    tx_target: &circuits.tx.light_tx_target,
                    tx_data: &circuits.tx.light_tx_data,
                    circuits,
                    created_at: block_ref.created_at,
                    native_metadata_hash: plan_ref.pre.state_metadata_hash,
                    native_initial_jump: plan_ref.initial_jump,
                    seed: light_seed_recv,
                    store: store_ref,
                    stats: stats_ref,
                    overlap_start_step: plan_ref.heavy.len() as u64,
                })
            })
            .expect("cannot start light path worker");

        let heavy_chain_proof = heavy_worker.join().expect("heavy path worker panicked");
        timer.mark("heavy path done");
        let light_chain_proof = light_worker.join().expect("light path worker panicked");
        timer.mark("light path done");
        let pre_proof = pre_worker.join().expect("pre-execution worker panicked");
        (pre_proof, light_chain_proof, heavy_chain_proof)
    });

    // The chunk Vecs live in the tx store (the pipelines only borrowed them);
    // the final block witness only asserts the chunk list is non-empty — the
    // tx data itself is already folded into the chain proofs — so restore a
    // sentinel.
    block.tx_chunks.push(Vec::new());

    // The background build had the whole leaf phase to finish; this wait is
    // expected to be ~zero (the mark makes any residual wait visible).
    let block_circuits = circuits.block();
    timer.mark("await block circuit");
    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    let proof = BlockCircuit::prove(
        &block_circuits.block_target,
        &block_circuits.block_data,
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
                let chunk_count = block.tx_chunks.len();
                // All-empty block: the pre-execution output state root is the
                // block's (unchanged) new state root, and every chunk is pure
                // padding, so every jump transition must be the identity.
                let (plan, chunks) = plan_paths(&mut block);
                assert_eq!(plan.pre.new_state_root, block.new_state_root);
                assert_eq!(
                    plan.pre.state_metadata_hash,
                    plan.pre.new_state_metadata.hash()
                );
                assert_eq!(chunks.len(), chunk_count, "txs move into the store");
                assert!(block.tx_chunks.is_empty(), "chunks move out of the block");
                assert_eq!(plan.heavy.len(), 3);
                assert_eq!(plan.light.len(), 49);
                for leaf in plan.heavy.iter().chain(&plan.light) {
                    assert_eq!(leaf.old_jump.to_vec(), plan.initial_jump.to_vec());
                    assert_eq!(leaf.new_jump.to_vec(), plan.initial_jump.to_vec());
                }
                // Per-path fixture order, and the two paths partition the
                // chunk indices exactly.
                assert!(plan.heavy.iter().map(|leaf| leaf.chunk_index).is_sorted());
                assert!(plan.light.iter().map(|leaf| leaf.chunk_index).is_sorted());
                let mut all_chunks = plan
                    .heavy
                    .iter()
                    .chain(&plan.light)
                    .map(|leaf| leaf.chunk_index)
                    .collect::<Vec<_>>();
                all_chunks.sort_unstable();
                assert_eq!(all_chunks, (0..chunk_count).collect::<Vec<_>>());
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

    /// End-to-end fallback drills on a shrunken (2 heavy + 2 light chunk)
    /// all-empty block. Debug builds skip them (full 2^16 proves are
    /// minutes-slow unoptimized); run with `cargo test -p bench --release`.
    /// The circuits build once (shared) and the drills serialize on a lock so
    /// their proving working sets don't stack.
    #[cfg(not(debug_assertions))]
    mod fallback_drills {
        use std::sync::LazyLock;

        use super::*;

        static CIRCUITS: LazyLock<Circuits> = LazyLock::new(Circuits::new);
        static DRILL_SLOT: Mutex<()> = Mutex::new(());

        fn shrunken_block() -> Block<F> {
            Block::<F>::from_json_with_empty_txs(
                include_bytes!("../bench_test.json"),
                HEAVY_TX_PER_PROOF,
                LIGHT_TX_PER_PROOF,
                2 * HEAVY_TX_PER_PROOF,
                2 * LIGHT_TX_PER_PROOF,
            )
            .expect("public fixture must parse")
        }

        /// Wrong native pre-execution outputs (the one place the optimistic
        /// pipeline trusts a native derivation): every witness of both paths
        /// chained from a wrong initial jump, the proof-derived seed exposes
        /// it at fold time, and both paths must discard their optimistic work
        /// and re-prove every leaf from proof-derived inputs — the
        /// zero-score → slow-pass net.
        #[test]
        fn native_pre_divergence_reproves_both_paths() {
            std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(|| {
                    let _slot = DRILL_SLOT.lock().expect("drill slot poisoned");
                    let block = shrunken_block();
                    let started = std::time::Instant::now();
                    let (proof, stats) = prove_block_with(block, &CIRCUITS, |plan| {
                        use plonky2::field::types::Field;
                        // A consistently wrong native pre state root: the
                        // initial jump and every (identity) transition planned
                        // from it. Padding chunks prove fine with any jump, so
                        // the divergence only becomes visible against the pre
                        // proof's outputs.
                        plan.initial_jump.prev_new_state_root.elements[0] += F::ONE;
                        let bad = plan.initial_jump;
                        for leaf in plan.heavy.iter_mut().chain(plan.light.iter_mut()) {
                            leaf.old_jump = bad;
                            leaf.new_jump = bad;
                        }
                    });
                    eprintln!(
                        "[drill] full re-prove of both paths took {:.3}s wall",
                        started.elapsed().as_secs_f64()
                    );
                    assert_eq!(
                        stats.reproved_leaves.load(Ordering::Relaxed),
                        4,
                        "both paths must re-prove all of their leaves"
                    );
                    CIRCUITS
                        .block()
                        .block_data
                        .verify(proof)
                        .expect("fallback run must still produce a verifying block proof");
                })
                .expect("fallback test thread must start")
                .join()
                .expect("fallback test thread must finish");
        }

        /// Wrong native per-step jump predictions cost nothing: the
        /// witness-extracted (ground truth) state threads the chain, so the
        /// run completes without a single re-prove and the proof verifies.
        #[test]
        fn native_prediction_divergence_is_absorbed_without_reproving() {
            std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(|| {
                    let _slot = DRILL_SLOT.lock().expect("drill slot poisoned");
                    let block = shrunken_block();
                    let (proof, stats) = prove_block_with(block, &CIRCUITS, |plan| {
                        use plonky2::field::types::Field;
                        // Light step 0: wrong predicted new_jump — detected at
                        // witness extraction (tripwire), nothing depends on it.
                        plan.light[0].new_jump.tx_count += F::ONE;
                        // Heavy step 1: stale predicted old_jump — equally
                        // harmless, the authority chain never reads it.
                        plan.heavy[1].old_jump.prev_new_state_root.elements[0] += F::ONE;
                    });
                    assert_eq!(
                        stats.reproved_leaves.load(Ordering::Relaxed),
                        0,
                        "prediction divergences must not cost any re-proving"
                    );
                    CIRCUITS
                        .block()
                        .block_data
                        .verify(proof)
                        .expect("run with diverged predictions must still verify");
                })
                .expect("fallback test thread must start")
                .join()
                .expect("fallback test thread must finish");
        }
    }
}
