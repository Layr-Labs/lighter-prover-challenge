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
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::generator::{ParallelWitnessGuard, PendingPartitionWitness};
#[cfg(test)]
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;

use crate::api::{Circuits, PROVER_THREAD_STACK_BYTES, Proof};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxPath {
    Heavy,
    Light,
}

const LIGHT_TX_PROOF_WINDOW: usize = 3;
// Keep only the first light proof serial while the fixed three-chunk heavy path ramps up.
// Starting bounded light overlap at step 2 exposes one more proof to idle cores without
// increasing the established three-proof memory window.
const LIGHT_TX_PROOF_OVERLAP_START_STEP: u64 = 2;

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
    circuits: &Circuits,
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
    let (tx_data, tx_target, chain_data, chain_target, dummy_proof) = match path {
        TxPath::Light => (
            &circuits.light_tx_data,
            &circuits.light_tx_target,
            &circuits.light_chain_data,
            &circuits.light_chain_target,
            &circuits.dummy_light_proof,
        ),
        TxPath::Heavy => (
            &circuits.heavy_tx_data,
            &circuits.heavy_tx_target,
            &circuits.heavy_chain_data,
            &circuits.heavy_chain_target,
            &circuits.dummy_heavy_proof,
        ),
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
          // Keep the predecessor-linked chain pipeline alive while draining buffered
          // transaction proofs. Each thread can prepare its proof-independent witness
          // inputs immediately, then waits for the prior chain proof only at the feed.
          // This removes synchronous witness preparation from the light-path tail.
          plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
          while let Some((chain_step, proof_handle)) = in_flight.pop_front() {
              let tx_proof = proof_handle
                  .join()
                  .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
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
                  .expect("chain step drain thread must start");
              chain = Some(ChainState::InFlight(handle));
          }
        let chain_proof = chain
            .map(ChainState::wait)
            .expect("transaction path must produce a chain proof");
        plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
        chain_proof
    })
}

pub fn prove_block(mut block: Block<F>, mut circuits: Circuits) -> Proof {
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

    let (light_chain_proof, heavy_chain_proof, block_target, block_data, block_pending) = {
        // The pipeline only ever reads the circuits; the borrow ends with this
        // block so the finished extensions can be released below.
        let circuits = &circuits;
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
                    )
                })
                .expect("heavy transaction chain thread must start");
            let block_ref = &block;
            let pre_proof_ref = &pre_proof;
            let block_circuit_handle = std::thread::Builder::new()
                .name("block-circuit-build".into())
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    let (block_target, block_data) = circuits.build_block_circuit();
                    let block_data: &'static CircuitData<F, C, D> =
                        Box::leak(Box::new(block_data));
                    let early = BlockCircuit::witness_inputs_early(
                        &block_target,
                        block_ref,
                        pre_proof_ref,
                    )
                    .expect("final block early witness inputs failed");
                    let mut pending = PendingPartitionWitness::start(
                        early,
                        &block_data.prover_only,
                        &block_data.common,
                    )
                    .expect("final block early witness phase failed");
                    let heavy_chain_proof = heavy_handle_outer
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
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
            let light_chain_proof = prove_path(
                TxPath::Light,
                light_chunks,
                circuits,
                block.block_number,
                block.created_at,
                block.old_account_delta_tree_root,
                &pre_output,
                state_metadata_hash,
            );
            let (block_target, block_data, block_pending, heavy_chain_proof) =
                block_circuit_handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
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
    block_pending
        .feed(
            BlockCircuit::witness_inputs_light_chain(&block_target, light_chain_input)
                .expect("final block light-chain witness inputs failed"),
        )
        .expect("final block light-chain witness feed failed");
    let _ = heavy_chain_input;
    let final_proof = BlockCircuit::prove_prepared(block_pending, block_data)
        .expect("final block proof failed");
    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
    final_proof
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::api::{
        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
    };

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
