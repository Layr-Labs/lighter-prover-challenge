// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, JumpState, JumpStateTarget};
use circuit::block_tx_chain_constraints::{
    cyclic_base_witness, BlockTxChainCircuit, BlockTxChainTarget,
};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::generator::{
    generate_partial_witness, ParallelWitnessGuard, PendingPartitionWitness,
};
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

const LIGHT_TX_PROOF_WINDOW: usize = 2;
// Keep the initial light proofs serial while the fixed three-chunk heavy path is active.
const LIGHT_TX_PROOF_OVERLAP_START_STEP: u64 = 3;

fn tx_proof_worker_count(path: TxPath) -> usize {
    match path {
        TxPath::Heavy => 1,
        TxPath::Light => LIGHT_TX_PROOF_WINDOW,
    }
}

fn max_tx_proofs_in_flight(path: TxPath, step: u64) -> usize {
    if path == TxPath::Light && step >= LIGHT_TX_PROOF_OVERLAP_START_STEP {
        LIGHT_TX_PROOF_WINDOW
    } else {
        1
    }
}

fn receive_tx_worker_result<'scope, T>(
    expected_step: u64,
    worker_slot: usize,
    result_receivers: &[std::sync::mpsc::Receiver<(u64, T)>],
    worker_handles: &mut [Option<std::thread::ScopedJoinHandle<'scope, ()>>],
    busy_steps: &mut [Option<u64>],
) -> T {
    let (returned_step, result) = result_receivers[worker_slot].recv().unwrap_or_else(|_| {
        let handle = worker_handles[worker_slot]
            .take()
            .expect("transaction proof worker handle must exist");
        match handle.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => {
                panic!("transaction proof worker exited before returning step {expected_step}")
            }
        }
    });
    assert_eq!(busy_steps[worker_slot], Some(expected_step));
    assert_eq!(returned_step, expected_step);
    busy_steps[worker_slot] = None;
    result
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
    let result = (|| {
        // Phase 1: run every generator that does not depend on the previous chain proof while
        // that proof may still be in flight.
        let early_inputs = BlockTxChainCircuit::witness_inputs_early(
            chain_target,
            chain_data,
            chain_step,
            dummy_proof,
            tx_proof,
        )?;
        let mut pending = PendingPartitionWitness::start(
            early_inputs,
            &chain_data.prover_only,
            &chain_data.common,
        )?;

        // Phase 2: wait for the previous chain proof, feed it, and prove.
        let previous_proof = previous.map(ChainState::wait);
        pending.feed(BlockTxChainCircuit::witness_inputs_cyclic(
            chain_target,
            previous_proof.as_ref().unwrap_or(base_proof),
        )?)?;
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
    let partial_witness =
        BlockTxCircuit::generate_witness(&block_tx, tx_target).unwrap_or_else(|error| {
            panic!("{path:?} block transaction chunk #{chunk_index} witness failed: {error:?}")
        });
    let partition_witness =
        generate_partial_witness::<F, C, D>(partial_witness, &tx_data.prover_only, &tx_data.common)
            .unwrap_or_else(|error| {
                panic!(
                    "{path:?} block transaction chunk #{chunk_index} generators failed: {error:?}"
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
        let worker_count = tx_proof_worker_count(path);
        let mut job_senders = Vec::with_capacity(worker_count);
        let mut result_receivers = Vec::with_capacity(worker_count);
        let mut worker_handles = Vec::with_capacity(worker_count);
        for worker_slot in 0..worker_count {
            let (job_sender, job_receiver) =
                std::sync::mpsc::sync_channel::<(u64, usize, PartitionWitness<'_, F>)>(1);
            let (result_sender, result_receiver) = std::sync::mpsc::sync_channel::<(u64, Proof)>(1);
            let handle = std::thread::Builder::new()
                .name(format!("{path:?}-tx-proof-worker-{worker_slot}"))
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    while let Ok((step, chunk_index, witness)) = job_receiver.recv() {
                        let proof = prove_tx_witness(path, chunk_index, tx_data, witness);
                        if result_sender.send((step, proof)).is_err() {
                            break;
                        }
                    }
                })
                .expect("transaction proof worker thread must start");
            job_senders.push(job_sender);
            result_receivers.push(result_receiver);
            worker_handles.push(Some(handle));
        }

        let mut chain: Option<ChainState<'_>> = None;
        let mut pending_tx: Option<(u64, Proof)> = None;
        let mut in_flight = std::collections::VecDeque::new();
        let mut busy_steps = vec![None; worker_count];
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

            let worker_slot = busy_steps
                .iter()
                .position(Option::is_none)
                .expect("transaction proof worker must be available");
            job_senders[worker_slot]
                .send((current_step, current_chunk_index, current_witness))
                .expect("transaction proof worker must accept a job");
            busy_steps[worker_slot] = Some(current_step);

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

            in_flight.push_back((current_step, worker_slot));
            let max_in_flight = max_tx_proofs_in_flight(path, current_step);
            if in_flight.len() >= max_in_flight {
                let (proof_step, worker_slot) = in_flight
                    .pop_front()
                    .expect("transaction proof window must not be empty");
                let tx_proof = receive_tx_worker_result(
                    proof_step,
                    worker_slot,
                    &result_receivers,
                    &mut worker_handles,
                    &mut busy_steps,
                );
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
        while let Some((chain_step, worker_slot)) = in_flight.pop_front() {
            let tx_proof = receive_tx_worker_result(
                chain_step,
                worker_slot,
                &result_receivers,
                &mut worker_handles,
                &mut busy_steps,
            );
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

        assert!(busy_steps.iter().all(Option::is_none));
        drop(job_senders);
        for handle in worker_handles.into_iter().flatten() {
            handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        }
        chain
            .map(ChainState::wait)
            .expect("transaction path must produce a chain proof")
    })
}

pub fn prove_block(mut block: Block<F>, circuits: &Circuits) -> Proof {
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(&block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
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
                        circuits,
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
                circuits,
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
    BlockCircuit::prove(
        &block_target,
        &block_data,
        &block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed")
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
        let tx_proof = prove_tx_witness(TxPath::Light, 0, &circuits.tx_data, witness);

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
            let proof = prove_with_partition_witness::<F, C, D>(
                &circuits.chain_data.prover_only,
                &circuits.chain_data.common,
                witness,
                &mut TimingTree::default(),
            )
            .expect("chain step proof failed");
            let prove_elapsed = prove_start.elapsed();

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

    #[test]
    fn transaction_proof_worker_counts_match_path_windows() {
        assert_eq!(tx_proof_worker_count(TxPath::Heavy), 1);
        assert_eq!(tx_proof_worker_count(TxPath::Light), 2);

        for step in 0..8 {
            assert_eq!(max_tx_proofs_in_flight(TxPath::Heavy, step), 1);
        }
        for step in 0..LIGHT_TX_PROOF_OVERLAP_START_STEP {
            assert_eq!(max_tx_proofs_in_flight(TxPath::Light, step), 1);
        }
        for step in LIGHT_TX_PROOF_OVERLAP_START_STEP..8 {
            assert_eq!(max_tx_proofs_in_flight(TxPath::Light, step), 2);
        }
    }

    #[test]
    fn transaction_proof_worker_slots_preserve_fifo_windows() {
        for path in [TxPath::Heavy, TxPath::Light] {
            let mut busy_steps = vec![None; tx_proof_worker_count(path)];
            let mut in_flight = std::collections::VecDeque::new();
            let mut consumed = Vec::new();

            for step in 0..8 {
                let worker_slot = busy_steps
                    .iter()
                    .position(Option::is_none)
                    .expect("a worker slot must be available");
                busy_steps[worker_slot] = Some(step);
                in_flight.push_back((step, worker_slot));

                if in_flight.len() >= max_tx_proofs_in_flight(path, step) {
                    let (expected_step, worker_slot) = in_flight.pop_front().unwrap();
                    assert_eq!(busy_steps[worker_slot], Some(expected_step));
                    busy_steps[worker_slot] = None;
                    consumed.push(expected_step);
                }
            }

            while let Some((expected_step, worker_slot)) = in_flight.pop_front() {
                assert_eq!(busy_steps[worker_slot], Some(expected_step));
                busy_steps[worker_slot] = None;
                consumed.push(expected_step);
            }

            assert_eq!(consumed, (0..8).collect::<Vec<_>>());
            assert!(busy_steps.iter().all(Option::is_none));
        }
    }

    #[test]
    fn transaction_proof_worker_panic_payload_is_resumed() {
        let panic = std::panic::catch_unwind(|| {
            std::thread::scope(|scope| {
                let (result_sender, result_receiver) =
                    std::sync::mpsc::sync_channel::<(u64, ())>(1);
                let handle = std::thread::Builder::new()
                    .name("panic-propagation-worker".into())
                    .spawn_scoped(scope, move || {
                        let _result_sender = result_sender;
                        std::panic::panic_any("fixed-worker-panic-payload");
                    })
                    .expect("panic propagation worker must start");
                let mut handles = vec![Some(handle)];
                let mut busy_steps = vec![Some(7)];

                receive_tx_worker_result(7, 0, &[result_receiver], &mut handles, &mut busy_steps);
            });
        })
        .expect_err("worker panic must escape the result receiver");

        assert_eq!(
            panic.downcast_ref::<&'static str>(),
            Some(&"fixed-worker-panic-payload")
        );
    }
}
