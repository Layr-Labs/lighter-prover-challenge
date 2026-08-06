// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::VecDeque;

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, JumpState, JumpStateTarget};
use circuit::block_tx_chain_constraints::{
    BlockTxChainCircuit, BlockTxChainTarget, Circuit as _, cyclic_base_witness,
};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;

use crate::api::{Circuits, PROVER_THREAD_STACK_BYTES, Proof};

/// Transaction proofs allowed to be in flight at once within a single transaction path.
///
/// Transaction proofs are independent of one another, so more than one can run at a time; the
/// witnesses feeding them are not, and are still produced one chunk at a time on the path thread.
/// Each in-flight proof owns a partition witness and the prover buffers derived from it, so this
/// constant is what bounds peak memory. A depth of 1 reproduces the previous serial behaviour.
const TX_PROOF_QUEUE_DEPTH: usize = 2;

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
    previous_proof: Option<&Proof>,
    base_proof: &Proof,
    dummy_proof: &Proof,
    tx_proof: &Proof,
) -> Proof {
    BlockTxChainCircuit::prove(
        chain_target,
        chain_data,
        chain_step,
        previous_proof.unwrap_or(base_proof),
        dummy_proof,
        tx_proof,
    )
    .unwrap_or_else(|error| {
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
        let mut chain: Option<ChainState<'_>> = None;
        let mut in_flight: VecDeque<(u64, std::thread::ScopedJoinHandle<'_, Proof>)> =
            VecDeque::with_capacity(TX_PROOF_QUEUE_DEPTH);
        let mut current_step = 0u64;

        let spawn_chain_step = |chain_step: u64, tx_proof: Proof, previous_proof: Option<Proof>| {
            std::thread::Builder::new()
                .name(format!("{path:?}-chain-step-{chain_step}"))
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    chain_step_proof(
                        path,
                        chain_target,
                        chain_data,
                        chain_step,
                        previous_proof.as_ref(),
                        base,
                        dummy_proof,
                        &tx_proof,
                    )
                })
                .expect("chain step pipeline thread must start")
        };

        loop {
            // Retire the oldest transaction proof before the queue can grow past its bound, so no
            // more than `TX_PROOF_QUEUE_DEPTH` proofs are ever alive at once. Popping the front
            // keeps chain steps fed in chunk order.
            if in_flight.len() == TX_PROOF_QUEUE_DEPTH {
                let (chain_step, handle) = in_flight
                    .pop_front()
                    .expect("full transaction proof queue must have an oldest entry");
                let tx_proof = handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                let previous_proof = chain.take().map(ChainState::wait);
                chain = Some(ChainState::InFlight(spawn_chain_step(
                    chain_step,
                    tx_proof,
                    previous_proof,
                )));
            }

            let witness = current_witness;
            let proof_handle = std::thread::Builder::new()
                .name(format!("{path:?}-tx-proof-{current_step}"))
                .stack_size(PROVER_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    prove_tx_witness(path, current_chunk_index, tx_data, witness)
                })
                .expect("transaction proof pipeline thread must start");
            in_flight.push_back((current_step, proof_handle));
            current_step += 1;

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

            match next_witness {
                Some((chunk_index, witness)) => {
                    current_chunk_index = chunk_index;
                    current_witness = witness;
                }
                None => break,
            }
        }

        while let Some((chain_step, handle)) = in_flight.pop_front() {
            let tx_proof = handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            let previous_proof = chain.take().map(ChainState::wait);
            chain = Some(if in_flight.is_empty() {
                ChainState::Ready(chain_step_proof(
                    path,
                    chain_target,
                    chain_data,
                    chain_step,
                    previous_proof.as_ref(),
                    base,
                    dummy_proof,
                    &tx_proof,
                ))
            } else {
                ChainState::InFlight(spawn_chain_step(chain_step, tx_proof, previous_proof))
            });
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

    let (light_chain_proof, heavy_chain_proof) = std::thread::scope(|scope| {
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
        (light_chain_proof, heavy_chain_proof)
    });

    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    BlockCircuit::prove(
        &circuits.block_target,
        &circuits.block_data,
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
}
