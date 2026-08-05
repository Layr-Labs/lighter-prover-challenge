// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::tx::Tx;
use circuit::types::config::{C, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::circuit_data::CircuitData;

use crate::api::{Circuits, Proof};

/// Stack size for proving helper threads. Witness generation and recursive
/// proving recurse deeply enough that a default thread stack is unsafe; the
/// orchestration test in this crate already needs an enlarged stack.
const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

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

/// A per-path chain proof that is either finished or still being produced by a
/// pipeline worker thread.
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

/// One cyclic chain recursion step. `previous_proof` is `None` for step zero.
/// The path's base proof is itself a valid proof of the path's dummy circuit,
/// so it fills both step-zero witness roles: the previous-proof slot until a
/// real chain proof exists, and the dummy slot at every step.
fn chain_step_proof(
    path: TxPath,
    chain_target: &BlockTxChainTarget,
    chain_data: &CircuitData<F, C, D>,
    chain_step: u64,
    previous_proof: Option<&Proof>,
    base_proof: &Proof,
    tx_proof: &Proof,
) -> Proof {
    BlockTxChainCircuit::prove(
        chain_target,
        chain_data,
        chain_step,
        previous_proof.unwrap_or(base_proof),
        base_proof,
        tx_proof,
    )
    .unwrap_or_else(|error| {
        panic!("{path:?} block transaction chain step #{chain_step} failed: {error:?}")
    })
}

/// Proves every transaction chunk of one path (heavy or light) and folds each
/// transaction proof into that path's cyclic chain.
///
/// The chain recursion step for transaction proof `i` only needs proof `i` and
/// the chain proof for steps `< i`, while transaction proof `i + 1` only needs
/// the jump state from proof `i`'s public inputs. The chain step therefore runs
/// on a worker thread while this thread proves the next transaction chunk, a
/// depth-1 pipeline holding at most two proofs in flight per path. Transaction
/// chunks arrive owned and are moved into `BlockTx`, never cloned.
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
    let (tx_data, tx_target, chain_data, chain_target, dummy_chain_circuit) = match path {
        TxPath::Light => (
            &circuits.light_tx_data,
            &circuits.light_tx_target,
            &circuits.light_chain_data,
            &circuits.light_chain_target,
            &circuits.dummy_light_chain_circuit,
        ),
        TxPath::Heavy => (
            &circuits.heavy_tx_data,
            &circuits.heavy_tx_target,
            &circuits.heavy_chain_data,
            &circuits.heavy_chain_target,
            &circuits.dummy_heavy_chain_circuit,
        ),
    };

    let base_proof = BlockTxChainCircuit::cyclic_base_proof(
        chain_data,
        dummy_chain_circuit,
        block_number,
        created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        old_account_delta_tree_root,
    );
    let mut jump = JumpState::initial(pre_output.new_state_root, old_account_delta_tree_root);

    let chain_proof = std::thread::scope(|scope| {
        let base = &base_proof;
        let mut chain: Option<ChainState<'_>> = None;
        let mut pending_tx: Option<(u64, Proof)> = None;

        for (step, (chunk_index, txs)) in chunks.into_iter().enumerate() {
            // Fold the previous transaction proof into the chain on a worker
            // thread while this thread proves the next transaction chunk.
            if let Some((chain_step, tx_proof)) = pending_tx.take() {
                let previous_proof = chain.take().map(ChainState::wait);
                let handle = std::thread::Builder::new()
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
                            &tx_proof,
                        )
                    })
                    .expect("chain step pipeline thread must start");
                chain = Some(ChainState::InFlight(handle));
            }

            let block_tx = BlockTx {
                created_at,
                state_metadata_hash,
                old_jump: jump,
                txs,
            };
            let tx_proof =
                BlockTxCircuit::prove(tx_data, &block_tx, tx_target).unwrap_or_else(|error| {
                    panic!("block transaction chunk #{chunk_index} proof failed: {error:?}")
                });
            jump = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs).new_jump;
            pending_tx = Some((step as u64, tx_proof));
        }

        // The last transaction proof's chain step has no next chunk to overlap
        // with; fold it inline.
        if let Some((chain_step, tx_proof)) = pending_tx.take() {
            let previous_proof = chain.take().map(ChainState::wait);
            chain = Some(ChainState::Ready(chain_step_proof(
                path,
                chain_target,
                chain_data,
                chain_step,
                previous_proof.as_ref(),
                base,
                &tx_proof,
            )));
        }
        chain.map(ChainState::wait)
    });

    // A path with no chunks falls back to its base proof for the final chain
    // input, exactly like the sequential prover.
    chain_proof.unwrap_or(base_proof)
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

    // Partition the parsed chunks by path, moving each transaction vector
    // (never cloning it) and preserving parsed block order within each path.
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
    // The final block circuit only asserts that the chunk vector is non-empty;
    // the transaction effects are carried by the recursively verified chains.
    block.tx_chunks = tx_chunks;
    block.tx_chunks.push(Vec::new());

    // The heavy and light chains only share the pre-execution output, so they
    // prove concurrently; the far shorter heavy path hides under the light one.
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
