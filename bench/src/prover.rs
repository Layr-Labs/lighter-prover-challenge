// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::tx::Tx;
use circuit::types::config::F;
use circuit::types::constants::TX_LIGHT;

use crate::api::{Circuits, Proof, PROVER_THREAD_STACK_BYTES};

fn chunk_is_light(txs: &[Tx<F>]) -> bool {
    txs.first()
        .expect("block transaction chunk must not be empty")
        .tx_circuit_type
        == TX_LIGHT
}

fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
    (light, heavy)
}

pub fn prove_block(mut block: Block<F>, circuits: &Circuits) -> Proof {
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(&block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);

    let (heavy_chain_proof, light_chain_proof) = rayon::join(
        || {
            BlockTxChainCircuit::cyclic_base_proof(
                &circuits.heavy_chain_data,
                &circuits.dummy_heavy_chain_circuit,
                block.block_number,
                block.created_at,
                pre_output.new_state_root,
                pre_output.new_validium_root,
                block.old_account_delta_tree_root,
            )
        },
        || {
            BlockTxChainCircuit::cyclic_base_proof(
                &circuits.light_chain_data,
                &circuits.dummy_light_chain_circuit,
                block.block_number,
                block.created_at,
                pre_output.new_state_root,
                pre_output.new_validium_root,
                block.old_account_delta_tree_root,
            )
        },
    );

    let mut heavy_jump =
        JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
    let mut light_jump = heavy_jump;
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let mut heavy_step = 0;
    let mut light_step = 0;
    let mut tx_chunks = std::mem::take(&mut block.tx_chunks);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let (light_chain_proof, heavy_chain_proof) = std::thread::scope(|scope| {
        let chain_worker = std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let mut heavy_chain_proof = heavy_chain_proof;
                let mut light_chain_proof = light_chain_proof;

                while let Ok((is_light, chain_step, tx_proof)) = receiver.recv() {
                    if is_light {
                        light_chain_proof = BlockTxChainCircuit::prove(
                            &circuits.light_chain_target,
                            &circuits.light_chain_data,
                            chain_step,
                            &light_chain_proof,
                            &circuits.dummy_light_proof,
                            &tx_proof,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "light block transaction chain step #{chain_step} failed: {error:?}"
                            )
                        });
                    } else {
                        heavy_chain_proof = BlockTxChainCircuit::prove(
                            &circuits.heavy_chain_target,
                            &circuits.heavy_chain_data,
                            chain_step,
                            &heavy_chain_proof,
                            &circuits.dummy_heavy_proof,
                            &tx_proof,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "heavy block transaction chain step #{chain_step} failed: {error:?}"
                            )
                        });
                    }
                }

                (light_chain_proof, heavy_chain_proof)
            })
            .expect("cannot spawn chain recursion worker");

        for (chunk_index, txs) in tx_chunks.drain(..).enumerate() {
            let is_light = chunk_is_light(&txs);
            let chain_step = if is_light {
                let step = light_step;
                light_step += 1;
                step
            } else {
                let step = heavy_step;
                heavy_step += 1;
                step
            };
            let block_tx = BlockTx {
                created_at: block.created_at,
                state_metadata_hash,
                old_jump: if is_light { light_jump } else { heavy_jump },
                txs,
            };

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
                panic!("block transaction chunk #{chunk_index} proof failed: {error:?}")
            });

            let tx_output = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs);
            if is_light {
                light_jump = tx_output.new_jump;
            } else {
                heavy_jump = tx_output.new_jump;
            }
            sender
                .send((is_light, chain_step, tx_proof))
                .expect("chain recursion worker stopped");
        }
        drop(sender);

        chain_worker
            .join()
            .expect("chain recursion worker panicked")
    });
    block.tx_chunks = tx_chunks;
    block.tx_chunks.push(Vec::new());

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
