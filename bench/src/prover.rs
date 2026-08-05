// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::types::config::F;
use circuit::types::constants::TX_LIGHT;

use crate::api::{Circuits, Proof};

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

pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proof {
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);

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

    let mut heavy_jump =
        JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
    let mut light_jump = heavy_jump;
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    // Chunk proofs are computed up front in route order and folded afterwards,
    // but the jump state must still thread per path: the chain circuit connects
    // the accumulated chain jump to each tx proof's old_jump public input, so
    // chunk k+1 on a path proves against chunk k's new_jump. Parallelizing the
    // chunk proofs would require precomputing those jumps, so they stay
    // sequential (rayon already parallelizes inside plonky2 ops).
    let routes = chunk_routes(block);
    let mut tx_proofs: Vec<Proof> = Vec::with_capacity(routes.len());
    for route in &routes {
        let is_light = route.path == TxPath::Light;
        let block_tx = BlockTx {
            created_at: block.created_at,
            state_metadata_hash,
            old_jump: if is_light { light_jump } else { heavy_jump },
            txs: block.tx_chunks[route.chunk_index].clone(),
        };
        let result = if is_light {
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
        };
        let tx_proof = result.unwrap_or_else(|error| {
            panic!(
                "block transaction chunk #{} proof failed: {error:?}",
                route.chunk_index
            )
        });
        let tx_output = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs);
        if is_light {
            light_jump = tx_output.new_jump;
        } else {
            heavy_jump = tx_output.new_jump;
        }
        tx_proofs.push(tx_proof);
    }

    for (route, tx_proof) in routes.iter().zip(tx_proofs.into_iter()) {
        let is_light = route.path == TxPath::Light;

        if is_light {
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
    }

    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    BlockCircuit::prove(
        &circuits.block_target,
        &circuits.block_data,
        block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed")
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
