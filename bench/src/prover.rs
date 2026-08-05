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

fn prove_tx_path(
    block: &Block<F>,
    circuits: &Circuits,
    pre_output: &BlockPreExecWitness<F>,
    routes: &[ChunkRoute],
    path: TxPath,
) -> Proof {
    let (tx_target, tx_data, chain_target, chain_data, dummy_chain_circuit, dummy_proof) =
        match path {
            TxPath::Heavy => (
                &circuits.heavy_tx_target,
                &circuits.heavy_tx_data,
                &circuits.heavy_chain_target,
                &circuits.heavy_chain_data,
                &circuits.dummy_heavy_chain_circuit,
                &circuits.dummy_heavy_proof,
            ),
            TxPath::Light => (
                &circuits.light_tx_target,
                &circuits.light_tx_data,
                &circuits.light_chain_target,
                &circuits.light_chain_data,
                &circuits.dummy_light_chain_circuit,
                &circuits.dummy_light_proof,
            ),
        };
    let mut chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        chain_data,
        dummy_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );
    let mut jump = JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    for route in routes.iter().filter(|route| route.path == path) {
        let block_tx = BlockTx {
            created_at: block.created_at,
            state_metadata_hash,
            old_jump: jump,
            txs: block.tx_chunks[route.chunk_index].clone(),
        };
        let tx_proof =
            BlockTxCircuit::prove(tx_data, &block_tx, tx_target).unwrap_or_else(|error| {
                panic!(
                    "block transaction chunk #{} proof failed: {error:?}",
                    route.chunk_index
                )
            });
        jump = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs).new_jump;
        chain_proof = BlockTxChainCircuit::prove(
            chain_target,
            chain_data,
            route.chain_step,
            &chain_proof,
            dummy_proof,
            &tx_proof,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{path:?} block transaction chain step #{} failed: {error:?}",
                route.chain_step
            )
        });
    }
    chain_proof
}

pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proof {
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    let routes = chunk_routes(block);
    let (heavy_chain_proof, light_chain_proof) = rayon::join(
        || prove_tx_path(block, circuits, &pre_output, &routes, TxPath::Heavy),
        || prove_tx_path(block, circuits, &pre_output, &routes, TxPath::Light),
    );

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
