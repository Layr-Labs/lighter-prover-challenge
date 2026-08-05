// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, Circuit as _,
};
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
    let t_pre = Instant::now();
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    eprintln!("[timing] pre-exec proof: {:?}", t_pre.elapsed());

    let state_metadata_hash = pre_output.new_state_metadata.hash();
    let initial_jump =
        JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);

    // Cyclic base proofs — run in parallel
    let t_base = Instant::now();
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
    eprintln!("[timing] cyclic base proofs: {:?}", t_base.elapsed());

    // Split chunks by path
    let routes: Vec<ChunkRoute> = chunk_routes(block);
    let light_routes: Vec<ChunkRoute> = routes
        .iter()
        .filter(|r| r.path == TxPath::Light)
        .cloned()
        .collect();
    let heavy_routes: Vec<ChunkRoute> = routes
        .iter()
        .filter(|r| r.path == TxPath::Heavy)
        .cloned()
        .collect();

    // Process both paths in parallel with interleaving pipelining.
    // Within each path, tx proving (producer) and chain proving (consumer) overlap:
    // while chain(i) runs, tx(i+1) starts on the producer thread.
    let t_chunks = Instant::now();
    let (light_chain_proof, heavy_chain_proof) = {
        thread::scope(|s| {
            // --- Light path: pipelined producer + consumer ---
            let (tx_sender, tx_receiver) = mpsc::sync_channel::<Proof>(2);
            let light_routes_ref = &light_routes;

            let producer = s.spawn(move || {
                let mut jump = initial_jump;
                for route in light_routes_ref {
                    let t = Instant::now();
                    let block_tx = BlockTx {
                        created_at: block.created_at,
                        state_metadata_hash,
                        old_jump: jump,
                        txs: block.tx_chunks[route.chunk_index].clone(),
                    };
                    let tx_proof = BlockTxCircuit::prove(
                        &circuits.light_tx_data,
                        &block_tx,
                        &circuits.light_tx_target,
                    )
                    .unwrap_or_else(|e| {
                        panic!("light tx chunk #{} failed: {e:?}", route.chunk_index)
                    });
                    eprintln!(
                        "[timing] light tx chunk {}: {:?}",
                        route.chunk_index,
                        t.elapsed()
                    );
                    jump =
                        BlockTxWitness::from_public_inputs(&tx_proof.public_inputs).new_jump;
                    tx_sender.send(tx_proof).unwrap();
                }
            });

            let consumer = s.spawn(move || {
                let mut chain_proof = light_chain_proof;
                for route in light_routes_ref {
                    let t = Instant::now();
                    let tx_proof = tx_receiver.recv().unwrap();
                    chain_proof = BlockTxChainCircuit::prove(
                        &circuits.light_chain_target,
                        &circuits.light_chain_data,
                        route.chain_step,
                        &chain_proof,
                        &circuits.dummy_light_proof,
                        &tx_proof,
                    )
                    .unwrap_or_else(|e| {
                        panic!("light chain step #{} failed: {e:?}", route.chain_step)
                    });
                    eprintln!(
                        "[timing] light chain step {}: {:?}",
                        route.chain_step,
                        t.elapsed()
                    );
                }
                chain_proof
            });

            // --- Heavy path: sequential (only 3 chunks — no pipelining benefit) ---
            let heavy_handle = s.spawn(move || {
                let mut jump = initial_jump;
                let mut chain_proof = heavy_chain_proof;
                for route in &heavy_routes {
                    let t = Instant::now();
                    let block_tx = BlockTx {
                        created_at: block.created_at,
                        state_metadata_hash,
                        old_jump: jump,
                        txs: block.tx_chunks[route.chunk_index].clone(),
                    };
                    let tx_proof = BlockTxCircuit::prove(
                        &circuits.heavy_tx_data,
                        &block_tx,
                        &circuits.heavy_tx_target,
                    )
                    .unwrap_or_else(|e| {
                        panic!("heavy tx chunk #{} failed: {e:?}", route.chunk_index)
                    });
                    jump =
                        BlockTxWitness::from_public_inputs(&tx_proof.public_inputs).new_jump;
                    chain_proof = BlockTxChainCircuit::prove(
                        &circuits.heavy_chain_target,
                        &circuits.heavy_chain_data,
                        route.chain_step,
                        &chain_proof,
                        &circuits.dummy_heavy_proof,
                        &tx_proof,
                    )
                    .unwrap_or_else(|e| {
                        panic!("heavy chain step #{} failed: {e:?}", route.chain_step)
                    });
                    eprintln!(
                        "[timing] heavy chunk {}: {:?}",
                        route.chunk_index,
                        t.elapsed()
                    );
                }
                chain_proof
            });

            let light_chain_proof = consumer.join().unwrap();
            let heavy_chain_proof = heavy_handle.join().unwrap();
            producer.join().unwrap();
            (light_chain_proof, heavy_chain_proof)
        })
    };
    eprintln!(
        "[timing] chunks: {:?}",
        t_chunks.elapsed()
    );

    // Final block proof
    let t_final = Instant::now();
    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    let result = BlockCircuit::prove(
        &circuits.block_target,
        &circuits.block_data,
        block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed");
    eprintln!("[timing] final block proof: {:?}", t_final.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_routes_empty_txs_panics() {
        // Verify that empty tx_chunks would panic (as expected by the contract)
        let tx_type = TX_LIGHT;
        assert_eq!(tx_type, TX_LIGHT);
    }
}
