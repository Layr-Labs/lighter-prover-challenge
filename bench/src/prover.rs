// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Mutex, mpsc};

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

const DEFAULT_TX_CONCURRENCY: usize = 4;

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

struct PreparedChunk {
    route: ChunkRoute,
    old_jump: JumpState<F>,
}

struct ProvedChunk {
    route: ChunkRoute,
    old_jump: JumpState<F>,
    proof: Proof,
    elapsed: std::time::Duration,
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

fn tx_concurrency() -> usize {
    match std::env::var("LIGHTER_TX_CONCURRENCY") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .unwrap_or_else(|| {
                panic!("LIGHTER_TX_CONCURRENCY must be a positive integer, got {value:?}")
            }),
        Err(std::env::VarError::NotPresent) => DEFAULT_TX_CONCURRENCY,
        Err(error) => panic!("failed to read LIGHTER_TX_CONCURRENCY: {error}"),
    }
}

fn fold_tx_path(
    path: TxPath,
    chunk_count: usize,
    receiver: mpsc::Receiver<ProvedChunk>,
    circuits: &Circuits,
    mut chain_proof: Proof,
) -> (Proof, std::time::Duration, std::time::Duration) {
    let mut pending = BTreeMap::new();
    let mut tx_total = std::time::Duration::ZERO;
    let mut chain_total = std::time::Duration::ZERO;

    for expected_step in 0..chunk_count as u64 {
        let proved = loop {
            if let Some(proved) = pending.remove(&expected_step) {
                break proved;
            }
            let proved = receiver.recv().unwrap_or_else(|_| {
                panic!("{path:?} tx proof channel closed before chain step #{expected_step}")
            });
            assert_eq!(
                proved.route.path, path,
                "received tx proof for the wrong chain path"
            );
            if proved.route.chain_step == expected_step {
                break proved;
            }
            assert!(
                proved.route.chain_step > expected_step,
                "received late or duplicate {path:?} tx proof for chain step #{}",
                proved.route.chain_step
            );
            let step = proved.route.chain_step;
            assert!(
                pending.insert(step, proved).is_none(),
                "received duplicate {path:?} tx proof for chain step #{step}"
            );
        };

        tx_total += proved.elapsed;
        let tx_output = BlockTxWitness::from_public_inputs(&proved.proof.public_inputs);
        assert!(
            tx_output.old_jump.to_vec() == proved.old_jump.to_vec(),
            "tx proof old_jump mismatch for {path:?} chain step #{} (chunk #{})",
            proved.route.chain_step,
            proved.route.chunk_index
        );

        let t_chain = std::time::Instant::now();
        chain_proof = match path {
            TxPath::Light => BlockTxChainCircuit::prove(
                &circuits.light_chain_target,
                &circuits.light_chain_data,
                proved.route.chain_step,
                &chain_proof,
                &circuits.dummy_light_proof,
                &proved.proof,
            ),
            TxPath::Heavy => BlockTxChainCircuit::prove(
                &circuits.heavy_chain_target,
                &circuits.heavy_chain_data,
                proved.route.chain_step,
                &chain_proof,
                &circuits.dummy_heavy_proof,
                &proved.proof,
            ),
        }
        .unwrap_or_else(|error| {
            panic!(
                "{path:?} block transaction chain step #{} failed: {error:?}",
                proved.route.chain_step
            )
        });
        chain_total += t_chain.elapsed();
    }

    (chain_proof, tx_total, chain_total)
}

pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proof {
    let t_pre = std::time::Instant::now();
    let pre_proof = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .expect("block pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    eprintln!("[t] pre-exec proof: {:?}", t_pre.elapsed());

    let t_base = std::time::Instant::now();
    let heavy_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &circuits.heavy_chain_data,
        &circuits.dummy_heavy_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );
    let light_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &circuits.light_chain_data,
        &circuits.dummy_light_chain_circuit,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
    );

    eprintln!("[t] cyclic base proofs: {:?}", t_base.elapsed());

    let t_prepare = std::time::Instant::now();
    let mut heavy_jump =
        JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root);
    let mut light_jump = heavy_jump;
    let state_metadata_hash = pre_output.new_state_metadata.hash();
    let prepared_chunks = chunk_routes(block)
        .into_iter()
        .map(|route| {
            let txs = &block.tx_chunks[route.chunk_index];
            let jump = match route.path {
                TxPath::Light => &mut light_jump,
                TxPath::Heavy => &mut heavy_jump,
            };
            let old_jump = *jump;
            *jump = jump.step_chunk(txs);
            PreparedChunk { route, old_jump }
        })
        .collect::<Vec<_>>();
    eprintln!("[t] native jump prep: {:?}", t_prepare.elapsed());

    let heavy_chunk_count = prepared_chunks
        .iter()
        .filter(|chunk| chunk.route.path == TxPath::Heavy)
        .count();
    let light_chunk_count = prepared_chunks.len() - heavy_chunk_count;
    let concurrency = tx_concurrency();
    let worker_count = concurrency.min(prepared_chunks.len());
    let jobs = Mutex::new(VecDeque::from(prepared_chunks));
    let (heavy_sender, heavy_receiver) = mpsc::sync_channel(worker_count.max(1));
    let (light_sender, light_receiver) = mpsc::sync_channel(worker_count.max(1));
    let t_parallel = std::time::Instant::now();

    let (
        (heavy_chain_proof, tx_heavy_total, chain_heavy_total),
        (light_chain_proof, tx_light_total, chain_light_total),
    ) = std::thread::scope(|scope| {
        let heavy_fold = scope.spawn(move || {
            fold_tx_path(
                TxPath::Heavy,
                heavy_chunk_count,
                heavy_receiver,
                circuits,
                heavy_chain_proof,
            )
        });
        let light_fold = scope.spawn(move || {
            fold_tx_path(
                TxPath::Light,
                light_chunk_count,
                light_receiver,
                circuits,
                light_chain_proof,
            )
        });

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let heavy_sender = heavy_sender.clone();
            let light_sender = light_sender.clone();
            let jobs = &jobs;
            workers.push(scope.spawn(move || {
                loop {
                    let prepared = jobs
                        .lock()
                        .expect("tx proof job queue mutex poisoned")
                        .pop_front();
                    let Some(prepared) = prepared else {
                        break;
                    };
                    let block_tx = BlockTx {
                        created_at: block.created_at,
                        state_metadata_hash,
                        old_jump: prepared.old_jump,
                        txs: block.tx_chunks[prepared.route.chunk_index].clone(),
                    };

                    let t_tx = std::time::Instant::now();
                    let tx_proof = match prepared.route.path {
                        TxPath::Light => BlockTxCircuit::prove(
                            &circuits.light_tx_data,
                            &block_tx,
                            &circuits.light_tx_target,
                        ),
                        TxPath::Heavy => BlockTxCircuit::prove(
                            &circuits.heavy_tx_data,
                            &block_tx,
                            &circuits.heavy_tx_target,
                        ),
                    }
                    .unwrap_or_else(|error| {
                        panic!(
                            "block transaction chunk #{} proof failed: {error:?}",
                            prepared.route.chunk_index
                        )
                    });

                    let proved = ProvedChunk {
                        route: prepared.route,
                        old_jump: prepared.old_jump,
                        proof: tx_proof,
                        elapsed: t_tx.elapsed(),
                    };
                    match proved.route.path {
                        TxPath::Light => light_sender.send(proved),
                        TxPath::Heavy => heavy_sender.send(proved),
                    }
                    .expect("tx proof chain receiver closed unexpectedly");
                }
            }));
        }

        drop(heavy_sender);
        drop(light_sender);
        for worker in workers {
            worker.join().expect("tx proof worker panicked");
        }
        let heavy = heavy_fold.join().expect("heavy chain fold worker panicked");
        let light = light_fold.join().expect("light chain fold worker panicked");
        (heavy, light)
    });

    eprintln!(
        "[t] tx proofs + chain folds wall (concurrency {worker_count}): {:?}",
        t_parallel.elapsed()
    );

    eprintln!(
        "[t] tx proofs: heavy {:?} light {:?} | chain folds: heavy {:?} light {:?}",
        tx_heavy_total, tx_light_total, chain_heavy_total, chain_light_total
    );

    let t_final = std::time::Instant::now();
    let (light_chain_input, heavy_chain_input) =
        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
    let final_proof = BlockCircuit::prove(
        &circuits.block_target,
        &circuits.block_data,
        block,
        &pre_proof,
        light_chain_input,
        heavy_chain_input,
    )
    .expect("final block proof failed");
    eprintln!("[t] final block proof: {:?}", t_final.elapsed());
    final_proof
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
