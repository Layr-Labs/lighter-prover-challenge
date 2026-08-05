// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Local-only A/B harness for a single light transaction-chunk proof.
//!
//! Whole-fixture wall clock on a laptop carries roughly +/-5% run-to-run noise
//! (thermal state, background load), which is the same size as the effects worth
//! measuring. This binary proves the *same* light chunk repeatedly inside one
//! process and reports min/median/mean, so a few dozen seconds of wall clock
//! give a statistic tight enough to compare two builds. Light chunk proofs are
//! 49 of the 53 chunk proofs in the public fixture and the single largest block
//! of proving time, so they track the parallel hot path closely.
//!
//! Not built by `setup.sh`, which compiles only `--bin prove`.

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;

use std::time::Instant;

use api::{HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT};
use circuit::block::Block;
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, JumpState};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::types::config::{CIRCUIT_CONFIG, F};
use circuit::types::constants::TX_LIGHT;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("iterations must be a number"))
        .unwrap_or(8);

    let json = include_bytes!("../../bench_test.json");
    let block = Block::<F>::from_json_with_empty_txs(
        json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("public fixture must parse");

    let build = Instant::now();
    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pre_target = pre.target;
    let pre_data = pre.builder.build::<circuit::types::config::C>();
    let light = BlockTxCircuit::define(
        CIRCUIT_CONFIG,
        LIGHT_TX_PER_PROOF,
        api::CHAIN_ID,
        api::LIGHT_TX_MODE,
    );
    let light_target = light.target;
    let light_data = light.builder.build::<circuit::types::config::C>();
    eprintln!("build {:.3} s", build.elapsed().as_secs_f64());

    let pre_proof =
        BlockPreExecutionCircuit::prove(&pre_data, &BlockPreExec::from_block(&block), &pre_target)
            .expect("pre-execution proof failed");
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);

    let chunk_index = block
        .tx_chunks
        .iter()
        .position(|txs| txs[0].tx_circuit_type == TX_LIGHT)
        .expect("fixture must contain a light chunk");
    let block_tx = BlockTx {
        created_at: block.created_at,
        state_metadata_hash: pre_output.new_state_metadata.hash(),
        old_jump: JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root),
        txs: block.tx_chunks[chunk_index].clone(),
    };

    // One untimed proof so any lazily initialised state (rayon pool, allocator
    // arenas, FFT root tables already live in circuit data) is warm.
    let _ = BlockTxCircuit::prove(&light_data, &block_tx, &light_target).expect("warmup failed");

    let mut samples = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let proof = BlockTxCircuit::prove(&light_data, &block_tx, &light_target)
            .expect("light chunk proof failed");
        let elapsed = start.elapsed().as_secs_f64();
        // Keep the proof alive past the timer so it cannot be optimised away.
        assert!(!proof.public_inputs.is_empty());
        samples.push(elapsed);
        eprintln!("  iter {i}: {elapsed:.4} s");
    }

    samples.sort_by(f64::total_cmp);
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "light_chunk_proof n={iterations} min={min:.4} median={median:.4} mean={mean:.4} max={:.4}",
        samples[samples.len() - 1]
    );
}
