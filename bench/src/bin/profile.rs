// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Local development harness: identical proving path to `prove`, but with the
//! plonky2 `TimingTree` output enabled and a wall-clock breakdown printed to
//! stderr. Never built by `setup.sh`, which compiles only `--bin prove`.

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../prover.rs"]
mod prover;

use std::env;
use std::fs;
use std::time::Instant;

use api::{
    Circuits, HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PROVER_THREAD_STACK_BYTES,
    PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
};
use circuit::block::Block;
use circuit::types::config::F;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .format_timestamp(None)
        .init();

    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: profile FIXTURE");

    let json = fs::read(fixture).expect("cannot read prover fixture");

    let build_start = Instant::now();
    let circuits = Circuits::new();
    let build_elapsed = build_start.elapsed();

    let block = Block::<F>::from_json_with_empty_txs(
        &json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");

    let prove_start = Instant::now();
    let proof = prover::prove_block(block, &circuits);
    let prove_elapsed = prove_start.elapsed();

    let bytes = bincode::serialize(&proof).expect("cannot serialize proof");
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);

    eprintln!("PROFILE circuits_build_s {:.3}", build_elapsed.as_secs_f64());
    eprintln!("PROFILE prove_block_s {:.3}", prove_elapsed.as_secs_f64());
    eprintln!("PROFILE proof_bytes {}", bytes.len());
    eprintln!("PROFILE proof_sha256 {}", hex::encode(digest));
}
