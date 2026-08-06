// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../prover.rs"]
mod prover;

use std::env;
use std::fs::{self, File};
use std::io::BufWriter;

use api::{
    Circuits, HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PROVER_THREAD_STACK_BYTES,
    PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
};
use circuit::block::Block;
use circuit::types::config::F;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Keep the promoted writer path while exercising a second submission from that baseline.
const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

fn main() {
    // Active ranked fixtures nest deeper than the public smoke fixture, and the
    // test suite already parses them on 32MiB stacks. Run the whole prover on a
    // generously sized worker thread so fixture parsing can never overflow the
    // 8MiB default main stack in the sandbox. (The rayon pool below gets the
    // same stack size for its workers.)
    std::thread::Builder::new()
        .name("prove-main".into())
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(prove_main)
        .expect("cannot spawn prover thread")
        .join()
        .expect("prover thread panicked");
}

fn prove_main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");

    // Local profiling only: the ranked sandbox clears the environment, so both
    // branches below are inert there.
    if env::var_os("RUST_LOG").is_some() {
        let _ = env_logger::try_init();
    }
    let mut timer = api::StageTimer::new();

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    let json = fs::read(fixture).expect("cannot read prover fixture");
    timer.mark("read fixture");
    let block = Block::<F>::from_json_with_empty_txs(
        &json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");
    timer.mark("parse fixture");
    // Only the leaf tx circuits are built here (they gate leaf proving); the
    // pre/chain/block circuits build on a background thread inside prove_block.
    let circuits = Circuits::new();
    timer.mark("tx circuits ready");
    let proof = prover::prove_block(block, &circuits);
    timer.mark("prove_block");
    bincode::serialize_into(
        BufWriter::with_capacity(
            PROOF_OUTPUT_BUFFER_BYTES,
            File::create(output).expect("cannot create proof output"),
        ),
        &proof,
    )
    .expect("cannot write proof output");
    timer.mark("write proof");
}
