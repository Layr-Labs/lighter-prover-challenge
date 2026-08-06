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
    env_logger::init();
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    // Startup is on the scored clock five times per ranked run. The three
    // independent pieces — one-time Metal shader compilation, the rayon-wide
    // circuit build, and the single-threaded fixture parse — overlap instead
    // of running back to back.
    std::thread::Builder::new()
        .name("metal-warmup".into())
        .spawn(plonky2::hash::poseidon2::warm_metal_context)
        .expect("metal warmup thread must start");
    let staged_handle = std::thread::Builder::new()
        .name("circuits-build".into())
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(Circuits::start)
        .expect("circuit build thread must start");

    let mut json = fs::read(fixture).expect("cannot read prover fixture");
    let block = Block::<F>::from_json_bytes_with_empty_txs(
        &mut json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");
    drop(json);

    // The staged build returns once the pre-execution circuit is ready, so the
    // pre-execution proof runs while the transaction/chain circuits finish.
    let staged = staged_handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    let (pre_target, pre_data) = staged.pre();
    let pre_proof = prover::prove_pre_execution(&block, pre_target, pre_data);
    let circuits = staged.finish();
    let proof = prover::prove_block_with_pre(block, &circuits, pre_proof);

    let mut writer = BufWriter::with_capacity(
        PROOF_OUTPUT_BUFFER_BYTES,
        File::create(output).expect("cannot create proof output"),
    );
    bincode::serialize_into(&mut writer, &proof).expect("cannot write proof output");
    writer
        .into_inner()
        .expect("cannot flush proof output")
        .sync_all()
        .expect("cannot sync proof output");
    // The proof file is durable; skip userland teardown (GB-scale circuit and
    // proof buffers would otherwise walk their destructors on the scored
    // clock).
    std::process::exit(0);
}
