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

// A worker that runs anywhere near this long has already lost (the ranked
// budget is 15 minutes for five fixtures). The trusted wrapper collapses the
// worker's exit status, so the stalled phase is encoded in the *time of
// death* instead: the watchdog samples the phase at four minutes and dies
// 60 s later per phase step. Phase codes: 44 startup, 45 fixture parsed,
// 46 loading circuits, 49 proving, 50 writing proof — deaths at
// 300/360/420/600/660 s respectively.
const STALL_WATCHDOG_BASE_SECONDS: u64 = 240;
const STALL_WATCHDOG_PHASE_STEP_SECONDS: u64 = 60;

fn main() {
    env_logger::init();
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");
    std::thread::Builder::new()
        .name("stall-watchdog".into())
        .spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(STALL_WATCHDOG_BASE_SECONDS));
            let phase = api::WORKER_PHASE.load(std::sync::atomic::Ordering::Relaxed) as u64;
            std::thread::sleep(std::time::Duration::from_secs(
                (phase.saturating_sub(43)) * STALL_WATCHDOG_PHASE_STEP_SECONDS,
            ));
            std::process::exit(phase as i32);
        })
        .expect("stall watchdog thread must start");

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    let json = fs::read(fixture).expect("cannot read prover fixture");
    let block = Block::<F>::from_json_with_empty_txs(
        &json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");
    api::set_worker_phase(45);
    api::set_worker_phase(46);
    let circuits = Circuits::new();
    api::set_worker_phase(49);
    let proof = prover::prove_block(block, &circuits);
    api::set_worker_phase(50);
    bincode::serialize_into(
        BufWriter::with_capacity(
            PROOF_OUTPUT_BUFFER_BYTES,
            File::create(output).expect("cannot create proof output"),
        ),
        &proof,
    )
    .expect("cannot write proof output");
}
