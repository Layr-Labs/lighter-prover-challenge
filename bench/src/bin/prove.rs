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

use api::{HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT};
use circuit::block::Block;
use circuit::types::config::F;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Keep the promoted writer path while exercising a second submission from that baseline.
const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

fn main() {
    // Only active when RUST_LOG is set (e.g. RUST_LOG=debug exposes plonky2
    // TimingTree phase breakdowns); the ranked harness clears the environment,
    // so official runs never log.
    let _ = env_logger::try_init();

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    let json = fs::read(fixture).expect("cannot read prover fixture");
    let block = api::stage("parse fixture", || {
        Block::<F>::from_json_with_empty_txs(
            &json,
            HEAVY_TX_PER_PROOF,
            LIGHT_TX_PER_PROOF,
            PUBLIC_HEAVY_TX_COUNT,
            PUBLIC_LIGHT_TX_COUNT,
        )
    })
    .expect("invalid prover fixture");
    let proof = api::stage("prove block (total)", || {
        prover::prove_block_pipelined(&block)
    });
    api::stage("serialize proof", || {
        bincode::serialize_into(
            BufWriter::with_capacity(
                PROOF_OUTPUT_BUFFER_BYTES,
                File::create(output).expect("cannot create proof output"),
            ),
            &proof,
        )
        .expect("cannot write proof output");
    });
}
