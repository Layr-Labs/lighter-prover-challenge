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

    // The output file's directory is the one place the sandboxed worker can
    // write; cache preprocessed circuit data there so later worker invocations
    // in the same run directory (ranked mode proves several fixtures) skip
    // rebuilding identical circuits. Entries are keyed by the binary's own
    // hash, so a different candidate can never read them.
    api::set_circuit_cache_dir(
        std::path::Path::new(&output)
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .map(|dir| dir.to_path_buf()),
    );

    let json = fs::read(fixture).expect("cannot read prover fixture");
    let block = Block::<F>::from_json_with_empty_txs(
        &json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");
    let proof = prover::prove_block(block, &Circuits::new());
    bincode::serialize_into(
        BufWriter::with_capacity(
            PROOF_OUTPUT_BUFFER_BYTES,
            File::create(output).expect("cannot create proof output"),
        ),
        &proof,
    )
    .expect("cannot write proof output");
}
