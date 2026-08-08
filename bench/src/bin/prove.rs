// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../embedded.rs"]
mod embedded;
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

#[cfg(not(target_env = "msvc"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: &[u8; 57] =
    b"dirty_decay_ms:-1,muzzy_decay_ms:-1,oversize_threshold:0\0";

const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

fn main() {
    env_logger::init();
    let metal_warm = std::thread::spawn(plonky2::hash::poseidon2::warm_up_gpu_context);
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    let (block, circuits) = rayon::join(
        || {
            let json = fs::read(&fixture).expect("cannot read prover fixture");
            Block::<F>::from_json_with_empty_txs(
                &json,
                HEAVY_TX_PER_PROOF,
                LIGHT_TX_PER_PROOF,
                PUBLIC_HEAVY_TX_COUNT,
                PUBLIC_LIGHT_TX_COUNT,
            )
            .expect("invalid prover fixture")
        },
        Circuits::load,
    );
    let _ = metal_warm.join();
    let proof = prover::prove_block(block, circuits);
    let mut writer = BufWriter::with_capacity(
        PROOF_OUTPUT_BUFFER_BYTES,
        File::create(output).expect("cannot create proof output"),
    );
    bincode::serialize_into(&mut writer, &proof).expect("cannot write proof output");
    let file = writer.into_inner().expect("cannot flush proof output");
    drop(file);
    std::process::exit(0);
}
