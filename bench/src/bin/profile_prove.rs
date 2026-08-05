// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Local profiling driver. Same code path as the `prove` worker, plus
//! `env_logger` so plonky2's `TimingTree` breakdown is printed, and coarse
//! stage timing around circuit construction. Never built by `setup.sh`, which
//! compiles only `--bin prove`.

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../prover.rs"]
mod prover;

use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::time::Instant;

use api::{
    Circuits, HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
};
use circuit::block::Block;
use circuit::types::config::F;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .format_timestamp_millis()
        .init();

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: profile_prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: profile_prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: profile_prove FIXTURE OUTPUT");

    let total = Instant::now();
    let json = fs::read(fixture).expect("cannot read prover fixture");
    let block = Block::<F>::from_json_with_empty_txs(
        &json,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
    .expect("invalid prover fixture");
    eprintln!("STAGE parse {:.3}", total.elapsed().as_secs_f64());

    let build = Instant::now();
    let circuits = Circuits::new();
    eprintln!("STAGE build_all {:.3}", build.elapsed().as_secs_f64());

    let proving = Instant::now();
    let proof = prover::prove_block(&block, &circuits);
    eprintln!("STAGE prove_all {:.3}", proving.elapsed().as_secs_f64());

    bincode::serialize_into(
        BufWriter::with_capacity(
            PROOF_OUTPUT_BUFFER_BYTES,
            File::create(output).expect("cannot create proof output"),
        ),
        &proof,
    )
    .expect("cannot write proof output");
    eprintln!("STAGE total {:.3}", total.elapsed().as_secs_f64());
}
