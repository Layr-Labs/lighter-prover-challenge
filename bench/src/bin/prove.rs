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

use api::Circuits;
use circuit::block::Block;
use circuit::types::config::F;

const TX_PER_PROOF: usize = 4;
const CHAIN_ID: u32 = 304;

fn main() {
    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    let json = fs::read_to_string(fixture).expect("cannot read prover fixture");
    let block: Block<F> = serde_json::from_str(&json).expect("invalid prover fixture");
    let proofs = prover::prove_block(&block, &Circuits::new(TX_PER_PROOF, CHAIN_ID), TX_PER_PROOF);
    bincode::serialize_into(
        BufWriter::new(File::create(output).expect("cannot create proof output")),
        &proofs,
    )
    .expect("cannot write proof output");
}
