#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../../submission/prover.rs"]
mod prover;

use std::env;
use std::fs::{self, File};

use api::Circuits;
use circuit::block::Block;
use circuit::types::config::F;

const TX_PER_PROOF: usize = 4;
const CHAIN_ID: u32 = 304;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "usage: prove FIXTURE OUTPUT");

    let json = fs::read_to_string(&args[1]).expect("cannot read prover fixture");
    let block: Block<F> = serde_json::from_str(&json).expect("invalid prover fixture");
    let proofs = prover::prove_block(&block, &Circuits::new(TX_PER_PROOF, CHAIN_ID));
    bincode::serialize_into(
        File::create(&args[2]).expect("cannot create proof output"),
        &proofs,
    )
    .expect("cannot write proof output");
}
