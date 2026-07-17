#![feature(stmt_expr_attributes)]

//! Trusted timer and verifier. Contestant code is not linked into this binary.

#[path = "../api.rs"]
mod api;

use std::env;
use std::fs::{self, File};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use api::{Circuits, Proofs};
use circuit::block::Block;
use circuit::block_pre_execution::BlockPreExecWitness;
use circuit::block_tx_chain::BlockTxChainWitness;
use circuit::types::config::F;
use env_logger::Env;
use log::info;
use serde_json::json;

const TX_PER_PROOF: usize = 4;
const CHAIN_ID: u32 = 304;
const MACHINE: &str = "Apple M4 Max, 14 cores, 36 GB, macOS 26.5";
const DEFAULT_PROVE_TIMEOUT: Duration = Duration::from_secs(15 * 60);

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info,circuit=error")).init();

    let args: Vec<String> = env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: bench MODE TRANSACTIONS COMMIT OUTPUT"
    );
    let (expected_transactions, baseline) = match args[1].as_str() {
        "local-iterate" => (32, 29.455_340_667),
        "local-submit" => (500, 489.222_507_667),
        mode => panic!("unknown benchmark mode: {mode}"),
    };
    let transactions: usize = args[2].parse().expect("invalid transaction count");
    assert_eq!(transactions, expected_transactions);

    let json = fs::read_to_string("bench_test.json").expect("cannot read bench_test.json");
    let block: Block<F> = serde_json::from_str(&json).expect("invalid prover fixture");
    assert_eq!(block.txs.len(), transactions);
    let proof_path = "proof.bin";
    let prove = env::current_exe().unwrap().with_file_name("prove");

    let timeout = prove_timeout();
    let started = Instant::now();
    let mut child = Command::new(prove)
        .args(["bench_test.json", proof_path])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("cannot start prover");
    let status = loop {
        match child.try_wait().expect("cannot wait for prover") {
            Some(status) => break status,
            None if started.elapsed() >= timeout => {
                child.kill().expect("cannot kill timed-out prover");
                let status = child.wait().expect("cannot reap timed-out prover");
                panic!(
                    "prover timed out after {}s: {status}",
                    timeout.as_secs()
                );
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    };
    let proving_time = started.elapsed();
    assert!(status.success(), "prover failed: {status}");

    let proof_file = File::open(proof_path).expect("missing proof output");
    assert!(
        proof_file.metadata().unwrap().len() <= 256 * 1024 * 1024,
        "proof output is too large"
    );
    let proofs: Proofs = bincode::deserialize_from(proof_file).expect("invalid proof output");
    verify(&block, &Circuits::new(TX_PER_PROOF, CHAIN_ID), &proofs);
    info!("TOTAL proving time: {proving_time:?}");

    let proving_seconds = proving_time.as_secs_f64();
    let speedup = baseline / proving_seconds;
    let score = json!({
        "score": speedup,
        "passed": true,
        "metrics": {
            "runtime": args[1],
            "commit": args[3],
            "transactions": transactions,
            "baseline_proving_seconds": baseline,
            "baseline_machine": MACHINE,
            "timing_authority": "trusted bench parent",
            "proving_seconds": proving_seconds,
            "transactions_per_second": transactions as f64 / proving_seconds,
            "speedup": speedup
        }
    });
    let rendered = serde_json::to_string_pretty(&score).unwrap();
    let output = Path::new(&args[4]);
    let temporary = output.with_extension("tmp");
    fs::write(&temporary, format!("{rendered}\n")).expect("cannot write score");
    fs::rename(temporary, output).expect("cannot publish score");
    println!("{rendered}");
}

fn prove_timeout() -> Duration {
    env::var("LIGHTER_PROVE_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_PROVE_TIMEOUT)
}

fn verify(block: &Block<F>, circuits: &Circuits, proofs: &Proofs) {
    circuits.pre_data.verify(proofs.pre.clone()).unwrap();
    circuits.chain_data.verify(proofs.chain.clone()).unwrap();

    let pre = BlockPreExecWitness::from_public_inputs(&proofs.pre.public_inputs);
    let chain = BlockTxChainWitness::from_public_inputs(&proofs.chain.public_inputs, 1, 1);

    assert_eq!(pre.block_number, block.block_number);
    assert_eq!(pre.created_at, block.created_at);
    assert_eq!(pre.old_state_root, block.old_state_root);
    assert_eq!(chain.block_number, block.block_number);
    assert_eq!(chain.created_at, block.created_at);
    assert_eq!(chain.old_state_root, pre.new_state_root);
    assert_eq!(chain.new_validium_root, block.new_validium_root);
    assert_eq!(chain.new_state_root, block.new_state_root);
    assert_eq!(
        chain.new_account_delta_tree_root,
        block.new_account_delta_tree_root
    );
    assert_eq!(
        chain.on_chain_operations_count,
        block.on_chain_operations_count
    );
    assert_eq!(
        chain.priority_operations_count,
        block.priority_operations_count
    );
}
