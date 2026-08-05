// Local validation: rebuild every circuit from scratch and check that the
// embedded compact caches reproduce them exactly.

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;

use std::time::Instant;

use api::{
    CHAIN_ID, HEAVY_TX_MODE, HEAVY_TX_PER_PROOF, LIGHT_TX_MODE, LIGHT_TX_PER_PROOF,
    ON_CHAIN_OPERATIONS_LIMIT, PROVER_THREAD_STACK_BYTES, cache_bytes,
};
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::circuit_cache::circuit_data_from_compact_bytes;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use plonky2::plonk::circuit_data::CircuitData;

fn check(label: &str, fresh: &CircuitData<F, C, D>, bytes: &[u8]) -> bool {
    let t = Instant::now();
    let loaded = match circuit_data_from_compact_bytes(bytes) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("[sizes] {label}: cache load FAILED: {error:?}");
            return false;
        }
    };
    let load = t.elapsed().as_secs_f64();
    let digest_ok = fresh.verifier_only.circuit_digest == loaded.verifier_only.circuit_digest;
    let cap_ok =
        fresh.verifier_only.constants_sigmas_cap == loaded.verifier_only.constants_sigmas_cap;
    let sigmas_ok = fresh.prover_only.sigmas == loaded.prover_only.sigmas;
    let reps_ok = fresh.prover_only.representative_map == loaded.prover_only.representative_map;
    let cap_tree_ok = fresh.prover_only.constants_sigmas_commitment.merkle_tree.cap
        == loaded.prover_only.constants_sigmas_commitment.merkle_tree.cap;
    let generators_ok = fresh.prover_only.generators.len() == loaded.prover_only.generators.len();
    let ok = digest_ok && cap_ok && sigmas_ok && reps_ok && cap_tree_ok && generators_ok;
    eprintln!(
        "[sizes] {label}: bytes={} ({:.1} MB) load={load:.2}s digest={digest_ok} cap={cap_ok} sigmas={sigmas_ok} reps={reps_ok} cap_tree={cap_tree_ok} generators={generators_ok}",
        bytes.len(),
        bytes.len() as f64 / 1e6,
    );
    ok
}

fn main() {
    std::thread::Builder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(worker_main)
        .expect("worker thread must start")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn worker_main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");

    let t = Instant::now();
    let heavy_tx = BlockTxCircuit::define(CIRCUIT_CONFIG, HEAVY_TX_PER_PROOF, CHAIN_ID, HEAVY_TX_MODE)
        .builder
        .build::<C>();
    let light_tx = BlockTxCircuit::define(CIRCUIT_CONFIG, LIGHT_TX_PER_PROOF, CHAIN_ID, LIGHT_TX_MODE)
        .builder
        .build::<C>();
    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG)
        .builder
        .build::<C>();
    let heavy_chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &heavy_tx, ON_CHAIN_OPERATIONS_LIMIT)
        .builder
        .build::<C>();
    let light_chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &light_tx, ON_CHAIN_OPERATIONS_LIMIT)
        .builder
        .build::<C>();
    let block = BlockCircuit::define(
        CIRCUIT_CONFIG,
        &pre,
        &light_chain,
        &heavy_chain,
        ON_CHAIN_OPERATIONS_LIMIT,
    )
    .builder
    .build::<C>();
    eprintln!("[sizes] from-scratch build: {:.3}s", t.elapsed().as_secs_f64());

    let all_ok = [
        check("heavy_tx", &heavy_tx, cache_bytes::HEAVY_TX),
        check("light_tx", &light_tx, cache_bytes::LIGHT_TX),
        check("pre", &pre, cache_bytes::PRE),
        check("heavy_chain", &heavy_chain, cache_bytes::HEAVY_CHAIN),
        check("light_chain", &light_chain, cache_bytes::LIGHT_CHAIN),
        check("block", &block, cache_bytes::BLOCK),
    ]
    .into_iter()
    .all(|ok| ok);
    eprintln!("[sizes] ALL_OK={all_ok}");
    if !all_ok {
        std::process::exit(1);
    }
}
