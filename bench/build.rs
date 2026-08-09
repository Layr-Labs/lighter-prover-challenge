// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Builds the fixed startup and final block circuits at compile time and
//! serializes them into OUT_DIR blobs that `src/embedded.rs` embeds into the
//! prove binary.
//!
//! Compilation runs in the benchmark's untimed CI job, so the multi-second
//! circuit construction here is free; the scored worker process then loads
//! the blobs in a fraction of the build time (`Circuits::from_embedded` and
//! `Circuits::load_block_circuit`).
//!
//! The circuits and their parameters must match `src/api.rs`
//! (`Circuits::new`/`PathCircuits::new`/`Circuits::build_block_circuit`)
//! exactly; the ignored test
//! `embedded_matches_rebuilt` in `src/embedded.rs` is the equality oracle for
//! that. Cargo re-runs this script whenever the `circuit` or `plonky2` crates
//! change (they are build-dependencies), so the blobs cannot go stale.
//!
//! Set `LIGHTER_SKIP_EMBED=1` to write empty blobs instead (the runtime then
//! falls back to building circuits from scratch); use this to A/B the
//! mechanism or to cut compile time while iterating on unrelated code.
//! `LIGHTER_SKIP_BLOCK_EMBED=1` stubs only the final block blob for a focused
//! compile-time A/B.

use std::path::{Path, PathBuf};

use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::embed::serialize_embedded;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::circuit_data::CircuitData;

// Mirrors of the `src/api.rs` constants (a build script cannot import from
// the crate it builds). Divergence is caught by `embedded_matches_rebuilt`:
// the freshly built and embedded circuits would differ in `circuit_digest`.
const CHAIN_ID: u32 = 304;
const HEAVY_TX_PER_PROOF: usize = 4;
const LIGHT_TX_PER_PROOF: usize = 10;
const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

const BLOB_NAMES: [&str; 6] = [
    "pre.embed",
    "heavy_tx.embed",
    "heavy_chain.embed",
    "light_tx.embed",
    "light_chain.embed",
    "block.embed",
];

fn write_blob(out_dir: &Path, name: &str, bytes: &[u8]) {
    let path = out_dir.join(name);
    std::fs::write(&path, bytes).unwrap_or_else(|error| {
        panic!("cannot write embedded circuit blob {}: {error}", path.display())
    });
    println!(
        "cargo:warning=embedded circuit blob {name}: {:.2} MiB",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
}

fn build_path_blobs(
    tx_per_proof: usize,
    tx_mode: u8,
) -> ((Vec<u8>, Vec<u8>), CircuitData<F, C, D>) {
    // Same construction as `PathCircuits::new`.
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
    let tx_target: BlockTxTarget = tx.target;
    let tx_data = tx.builder.build::<C>();

    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
    let chain_target = chain.target;
    let mut chain_data = chain.builder.build::<C>();

    let tx_blob = serialize_embedded(&tx_target, &tx_data)
        .expect("serializing block transaction circuit for embedding");
    let chain_blob = serialize_embedded(&chain_target, &chain_data)
        .expect("serializing block transaction chain circuit for embedding");
    release_preprocessing(&mut chain_data);
    ((tx_blob, chain_blob), chain_data)
}

/// Releases preprocessing that the final block definition cannot read. It
/// consumes only each input circuit's common and verifier-only data.
fn release_preprocessing(data: &mut CircuitData<F, C, D>) {
    data.prover_only.constants_sigmas_commitment = PolynomialBatch::default();
    data.prover_only.constants_sigmas_quotient_cache = None;
}

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_BLOCK_EMBED");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    if std::env::var_os("LIGHTER_SKIP_EMBED").is_some_and(|v| v == "1") {
        for name in BLOB_NAMES {
            write_blob(&out_dir, name, &[]);
        }
        println!("cargo:warning=LIGHTER_SKIP_EMBED=1: embedded circuit blobs are empty stubs");
        return;
    }

    let skip_block =
        std::env::var_os("LIGHTER_SKIP_BLOCK_EMBED").is_some_and(|value| value == "1");

    // Circuit construction needs deep stacks (recursive gadget definition) on
    // both the spawning thread and the rayon workers, exactly like the prove
    // binary configures them at startup.
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure build-script thread pool");

    std::thread::Builder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(move || {
            // Same layout as `Circuits::new`: pre-execution circuit in
            // parallel with the heavy and light transaction paths.
            let ((pre_blob, pre_data), (heavy, light)) = rayon::join(
                || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    let pre_target = pre.target;
                    let mut pre_data = pre.builder.build::<C>();
                    let pre_blob = serialize_embedded(&pre_target, &pre_data)
                        .expect("serializing block pre-execution circuit for embedding");
                    release_preprocessing(&mut pre_data);
                    (pre_blob, pre_data)
                },
                || {
                    rayon::join(
                        || build_path_blobs(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || build_path_blobs(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );
            let (heavy_blobs, heavy_chain_data) = heavy;
            let (light_blobs, light_chain_data) = light;

            write_blob(&out_dir, "pre.embed", &pre_blob);
            write_blob(&out_dir, "heavy_tx.embed", &heavy_blobs.0);
            write_blob(&out_dir, "heavy_chain.embed", &heavy_blobs.1);
            write_blob(&out_dir, "light_tx.embed", &light_blobs.0);
            write_blob(&out_dir, "light_chain.embed", &light_blobs.1);
            drop(pre_blob);
            drop(heavy_blobs);
            drop(light_blobs);

            let block_blob = if skip_block {
                Vec::new()
            } else {
                let block = BlockCircuit::define(
                    CIRCUIT_CONFIG,
                    &pre_data,
                    &light_chain_data,
                    &heavy_chain_data,
                    ON_CHAIN_OPERATIONS_LIMIT,
                );
                drop(pre_data);
                drop(light_chain_data);
                drop(heavy_chain_data);
                let block_target = block.target;
                let block_data = block.builder.build::<C>();
                serialize_embedded(&block_target, &block_data)
                    .expect("serializing final block circuit for embedding")
            };
            write_blob(&out_dir, "block.embed", &block_blob);
            if block_blob.is_empty() {
                println!(
                    "cargo:warning=LIGHTER_SKIP_BLOCK_EMBED=1: final block blob is an empty stub"
                );
            }
        })
        .expect("circuit build thread must start")
        .join()
        .expect("circuit build thread must finish");
}
