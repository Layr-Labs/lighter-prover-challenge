// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Builds the five startup circuits and the final block circuit at compile time
//! and serializes them into
//! OUT_DIR blobs that `src/embedded.rs` embeds into the prove binary.
//!
//! Compilation runs in the benchmark's untimed CI job, so the multi-second
//! circuit construction here is free; the scored worker process then loads
//! the blobs in a fraction of the build time (`Circuits::from_embedded`).
//!
//! The circuits and their parameters must match `src/api.rs`
//! (`Circuits::new`/`PathCircuits::new`) exactly; the ignored test
//! `embedded_matches_rebuilt` in `src/embedded.rs` is the equality oracle for
//! that. Cargo re-runs this script whenever the `circuit` or `plonky2` crates
//! change (they are build-dependencies), so the blobs cannot go stale.
//!
//! Set `LIGHTER_SKIP_EMBED=1` to write empty blobs instead (the runtime then
//! falls back to building circuits from scratch); use this to A/B the
//! mechanism or to cut compile time while iterating on unrelated code.

use std::path::{Path, PathBuf};

use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::embed::serialize_embedded;
use circuit::types::config::{C, CIRCUIT_CONFIG};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

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

struct PathBuild {
    tx_blob: Vec<u8>,
    chain: BlockTxChainCircuit,
}

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

fn prepare_path(tx_per_proof: usize, tx_mode: u8) -> PathBuild {
    // Same construction as `PathCircuits::new`.
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
    let tx_target: BlockTxTarget = tx.target;
    let tx_data = tx.builder.build::<C>();

    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
    let tx_blob = serialize_embedded(&tx_target, &tx_data)
        .expect("serializing block transaction circuit for embedding");
    PathBuild { tx_blob, chain }
}

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    if std::env::var_os("LIGHTER_SKIP_EMBED").is_some_and(|v| v == "1") {
        for name in BLOB_NAMES {
            write_blob(&out_dir, name, &[]);
        }
        println!("cargo:warning=LIGHTER_SKIP_EMBED=1: embedded circuit blobs are empty stubs");
        return;
    }

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
                    let pre_data = pre.builder.build::<C>();
                    let pre_blob = serialize_embedded(&pre_target, &pre_data)
                        .expect("serializing block pre-execution circuit for embedding");
                    (pre_blob, pre_data)
                },
                || {
                    rayon::join(
                        || prepare_path(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || prepare_path(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );

            let PathBuild {
                tx_blob: heavy_tx_blob,
                chain: heavy_chain,
            } = heavy;
            let PathBuild {
                tx_blob: light_tx_blob,
                chain: light_chain,
            } = light;
            let (heavy_chain, light_chain) = rayon::join(
                || {
                    let chain_target = heavy_chain.target;
                    let chain_data = heavy_chain.builder.build::<C>();
                    let chain_blob = serialize_embedded(&chain_target, &chain_data)
                        .expect("serializing heavy transaction chain circuit for embedding");
                    (chain_blob, chain_data)
                },
                || {
                    let chain_target = light_chain.target;
                    let chain_data = light_chain.builder.build::<C>();
                    let chain_blob = serialize_embedded(&chain_target, &chain_data)
                        .expect("serializing light transaction chain circuit for embedding");
                    (chain_blob, chain_data)
                },
            );
            let (heavy_chain_blob, heavy_chain_data) = heavy_chain;
            let (light_chain_blob, light_chain_data) = light_chain;

            // The final circuit depends on the three verifier circuits, but
            // not their witnesses or targets. Build and serialize it only
            // after the independent startup circuits finish so compile-time
            // parallelism does not increase peak memory unnecessarily.
            let block = BlockCircuit::define(
                CIRCUIT_CONFIG,
                &pre_data,
                &light_chain_data,
                &heavy_chain_data,
                ON_CHAIN_OPERATIONS_LIMIT,
            );
            let block_target = block.target;
            let block_data = block.builder.build::<C>();
            let block_blob = serialize_embedded(&block_target, &block_data)
                .expect("serializing final block circuit for embedding");

            write_blob(&out_dir, "pre.embed", &pre_blob);
            write_blob(&out_dir, "heavy_tx.embed", &heavy_tx_blob);
            write_blob(&out_dir, "heavy_chain.embed", &heavy_chain_blob);
            write_blob(&out_dir, "light_tx.embed", &light_tx_blob);
            write_blob(&out_dir, "light_chain.embed", &light_chain_blob);
            write_blob(&out_dir, "block.embed", &block_blob);
        })
        .expect("circuit build thread must start")
        .join()
        .expect("circuit build thread must finish");
}
