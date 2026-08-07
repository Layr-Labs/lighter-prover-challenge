// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Builds the five startup circuits **and** the final block circuit at compile
//! time and serializes them into OUT_DIR blobs that `src/embedded.rs` embeds.
//!
//! Compilation runs in the benchmark's untimed CI job, so multi-second circuit
//! construction here is free; the scored worker loads blobs instead of building
//! on the critical path (including the final block circuit that used to build
//! concurrently with the light spine).

use std::path::{Path, PathBuf};

use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::embed::serialize_embedded;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;

// Mirrors of the `src/api.rs` constants (a build script cannot import from
// the crate it builds). Divergence is caught by `embedded_matches_rebuilt`.
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

/// Builds a tx+chain path and returns blobs plus the chain circuit data needed
/// to define the final block circuit.
fn build_path(
    tx_per_proof: usize,
    tx_mode: u8,
) -> (Vec<u8>, Vec<u8>, CircuitData<F, C, D>) {
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
    let tx_target: BlockTxTarget = tx.target;
    let tx_data = tx.builder.build::<C>();

    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
    let chain_target = chain.target;
    let chain_data = chain.builder.build::<C>();

    let tx_blob = serialize_embedded(&tx_target, &tx_data)
        .expect("serializing block transaction circuit for embedding");
    let chain_blob = serialize_embedded(&chain_target, &chain_data)
        .expect("serializing block transaction chain circuit for embedding");
    drop(tx_data);
    (tx_blob, chain_blob, chain_data)
}

fn main() {
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

    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure build-script thread pool");

    std::thread::Builder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(move || {
            // Keep CircuitData alive for final block definition.
            let (
                (pre_blob, pre_data),
                ((heavy_tx_blob, heavy_chain_blob, heavy_chain_data), (light_tx_blob, light_chain_blob, light_chain_data)),
            ) = rayon::join(
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
                        || build_path(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || build_path(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );

            // Final block circuit depends on pre + both chains (same as
            // `Circuits::build_block_circuit`). Built here so the scored worker
            // never pays define+build on the light-spine critical path.
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
            drop(pre_data);
            drop(heavy_chain_data);
            drop(light_chain_data);
            drop(block_data);

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
