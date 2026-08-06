// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Pregenerates compact circuit data at compile time.
//!
//! `Circuits::new` previously rebuilt every circuit on each worker start,
//! paying gate placement, copy-constraint resolution, and sigma/selector
//! polynomial construction inside the benchmark's timed window — once per
//! fixture. This build script runs that construction once, at compile time,
//! and stores each circuit in the compact cache format
//! (`circuit::circuit_cache`); the worker embeds the bytes and loads them,
//! recomputing only the LDE/Merkle commitment, sigma transpose, subgroup, and
//! FFT root table. The chain circuits' shared recursion common data is also
//! written, so the runtime chain defines skip the three-stage throwaway build.
//!
//! Constants here must match `bench/src/api.rs`; both sides assert the same
//! pinned production parameters, and the `verify_caches` binary validates the
//! loaded data against freshly built circuits.

use std::env;
use std::fs;
use std::path::Path;

use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::circuit_cache::{circuit_data_to_compact_bytes, recursion_common_to_bytes};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;

const CHAIN_ID: u32 = 304;
const HEAVY_TX_PER_PROOF: usize = 4;
const LIGHT_TX_PER_PROOF: usize = 10;
const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
const BUILD_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// zstd level for embedded caches. 19 keeps the total embedded payload around
/// 370 MiB (the harness rejects staged binaries over 512 MiB); decompression
/// stays sub-second per cache.
const CACHE_ZSTD_LEVEL: i32 = 19;

fn write_framed(out_dir: &Path, name: &str, bytes: &[u8]) {
    let compressed = zstd::bulk::compress(bytes, CACHE_ZSTD_LEVEL)
        .unwrap_or_else(|error| panic!("cannot compress {name} circuit cache: {error:?}"));
    let mut framed = Vec::with_capacity(8 + compressed.len());
    framed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    framed.extend_from_slice(&compressed);
    let path = out_dir.join(format!("{name}.bin.zst"));
    fs::write(&path, framed)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}

fn write_circuit(out_dir: &Path, name: &str, data: &CircuitData<F, C, D>) {
    let bytes = circuit_data_to_compact_bytes(data)
        .unwrap_or_else(|error| panic!("cannot serialize {name} circuit cache: {error:?}"));
    write_framed(out_dir, name, &bytes);
}

fn generate(out_dir: &Path) {
    struct PathData {
        tx_data: CircuitData<F, C, D>,
        chain_data: CircuitData<F, C, D>,
    }

    let build_path = |tx_per_proof: usize, tx_mode: u8| -> PathData {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
        let tx_data = tx.builder.build::<C>();
        let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_data = chain.builder.build::<C>();
        PathData {
            tx_data,
            chain_data,
        }
    };

    let ((pre_data, heavy), light) = rayon::join(
        || {
            rayon::join(
                || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    pre.builder.build::<C>()
                },
                || build_path(HEAVY_TX_PER_PROOF, TX_HEAVY),
            )
        },
        || build_path(LIGHT_TX_PER_PROOF, TX_LIGHT),
    );

    let block = BlockCircuit::define(
        CIRCUIT_CONFIG,
        &pre_data,
        &light.chain_data,
        &heavy.chain_data,
        ON_CHAIN_OPERATIONS_LIMIT,
    );
    let block_data = block.builder.build::<C>();

    // The chain defines above populated the recursion common-data cache with
    // the same value the workers will install at startup.
    let recursion_common = circuit::block_tx_chain_constraints::recursion_common_data();
    let recursion_common_bytes = recursion_common_to_bytes(&recursion_common)
        .expect("cannot serialize recursion common data");

    rayon::scope(|scope| {
        scope.spawn(|_| write_circuit(out_dir, "heavy_tx", &heavy.tx_data));
        scope.spawn(|_| write_circuit(out_dir, "light_tx", &light.tx_data));
        scope.spawn(|_| write_circuit(out_dir, "pre", &pre_data));
        scope.spawn(|_| write_circuit(out_dir, "heavy_chain", &heavy.chain_data));
        scope.spawn(|_| write_circuit(out_dir, "light_chain", &light.chain_data));
        scope.spawn(|_| write_circuit(out_dir, "block", &block_data));
        scope.spawn(|_| write_framed(out_dir, "recursion_common", &recursion_common_bytes));
    });
}

fn main() {
    // Regenerate only when the inputs that shape the circuits change; the
    // circuit and plonky2 crates are build-dependencies, so their changes
    // already invalidate the script itself.
    println!("cargo::rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set");
    let out_dir = Path::new(&out_dir).to_owned();

    // Circuit construction recurses deeply; keep it off the default stack.
    std::thread::Builder::new()
        .stack_size(BUILD_THREAD_STACK_BYTES)
        .spawn(move || generate(&out_dir))
        .expect("circuit generation thread must start")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
