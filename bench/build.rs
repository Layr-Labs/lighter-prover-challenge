// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Builds the five startup circuits at compile time and serializes them into
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

const BLOB_NAMES: [&str; 5] = [
    "pre.embed",
    "heavy_tx.embed",
    "heavy_chain.embed",
    "light_tx.embed",
    "light_chain.embed",
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

fn build_path_blobs(tx_per_proof: usize, tx_mode: u8) -> (Vec<u8>, Vec<u8>) {
    // Same construction as `PathCircuits::new`.
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
    (tx_blob, chain_blob)
}

/// Compiles the Poseidon2 Metal kernels into a serialized pipeline binary
/// archive and points the prove binary at it.
///
/// The worker compiles its Metal shaders at runtime. Under the benchmark's
/// sandbox profile that is expensive and unavoidable there: writes to the OS
/// shader cache are denied, which disables the cache for reads too, so every
/// worker process lowers all eight kernels from scratch. Measured cold under
/// the profile: 415 ms for the MSL-to-AIR compile plus 2.69 s of pipeline
/// lowering.
///
/// The lowering can be moved here instead. This script runs in the untimed
/// build job with no sandbox, so it can build an `MTLBinaryArchive` and
/// serialize it; the worker then creates its pipelines from that archive
/// instead of paying the lowering again. Building the archive *inside* the
/// sandbox is not an option — `addComputePipelineFunctionsWithDescriptor:`
/// dies with SIGSEGV there — which is why this belongs at build time.
///
/// Everything here is scheduling: the archive holds the same kernels compiled
/// from the same source, so no value the GPU computes can change. A failure
/// at any step is reported and skipped, and the worker keeps its existing
/// runtime-compile path.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn build_pipeline_archive(out_dir: &Path) {
    use metal::{BinaryArchiveDescriptor, CompileOptions, ComputePipelineDescriptor, Device, URL};

    const KERNELS: [&str; 8] = [
        "poseidon2_hash_leaves",
        "poseidon2_hash_leaves_colmajor",
        "poseidon2_hash_parents",
        "ntt_prepare",
        "ntt_stage",
        "ifft_finalize",
        "poseidon2_gate_quotient",
        "range_check_gate_quotient",
    ];
    const SHADER: &str = "../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal";

    println!("cargo:rerun-if-changed={SHADER}");

    let skip = |reason: &str| {
        println!("cargo:warning=pipeline archive not built ({reason}); worker will compile kernels at startup");
    };

    let source = match std::fs::read_to_string(SHADER) {
        Ok(source) => source,
        Err(error) => return skip(&format!("cannot read shader source: {error}")),
    };
    let Some(device) = Device::system_default() else {
        return skip("no Metal device on the build host");
    };
    let library = match device.new_library_with_source(&source, &CompileOptions::new()) {
        Ok(library) => library,
        Err(error) => return skip(&format!("shader compilation failed: {error}")),
    };
    let archive = match device.new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new()) {
        Ok(archive) => archive,
        Err(error) => return skip(&format!("cannot create binary archive: {error}")),
    };
    for name in KERNELS {
        let function = match library.get_function(name, None) {
            Ok(function) => function,
            Err(error) => return skip(&format!("kernel {name} unavailable: {error}")),
        };
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        if let Err(error) = archive.add_compute_pipeline_functions_with_descriptor(&descriptor) {
            return skip(&format!("cannot add {name} to the archive: {error}"));
        }
    }

    let path = out_dir.join("poseidon2_pipelines.metalar");
    // Serializing onto an existing file fails, so a rebuild starts clean.
    let _ = std::fs::remove_file(&path);
    // `URLWithString:` hands back an autoreleased object that the binding
    // wraps as owned and would release a second time.
    let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
        "file://{}",
        path.display()
    )));
    if let Err(error) = archive.serialize_to_url(&url) {
        return skip(&format!("cannot serialize the archive: {error}"));
    }

    let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
    println!("cargo:warning=pipeline archive: {:.2} MiB", bytes as f64 / (1024.0 * 1024.0));
    println!("cargo:rustc-env=LIGHTER_PIPELINE_ARCHIVE={}", path.display());
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn build_pipeline_archive(_out_dir: &Path) {}

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    build_pipeline_archive(&out_dir);

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
            let (pre_blob, (heavy_blobs, light_blobs)) = rayon::join(
                || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    let pre_target = pre.target;
                    let pre_data = pre.builder.build::<C>();
                    serialize_embedded(&pre_target, &pre_data)
                        .expect("serializing block pre-execution circuit for embedding")
                },
                || {
                    rayon::join(
                        || build_path_blobs(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || build_path_blobs(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );

            write_blob(&out_dir, "pre.embed", &pre_blob);
            write_blob(&out_dir, "heavy_tx.embed", &heavy_blobs.0);
            write_blob(&out_dir, "heavy_chain.embed", &heavy_blobs.1);
            write_blob(&out_dir, "light_tx.embed", &light_blobs.0);
            write_blob(&out_dir, "light_chain.embed", &light_blobs.1);
        })
        .expect("circuit build thread must start")
        .join()
        .expect("circuit build thread must finish");
}
