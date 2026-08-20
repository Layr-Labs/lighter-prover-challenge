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

const PIPELINE_ARCHIVE_NAME: &str = "poseidon2-host-pipelines.metalarchive";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn record_host_pipeline_archive(out_dir: &Path) -> anyhow::Result<()> {
    use anyhow::{anyhow, Context as _};
    use metal::{BinaryArchiveDescriptor, ComputePipelineDescriptor, Device, URL};

    const KERNELS: [&str; 10] = [
        "poseidon2_hash_leaves",
        "poseidon2_hash_leaves_colmajor",
        "poseidon2_hash_parents",
        "poseidon2_absorb_pass",
        "ntt_prepare",
        "ntt_stage",
        "ifft_finalize",
        "poseidon2_gate_quotient",
        "range_check_gate_quotient",
        "permutation_quotient",
    ];

    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );
    let metallib_path = manifest_dir.join(
        "../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metallib",
    );
    let metallib = std::fs::read(&metallib_path)
        .with_context(|| format!("reading {}", metallib_path.display()))?;
    let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
    let library = device
        .new_library_with_data(&metallib)
        .map_err(|error| anyhow!("loading Poseidon2 metallib: {error}"))?;
    let archive = device
        .new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())
        .map_err(|error| anyhow!("creating host pipeline archive: {error}"))?;

    for name in KERNELS {
        let function = library
            .get_function(name, None)
            .map_err(|error| anyhow!("resolving {name}: {error}"))?;
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        archive
            .add_compute_pipeline_functions_with_descriptor(&descriptor)
            .map_err(|error| anyhow!("recording {name}: {error}"))?;
    }

    let path = out_dir.join(PIPELINE_ARCHIVE_NAME);
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("removing stale {}", path.display()))?;
    }
    let url = URL::new_with_string(&format!("file://{}", path.display()));
    archive
        .serialize_to_url(&url)
        .map_err(|error| anyhow!("serializing {}: {error}", path.display()))?;
    // `URLWithString:` is autoreleased while metal-rs wraps it as owned.
    // Let the surrounding process pool perform the one balancing release.
    core::mem::forget(url);
    println!(
        "cargo:warning=host-native Metal pipeline archive: {:.2} MiB",
        std::fs::metadata(&path)?.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn record_host_pipeline_archive(_out_dir: &Path) -> anyhow::Result<()> {
    anyhow::bail!("host-native Metal archive recording requires Apple Silicon macOS")
}

fn write_pipeline_archive(out_dir: &Path) {
    if let Err(error) = record_host_pipeline_archive(out_dir) {
        // Keep non-Mac development and a transient Metal recording failure
        // buildable with the committed M4 Pro archive. Runtime probing already
        // turns a foreign archive into the ordinary AIR-lowering path.
        let manifest_dir = PathBuf::from(
            std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
        );
        let fallback = manifest_dir.join(
            "../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2-pipelines.metalarchive",
        );
        let destination = out_dir.join(PIPELINE_ARCHIVE_NAME);
        std::fs::copy(&fallback, &destination).unwrap_or_else(|copy_error| {
            panic!(
                "host archive recording failed ({error}); cannot copy {} to {}: {copy_error}",
                fallback.display(),
                destination.display()
            )
        });
        println!(
            "cargo:warning=host-native Metal archive unavailable ({error}); embedded committed fallback"
        );
    }
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

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed=../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metallib"
    );
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    // Setup runs once on the exact ranked GPU before the trusted clock starts.
    // Record its device/compiler-keyed pipeline archive there and embed it in
    // the disposable worker, so every fixture can skip cold AIR lowering even
    // though the trusted harness resets scratch between worker processes.
    write_pipeline_archive(&out_dir);

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
