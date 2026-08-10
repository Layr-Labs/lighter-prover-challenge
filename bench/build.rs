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
use sha2::{Digest, Sha256};

// Mirrors of the `src/api.rs` constants (a build script cannot import from
// the crate it builds). Divergence is caught by `embedded_matches_rebuilt`:
// the freshly built and embedded circuits would differ in `circuit_digest`.
const CHAIN_ID: u32 = 304;
const HEAVY_TX_PER_PROOF: usize = 4;
const LIGHT_TX_PER_PROOF: usize = 10;
const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;
const K6_DYLIB_NAME: &str = "liblighter_k6.metallib";
const K6_DYLIB_META_NAME: &str = "k6_dynamic_library_meta.rs";
const K6_DYLIB_HELPER_MODE: &str = "LIGHTER_K6_DYLIB_HELPER";
const K6_DYLIB_HELPER_PATH: &str = "LIGHTER_K6_DYLIB_PATH";
const K6_DYLIB_HELPER_FORCE_ABORT: &str = "LIGHTER_K6_DYLIB_HELPER_FORCE_ABORT";
const K6_DYLIB_HELPER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

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

fn write_k6_dynamic_library_metadata(out_dir: &Path, bytes: &[u8]) {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut source = String::from("pub const K6_DYNAMIC_LIBRARY_SHA256: [u8; 32] = [\n    ");
    for (index, byte) in digest.iter().enumerate() {
        if index != 0 && index % 8 == 0 {
            source.push_str("\n    ");
        }
        write!(&mut source, "0x{byte:02x}, ").expect("writing generated K6 digest source");
    }
    source.push_str("\n];\n");
    std::fs::write(out_dir.join(K6_DYLIB_META_NAME), source)
        .expect("write generated K6 dynamic-library metadata");
}

fn wait_for_k6_dynamic_library_helper(
    mut child: std::process::Child,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let deadline = std::time::Instant::now() + K6_DYLIB_HELPER_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn build_k6_dynamic_library(out_dir: &Path) {
    let path = out_dir.join(K6_DYLIB_NAME);
    let _ = std::fs::remove_file(&path);

    let bytes = if std::env::var_os("LIGHTER_SKIP_K6_DYLIB").is_some_and(|value| value == "1") {
        println!("cargo:warning=LIGHTER_SKIP_K6_DYLIB=1: K6 dynamic library is an empty stub");
        Vec::new()
    } else {
        // The official build bridge has previously SIGSEGV'd inside Metal
        // archive calls. Keep every GPU-touching API in a copy of this build
        // script so a signal, denial, or hang degrades to an empty payload
        // without terminating Cargo or retaining a partial artifact.
        let child = std::env::current_exe().and_then(|executable| {
            std::process::Command::new(executable)
                .env(K6_DYLIB_HELPER_MODE, "1")
                .env(K6_DYLIB_HELPER_PATH, &path)
                .spawn()
        });
        let status = child.and_then(wait_for_k6_dynamic_library_helper);
        match status {
            Ok(Some(status)) if status.success() => std::fs::read(&path).unwrap_or_default(),
            Ok(None) => {
                println!("cargo:warning=K6 Metal helper timed out; embedding empty fallback");
                Vec::new()
            }
            _ => {
                println!("cargo:warning=K6 Metal helper unavailable; embedding empty fallback");
                Vec::new()
            }
        }
    };

    // Overwrite any partial child output. The include_bytes! consumer and its
    // sentinel metadata therefore always describe the same exact payload.
    std::fs::write(&path, &bytes).expect("write embedded K6 dynamic library");
    write_k6_dynamic_library_metadata(out_dir, &bytes);
    println!(
        "cargo:warning=build-host K6 dynamic library: {} bytes{}",
        bytes.len(),
        if bytes.is_empty() { " (runtime-source fallback)" } else { "" }
    );
}

fn run_k6_dynamic_library_helper() -> ! {
    if std::env::var_os(K6_DYLIB_HELPER_FORCE_ABORT).is_some_and(|value| value == "1") {
        std::process::abort();
    }
    let path = PathBuf::from(
        std::env::var_os(K6_DYLIB_HELPER_PATH)
            .expect("K6 dynamic-library helper requires LIGHTER_K6_DYLIB_PATH"),
    );
    match plonky2::hash::poseidon2::write_k6_dynamic_library(&path) {
        Ok(_) => std::process::exit(0),
        Err(error) => {
            eprintln!("K6 dynamic-library helper failed: {error}");
            std::process::exit(1)
        }
    }
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
    if std::env::var_os(K6_DYLIB_HELPER_MODE).is_some_and(|value| value == "1") {
        run_k6_dynamic_library_helper();
    }

    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_K6_DYLIB");
    println!("cargo:rerun-if-env-changed=LIGHTER_K6_DYLIB_HELPER_FORCE_ABORT");
    println!("cargo:rerun-if-changed=../vendor/plonky2/plonky2/src/hash/poseidon2/k6_residual.metal");
    println!("cargo:rerun-if-changed=../vendor/plonky2/plonky2/src/hash/poseidon2/k6_residual_wrapper.metal");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    build_k6_dynamic_library(&out_dir);

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
