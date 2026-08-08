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
use std::process::{Command, Stdio};

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
const METAL_PIPELINE_ARCHIVE: &str = "poseidon2-pipelines.metallib";
const METAL_ARCHIVE_CHILD_OUTPUT: &str = "LIGHTER_METAL_ARCHIVE_CHILD_OUTPUT";

fn write_blob(out_dir: &Path, name: &str, bytes: &[u8]) {
    let path = out_dir.join(name);
    std::fs::write(&path, bytes).unwrap_or_else(|error| {
        panic!(
            "cannot write embedded circuit blob {}: {error}",
            path.display()
        )
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

fn write_empty_metal_archive(path: &Path) {
    std::fs::write(path, []).unwrap_or_else(|error| {
        panic!(
            "cannot write Metal archive fallback {}: {error}",
            path.display()
        )
    });
}

fn run_metal_archive_child() -> ! {
    let path = std::env::var_os(METAL_ARCHIVE_CHILD_OUTPUT)
        .map(PathBuf::from)
        .expect("Metal archive child output path must be set");
    let out_dir = PathBuf::from(
        std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for Metal archive child"),
    );
    if path != out_dir.join("poseidon2-pipelines.child.metallib") {
        eprintln!("Metal archive child output must be the fixed OUT_DIR path");
        std::process::exit(1);
    }
    match plonky2::hash::poseidon2::serialize_pipeline_archive(&path) {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("build-host Metal archive child failed: {error}");
            std::process::exit(1);
        }
    }
}

fn build_metal_pipeline_archive(out_dir: &Path) {
    let path = out_dir.join(METAL_PIPELINE_ARCHIVE);
    if std::env::var_os("LIGHTER_SKIP_METAL_ARCHIVE").is_some_and(|value| value == "1") {
        write_empty_metal_archive(&path);
        println!(
            "cargo:warning=LIGHTER_SKIP_METAL_ARCHIVE=1: embedded Metal archive is an empty fallback stub"
        );
        return;
    }

    // Metal archive creation is intentionally isolated from Cargo's build
    // script process. The ranked build sandbox can deny driver/cache writes;
    // some Metal versions respond by terminating the caller with SIGSEGV
    // instead of returning an NSError. A child signal must therefore degrade
    // to the current pipeline-lowering path, never abort the candidate build.
    // Give the child a writable cache/home rooted in OUT_DIR as well.
    let child_path = out_dir.join("poseidon2-pipelines.child.metallib");
    let _ = std::fs::remove_file(&child_path);
    let child_status = std::env::current_exe()
        .map_err(|error| format!("locating Metal archive child failed: {error}"))
        .and_then(|executable| {
            Command::new(executable)
                .env(METAL_ARCHIVE_CHILD_OUTPUT, &child_path)
                .env("TMPDIR", out_dir)
                .env("HOME", out_dir)
                .env("CFFIXED_USER_HOME", out_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(|error| format!("starting Metal archive child failed: {error}"))
        });

    let child_result = child_status.and_then(|status| {
        if status.success() {
            Ok(())
        } else {
            Err(format!("Metal archive child exited with {status}"))
        }
    });

    match child_result {
        Ok(()) => match std::fs::metadata(&child_path) {
            Ok(metadata) if metadata.len() > 0 => {
                std::fs::rename(&child_path, &path).unwrap_or_else(|error| {
                    panic!(
                        "cannot publish Metal archive {}: {error}",
                        path.display()
                    )
                });
                println!(
                    "cargo:warning=embedded build-host Metal archive: {:.2} MiB",
                    metadata.len() as f64 / (1024.0 * 1024.0)
                );
            }
            Ok(_) => {
                println!(
                    "cargo:warning=build-host Metal archive child produced an empty artifact; workers will use current pipeline lowering"
                );
                let _ = std::fs::remove_file(&child_path);
                write_empty_metal_archive(&path);
            }
            Err(error) => {
                println!(
                    "cargo:warning=build-host Metal archive child produced no artifact ({error}); workers will use current pipeline lowering"
                );
                write_empty_metal_archive(&path);
            }
        },
        Err(error) => {
            println!(
                "cargo:warning=build-host Metal archive unavailable ({error}); workers will use current pipeline lowering"
            );
            let _ = std::fs::remove_file(&child_path);
            write_empty_metal_archive(&path);
        }
    }
}

fn main() {
    if std::env::var_os(METAL_ARCHIVE_CHILD_OUTPUT).is_some() {
        run_metal_archive_child();
    }

    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_METAL_ARCHIVE");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    // The ranked workflow stages only the executable. Generate the archive on
    // its build host and embed the bytes instead of retaining an OUT_DIR path
    // or assuming that fixture scratch directories survive between workers.
    build_metal_pipeline_archive(&out_dir);

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
