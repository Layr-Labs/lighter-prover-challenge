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
use circuit::cs_merkle;
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

const CS_MERKLE_NAMES: [&str; 5] = [
    "pre.csmerkle",
    "heavy_tx.csmerkle",
    "heavy_chain.csmerkle",
    "light_tx.csmerkle",
    "light_chain.csmerkle",
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

/// `OUT_DIR` is `target/{profile}/build/{pkg}-{hash}/out`. Walk up to the
/// profile directory so the worker can read sidecars next to the binary
/// (`target/release/csmerkle/`), which the ranked sandbox allowlists as
/// `dirname(worker)`.
fn sidecar_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|profile| profile.join("csmerkle"))
        .unwrap_or_else(|| out_dir.join("csmerkle"))
}

fn write_cs_merkle(out_dir: &Path, sidecars: &Path, name: &str, bytes: &[u8]) {
    write_blob(out_dir, name, bytes);
    let _ = std::fs::create_dir_all(sidecars);
    write_blob(sidecars, name, bytes);
}

fn build_path_blobs(tx_per_proof: usize, tx_mode: u8) -> ((Vec<u8>, Vec<u8>), (Vec<u8>, Vec<u8>)) {
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
    let tx_cs = cs_merkle::encode(&tx_data.prover_only.constants_sigmas_commitment.merkle_tree)
        .expect("encoding transaction constants/sigmas Merkle sidecar");
    let chain_cs = cs_merkle::encode(&chain_data.prover_only.constants_sigmas_commitment.merkle_tree)
        .expect("encoding chain constants/sigmas Merkle sidecar");
    ((tx_blob, tx_cs), (chain_blob, chain_cs))
}

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));

    let sidecars = sidecar_dir(&out_dir);
    if std::env::var_os("LIGHTER_SKIP_EMBED").is_some_and(|v| v == "1") {
        for name in BLOB_NAMES {
            write_blob(&out_dir, name, &[]);
        }
        for name in CS_MERKLE_NAMES {
            write_cs_merkle(&out_dir, &sidecars, name, &[]);
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
            let (pre, (heavy, light)) = rayon::join(
                || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    let pre_target = pre.target;
                    let pre_data = pre.builder.build::<C>();
                    let embed = serialize_embedded(&pre_target, &pre_data)
                        .expect("serializing block pre-execution circuit for embedding");
                    let cs = cs_merkle::encode(
                        &pre_data.prover_only.constants_sigmas_commitment.merkle_tree,
                    )
                    .expect("encoding pre-execution constants/sigmas Merkle sidecar");
                    (embed, cs)
                },
                || {
                    rayon::join(
                        || build_path_blobs(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || build_path_blobs(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );

            write_blob(&out_dir, "pre.embed", &pre.0);
            write_cs_merkle(&out_dir, &sidecars, "pre.csmerkle", &pre.1);
            write_blob(&out_dir, "heavy_tx.embed", &heavy.0.0);
            write_cs_merkle(&out_dir, &sidecars, "heavy_tx.csmerkle", &heavy.0.1);
            write_blob(&out_dir, "heavy_chain.embed", &heavy.1.0);
            write_cs_merkle(&out_dir, &sidecars, "heavy_chain.csmerkle", &heavy.1.1);
            write_blob(&out_dir, "light_tx.embed", &light.0.0);
            write_cs_merkle(&out_dir, &sidecars, "light_tx.csmerkle", &light.0.1);
            write_blob(&out_dir, "light_chain.embed", &light.1.0);
            write_cs_merkle(&out_dir, &sidecars, "light_chain.csmerkle", &light.1.1);
        })
        .expect("circuit build thread must start")
        .join()
        .expect("circuit build thread must finish");
}
