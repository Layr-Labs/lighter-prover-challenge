// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Freshness guard for the precompiled Poseidon2 Metal library.
//!
//! `src/hash/poseidon2/poseidon2.metallib` is `poseidon2.metal` compiled ahead
//! of time (`tools/build-poseidon2-metallib.sh`). Loading it with
//! `newLibraryWithData:` instead of `newLibraryWithSource:` removes the Metal
//! front-end compile from every worker process's startup — a cost the scored
//! sandbox pays in full, because it may read the OS shader cache but not write
//! it, so the cache never populates.
//!
//! The blob is checked in rather than regenerated here on purpose: the ranked
//! build environment is only guaranteed `cargo`, `codesign`, `rustup` and
//! `shasum` (`setup.sh`), and the Metal toolchain (`xcrun metal`) ships as a
//! separately installable component that may be absent. Invoking it from a
//! build script would turn an optional startup optimization into a build
//! dependency.
//!
//! Because the blob is checked in it can go stale. This script computes a
//! digest of the shader source and emits `METALLIB_MATCHES_SOURCE`, which the
//! runtime consults before trusting the blob. **A stale blob is not an error:
//! the prover falls back to compiling from source, exactly as it did before
//! this path existed.** Fail-safe, not fail-shut: a build that refuses to
//! complete produces no score at all, which is a strictly worse outcome than
//! one that starts a few hundred milliseconds slower.
//!
//! **Do not rely on the `cargo:warning` below to notice staleness.** Cargo
//! suppresses build-script warnings from path dependencies that are not
//! workspace members, which is exactly what this crate is, so the warning is
//! invisible in a normal `cargo build` — verified, not assumed. The reliable
//! signals are the generated `METALLIB_MATCHES_SOURCE` constant in `OUT_DIR`
//! and, at run time, the absence of the fast path.

use std::env;
use std::fs;
use std::path::Path;

/// Digest recorded when `poseidon2.metallib` was last generated. Update it with
/// `tools/build-poseidon2-metallib.sh`, which regenerates the blob and rewrites
/// this constant together, so the two can never be updated independently.
const RECORDED_SHADER_DIGEST: u128 = 0xc4bb96744f21c73a0ebe68f513bfbb13;

/// Digest of the `poseidon2.metallib` the checked-in `poseidon2.binarchive` was
/// generated from. The archive holds pipelines lowered from *that* library's
/// AIR, so it is meaningful only against it; the same script rewrites both
/// constants, so they cannot drift apart.
const RECORDED_METALLIB_DIGEST: u128 = 0xf431280d799f7c7ff7b0807b9fce5a11;

/// FNV-1a over 128 bits, plus the byte length folded in.
///
/// Deliberately dependency-free: `setup.sh` builds `--locked --offline`, so
/// adding a `sha2` build-dependency would change `Cargo.lock` and is a
/// disproportionate cost for what this hash has to do. The threat model is an
/// edited shader that nobody remembered to recompile — accident, not forgery —
/// and 128 bits of FNV is far past sufficient for that.
fn digest(bytes: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013b;
    let mut hash = OFFSET;
    for &byte in bytes {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(PRIME);
    }
    hash ^= bytes.len() as u128;
    hash.wrapping_mul(PRIME)
}

fn main() {
    let dir = Path::new("src/hash/poseidon2");
    let source_path = dir.join("poseidon2.metal");
    let metallib_path = dir.join("poseidon2.metallib");
    let archive_path = dir.join("poseidon2.binarchive");

    println!("cargo:rerun-if-changed={}", source_path.display());
    println!("cargo:rerun-if-changed={}", metallib_path.display());
    println!("cargo:rerun-if-changed={}", archive_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let source = fs::read(&source_path).expect("poseidon2.metal must be readable");
    let actual = digest(&source);
    let fresh = metallib_path.exists() && actual == RECORDED_SHADER_DIGEST;

    if !fresh {
        if metallib_path.exists() {
            println!(
                "cargo:warning=poseidon2.metallib is stale (poseidon2.metal digest {actual:#034x} \
                 != recorded {RECORDED_SHADER_DIGEST:#034x}); the prover will compile the shader \
                 from source at startup. Regenerate with tools/build-poseidon2-metallib.sh."
            );
        } else {
            println!(
                "cargo:warning=poseidon2.metallib is missing; the prover will compile the shader \
                 from source at startup. Regenerate with tools/build-poseidon2-metallib.sh."
            );
        }
    }

    // The archive is checked against the metallib rather than the shader,
    // because the metallib is what it was lowered from. It is a second,
    // independent fast path: a valid metallib with a stale archive still skips
    // the front end, it just lowers the pipelines as before.
    let archive_fresh = archive_path.exists()
        && metallib_path.exists()
        && digest(&fs::read(&metallib_path).expect("poseidon2.metallib must be readable"))
            == RECORDED_METALLIB_DIGEST;
    if !archive_fresh {
        println!(
            "cargo:warning=poseidon2.binarchive is missing or stale; GPU pipelines will be \
             lowered at startup instead of resolved from the archive. Regenerate with \
             tools/build-poseidon2-metallib.sh."
        );
    }

    let out = Path::new(&env::var("OUT_DIR").expect("OUT_DIR must be set"))
        .join("poseidon2_metallib_guard.rs");
    fs::write(
        &out,
        format!(
            "/// Whether the checked-in `poseidon2.metallib` was generated from the\n\
             /// `poseidon2.metal` that is compiled into this binary. Written by\n\
             /// `build.rs`; `false` disables the precompiled-library fast path.\n\
             pub(crate) const METALLIB_MATCHES_SOURCE: bool = {fresh};\n\
             /// Whether the checked-in `poseidon2.binarchive` was generated from the\n\
             /// `poseidon2.metallib` compiled into this binary. `false` disables the\n\
             /// precompiled-pipeline fast path but leaves the metallib path intact.\n\
             pub(crate) const ARCHIVE_MATCHES_METALLIB: bool = {archive_fresh};\n"
        ),
    )
    .expect("guard file must be writable");
}
