// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Staleness guard for the committed startup-circuit blobs.
//!
//! The five circuits used to be *constructed* here, at compile time, and written
//! into `OUT_DIR` for `src/embedded.rs` to embed. They are a deterministic
//! function of `circuit/` and `vendor/plonky2/`, so they are committed under
//! `embedded/` instead and `src/embedded.rs` embeds them from there.
//!
//! That trades away a guarantee Cargo used to provide for free. While `circuit`
//! was a build-dependency, any change to it re-ran this script and the blobs
//! could not go stale. Committed blobs can, and a stale blob fails *silently*:
//! `Circuits::load` falls back to `Circuits::new` when
//! `Circuits::from_embedded` rejects a blob on its commitment-cap check, and the
//! `log::warn!` announcing that fallback is compiled out of release builds by
//! `log/release_max_level_off`. The worker would quietly rebuild all five
//! circuits at startup with nothing in any log to say so.
//!
//! So this script re-establishes the guarantee without re-introducing the
//! dependency: it hashes every source that determines circuit shape and compares
//! the digest with `embedded/STAMP`, failing the build on any mismatch. It has no
//! build-dependencies and no crate imports.
//!
//! Regenerate after any change under `circuit/` or `vendor/plonky2/`:
//!
//! ```text
//! cargo run --release -p bench --bin embed_gen
//! ```

include!("stamp_hash.rs");

fn main() {
    // Cargo walks directories given to `rerun-if-changed` recursively, so these
    // entries cover every file the stamp hashes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=stamp_hash.rs");
    println!("cargo:rerun-if-changed=embedded/STAMP");
    for root in STAMP_ROOTS {
        println!("cargo:rerun-if-changed={root}");
    }

    let bench_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );

    let stamp_path = bench_dir.join("embedded/STAMP");
    let recorded = std::fs::read_to_string(&stamp_path)
        .unwrap_or_else(|error| {
            panic!(
                "cannot read {}: {error}\n\
                 The committed startup-circuit blobs are missing their stamp. Regenerate with:\n    \
                 cargo run --release -p bench --bin embed_gen",
                stamp_path.display()
            )
        })
        .trim()
        .to_owned();

    let actual = compute_stamp(&bench_dir);
    assert!(
        recorded == actual,
        "committed startup-circuit blobs are STALE.\n\
         \n\
         embedded/STAMP records {recorded}, but circuit/ + vendor/plonky2/ now hash to {actual}.\n\
         The blobs under bench/embedded/ were built from different sources, so\n\
         `Circuits::from_embedded` would fail its commitment-cap check at run time and fall back\n\
         to building all five circuits from scratch in the scored worker -- silently, because the\n\
         warning is compiled out of release builds.\n\
         \n\
         Regenerate the blobs and the stamp:\n    \
         cargo run --release -p bench --bin embed_gen\n"
    );
}
