// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Developer tool: writes the Poseidon2 GPU binary archive.
//!
//! Driven by `vendor/plonky2/plonky2/tools/build-poseidon2-metallib.sh`, which
//! also records the digest that `build.rs` checks. It exists as a `bench`
//! binary because `vendor/plonky2` is a path dependency rather than a workspace
//! member, so building an example or test inside it would create
//! `vendor/plonky2/target` — which the measurement protocol forbids.
//!
//! Never built or run by `setup.sh`, which builds only `--bin prove`.

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gen-gpu-archive OUTPUT.binarchive");
    assert!(args.next().is_none(), "usage: gen-gpu-archive OUTPUT.binarchive");
    let path = std::path::PathBuf::from(path);
    match plonky2::hash::poseidon2::generate_binary_archive(&path) {
        Ok(()) => println!("wrote {}", path.display()),
        Err(error) => {
            eprintln!("gen-gpu-archive: {error}");
            std::process::exit(1);
        }
    }
}
