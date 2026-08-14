// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

// Deterministic content stamp over every input that determines the shape of
// the five embedded startup circuits.
//
// Included by both `bench/build.rs` (which verifies the stamp) and
// `bench/src/bin/embed_gen.rs` (which regenerates the blobs and writes it), so
// the two can never disagree about what was hashed.
//
// The hash is a hand-rolled FNV-1a over 8-byte little-endian words on purpose:
// it must produce the same digest on the pinned local toolchain and on
// whatever toolchain the ranked bridge pins, so nothing from `std`'s hashing
// (whose algorithm carries no stability guarantee) may participate. Reading a
// word at a time rather than a byte at a time keeps the walk well under a
// second even though build scripts compile at `opt-level = 0`.

use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_word(hash: &mut u64, word: u64) {
    *hash ^= word;
    *hash = hash.wrapping_mul(FNV_PRIME);
}

fn fnv_bytes(hash: &mut u64, bytes: &[u8]) {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        fnv_word(hash, u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    let mut tail = [0u8; 8];
    let rest = chunks.remainder();
    tail[..rest.len()].copy_from_slice(rest);
    fnv_word(hash, u64::from_le_bytes(tail));
    fnv_word(hash, bytes.len() as u64);
}

/// Recursively collects the circuit-defining sources under `dir`.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Build outputs are not inputs; `.`-prefixed entries are tooling.
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect(&path, out);
        } else if name.ends_with(".rs") || name == "Cargo.toml" {
            out.push(path);
        }
    }
}

/// The roots whose contents fix the embedded circuits, relative to `bench/`.
/// Deliberately excludes the workspace and `bench` manifests: they carry
/// `[profile.*]` settings, which change codegen but never circuit shape, and a
/// stamp that tripped on a profile edit would be pure friction.
const STAMP_ROOTS: [&str; 4] = [
    "../circuit/Cargo.toml",
    "../circuit/src",
    "../vendor/plonky2",
    "src/bin/embed_gen.rs",
];

/// Computes the stamp for the tree rooted at `bench_dir`.
pub fn compute_stamp(bench_dir: &Path) -> String {
    let mut files = Vec::new();
    for root in STAMP_ROOTS {
        let path = bench_dir.join(root);
        if path.is_dir() {
            collect(&path, &mut files);
        } else if path.is_file() {
            files.push(path);
        }
    }
    // Sort on the path relative to `bench/` so the digest does not depend on
    // where the tree is checked out or on `read_dir` order.
    let mut keyed: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| {
            let rel = path
                .strip_prefix(bench_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, path)
        })
        .collect();
    keyed.sort();

    let mut hash = FNV_OFFSET;
    fnv_word(&mut hash, keyed.len() as u64);
    for (rel, path) in &keyed {
        fnv_bytes(&mut hash, rel.as_bytes());
        match std::fs::read(path) {
            Ok(bytes) => fnv_bytes(&mut hash, &bytes),
            Err(error) => panic!("cannot read circuit stamp input {}: {error}", path.display()),
        }
    }
    format!("{hash:016x}")
}
