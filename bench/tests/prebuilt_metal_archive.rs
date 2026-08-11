// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#[path = "../prebuilt_metal_archive.rs"]
mod prebuilt_metal_archive;

use std::path::PathBuf;

use prebuilt_metal_archive::{read_verified, validate_bytes, FILE_NAME, MAX_COMMITTED_BYTES};

#[test]
fn archive_integrity_gate_rejects_absent_and_stale_inputs() {
    let absent = std::env::temp_dir().join(format!(
        "lighter-absent-metal-archive-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&absent);
    assert!(read_verified(&absent).is_err());

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    validate_bytes(b"abc", 3, ABC_SHA256).expect("matching size and digest must pass");
    assert!(validate_bytes(b"ab", 3, ABC_SHA256).is_err());
    assert!(validate_bytes(b"abd", 3, ABC_SHA256).is_err());
    assert!(validate_bytes(b"abc", MAX_COMMITTED_BYTES + 1, ABC_SHA256).is_err());
}

/// Generation/release gate. It is ignored while the branch intentionally has
/// no fabricated placeholder; the archive-producing macOS workflow must run
/// it explicitly before its artifact is accepted for commit.
#[test]
#[ignore = "requires a generated, pinned Metal archive"]
fn committed_archive_is_present_pinned_and_current() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FILE_NAME);
    let bytes = read_verified(&path).unwrap_or_else(|error| panic!("{error}"));
    assert!(!bytes.is_empty());
    assert!(bytes.len() as u64 <= MAX_COMMITTED_BYTES);
}
