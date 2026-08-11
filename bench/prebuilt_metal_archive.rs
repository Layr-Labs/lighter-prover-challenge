// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Integrity gate for the source-tracked Metal binary archive.
//!
//! A Metal binary archive is device-family-specific, opaque compiler output.
//! Never accept a file merely because it has the expected name: a partial,
//! stale, or accidentally replaced archive must remain a soft runtime miss.

use std::path::Path;

use sha2::{Digest, Sha256};

pub const FILE_NAME: &str = "poseidon2-pipelines.binary.metallib";

/// Hard package-size ceiling for the committed archive. The current Yukon
/// source package is under 3 MiB and the upload limit is 25 MiB, so 16 MiB
/// leaves margin for tar metadata and future source changes.
pub const MAX_COMMITTED_BYTES: u64 = 16 * 1024 * 1024;

/// Generated offline on a GitHub-hosted arm64 runner with Xcode 26.5 / AIR 2.8
/// for macOS 26 deployment and the exact eleven-pipeline roster, then reduced
/// to the M4 `g16g`/`g16s` slices. Runtime `FailOnBinaryArchiveMiss` probes
/// remain authoritative.
pub const EXPECTED_BYTES: Option<u64> = Some(1_364_368);
pub const EXPECTED_SHA256: Option<&str> =
    Some("83d04aeeaaf75a9cd2977f431cd832dfd5cfc18ffa0184d7cac72222e13de83a");

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn validate_bytes(
    bytes: &[u8],
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    if expected_bytes == 0 || expected_bytes > MAX_COMMITTED_BYTES {
        return Err(format!(
            "pinned Metal archive size {expected_bytes} is outside 1..={MAX_COMMITTED_BYTES}"
        ));
    }
    if bytes.len() as u64 != expected_bytes {
        return Err(format!(
            "Metal archive size mismatch: got {}, pinned {expected_bytes}",
            bytes.len()
        ));
    }
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("pinned Metal archive SHA-256 must be 64 lowercase hex digits".to_owned());
    }
    let actual_sha256 = sha256_hex(bytes);
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "Metal archive SHA-256 mismatch: got {actual_sha256}, pinned {expected_sha256}"
        ));
    }
    Ok(())
}

pub fn read_verified(path: &Path) -> Result<Vec<u8>, String> {
    let expected_bytes = EXPECTED_BYTES.ok_or_else(|| {
        "committed Metal archive size is not pinned; set EXPECTED_BYTES with the artifact"
            .to_owned()
    })?;
    let expected_sha256 = EXPECTED_SHA256.ok_or_else(|| {
        "committed Metal archive SHA-256 is not pinned; set EXPECTED_SHA256 with the artifact"
            .to_owned()
    })?;
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "reading committed Metal archive {} failed: {error}",
            path.display()
        )
    })?;
    validate_bytes(&bytes, expected_bytes, expected_sha256)?;
    Ok(bytes)
}
