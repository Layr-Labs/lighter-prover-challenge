// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Build-time constants/sigmas Merkle digest sidecars.
//!
//! The shrunken embed deliberately omits the CS Merkle interiors: they are
//! incompressible Poseidon2 nodes (~80 MiB across the four large trees) and
//! would both blow the 25 MiB submission archive and add ~7.5 ms/MB of
//! macOS code-signature validation if `include_bytes!`'d into the prove
//! binary. `bench/build.rs` therefore writes them next to the untimed
//! circuit blobs (OUT_DIR and `target/{profile}/csmerkle/`). The worker
//! mmaps/reads those sidecars at load, adopts the nodes, and recomputes
//! only the LDE columns. Isolate marker: cs-digest-blob-1786682000.

use anyhow::{Context, Result, bail, ensure};
use plonky2::hash::hash_types::{HashOut, NUM_HASH_OUT_ELTS, RichField};
use plonky2::hash::merkle_tree::{AdoptedCsMerkle, DigestStore, LevelOrderDigests, MerkleTree};
use plonky2::plonk::config::Hasher;
use sha2::{Digest, Sha256};

const MAGIC: u32 = 0x4353_4431; // "CSD1"
const VERSION: u32 = 1;
const TAG_LEVEL: u8 = 1;
const TAG_INTERLEAVED: u8 = 2;

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    ensure!(*pos + 8 <= bytes.len(), "cs-merkle blob truncated (u64)");
    let value = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(value)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    ensure!(*pos + 4 <= bytes.len(), "cs-merkle blob truncated (u32)");
    let value = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(value)
}

fn write_hash<F: RichField>(out: &mut Vec<u8>, hash: &HashOut<F>) {
    for element in hash.elements {
        write_u64(out, element.to_noncanonical_u64());
    }
}

fn read_hash<F: RichField>(bytes: &[u8], pos: &mut usize) -> Result<HashOut<F>> {
    let mut elements = [F::ZERO; NUM_HASH_OUT_ELTS];
    for element in &mut elements {
        *element = F::from_noncanonical_u64(read_u64(bytes, pos)?);
    }
    Ok(HashOut { elements })
}

fn write_hashes<F: RichField>(out: &mut Vec<u8>, hashes: &[HashOut<F>]) {
    write_u64(out, hashes.len() as u64);
    for hash in hashes {
        write_hash(out, hash);
    }
}

fn read_hashes<F: RichField>(bytes: &[u8], pos: &mut usize) -> Result<Vec<HashOut<F>>> {
    let n = usize::try_from(read_u64(bytes, pos)?).context("cs-merkle hash count")?;
    let mut hashes = Vec::with_capacity(n);
    for _ in 0..n {
        hashes.push(read_hash(bytes, pos)?);
    }
    Ok(hashes)
}

/// Encode a constants/sigmas Merkle tree for later adoption.
pub fn encode<F: RichField, H: Hasher<F, Hash = HashOut<F>>>(
    tree: &MerkleTree<F, H>,
) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    write_u64(&mut payload, tree.num_leaves as u64);
    write_hashes(&mut payload, &tree.cap.0);
    if let Some(levels) = &tree.level_digests {
        payload.push(TAG_LEVEL);
        write_u64(&mut payload, levels.level_offsets.len() as u64);
        for offset in &levels.level_offsets {
            write_u64(&mut payload, *offset as u64);
        }
        write_hashes(&mut payload, &levels.nodes);
    } else {
        payload.push(TAG_INTERLEAVED);
        write_hashes(&mut payload, &tree.digests);
    }
    let digest = Sha256::digest(&payload);
    let mut out = Vec::with_capacity(8 + payload.len() + 32);
    write_u32(&mut out, MAGIC);
    write_u32(&mut out, VERSION);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&digest);
    Ok(out)
}

/// Decode a blob. Any checksum or layout error is `Err` (caller falls back).
pub fn decode<F: RichField, H: Hasher<F, Hash = HashOut<F>>>(
    bytes: &[u8],
) -> Result<AdoptedCsMerkle<F, H>> {
    ensure!(bytes.len() >= 8 + 32, "cs-merkle blob too short");
    let mut pos = 0usize;
    let magic = read_u32(bytes, &mut pos)?;
    ensure!(magic == MAGIC, "cs-merkle magic mismatch");
    let version = read_u32(bytes, &mut pos)?;
    ensure!(version == VERSION, "cs-merkle version mismatch");
    let checksum_at = bytes.len() - 32;
    ensure!(checksum_at >= pos, "cs-merkle blob missing payload");
    let payload = &bytes[pos..checksum_at];
    let expected = Sha256::digest(payload);
    ensure!(expected.as_slice() == &bytes[checksum_at..], "cs-merkle checksum mismatch");
    let mut pos = 0usize;
    let num_leaves = usize::try_from(read_u64(payload, &mut pos)?).context("num_leaves")?;
    let cap = read_hashes::<F>(payload, &mut pos)?;
    ensure!(!cap.is_empty() && cap.len().is_power_of_two(), "cs-merkle cap");
    ensure!(pos < payload.len(), "cs-merkle missing tag");
    let tag = payload[pos];
    pos += 1;
    match tag {
        TAG_LEVEL => {
            let n_off = usize::try_from(read_u64(payload, &mut pos)?).context("offsets")?;
            let mut level_offsets = Vec::with_capacity(n_off);
            for _ in 0..n_off {
                level_offsets.push(usize::try_from(read_u64(payload, &mut pos)?)?);
            }
            let nodes = read_hashes::<F>(payload, &mut pos)?;
            ensure!(pos == payload.len(), "cs-merkle level payload trailing");
            Ok(AdoptedCsMerkle {
                num_leaves,
                cap,
                level_digests: Some(LevelOrderDigests {
                    nodes: DigestStore::Owned(nodes),
                    level_offsets,
                }),
                interleaved: None,
            })
        }
        TAG_INTERLEAVED => {
            let interleaved = read_hashes::<F>(payload, &mut pos)?;
            ensure!(pos == payload.len(), "cs-merkle interleaved trailing");
            Ok(AdoptedCsMerkle {
                num_leaves,
                cap,
                level_digests: None,
                interleaved: Some(interleaved),
            })
        }
        _ => bail!("cs-merkle unknown tag {tag}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::hash::poseidon2::hash::Poseidon2Hash;

    type F = GoldilocksField;
    type H = Poseidon2Hash;

    #[test]
    fn decode_rejects_truncated_and_bad_checksum() {
        assert!(decode::<F, H>(&[]).is_err());
        assert!(decode::<F, H>(&[0u8; 16]).is_err());
        let mut blob = encode(&MerkleTree::<F, H> {
            leaves: plonky2::hash::merkle_tree::MerkleLeaves::Rows {
                data: Vec::new(),
                width: 0,
            },
            num_leaves: 2,
            digests: vec![HashOut::from_partial(&[F::from_canonical_u64(1)])],
            level_digests: None,
            cap: plonky2::hash::merkle_tree::MerkleCap(vec![
                HashOut::from_partial(&[F::from_canonical_u64(2)]),
                HashOut::from_partial(&[F::from_canonical_u64(3)]),
            ]),
        })
        .unwrap();
        let last = blob.len() - 1;
        blob[last] ^= 0xff;
        assert!(decode::<F, H>(&blob).is_err());
    }

    #[test]
    fn encode_decode_interleaved_round_trip() {
        let cap0 = HashOut::from_partial(&[F::from_canonical_u64(11)]);
        let cap1 = HashOut::from_partial(&[F::from_canonical_u64(12)]);
        let d0 = HashOut::from_partial(&[F::from_canonical_u64(21)]);
        let d1 = HashOut::from_partial(&[F::from_canonical_u64(22)]);
        let tree = MerkleTree::<F, H> {
            leaves: plonky2::hash::merkle_tree::MerkleLeaves::Rows {
                data: Vec::new(),
                width: 0,
            },
            num_leaves: 4,
            digests: vec![d0, d1],
            level_digests: None,
            cap: plonky2::hash::merkle_tree::MerkleCap(vec![cap0, cap1]),
        };
        let blob = encode(&tree).unwrap();
        let adopted = decode::<F, H>(&blob).unwrap();
        assert_eq!(adopted.num_leaves, 4);
        assert_eq!(adopted.cap, vec![cap0, cap1]);
        assert_eq!(adopted.interleaved.as_deref(), Some(&[d0, d1][..]));
        assert!(adopted.level_digests.is_none());
    }
}
