// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Compile-time circuit embedding ("shrunken embed").
//!
//! [`serialize_embedded`] turns a built [`CircuitData`] plus its target struct
//! into a compact blob at *compile time* (from a build script, where time is
//! untimed), and [`deserialize_embedded`] reconstitutes the exact same
//! `CircuitData` at runtime far faster than re-running circuit construction.
//!
//! The blob deliberately omits everything that is cheap to recompute and
//! expensive to store, keeping the binary small enough that macOS's per-exec
//! code-signature validation (~7.5 ms/MB on every fresh inode) does not eat
//! the startup win:
//!
//! * the 80 sigma coefficient polynomials (~40 MiB/tx circuit) are **not**
//!   stored — sigma *values* are re-derived from the representative map with
//!   the same [`Forest::wire_partition`] + [`WirePartition::get_sigma_polys`]
//!   code the builder itself uses, and the constants/sigmas commitment is
//!   recomputed through [`PolynomialBatch::from_values`], the builder's own
//!   commitment path, guaranteeing a bit-identical Merkle cap;
//! * the representative map is stored as zigzag-varint deltas against the
//!   identity permutation (mostly zeros) instead of 8-byte usizes;
//! * the generator watch index is stored in its CSR form (varint-delta
//!   offsets + `u32` watcher ids);
//! * constant polynomials are stored as *values* (step-function selectors,
//!   long constant runs) rather than incompressible coefficients;
//! * every bulky section is independently zstd-compressed, keeping parallel
//!   load memory bounded without a second whole-blob decompression pass.
//!
//! Everything recomputed is validated at load: the recomputed commitment cap
//! must equal the embedded verifier data's cap, which transitively pins the
//! circuit digest. On any mismatch the loader errors and callers fall back to
//! building circuits from scratch.

use anyhow::{Context, Result, bail, ensure};
use plonky2::field::fft::{cached_fft_root_table, cached_two_adic_subgroup};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::circuit_data::{
    CircuitData, GeneratorWatchIndex, ProverOnlyCircuitData, VerifierOnlyCircuitData,
};
use plonky2::plonk::permutation_argument::{fixed_routed_wire_mask, Forest};
use plonky2::util::serialization::{Buffer, Read as _, Write as _};
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, transpose_poly_values_ref};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::circuit_serializer::{BlockGateSerializer, BlockGeneratorSerializer};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::types::config::{C, D, F};

/// Generator registry used for embedded blobs: the block registry already
/// covers every generator of the five embedded circuits (the chain circuits
/// use `dummy_proof_and_constant_vk_no_generator`, so no `DummyProofGenerator`
/// ever reaches a generator list).
pub type EmbedGeneratorSerializer = BlockGeneratorSerializer<C, D, Secp256K1>;

fn embed_generator_serializer() -> EmbedGeneratorSerializer {
    EmbedGeneratorSerializer {
        _phantom: Default::default(),
        _phantom2: Default::default(),
    }
}

const EMBED_MAGIC: u32 = 0x4C45_4331; // "LEC1"
const EMBED_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Primitive encoding helpers
// ---------------------------------------------------------------------------

fn write_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_uvarint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(*pos)
            .context("varint stream truncated in embedded circuit blob")?;
        *pos += 1;
        ensure!(shift < 64, "varint overflow in embedded circuit blob");
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

const fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

const fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Frames `bytes` as `[u64 LE length][bytes]`.
fn write_section(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_section<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    ensure!(
        bytes.len() >= *pos + 8,
        "embedded circuit blob truncated at section header"
    );
    let len = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap()) as usize;
    *pos += 8;
    ensure!(
        bytes.len() >= *pos + len,
        "embedded circuit blob truncated inside section"
    );
    let section = &bytes[*pos..*pos + len];
    *pos += len;
    Ok(section)
}

fn write_compressed_section(out: &mut Vec<u8>, raw: &[u8]) {
    // Compression happens in the untimed build job. Encoding each bulky
    // section directly with zstd avoids the former runtime double decode
    // (whole-blob zstd followed by per-section LZ4) while retaining bounded
    // per-section peak memory during parallel circuit loading.
    let compressed = zstd::bulk::compress(raw, 19)
        .expect("zstd-compressing embedded circuit section");
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    write_section(out, &compressed);
}

fn read_compressed_section(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>> {
    ensure!(
        bytes.len() >= *pos + 8,
        "embedded circuit blob truncated at compressed section header"
    );
    let raw_len = usize::try_from(u64::from_le_bytes(
        bytes[*pos..*pos + 8].try_into().unwrap(),
    ))
    .context("embedded circuit compressed section length exceeds usize")?;
    *pos += 8;
    let compressed = read_section(bytes, pos)?;
    let raw = zstd::bulk::decompress(compressed, raw_len)
        .context("embedded circuit blob failed zstd section decompression")?;
    ensure!(
        raw.len() == raw_len,
        "embedded circuit compressed section expanded to {} bytes, expected {raw_len}",
        raw.len()
    );
    Ok(raw)
}

// ---------------------------------------------------------------------------
// Write side (build script)
// ---------------------------------------------------------------------------

/// Serializes a built circuit (target struct + [`CircuitData`]) into an
/// embeddable blob. Companion of [`deserialize_embedded`].
pub fn serialize_embedded<T: Serialize>(target: &T, data: &CircuitData<F, C, D>) -> Result<Vec<u8>> {
    let gate_serializer = BlockGateSerializer;
    let generator_serializer = embed_generator_serializer();
    let common = &data.common;
    let prover = &data.prover_only;
    let degree = 1usize << common.degree_bits();
    let num_constants = common.num_constants;
    let num_routed = common.config.num_routed_wires;
    let polys = &prover.constants_sigmas_commitment.polynomials;
    ensure!(
        polys.len() == num_constants + num_routed,
        "constants/sigmas commitment holds {} polynomials, expected {} constants + {} sigmas \
         (was the commitment released or built with commit_to_sigma = false?)",
        polys.len(),
        num_constants,
        num_routed,
    );

    let mut out = Vec::new();
    out.extend_from_slice(&EMBED_MAGIC.to_le_bytes());
    out.extend_from_slice(&EMBED_VERSION.to_le_bytes());

    // common
    let mut buf = Vec::new();
    buf.write_common_circuit_data(common, &gate_serializer)
        .map_err(|e| anyhow::anyhow!("serializing common circuit data: {e:?}"))?;
    write_section(&mut out, &buf);

    // verifier_only
    let mut buf = Vec::new();
    buf.write_verifier_only_circuit_data(&data.verifier_only)
        .map_err(|e| anyhow::anyhow!("serializing verifier-only circuit data: {e:?}"))?;
    write_section(&mut out, &buf);

    // target struct
    let target_bytes = bincode::serialize(target).context("serializing circuit target struct")?;
    write_compressed_section(&mut out, &target_bytes);

    // public inputs
    let mut buf = Vec::new();
    buf.write_target_vec(&prover.public_inputs)
        .map_err(|e| anyhow::anyhow!("serializing public inputs: {e:?}"))?;
    write_compressed_section(&mut out, &buf);

    // lookups (none in practice for the embedded circuits, serialized
    // generically so a future lookup-bearing circuit fails loudly on digest
    // rather than silently losing data)
    let mut buf = Vec::new();
    buf.write_usize(prover.lookup_rows.len()).unwrap();
    for wire in &prover.lookup_rows {
        buf.write_usize(wire.last_lu_gate).unwrap();
        buf.write_usize(wire.last_lut_gate).unwrap();
        buf.write_usize(wire.first_lut_gate).unwrap();
    }
    buf.write_usize(prover.lut_to_lookups.len()).unwrap();
    for lut in &prover.lut_to_lookups {
        buf.write_target_lut(lut)
            .map_err(|e| anyhow::anyhow!("serializing lookup table: {e:?}"))?;
    }
    write_section(&mut out, &buf);

    // generators
    let mut buf = Vec::new();
    buf.write_usize(prover.generators.len()).unwrap();
    for generator in &prover.generators {
        buf.write_generator::<F, D>(generator, &generator_serializer, common)
            .map_err(|e| {
                anyhow::anyhow!(
                    "serializing generator {:?} (missing from EmbedGeneratorSerializer registry?): {e:?}",
                    generator.0.id()
                )
            })?;
    }
    write_compressed_section(&mut out, &buf);

    // watch index CSR: offsets as varint deltas (mostly zero), watchers as u32
    let offsets = prover.generator_indices_by_watches.offsets();
    let watchers = prover.generator_indices_by_watches.watchers();
    let mut buf = Vec::new();
    write_uvarint(&mut buf, offsets.len() as u64);
    let mut previous = 0u32;
    for &offset in offsets {
        ensure!(offset >= previous, "watch index offsets must be sorted");
        write_uvarint(&mut buf, u64::from(offset - previous));
        previous = offset;
    }
    write_compressed_section(&mut out, &buf);

    let mut buf = Vec::with_capacity(4 * watchers.len() + 8);
    write_uvarint(&mut buf, watchers.len() as u64);
    for &watcher in watchers {
        let watcher = u32::try_from(watcher).context("generator index exceeds u32")?;
        buf.extend_from_slice(&watcher.to_le_bytes());
    }
    write_compressed_section(&mut out, &buf);

    // constant polynomial *values* (fft of the committed coefficients; the
    // loader inverts this with the same exact-arithmetic ifft the builder's
    // from_values uses, so the round trip is bit-exact)
    let mut buf = Vec::new();
    buf.write_usize(num_constants).unwrap();
    buf.write_usize(degree).unwrap();
    for poly in &polys[..num_constants] {
        ensure!(poly.coeffs.len() == degree, "constant polynomial length");
        let values = poly.clone().fft();
        buf.write_field_vec(&values.values)
            .map_err(|e| anyhow::anyhow!("serializing constant polynomial values: {e:?}"))?;
    }
    write_compressed_section(&mut out, &buf);

    // representative map: zigzag varint deltas against the identity map
    let mut buf = Vec::with_capacity(2 * prover.representative_map.len() + 8);
    write_uvarint(&mut buf, prover.representative_map.len() as u64);
    for (index, &parent) in prover.representative_map.iter().enumerate() {
        write_uvarint(&mut buf, zigzag(i64::from(parent) - index as i64));
    }
    write_compressed_section(&mut out, &buf);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Read side (runtime)
// ---------------------------------------------------------------------------

/// Reconstructs the target struct and the full [`CircuitData`] from a blob
/// produced by [`serialize_embedded`].
///
/// The returned `CircuitData` is value-identical to the freshly built one:
/// deserialized components are byte round trips, and every recomputed
/// component (subgroup, FFT root table, sigma values/transpose, watch counts,
/// constants/sigmas commitment) is derived by the same code paths the builder
/// itself runs, from the same inputs. The recomputed commitment cap is checked
/// against the embedded verifier data before returning.
pub fn deserialize_embedded<T: DeserializeOwned>(bytes: &[u8]) -> Result<(T, CircuitData<F, C, D>)> {
    let gate_serializer = BlockGateSerializer;
    let generator_serializer = embed_generator_serializer();

    // Blobs are zstd-wrapped as a whole (frame magic 0x28 B5 2F FD); unwrap
    // before parsing. Unwrapped blobs still parse directly, so a build with
    // the wrap disabled degrades to the previous format instead of failing.
    let unwrapped;
    let bytes = if bytes.len() >= 4 && bytes[0..4] == [0x28, 0xB5, 0x2F, 0xFD] {
        unwrapped = zstd::stream::decode_all(bytes)
            .context("embedded circuit blob failed zstd unwrap")?;
        &unwrapped[..]
    } else {
        bytes
    };

    ensure!(bytes.len() >= 8, "embedded circuit blob too short");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    ensure!(magic == EMBED_MAGIC, "embedded circuit blob magic mismatch");
    ensure!(
        version == EMBED_VERSION,
        "embedded circuit blob version {version} unsupported"
    );
    let mut pos = 8usize;

    // common
    let section = read_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(section);
    let common = reader
        .read_common_circuit_data::<F, D>(&gate_serializer)
        .map_err(|e| anyhow::anyhow!("deserializing common circuit data: {e:?}"))?;

    // verifier_only
    let section = read_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(section);
    let verifier_only: VerifierOnlyCircuitData<C, D> = reader
        .read_verifier_only_circuit_data()
        .map_err(|e| anyhow::anyhow!("deserializing verifier-only circuit data: {e:?}"))?;

    // target struct
    let section = read_compressed_section(bytes, &mut pos)?;
    let target: T =
        bincode::deserialize(&section).context("deserializing circuit target struct")?;

    // public inputs
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(&section);
    let public_inputs = reader
        .read_target_vec()
        .map_err(|e| anyhow::anyhow!("deserializing public inputs: {e:?}"))?;

    // lookups
    let section = read_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(section);
    let lookup_count = reader
        .read_usize()
        .map_err(|e| anyhow::anyhow!("deserializing lookup rows: {e:?}"))?;
    let mut lookup_rows = Vec::with_capacity(lookup_count);
    for _ in 0..lookup_count {
        let read = |r: &mut Buffer| -> Result<usize> {
            r.read_usize()
                .map_err(|e| anyhow::anyhow!("deserializing lookup rows: {e:?}"))
        };
        lookup_rows.push(plonky2::plonk::circuit_builder::LookupWire {
            last_lu_gate: read(&mut reader)?,
            last_lut_gate: read(&mut reader)?,
            first_lut_gate: read(&mut reader)?,
        });
    }
    let lut_count = reader
        .read_usize()
        .map_err(|e| anyhow::anyhow!("deserializing lookup tables: {e:?}"))?;
    let mut lut_to_lookups = Vec::with_capacity(lut_count);
    for _ in 0..lut_count {
        lut_to_lookups.push(
            reader
                .read_target_lut()
                .map_err(|e| anyhow::anyhow!("deserializing lookup tables: {e:?}"))?,
        );
    }

    // generators
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(&section);
    let generator_count = reader
        .read_usize()
        .map_err(|e| anyhow::anyhow!("deserializing generators: {e:?}"))?;
    let mut generators = Vec::with_capacity(generator_count);
    for _ in 0..generator_count {
        generators.push(
            reader
                .read_generator::<F, D>(&generator_serializer, &common)
                .map_err(|e| anyhow::anyhow!("deserializing generator: {e:?}"))?,
        );
    }
    let generator_readiness_is_authoritative = generators
        .iter()
        .map(|generator| generator.0.readiness_is_authoritative())
        .collect();

    // watch index
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut vpos = 0usize;
    let offsets_len = read_uvarint(&section, &mut vpos)? as usize;
    let mut offsets = Vec::with_capacity(offsets_len);
    let mut running = 0u64;
    for _ in 0..offsets_len {
        running += read_uvarint(&section, &mut vpos)?;
        offsets.push(u32::try_from(running).context("watch index offset exceeds u32")?);
    }
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut vpos = 0usize;
    let watchers_len = read_uvarint(&section, &mut vpos)? as usize;
    ensure!(
        section.len() == vpos + 4 * watchers_len,
        "watch index watcher section length mismatch"
    );
    let mut watchers = Vec::with_capacity(watchers_len);
    for chunk in section[vpos..].chunks_exact(4) {
        let watcher = u32::from_le_bytes(chunk.try_into().unwrap()) as usize;
        ensure!(watcher < generator_count, "watcher index out of range");
        watchers.push(watcher);
    }
    // Watch counts are a pure function of the (deduplicated) watcher lists;
    // this mirrors `read_prover_only_circuit_data`'s reconstruction.
    let mut generator_watch_counts = vec![0usize; generator_count];
    for &watcher in &watchers {
        generator_watch_counts[watcher] += 1;
    }
    let generator_indices_by_watches = GeneratorWatchIndex::from_parts(offsets, watchers);

    // constant polynomial values
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut reader = Buffer::new(&section);
    let num_constants = reader
        .read_usize()
        .map_err(|e| anyhow::anyhow!("deserializing constant polynomials: {e:?}"))?;
    let degree = reader
        .read_usize()
        .map_err(|e| anyhow::anyhow!("deserializing constant polynomials: {e:?}"))?;
    ensure!(
        num_constants == common.num_constants,
        "embedded constant polynomial count diverges from common circuit data"
    );
    ensure!(
        degree == 1usize << common.degree_bits(),
        "embedded constant polynomial degree diverges from common circuit data"
    );
    let mut constant_values = Vec::with_capacity(num_constants);
    for _ in 0..num_constants {
        constant_values.push(PolynomialValues::new(
            reader
                .read_field_vec::<F>(degree)
                .map_err(|e| anyhow::anyhow!("deserializing constant polynomials: {e:?}"))?,
        ));
    }

    // representative map
    let section = read_compressed_section(bytes, &mut pos)?;
    let mut vpos = 0usize;
    let repmap_len = read_uvarint(&section, &mut vpos)? as usize;
    let mut representative_map = Vec::with_capacity(repmap_len);
    for index in 0..repmap_len {
        let delta = unzigzag(read_uvarint(&section, &mut vpos)?);
        let parent = index as i64 + delta;
        representative_map
            .push(u32::try_from(parent).context("representative map entry out of range")?);
    }
    ensure!(pos == bytes.len(), "trailing bytes in embedded circuit blob");

    // ---- recompute the derived prover-only components ----
    let degree_bits = common.degree_bits();
    let rate_bits = common.config.fri_config.rate_bits;
    let cap_height = common.config.fri_config.cap_height;
    let num_wires = common.config.num_wires;
    let num_routed = common.config.num_routed_wires;

    // The embedded loads run concurrently and several circuits share a degree
    // or FFT-domain size, so route these deterministic derivations through the
    // process-wide caches instead of recomputing the primitive-root power
    // chains per load. The cached values are value-identical to a fresh
    // computation (the cache stores exactly what `two_adic_subgroup` /
    // `fft_root_table` produce), so this is startup-only deduplication.
    let subgroup = cached_two_adic_subgroup::<F>(degree_bits).as_ref().clone();

    // Same table size expression as `try_build_with_options`.
    let max_fft_points =
        1usize << (degree_bits + rate_bits.max(log2_ceil(common.quotient_degree_factor)));
    let root_table = cached_fft_root_table::<F>(max_fft_points);

    // Sigma values from the representative map, through the builder's own
    // forest partition code (`sigma_vecs` post-`compress_paths` state).
    let mut forest = Forest::from_parents(representative_map, num_wires, num_routed, degree);
    let wire_partition = forest.wire_partition();
    let sigma_vecs = wire_partition.get_sigma_polys(degree_bits, &common.k_is, &subgroup);
    let representative_map = forest.into_parents();
    let fixed_routed_wires =
        fixed_routed_wire_mask(&representative_map, num_wires, num_routed, degree)
            .context("embedded circuit has an invalid compressed representative map")?;

    // `prover_only.sigmas` is the transpose of the sigma *values*, and the
    // commitment below consumes those same values. Transposing first reads the
    // columns in place, so they can then be moved into the commitment instead
    // of cloned; the clone was one extra full copy of the sigma columns
    // (`num_routed_wires * degree` field elements) per circuit. Only the order
    // of two independent reads changes — no quantity is computed differently.
    let sigmas = transpose_poly_values_ref(&sigma_vecs);

    // The builder's commitment path: values in, IFFT inside, LDE + Merkle.
    // `PlonkOracle::CONSTANTS_SIGMAS.blinding` is `false` (non-ZK circuits).
    let mut constants_sigmas_vecs = constant_values;
    constants_sigmas_vecs.extend(sigma_vecs);
    let constants_sigmas_commitment = PolynomialBatch::<F, C, D>::from_values(
        constants_sigmas_vecs,
        rate_bits,
        false,
        cap_height,
        &mut TimingTree::default(),
        Some(&root_table),
    );
    if constants_sigmas_commitment.merkle_tree.cap != verifier_only.constants_sigmas_cap {
        bail!(
            "recomputed constants/sigmas commitment cap diverges from the embedded verifier data \
             (stale or corrupt embedded circuit blob)"
        );
    }

    let circuit_digest = verifier_only.circuit_digest;

    // Mirror the builder's quotient-domain constants/sigmas cache (added by the
    // Metal quotient-gate union frontier). It is a pure derivation from the
    // freshly recomputed column-backed commitment — the same extraction the
    // builder performs — and the documented `None` fallback keeps the quotient
    // path correct if extraction declines.
    // Skipped entirely at `step == 1`, where the cache cannot pay for itself:
    // it exists to turn a strided gather into a contiguous copy, and at stride
    // one the gather is *already* contiguous. Concretely,
    // `extract_lde_batch_columns(1, range, domain)` memcpys
    // `columns.col(c)[..domain]` per column, while the uncached quotient path
    // reaches `fill_lde_batch` with `BatchLayout::PolyMajor`, `step == 1` and
    // consecutive indices — which routes to `fill_lde_batch_contiguous` and
    // copies `columns.col(c)[start..end]`. Same bytes out of the same buffer,
    // one `copy_from_slice` per column either way. So the cache is a bit-exact
    // duplicate of storage the commitment already retains, and building it
    // costs one extra full-LDE allocation plus copy per circuit and holds that
    // duplicate resident for the rest of the process.
    let quotient_degree_bits = plonky2::util::log2_ceil(common.quotient_degree_factor);
    let (constants_sigmas_quotient_cache, constants_sigmas_quotient_step, constants_sigmas_quotient_domain) = {
        let step = 1 << (common.config.fri_config.rate_bits - quotient_degree_bits);
        let domain = 1 << (common.degree_bits() + quotient_degree_bits);
        let cols = common.constants_range().len() + common.sigmas_range().len();
        if step != 1 && cols.saturating_mul(domain) * core::mem::size_of::<F>() <= 1 << 30 {
            match (
                constants_sigmas_commitment.extract_lde_batch_columns(
                    step,
                    common.constants_range(),
                    domain,
                ),
                constants_sigmas_commitment.extract_lde_batch_columns(
                    step,
                    common.sigmas_range(),
                    domain,
                ),
            ) {
                (Some(constants), Some(sigmas_cache)) => {
                    let mut cache: Vec<F> = constants;
                    cache.extend(sigmas_cache);
                    (Some(cache), step, domain)
                }
                _ => (None, step, domain),
            }
        } else {
            (None, step, domain)
        }
    };

    let prover_only = ProverOnlyCircuitData::<F, C, D> {
        constants_sigmas_quotient_cache,
        constants_sigmas_quotient_step,
        constants_sigmas_quotient_domain,
        generators,
        generator_readiness_is_authoritative,
        generator_indices_by_watches,
        generator_watch_counts,
        constants_sigmas_commitment,
        sigmas,
        subgroup,
        public_inputs,
        representative_map,
        fixed_routed_wires,
        fft_root_table: Some(root_table),
        circuit_digest,
        lookup_rows,
        lut_to_lookups,
    };

    Ok((
        target,
        CircuitData {
            prover_only,
            verifier_only,
            common,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trips() {
        let values = [
            0u64,
            1,
            127,
            128,
            300,
            16383,
            16384,
            u64::from(u32::MAX),
            u64::MAX,
        ];
        let mut buf = Vec::new();
        for &v in &values {
            write_uvarint(&mut buf, v);
        }
        let mut pos = 0;
        for &v in &values {
            assert_eq!(read_uvarint(&buf, &mut pos).unwrap(), v);
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn zigzag_round_trips() {
        for v in [0i64, 1, -1, 2, -2, i64::MAX, i64::MIN, 1 << 40, -(1 << 40)] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
        // Small magnitudes encode small: identity-adjacent deltas stay 1 byte.
        assert_eq!(zigzag(0), 0);
        assert_eq!(zigzag(-1), 1);
        assert_eq!(zigzag(1), 2);
    }

    #[test]
    fn compressed_section_round_trips() {
        let input = (0..1_000_000usize)
            .map(|i| ((i.wrapping_mul(131) ^ (i >> 3)) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut framed = Vec::new();
        write_compressed_section(&mut framed, &input);
        let mut pos = 0;
        let decoded = read_compressed_section(&framed, &mut pos).unwrap();
        assert_eq!(decoded, input);
        assert_eq!(pos, framed.len());
    }
}
