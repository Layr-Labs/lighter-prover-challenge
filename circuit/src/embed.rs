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
//! The blob is laid out so that a load is *parallel in two dimensions*:
//!
//! * every section carries its own length, so [`deserialize_embedded`] resolves
//!   the whole section table in one pass over the headers and then decompresses
//!   and parses all ten sections concurrently instead of walking them in
//!   sequence;
//! * the three sections that are long streams of self-delimiting records — the
//!   generators, the watch-index offsets and the representative map, tens of
//!   millions of entries on a transaction circuit — carry a *chunk directory*:
//!   the byte offset (and, where reconstruction is a running sum, the base
//!   value) at every `EMBED_CHUNK_RECORDS`-th record. The directory is built by
//!   the encoder, which runs in the untimed build job, so at load time each
//!   chunk is an independent decode and the streams go through `par_iter`. A
//!   chunk that does not end exactly on the next chunk's offset is a hard
//!   error, which is what makes a damaged directory fail loudly instead of
//!   silently producing a different map.
//!
//! Everything recomputed is validated at load: the recomputed commitment cap
//! must equal the embedded verifier data's cap, which transitively pins the
//! circuit digest. On any mismatch the loader errors and callers fall back to
//! building circuits from scratch.

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use plonky2::field::fft::{cached_fft_root_table, cached_two_adic_subgroup};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::gates::lookup::Lookup;
use plonky2::iop::generator::WitnessGeneratorRef;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::LookupWire;
use plonky2::plonk::circuit_data::{
    CircuitData, CommonCircuitData, GeneratorWatchIndex, ProverOnlyCircuitData,
    VerifierOnlyCircuitData,
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
// v2 adds the chunk directories to the generator, watch-offset and
// representative-map sections. Writer and reader ship together and the blobs
// live in OUT_DIR, so a version bump can never meet a stale blob at runtime —
// it exists so that a mismatched pair fails immediately and loudly instead of
// misparsing.
const EMBED_VERSION: u32 = 2;

/// Records per independently decodable chunk in the streamed sections. A
/// transaction circuit has ~10^7 representative-map entries, so this is a few
/// hundred chunks: enough to fill the machine, few enough that the directory
/// costs one varint per chunk against a multi-megabyte payload.
const EMBED_CHUNK_RECORDS: usize = 1 << 16;

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

/// A compressed section that has been *located* in the blob but not yet
/// expanded. Splitting location from expansion is what lets the loader resolve
/// the whole section table up front and then decompress the sections in
/// parallel: locating is a walk over the length headers only.
#[derive(Clone, Copy)]
struct CompressedSection<'a> {
    raw_len: usize,
    bytes: &'a [u8],
}

impl CompressedSection<'_> {
    fn decompress(self) -> Result<Vec<u8>> {
        let raw = zstd::bulk::decompress(self.bytes, self.raw_len)
            .context("embedded circuit blob failed zstd section decompression")?;
        ensure!(
            raw.len() == self.raw_len,
            "embedded circuit compressed section expanded to {} bytes, expected {}",
            raw.len(),
            self.raw_len
        );
        Ok(raw)
    }
}

fn locate_compressed_section<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
) -> Result<CompressedSection<'a>> {
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
    Ok(CompressedSection {
        raw_len,
        bytes: compressed,
    })
}

fn read_compressed_section(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>> {
    locate_compressed_section(bytes, pos)?.decompress()
}

// ---------------------------------------------------------------------------
// Chunk directories
// ---------------------------------------------------------------------------
//
// A chunked section is `[directory][payload length][payload]`, where the
// payload is the same record stream the section always held and the directory
// records, for every `EMBED_CHUNK_RECORDS`-th record, the payload byte offset
// at which it starts. Offsets are stored as varint deltas and are therefore
// monotone by construction.

/// Writes the `[record count][chunk size][chunk count][offset deltas]` header
/// of a chunked section. `offsets` must be non-decreasing payload byte offsets,
/// one per chunk.
fn write_chunk_directory(out: &mut Vec<u8>, records: usize, offsets: &[u64]) {
    write_uvarint(out, records as u64);
    write_uvarint(out, EMBED_CHUNK_RECORDS as u64);
    write_uvarint(out, offsets.len() as u64);
    let mut previous = 0u64;
    for &offset in offsets {
        write_uvarint(out, offset - previous);
        previous = offset;
    }
}

/// Reads a header written by [`write_chunk_directory`], returning
/// `(record count, chunk size, chunk start offsets)`.
fn read_chunk_directory(raw: &[u8], pos: &mut usize) -> Result<(usize, usize, Vec<usize>)> {
    let records = usize::try_from(read_uvarint(raw, pos)?)
        .context("chunked section record count exceeds usize")?;
    let chunk_size = usize::try_from(read_uvarint(raw, pos)?)
        .context("chunked section chunk size exceeds usize")?;
    ensure!(chunk_size > 0, "chunked section chunk size must be positive");
    let chunk_count = usize::try_from(read_uvarint(raw, pos)?)
        .context("chunked section chunk count exceeds usize")?;
    ensure!(
        chunk_count == records.div_ceil(chunk_size),
        "chunked section declares {chunk_count} chunks for {records} records at {chunk_size} \
         records per chunk"
    );
    let mut offsets = Vec::with_capacity(chunk_count);
    let mut running = 0u64;
    for _ in 0..chunk_count {
        running += read_uvarint(raw, pos)?;
        offsets.push(usize::try_from(running).context("chunk start offset exceeds usize")?);
    }
    Ok((records, chunk_size, offsets))
}

/// Writes `[payload length][payload]`, the tail of a chunked section.
fn write_chunk_payload(out: &mut Vec<u8>, payload: &[u8]) {
    write_uvarint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Reads the payload written by [`write_chunk_payload`]. The payload must run
/// to the end of the section.
fn read_chunk_payload<'a>(raw: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let len = usize::try_from(read_uvarint(raw, pos)?)
        .context("chunked section payload length exceeds usize")?;
    ensure!(
        raw.len() == *pos + len,
        "chunked section payload is {} bytes, expected {len}",
        raw.len().saturating_sub(*pos)
    );
    let payload = &raw[*pos..];
    *pos += len;
    Ok(payload)
}

/// Byte range of chunk `index` inside `payload`, validated against the payload
/// bounds. The end is the next chunk's start (or the payload end for the last
/// chunk), which is what every chunk decoder checks it consumed exactly.
fn chunk_bounds(offsets: &[usize], index: usize, payload_len: usize) -> Result<(usize, usize)> {
    let start = offsets[index];
    let end = offsets.get(index + 1).copied().unwrap_or(payload_len);
    ensure!(
        start <= end && end <= payload_len,
        "chunk {index} spans payload bytes {start}..{end}, outside a {payload_len}-byte payload"
    );
    Ok((start, end))
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

    // generators, with a chunk directory over the record stream
    let mut payload = Vec::new();
    let mut chunk_offsets = Vec::with_capacity(prover.generators.len().div_ceil(EMBED_CHUNK_RECORDS));
    for (index, generator) in prover.generators.iter().enumerate() {
        if index % EMBED_CHUNK_RECORDS == 0 {
            chunk_offsets.push(payload.len() as u64);
        }
        payload
            .write_generator::<F, D>(generator, &generator_serializer, common)
            .map_err(|e| {
                anyhow::anyhow!(
                    "serializing generator {:?} (missing from EmbedGeneratorSerializer registry?): {e:?}",
                    generator.0.id()
                )
            })?;
    }
    let mut buf = Vec::with_capacity(payload.len() + 2 * chunk_offsets.len() + 32);
    write_chunk_directory(&mut buf, prover.generators.len(), &chunk_offsets);
    write_chunk_payload(&mut buf, &payload);
    write_compressed_section(&mut out, &buf);

    // watch index CSR: offsets as varint deltas (mostly zero), watchers as u32.
    // Reconstructing the offsets is a running sum, so each chunk additionally
    // carries the sum at its first record; that is what makes the chunks
    // independent.
    let offsets = prover.generator_indices_by_watches.offsets();
    let watchers = prover.generator_indices_by_watches.watchers();
    write_compressed_section(&mut out, &encode_watch_offsets(offsets)?);

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

    // representative map: zigzag varint deltas against the identity map, with a
    // chunk directory. Each entry's base is its own index, so the chunks need
    // nothing beyond their start offsets to be independent.
    write_compressed_section(&mut out, &encode_representative_map(&prover.representative_map));

    Ok(out)
}

/// Encodes the watch-index CSR offsets as
/// `[chunk directory][chunk base deltas][payload]`. Reconstruction is a running
/// sum, so each chunk records the sum at its own first entry.
fn encode_watch_offsets(offsets: &[u32]) -> Result<Vec<u8>> {
    let chunks = offsets.len().div_ceil(EMBED_CHUNK_RECORDS);
    let mut payload = Vec::with_capacity(offsets.len() + 8);
    let mut chunk_offsets = Vec::with_capacity(chunks);
    let mut chunk_bases = Vec::with_capacity(chunks);
    let mut previous = 0u32;
    for (index, &offset) in offsets.iter().enumerate() {
        if index % EMBED_CHUNK_RECORDS == 0 {
            chunk_offsets.push(payload.len() as u64);
            chunk_bases.push(u64::from(previous));
        }
        ensure!(offset >= previous, "watch index offsets must be sorted");
        write_uvarint(&mut payload, u64::from(offset - previous));
        previous = offset;
    }
    let mut buf = Vec::with_capacity(payload.len() + 4 * chunks + 32);
    write_chunk_directory(&mut buf, offsets.len(), &chunk_offsets);
    let mut previous_base = 0u64;
    for &base in &chunk_bases {
        write_uvarint(&mut buf, base - previous_base);
        previous_base = base;
    }
    write_chunk_payload(&mut buf, &payload);
    Ok(buf)
}

/// Encodes a representative map as `[chunk directory][payload]`, the payload
/// being zigzag varint deltas against the identity permutation.
fn encode_representative_map(map: &[u32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 * map.len());
    let mut chunk_offsets = Vec::with_capacity(map.len().div_ceil(EMBED_CHUNK_RECORDS));
    for (index, &parent) in map.iter().enumerate() {
        if index % EMBED_CHUNK_RECORDS == 0 {
            chunk_offsets.push(payload.len() as u64);
        }
        write_uvarint(&mut payload, zigzag(i64::from(parent) - index as i64));
    }
    let mut buf = Vec::with_capacity(payload.len() + 2 * chunk_offsets.len() + 32);
    write_chunk_directory(&mut buf, map.len(), &chunk_offsets);
    write_chunk_payload(&mut buf, &payload);
    buf
}

// ---------------------------------------------------------------------------
// Read side (runtime)
// ---------------------------------------------------------------------------
//
// One decoder per section, so that [`deserialize_embedded`] can run them
// concurrently. Each is a byte-for-byte inverse of the corresponding writer
// block above.

fn decode_verifier_only(section: &[u8]) -> Result<VerifierOnlyCircuitData<C, D>> {
    Buffer::new(section)
        .read_verifier_only_circuit_data()
        .map_err(|e| anyhow::anyhow!("deserializing verifier-only circuit data: {e:?}"))
}

fn decode_target<T: DeserializeOwned>(section: CompressedSection<'_>) -> Result<T> {
    bincode::deserialize(&section.decompress()?).context("deserializing circuit target struct")
}

fn decode_public_inputs(section: CompressedSection<'_>) -> Result<Vec<Target>> {
    let raw = section.decompress()?;
    Buffer::new(&raw)
        .read_target_vec()
        .map_err(|e| anyhow::anyhow!("deserializing public inputs: {e:?}"))
}

fn decode_lookups(section: &[u8]) -> Result<(Vec<LookupWire>, Vec<Lookup>)> {
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
        lookup_rows.push(LookupWire {
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
    Ok((lookup_rows, lut_to_lookups))
}

/// Decodes the generator stream chunk by chunk. Generator records are
/// self-delimiting given `common`, and the chunk directory says where each
/// group of `EMBED_CHUNK_RECORDS` of them starts, so the chunks decode
/// independently and are concatenated in order.
fn decode_generators(
    section: CompressedSection<'_>,
    common: &CommonCircuitData<F, D>,
) -> Result<Vec<WitnessGeneratorRef<F, D>>> {
    let raw = section.decompress()?;
    let mut pos = 0usize;
    let (count, chunk_size, offsets) = read_chunk_directory(&raw, &mut pos)
        .context("deserializing generator chunk directory")?;
    let payload = read_chunk_payload(&raw, &mut pos).context("deserializing generators")?;

    let chunks = offsets
        .par_iter()
        .enumerate()
        .map(|(chunk, _)| -> Result<Vec<WitnessGeneratorRef<F, D>>> {
            let generator_serializer = embed_generator_serializer();
            let (start, end) = chunk_bounds(&offsets, chunk, payload.len())
                .context("deserializing generators")?;
            let records = chunk_size.min(count - chunk * chunk_size);
            let mut reader = Buffer::new(&payload[start..end]);
            let mut generators = Vec::with_capacity(records);
            for _ in 0..records {
                generators.push(
                    reader
                        .read_generator::<F, D>(&generator_serializer, common)
                        .map_err(|e| anyhow::anyhow!("deserializing generator: {e:?}"))?,
                );
            }
            ensure!(
                reader.pos() == end - start,
                "generator chunk {chunk} consumed {} of its {} payload bytes",
                reader.pos(),
                end - start
            );
            Ok(generators)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut generators = Vec::with_capacity(count);
    for chunk in chunks {
        generators.extend(chunk);
    }
    Ok(generators)
}

/// Decodes the watch-index CSR offsets. Each offset is a varint delta on its
/// predecessor, so the chunk directory carries the running sum at every chunk
/// boundary alongside the byte offset.
fn decode_watch_offsets(section: CompressedSection<'_>) -> Result<Vec<u32>> {
    decode_watch_offsets_bytes(&section.decompress()?)
}

fn decode_watch_offsets_bytes(raw: &[u8]) -> Result<Vec<u32>> {
    let mut pos = 0usize;
    let (count, chunk_size, offsets) = read_chunk_directory(raw, &mut pos)
        .context("deserializing watch index chunk directory")?;
    let mut bases = Vec::with_capacity(offsets.len());
    let mut running = 0u64;
    for _ in 0..offsets.len() {
        running += read_uvarint(raw, &mut pos)?;
        bases.push(u32::try_from(running).context("watch index chunk base exceeds u32")?);
    }
    let payload = read_chunk_payload(raw, &mut pos).context("deserializing watch index")?;

    let mut values = vec![0u32; count];
    values
        .par_chunks_mut(chunk_size)
        .zip(bases.par_iter())
        .enumerate()
        .try_for_each(|(chunk, (slots, &base))| -> Result<()> {
            let (start, end) = chunk_bounds(&offsets, chunk, payload.len())
                .context("deserializing watch index")?;
            let mut cursor = start;
            let mut running = u64::from(base);
            for slot in slots.iter_mut() {
                running += read_uvarint(payload, &mut cursor)?;
                *slot = u32::try_from(running).context("watch index offset exceeds u32")?;
            }
            ensure!(
                cursor == end,
                "watch index chunk {chunk} ends at payload byte {cursor}, expected {end}"
            );
            Ok(())
        })?;
    Ok(values)
}

/// Watcher ids are fixed-width little-endian `u32`, so this section needs no
/// directory: the byte offset of record `i` is `4 * i`.
fn decode_watchers(section: CompressedSection<'_>) -> Result<Vec<usize>> {
    let raw = section.decompress()?;
    let mut pos = 0usize;
    let count = usize::try_from(read_uvarint(&raw, &mut pos)?)
        .context("watch index watcher count exceeds usize")?;
    ensure!(
        raw.len() == pos + 4 * count,
        "watch index watcher section length mismatch"
    );
    Ok(raw[pos..]
        .par_chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        .collect())
}

fn decode_constant_values(
    section: CompressedSection<'_>,
    common: &CommonCircuitData<F, D>,
) -> Result<Vec<PolynomialValues<F>>> {
    let raw = section.decompress()?;
    let mut reader = Buffer::new(&raw);
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
    Ok(constant_values)
}

fn decode_representative_map(section: CompressedSection<'_>) -> Result<Vec<u32>> {
    decode_representative_map_bytes(&section.decompress()?)
}

/// Inverse of [`encode_representative_map`]: each chunk's entries are indexed
/// from the chunk's first record, so a chunk decodes with no reference to its
/// predecessors.
fn decode_representative_map_bytes(raw: &[u8]) -> Result<Vec<u32>> {
    let mut pos = 0usize;
    let (count, chunk_size, offsets) = read_chunk_directory(raw, &mut pos)
        .context("deserializing representative map chunk directory")?;
    let payload = read_chunk_payload(raw, &mut pos).context("deserializing representative map")?;

    let mut map = vec![0u32; count];
    map.par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk, slots)| -> Result<()> {
            let (start, end) = chunk_bounds(&offsets, chunk, payload.len())
                .context("deserializing representative map")?;
            let base = chunk * chunk_size;
            let mut cursor = start;
            for (offset, slot) in slots.iter_mut().enumerate() {
                let delta = unzigzag(read_uvarint(payload, &mut cursor)?);
                let parent = (base + offset) as i64 + delta;
                *slot = u32::try_from(parent).context("representative map entry out of range")?;
            }
            ensure!(
                cursor == end,
                "representative map chunk {chunk} ends at payload byte {cursor}, expected {end}"
            );
            Ok(())
        })?;
    Ok(map)
}

/// Reconstructs the target struct and the full [`CircuitData`] from a blob
/// produced by [`serialize_embedded`].
///
/// The returned `CircuitData` is value-identical to the freshly built one:
/// deserialized components are byte round trips, and every recomputed
/// component (subgroup, FFT root table, sigma values/transpose, watch counts,
/// constants/sigmas commitment) is derived by the same code paths the builder
/// itself runs, from the same inputs. The recomputed commitment cap is checked
/// against the embedded verifier data before returning.
pub fn deserialize_embedded<T: DeserializeOwned + Send>(
    bytes: &[u8],
) -> Result<(T, CircuitData<F, C, D>)> {
    let gate_serializer = BlockGateSerializer;
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

    // Resolve the whole section table first. Every section carries its own
    // length, so this is a walk over ten headers — microseconds — and it turns
    // the decode below from a sequential walk into a fan-out.
    let common_section = read_section(bytes, &mut pos)?;
    let verifier_section = read_section(bytes, &mut pos)?;
    let target_section = locate_compressed_section(bytes, &mut pos)?;
    let public_inputs_section = locate_compressed_section(bytes, &mut pos)?;
    let lookups_section = read_section(bytes, &mut pos)?;
    let generators_section = locate_compressed_section(bytes, &mut pos)?;
    let watch_offsets_section = locate_compressed_section(bytes, &mut pos)?;
    let watchers_section = locate_compressed_section(bytes, &mut pos)?;
    let constants_section = locate_compressed_section(bytes, &mut pos)?;
    let representative_map_section = locate_compressed_section(bytes, &mut pos)?;
    ensure!(pos == bytes.len(), "trailing bytes in embedded circuit blob");

    // `common` is the blob's one cross-section dependency: the generator stream
    // and the constant-polynomial header are both parsed against it. It is a
    // small uncompressed section, so decoding it up front costs microseconds and
    // unblocks everything else.
    let mut reader = Buffer::new(common_section);
    let common = reader
        .read_common_circuit_data::<F, D>(&gate_serializer)
        .map_err(|e| anyhow::anyhow!("deserializing common circuit data: {e:?}"))?;

    // The remaining nine sections are mutually independent, so they decompress
    // and parse concurrently instead of one after another. Ordered longest-first
    // within each arm of the join tree.
    let (
        (generators, representative_map),
        ((watch_offsets, watchers), ((constants, target), (public_inputs, (verifier_only, lookups)))),
    ) = rayon::join(
        || {
            rayon::join(
                || decode_generators(generators_section, &common),
                || decode_representative_map(representative_map_section),
            )
        },
        || {
            rayon::join(
                || {
                    rayon::join(
                        || decode_watch_offsets(watch_offsets_section),
                        || decode_watchers(watchers_section),
                    )
                },
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || decode_constant_values(constants_section, &common),
                                || decode_target::<T>(target_section),
                            )
                        },
                        || {
                            rayon::join(
                                || decode_public_inputs(public_inputs_section),
                                || {
                                    rayon::join(
                                        || decode_verifier_only(verifier_section),
                                        || decode_lookups(lookups_section),
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    );
    let generators = generators?;
    let representative_map = representative_map?;
    let offsets = watch_offsets?;
    let watchers = watchers?;
    let constant_values = constants?;
    let target = target?;
    let public_inputs = public_inputs?;
    let verifier_only = verifier_only?;
    let (lookup_rows, lut_to_lookups) = lookups?;

    // Watch counts are a pure function of the (deduplicated) watcher lists;
    // this mirrors `read_prover_only_circuit_data`'s reconstruction. The
    // in-range check on each watcher rides along on the same pass, now that the
    // generator count is only known once both sections are decoded.
    let mut generator_watch_counts = vec![0usize; generators.len()];
    for &watcher in &watchers {
        *generator_watch_counts
            .get_mut(watcher)
            .context("watcher index out of range")? += 1;
    }
    let generator_indices_by_watches = GeneratorWatchIndex::from_parts(offsets, watchers);

    // ---- recompute the derived prover-only components ----
    let degree_bits = common.degree_bits();
    // Same value the constant-polynomial section header carries, and checked
    // against it by `decode_constant_values`.
    let degree = 1usize << degree_bits;
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

    /// Reference decoder for the representative map: ignores the chunk
    /// directory entirely and walks the payload from the front, exactly as the
    /// pre-chunk-directory loader did. The differential against
    /// [`decode_representative_map_bytes`] is the oracle that the chunk
    /// directory is pure metadata — the record stream it indexes is unchanged.
    fn decode_representative_map_serially(raw: &[u8]) -> Vec<u32> {
        let mut pos = 0usize;
        let (count, _chunk_size, _offsets) = read_chunk_directory(raw, &mut pos).unwrap();
        let payload = read_chunk_payload(raw, &mut pos).unwrap();
        let mut cursor = 0usize;
        (0..count)
            .map(|index| {
                let delta = unzigzag(read_uvarint(payload, &mut cursor).unwrap());
                u32::try_from(index as i64 + delta).unwrap()
            })
            .collect()
    }

    /// A representative map with the shapes the real ones have: long identity
    /// runs (zero deltas, one byte each) punctuated by far-away parents, and a
    /// length that is not a multiple of the chunk size.
    fn sample_representative_map() -> Vec<u32> {
        let len = 5 * EMBED_CHUNK_RECORDS + 12_345;
        (0..len)
            .map(|index| {
                if index % 97 == 0 {
                    (index / 97 * 3) as u32
                } else if index % 5 == 0 {
                    (len - 1 - index) as u32
                } else {
                    index as u32
                }
            })
            .collect()
    }

    #[test]
    fn representative_map_chunks_round_trip() {
        let map = sample_representative_map();
        let encoded = encode_representative_map(&map);
        assert_eq!(decode_representative_map_bytes(&encoded).unwrap(), map);
        // Differential: the parallel chunked decode and a decoder that ignores
        // the directory must agree on the same bytes.
        assert_eq!(decode_representative_map_serially(&encoded), map);
    }

    #[test]
    fn watch_offsets_chunks_round_trip() {
        // CSR offsets: non-decreasing, mostly small steps, spanning several
        // chunks with a partial last one.
        let mut offsets = Vec::with_capacity(3 * EMBED_CHUNK_RECORDS + 777);
        let mut running = 0u32;
        for index in 0..3 * EMBED_CHUNK_RECORDS + 777 {
            if index % 11 == 0 {
                running += 1;
            }
            if index % 1_009 == 0 {
                running += 300;
            }
            offsets.push(running);
        }
        let encoded = encode_watch_offsets(&offsets).unwrap();
        assert_eq!(decode_watch_offsets_bytes(&encoded).unwrap(), offsets);
        assert!(encode_watch_offsets(&[]).is_ok());
        assert!(
            decode_watch_offsets_bytes(&encode_watch_offsets(&[]).unwrap())
                .unwrap()
                .is_empty()
        );

        // Sabotage: shifting a chunk's declared start must be rejected.
        let mut pos = 0usize;
        for _ in 0..5 {
            read_uvarint(&encoded, &mut pos).unwrap();
        }
        let mut corrupted = encoded.clone();
        corrupted[pos - 1] = corrupted[pos - 1].wrapping_add(1);
        assert!(
            decode_watch_offsets_bytes(&corrupted).is_err(),
            "a corrupted watch-offset chunk directory must not decode silently"
        );
    }

    #[test]
    fn representative_map_empty_round_trips() {
        let encoded = encode_representative_map(&[]);
        assert!(decode_representative_map_bytes(&encoded).unwrap().is_empty());
    }

    /// Sabotage oracle: a damaged chunk offset must fail loudly. Varints are
    /// self-synchronizing, so a wrong offset still decodes *something*; the
    /// end-of-chunk check is what turns that into an error.
    #[test]
    fn representative_map_chunk_offset_corruption_is_detected() {
        let map = sample_representative_map();
        let encoded = encode_representative_map(&map);
        // Locate the first chunk-offset delta: after count, chunk size and
        // chunk count, the directory holds one varint per chunk. The first
        // offset is always zero, so the second is the first corruptible one.
        let mut pos = 0usize;
        for _ in 0..5 {
            read_uvarint(&encoded, &mut pos).unwrap();
        }
        let mut corrupted = encoded.clone();
        // `pos` now sits just past the *second* chunk's offset delta (the first
        // is always zero); nudging the byte before it moves that chunk's start,
        // which is also the first chunk's declared end.
        corrupted[pos - 1] = corrupted[pos - 1].wrapping_add(1);
        let error = decode_representative_map_bytes(&corrupted)
            .expect_err("a corrupted chunk offset must not decode silently");
        let message = format!("{error:#}");
        assert!(
            message.contains("representative map") || message.contains("chunk"),
            "unexpected error: {message}"
        );
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
