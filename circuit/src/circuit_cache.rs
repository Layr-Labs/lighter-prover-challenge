// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Compact [`CircuitData`] cache.
//!
//! Serializes prover circuit data while omitting everything that is cheaper to
//! recompute at load time than to store: the constants/sigmas LDE and its
//! Merkle tree (rebuilt via [`PolynomialBatch::from_coeffs`], which uses the
//! hasher's accelerated Merkle path when available), the transposed sigma
//! values, the subgroup, and the FFT root table. Loading a cached circuit is a
//! small fraction of the cost of rebuilding it with
//! `CircuitBuilder::build`, which also re-runs gate placement, copy-constraint
//! resolution, and selector/sigma polynomial construction.
//!
//! The compact format stores, per circuit: common data, verifier data, witness
//! generators, the generator watch index, the constants/sigmas polynomial
//! coefficients, public-input targets, the representative map (as `u32`), the
//! circuit digest, and lookup metadata.

use core::cmp::max;

use plonky2::field::fft::fft_root_table;
use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2::field::types::Field;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::circuit_data::{CircuitData, ProverOnlyCircuitData};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, transpose};

use crate::circuit_serializer::{BlockGateSerializer, BlockGeneratorSerializer};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::types::config::{C, D, F};

/// Serializes the chain circuits' shared recursion common data, so workers can
/// install it via
/// [`crate::block_tx_chain_constraints::install_recursion_common_data`] and
/// skip the three-stage throwaway build inside the first chain define.
pub fn recursion_common_to_bytes(
    common: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
) -> IoResult<Vec<u8>> {
    let mut out = Vec::new();
    out.write_common_circuit_data(common, &BlockGateSerializer)?;
    Ok(out)
}

/// Deserializes the recursion common data written by
/// [`recursion_common_to_bytes`].
pub fn recursion_common_from_bytes(
    bytes: &[u8],
) -> IoResult<plonky2::plonk::circuit_data::CommonCircuitData<F, D>> {
    let mut buffer = Buffer::new(bytes);
    buffer.read_common_circuit_data(&BlockGateSerializer)
}

/// Serializes `data` into the compact cache format.
pub fn circuit_data_to_compact_bytes(data: &CircuitData<F, C, D>) -> IoResult<Vec<u8>> {
    let gate_serializer = BlockGateSerializer;
    let generator_serializer = BlockGeneratorSerializer::<C, D, Secp256K1>::default();
    let mut out = Vec::new();

    out.write_common_circuit_data(&data.common, &gate_serializer)?;
    out.write_verifier_only_circuit_data(&data.verifier_only)?;

    let prover_only = &data.prover_only;
    out.write_usize(prover_only.generators.len())?;
    for generator in &prover_only.generators {
        out.write_generator::<F, D>(generator, &generator_serializer, &data.common)?;
    }

    out.write_usize(prover_only.generator_indices_by_watches.len())?;
    for (watch, indices) in &prover_only.generator_indices_by_watches {
        out.write_usize(*watch)?;
        out.write_usize_vec(indices)?;
    }

    let batch = &prover_only.constants_sigmas_commitment;
    out.write_usize(batch.polynomials.len())?;
    for poly in &batch.polynomials {
        out.write_usize(poly.coeffs.len())?;
        out.write_field_vec(&poly.coeffs)?;
    }
    out.write_bool(batch.blinding)?;

    out.write_target_vec(&prover_only.public_inputs)?;

    out.write_usize(prover_only.representative_map.len())?;
    for &representative in &prover_only.representative_map {
        out.write_u32(
            u32::try_from(representative).expect("representative index exceeds u32"),
        )?;
    }

    out.write_hash::<F, <C as plonky2::plonk::config::GenericConfig<D>>::Hasher>(
        prover_only.circuit_digest,
    )?;

    out.write_usize(prover_only.lookup_rows.len())?;
    for wire in &prover_only.lookup_rows {
        out.write_usize(wire.last_lu_gate)?;
        out.write_usize(wire.last_lut_gate)?;
        out.write_usize(wire.first_lut_gate)?;
    }
    out.write_usize(prover_only.lut_to_lookups.len())?;
    for lookup in &prover_only.lut_to_lookups {
        out.write_target_lut(lookup)?;
    }

    Ok(out)
}

/// Deserializes a circuit from the compact cache format, recomputing the
/// constants/sigmas commitment (LDE + Merkle tree), transposed sigma values,
/// subgroup, and FFT root table.
pub fn circuit_data_from_compact_bytes(bytes: &[u8]) -> IoResult<CircuitData<F, C, D>> {
    let gate_serializer = BlockGateSerializer;
    let generator_serializer = BlockGeneratorSerializer::<C, D, Secp256K1>::default();
    let mut buffer = Buffer::new(bytes);

    let common = buffer.read_common_circuit_data(&gate_serializer)?;
    let verifier_only = buffer.read_verifier_only_circuit_data()?;

    let generators_len = buffer.read_usize()?;
    let mut generators = Vec::with_capacity(generators_len);
    for _ in 0..generators_len {
        generators.push(buffer.read_generator::<F, D>(&generator_serializer, &common)?);
    }

    let watches_len = buffer.read_usize()?;
    let mut generator_indices_by_watches = std::collections::BTreeMap::new();
    for _ in 0..watches_len {
        let watch = buffer.read_usize()?;
        let indices = buffer.read_usize_vec()?;
        generator_indices_by_watches.insert(watch, indices);
    }

    let poly_len = buffer.read_usize()?;
    let mut polynomials = Vec::with_capacity(poly_len);
    for _ in 0..poly_len {
        let coeff_len = buffer.read_usize()?;
        polynomials.push(PolynomialCoeffs::new(buffer.read_field_vec(coeff_len)?));
    }
    let blinding = buffer.read_bool()?;

    let public_inputs = buffer.read_target_vec()?;

    let representative_len = buffer.read_usize()?;
    let mut representative_map = Vec::with_capacity(representative_len);
    for _ in 0..representative_len {
        representative_map.push(buffer.read_u32()? as usize);
    }

    let circuit_digest =
        buffer.read_hash::<F, <C as plonky2::plonk::config::GenericConfig<D>>::Hasher>()?;

    let lookup_rows_len = buffer.read_usize()?;
    let mut lookup_rows = Vec::with_capacity(lookup_rows_len);
    for _ in 0..lookup_rows_len {
        lookup_rows.push(plonky2::plonk::circuit_builder::LookupWire {
            last_lu_gate: buffer.read_usize()?,
            last_lut_gate: buffer.read_usize()?,
            first_lut_gate: buffer.read_usize()?,
        });
    }
    let luts_len = buffer.read_usize()?;
    let mut lut_to_lookups = Vec::with_capacity(luts_len);
    for _ in 0..luts_len {
        lut_to_lookups.push(buffer.read_target_lut()?);
    }

    // Recompute everything the compact format omits, mirroring
    // `CircuitBuilder::build`.
    let degree_bits = common.degree_bits();
    let rate_bits = common.config.fri_config.rate_bits;
    let cap_height = common.config.fri_config.cap_height;
    let quotient_degree_factor = common.config.max_quotient_degree_factor;
    let max_fft_points = 1 << (degree_bits + max(rate_bits, log2_ceil(quotient_degree_factor)));
    let root_table = fft_root_table(max_fft_points);

    // Transposed sigma values: the sigma polynomials are the last
    // `num_routed_wires` entries of the constants/sigmas batch; their values
    // on the subgroup are the FFT of the stored coefficients. The FFT here is
    // over the base domain, so it needs a base-sized root table rather than
    // the LDE-sized one.
    let base_root_table = fft_root_table(1 << degree_bits);
    let num_routed_wires = common.config.num_routed_wires;
    let sigma_start = polynomials.len() - num_routed_wires;
    let sigma_values = {
        use rayon::prelude::*;
        polynomials[sigma_start..]
            .par_iter()
            .map(|poly| {
                poly.clone()
                    .fft_with_options(None, Some(&base_root_table))
                    .values
            })
            .collect::<Vec<_>>()
    };
    let sigmas = transpose(&sigma_values);

    let mut timing = TimingTree::default();
    let constants_sigmas_commitment = PolynomialBatch::<F, C, D>::from_coeffs(
        polynomials,
        rate_bits,
        blinding,
        cap_height,
        &mut timing,
        Some(&root_table),
    );

    let subgroup = F::two_adic_subgroup(degree_bits);

    let prover_only = ProverOnlyCircuitData::<F, C, D> {
        generators,
        generator_indices_by_watches,
        constants_sigmas_commitment,
        sigmas,
        subgroup,
        public_inputs,
        representative_map,
        fft_root_table: Some(root_table),
        circuit_digest,
        lookup_rows,
        lut_to_lookups,
    };

    Ok(CircuitData {
        prover_only,
        verifier_only,
        common,
    })
}
