//! GPU-accelerated quotient polynomial evaluation via Metal.
//!
//! Moves the entire `compute_quotient_polys` inner loop to a single GPU kernel:
//! gate constraints + permutation constraints + alpha reduction + Z_H division.
//! CPU only handles leaf flattening (pre-process) and coset iFFT (post-process).

use std::time::Instant;

use metal::*;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::{Field, PrimeField64};
use plonky2_maybe_rayon::*;

use super::runtime::RUNTIME;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::{PolynomialBatch, SALT_SIZE};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher, Poseidon2GoldilocksConfig};
use crate::util::{log2_ceil, reverse_bits, transpose};

type F = GoldilocksField;
type C = Poseidon2GoldilocksConfig;
const D: usize = 2;

//===================================================================
// Gate descriptor matching the Metal shader struct
//===================================================================
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GateDescriptorGpu {
    pub gate_type: u32,
    pub row: u32,
    pub selector_index: u32,
    pub group_start: u32,
    pub group_end: u32,
    pub num_constraints: u32,
    pub constraint_offset: u32,
    pub param0: u32,
    pub param1: u32,
    pub param2: u32,
    pub param3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QuotientUniforms {
    pub lde_size: u32,
    pub degree_bits: u32,
    pub quotient_degree_bits: u32,
    pub num_wires: u32,
    pub num_constants: u32,
    pub num_routed_wires: u32,
    pub num_challenges: u32,
    pub num_partial_products: u32,
    pub num_gate_constraints: u32,
    pub max_degree: u32,
    pub num_gates: u32,
    pub num_selectors: u32,
    pub num_lookup_selectors: u32,
    pub zs_stride: u32,
    pub wires_stride: u32,
    pub cs_stride: u32,
}

// Gate type enum values matching the Metal shader
const GATE_NOOP: u32 = 0;
const GATE_ARITHMETIC: u32 = 1;
const GATE_ARITHMETIC_EXTENSION: u32 = 2;
const GATE_MUL_EXTENSION: u32 = 3;
const GATE_CONSTANT: u32 = 4;
const GATE_PUBLIC_INPUT: u32 = 5;
const GATE_BASE_SUM: u32 = 6;
const GATE_REDUCING: u32 = 7;
const GATE_REDUCING_EXTENSION: u32 = 8;
const GATE_RANDOM_ACCESS: u32 = 9;
const GATE_COSET_INTERPOLATION: u32 = 10;
const GATE_POSEIDON2: u32 = 11;
const GATE_ADDITION: u32 = 12;
const GATE_SELECTION: u32 = 13;
const GATE_EQUALITY: u32 = 14;
const GATE_EXPONENTIATION: u32 = 15;
const GATE_MULTIPLICATION: u32 = 16;
const GATE_BYTE_DECOMPOSITION: u32 = 17;
const GATE_U48_SUBTRACTION: u32 = 18;
const GATE_U32_SUBTRACTION: u32 = 19;
const GATE_U16_ADD_MANY: u32 = 20;
const GATE_RANGE_CHECK: u32 = 21;
#[allow(dead_code)]
const GATE_U16_SUBTRACTION: u32 = 22;
const GATE_QUINTIC_MULTIPLICATION: u32 = 23;
const GATE_QUINTIC_SQUARING: u32 = 24;
const GATE_U32_ADD_MANY: u32 = 25;
const GATE_U32_ARITHMETIC: u32 = 26;

//===================================================================
// Parse gate id string to extract parameters
//===================================================================
fn parse_param_from_id(id: &str, key: &str) -> Option<u32> {
    // Parse "key: value" from debug format like "ArithmeticGate { num_ops: 20 }"
    if let Some(pos) = id.find(key) {
        let after = &id[pos + key.len()..];
        // Skip ": "
        let after = after.trim_start_matches(|c: char| c == ':' || c == ' ');
        // Parse number
        let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        num_str.parse().ok()
    } else {
        None
    }
}

//===================================================================
// Build gate descriptors from CommonCircuitData
//===================================================================
pub(crate) fn build_gate_descriptors(
    common_data: &CommonCircuitData<F, D>,
) -> (Vec<GateDescriptorGpu>, Vec<u64>) {
    let mut descriptors = Vec::with_capacity(common_data.gates.len());
    let mut coset_interp_data: Vec<u64> = Vec::new();

    for (i, gate) in common_data.gates.iter().enumerate() {
        let id = gate.0.id();
        let selector_index = common_data.selectors_info.selector_indices[i];
        let group = &common_data.selectors_info.groups[selector_index];
        let num_constraints = gate.0.num_constraints() as u32;

        // NOTE: All gates share constraint offset 0 because the CPU accumulates
        // all gates' filtered constraints into the same positions (overlapping).
        // The selector filter ensures only one gate is active per point.
        let mut desc = GateDescriptorGpu {
            row: i as u32,
            selector_index: selector_index as u32,
            group_start: group.start as u32,
            group_end: group.end as u32,
            num_constraints,
            constraint_offset: 0,
            ..Default::default()
        };

        if id.starts_with("NoopGate") {
            desc.gate_type = GATE_NOOP;
        } else if id.starts_with("ArithmeticExtensionGate") {
            desc.gate_type = GATE_ARITHMETIC_EXTENSION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("ArithmeticGate") {
            desc.gate_type = GATE_ARITHMETIC;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("MulExtensionGate") {
            desc.gate_type = GATE_MUL_EXTENSION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("ConstantGate") {
            desc.gate_type = GATE_CONSTANT;
            desc.param0 = parse_param_from_id(&id, "num_consts").unwrap_or(0);
        } else if id.starts_with("PublicInputGate") {
            desc.gate_type = GATE_PUBLIC_INPUT;
        } else if id.starts_with("BaseSumGate") {
            desc.gate_type = GATE_BASE_SUM;
            desc.param0 = parse_param_from_id(&id, "num_limbs").unwrap_or(0);
            // Extract base from "+ Base: N" — must use "+ Base" to avoid matching "BaseSumGate"
            desc.param1 = parse_param_from_id(&id, "+ Base").unwrap_or(2);
        } else if id.starts_with("ReducingExtensionGate") {
            desc.gate_type = GATE_REDUCING_EXTENSION;
            desc.param0 = parse_param_from_id(&id, "num_coeffs").unwrap_or(0);
        } else if id.starts_with("ReducingGate") {
            desc.gate_type = GATE_REDUCING;
            desc.param0 = parse_param_from_id(&id, "num_coeffs").unwrap_or(0);
        } else if id.starts_with("RandomAccessGate") {
            desc.gate_type = GATE_RANDOM_ACCESS;
            let bits = parse_param_from_id(&id, "bits").unwrap_or(0);
            let num_copies = parse_param_from_id(&id, "num_copies").unwrap_or(0);
            let num_extra_constants = parse_param_from_id(&id, "num_extra_constants").unwrap_or(0);
            let vec_size = 1u32 << bits;
            // num_routed_wires = start_extra_constants + num_extra_constants
            //                  = (2 + vec_size) * num_copies + num_extra_constants
            let num_routed_wires = (2 + vec_size) * num_copies + num_extra_constants;
            desc.param0 = bits;
            desc.param1 = num_copies;
            desc.param2 = num_extra_constants;
            desc.param3 = num_routed_wires;
        } else if id.starts_with("CosetInterpolationGate") {
            desc.gate_type = GATE_COSET_INTERPOLATION;
            let subgroup_bits = parse_param_from_id(&id, "subgroup_bits").unwrap_or(0);
            let degree = parse_param_from_id(&id, "degree").unwrap_or(2);
            desc.param0 = subgroup_bits;
            desc.param1 = degree;

            // Store data offset and pack subgroup points + barycentric weights
            desc.param2 = coset_interp_data.len() as u32;
            let _n = 1usize << subgroup_bits;

            // Generate the two-adic subgroup points
            let subgroup = F::two_adic_subgroup(subgroup_bits as usize);
            for pt in &subgroup {
                coset_interp_data.push(pt.to_canonical_u64());
            }

            // Compute barycentric weights
            let bary_weights = crate::field::interpolation::barycentric_weights(
                &subgroup.iter().map(|x| (*x, F::ZERO)).collect::<Vec<_>>(),
            );
            for w in &bary_weights {
                coset_interp_data.push(w.to_canonical_u64());
            }
        } else if id.starts_with("Poseidon2Gate") {
            desc.gate_type = GATE_POSEIDON2;
        } else if id.starts_with("AdditionGate") {
            desc.gate_type = GATE_ADDITION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("SelectionGate") {
            desc.gate_type = GATE_SELECTION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("EqualityGate") {
            desc.gate_type = GATE_EQUALITY;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("ExponentiationGate") {
            desc.gate_type = GATE_EXPONENTIATION;
            desc.param0 = parse_param_from_id(&id, "num_power_bits").unwrap_or(0);
        } else if id.starts_with("MultiplicationGate") || id.starts_with("MulBaseGate") {
            desc.gate_type = GATE_MULTIPLICATION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("ByteDecompositionGate") {
            desc.gate_type = GATE_BYTE_DECOMPOSITION;
            desc.param0 = parse_param_from_id(&id, "num_limbs").unwrap_or(0);
            desc.param1 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("U48SubtractionGate") {
            desc.gate_type = GATE_U48_SUBTRACTION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
            desc.param1 = 48; // bit_width
        } else if id.starts_with("U32SubtractionGate") {
            desc.gate_type = GATE_U32_SUBTRACTION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
            desc.param1 = 32; // bit_width
        } else if id.starts_with("U16AddManyGate") {
            desc.gate_type = GATE_U16_ADD_MANY;
            desc.param0 = parse_param_from_id(&id, "num_addends").unwrap_or(0);
            desc.param1 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("RangeCheckGate") {
            desc.gate_type = GATE_RANGE_CHECK;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
            desc.param1 = parse_param_from_id(&id, "bit_size").unwrap_or(0);
        } else if id.starts_with("U16SubtractionGate") {
            // Same structure as U48/U32 subtraction, reuse the same gate type
            desc.gate_type = GATE_U48_SUBTRACTION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
            desc.param1 = 16; // bit_width
        } else if id.starts_with("QuinticMultiplicationGate") {
            desc.gate_type = GATE_QUINTIC_MULTIPLICATION;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("QuinticSquaringGate") {
            desc.gate_type = GATE_QUINTIC_SQUARING;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("U32AddManyGate") {
            desc.gate_type = GATE_U32_ADD_MANY;
            desc.param0 = parse_param_from_id(&id, "num_addends").unwrap_or(0);
            desc.param1 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else if id.starts_with("U32ArithmeticGate") {
            desc.gate_type = GATE_U32_ARITHMETIC;
            desc.param0 = parse_param_from_id(&id, "num_ops").unwrap_or(0);
        } else {
            // Unknown gate — fall back to CPU
            log::info!(
                "Unknown gate type for GPU quotient: {}. Falling back to CPU.",
                id
            );
            return (vec![], vec![]);
        }

        descriptors.push(desc);
    }

    (descriptors, coset_interp_data)
}

//===================================================================
// Flatten LDE leaves for GPU: one contiguous buffer per commitment
//===================================================================
fn flatten_lde_leaves(batch: &PolynomialBatch<F, C, D>, lde_size: usize, step: usize) -> Vec<u64> {
    let leaf_len = batch.merkle_tree.leaves[0].len() - if batch.blinding { SALT_SIZE } else { 0 };
    let mut flat = vec![0u64; lde_size * leaf_len];
    flat.par_chunks_mut(leaf_len)
        .enumerate()
        .for_each(|(i, chunk)| {
            let bit_rev_idx = i * step;
            let bit_rev = reverse_bits(bit_rev_idx, batch.degree_log + batch.rate_bits);
            let src = &batch.merkle_tree.leaves[bit_rev];
            for j in 0..leaf_len {
                chunk[j] = src[j].to_canonical_u64();
            }
        });
    flat
}

/// Flatten the "next" ZS values (for next_step offset).
fn flatten_lde_leaves_next(
    batch: &PolynomialBatch<F, C, D>,
    lde_size: usize,
    step: usize,
    next_step: usize,
) -> Vec<u64> {
    let leaf_len = batch.merkle_tree.leaves[0].len() - if batch.blinding { SALT_SIZE } else { 0 };
    let mut flat = vec![0u64; lde_size * leaf_len];
    flat.par_chunks_mut(leaf_len)
        .enumerate()
        .for_each(|(i, chunk)| {
            let i_next = (i + next_step) % lde_size;
            let bit_rev_idx = i_next * step;
            let bit_rev = reverse_bits(bit_rev_idx, batch.degree_log + batch.rate_bits);
            let src = &batch.merkle_tree.leaves[bit_rev];
            for j in 0..leaf_len {
                chunk[j] = src[j].to_canonical_u64();
            }
        });
    flat
}

//===================================================================
// Main GPU quotient polynomial computation
//===================================================================
pub(crate) fn compute_quotient_polys_gpu(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_commitment: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let start = Instant::now();
    let num_challenges = common_data.config.num_challenges;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();

    let z_h_on_coset = ZeroPolyOnCoset::<F>::new(common_data.degree_bits(), quotient_degree_bits);

    // Build gate descriptors
    let (gate_descs, coset_interp_data) = build_gate_descriptors(common_data);
    if gate_descs.is_empty() {
        // Fallback: unknown gate encountered
        log::warn!("GPU quotient: falling back to CPU due to unknown gate");
        return cpu_fallback(
            common_data,
            prover_data,
            public_inputs_hash,
            wires_commitment,
            zs_partial_products_commitment,
            betas,
            gammas,
            alphas,
        );
    }

    log::info!(
        "GPU quotient: lde_size={}, gates={}",
        lde_size,
        gate_descs.len()
    );

    // 1. Flatten leaves (parallelized with Rayon)
    let flat_t = Instant::now();

    let wires_stride = wires_commitment.merkle_tree.leaves[0].len()
        - if wires_commitment.blinding {
            SALT_SIZE
        } else {
            0
        };
    let cs_stride = prover_data.constants_sigmas_commitment.merkle_tree.leaves[0].len()
        - if prover_data.constants_sigmas_commitment.blinding {
            SALT_SIZE
        } else {
            0
        };
    let zs_stride = zs_partial_products_commitment.merkle_tree.leaves[0].len()
        - if zs_partial_products_commitment.blinding {
            SALT_SIZE
        } else {
            0
        };

    let flat_wires = flatten_lde_leaves(wires_commitment, lde_size, step);
    let flat_cs = flatten_lde_leaves(&prover_data.constants_sigmas_commitment, lde_size, step);
    let flat_zs = flatten_lde_leaves(zs_partial_products_commitment, lde_size, step);
    let flat_zs_next =
        flatten_lde_leaves_next(zs_partial_products_commitment, lde_size, step, next_step);

    log::info!(
        "GPU quotient: flatten {:.0}ms ({:.0}MB total)",
        flat_t.elapsed().as_millis(),
        ((flat_wires.len() + flat_cs.len() + flat_zs.len() + flat_zs_next.len()) * 8) as f64 / 1e6,
    );

    // 2. Precompute shifted points, L_0, Z_H inverses
    // CRITICAL: CPU evaluates everything on the COSET (coset_shift * x), not the raw subgroup.
    let shifted_points_u64: Vec<u64> = points
        .iter()
        .map(|&p| (F::coset_shift() * p).to_canonical_u64())
        .collect();
    let rate = 1usize << quotient_degree_bits;
    let z_h_inv: Vec<u64> = (0..rate)
        .map(|i| z_h_on_coset.eval_inverse(i).to_canonical_u64())
        .collect();

    // L_0(shifted_x) = Z_H(shifted_x) / (n * (shifted_x - 1))
    // shifted_x is never 1 (coset_shift=7), so no division by zero.
    let l_0_values: Vec<u64> = points
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let shifted_x = F::coset_shift() * x;
            z_h_on_coset.eval_l_0(i, shifted_x).to_canonical_u64()
        })
        .collect();

    // k_is for permutation
    let k_is_u64: Vec<u64> = common_data
        .k_is
        .iter()
        .map(|k| k.to_canonical_u64())
        .collect();
    let betas_u64: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
    let gammas_u64: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
    let alphas_u64: Vec<u64> = alphas.iter().map(|a| a.to_canonical_u64()).collect();

    // Public inputs hash - use GenericHashOut trait for to_vec()
    let pih_elements = GenericHashOut::<F>::to_vec(public_inputs_hash);
    let pub_hash_u64: Vec<u64> = pih_elements.iter().map(|e| e.to_canonical_u64()).collect();

    // 3. Build uniforms
    let uniforms = QuotientUniforms {
        lde_size: lde_size as u32,
        degree_bits: common_data.degree_bits() as u32,
        quotient_degree_bits: quotient_degree_bits as u32,
        num_wires: common_data.config.num_wires as u32,
        num_constants: common_data.num_constants as u32,
        num_routed_wires: common_data.config.num_routed_wires as u32,
        num_challenges: num_challenges as u32,
        num_partial_products: common_data.num_partial_products as u32,
        num_gate_constraints: common_data.num_gate_constraints as u32,
        max_degree: common_data.quotient_degree_factor as u32,
        num_gates: gate_descs.len() as u32,
        num_selectors: common_data.selectors_info.num_selectors() as u32,
        num_lookup_selectors: common_data.num_lookup_selectors as u32,
        zs_stride: zs_stride as u32,
        wires_stride: wires_stride as u32,
        cs_stride: cs_stride as u32,
    };

    // 4. Dispatch GPU kernel on the dedicated GPU thread.
    // All Metal buffer creation, encoding, dispatch, and result reading happens
    // on the same thread as the Merkle builder — with autoreleasepool, shared
    // command queue, and proper ObjC object lifecycle management.
    let gpu_t = Instant::now();
    let output_len = lde_size * num_challenges;

    let coset_data = if coset_interp_data.is_empty() {
        vec![0u64; 1]
    } else {
        coset_interp_data
    };

    // Dispatch on the GPU thread (same thread as Merkle builder) using
    // RUNTIME's shared command queue and pre-built PSO. This ensures proper
    // GPU-side command ordering after Merkle operations.
    use super::gpu_thread::GPU_DISPATCHER;
    let lde_size_cap = lde_size;
    let output_data_tuple = GPU_DISPATCHER.run_on_gpu_thread(move || {
        let runtime = &*RUNTIME;
        let device = runtime.device.lock().unwrap();

        // Create Metal buffers from the raw data
        let buf_wires = device.new_buffer_with_data(
            flat_wires.as_ptr() as *const std::ffi::c_void,
            (flat_wires.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_cs = device.new_buffer_with_data(
            flat_cs.as_ptr() as *const std::ffi::c_void,
            (flat_cs.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_zs = device.new_buffer_with_data(
            flat_zs.as_ptr() as *const std::ffi::c_void,
            (flat_zs.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_zs_next = device.new_buffer_with_data(
            flat_zs_next.as_ptr() as *const std::ffi::c_void,
            (flat_zs_next.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_gates = device.new_buffer_with_data(
            gate_descs.as_ptr() as *const std::ffi::c_void,
            (gate_descs.len() * std::mem::size_of::<GateDescriptorGpu>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_k_is = device.new_buffer_with_data(
            k_is_u64.as_ptr() as *const std::ffi::c_void,
            (k_is_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_betas = device.new_buffer_with_data(
            betas_u64.as_ptr() as *const std::ffi::c_void,
            (betas_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_gammas = device.new_buffer_with_data(
            gammas_u64.as_ptr() as *const std::ffi::c_void,
            (gammas_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_alphas = device.new_buffer_with_data(
            alphas_u64.as_ptr() as *const std::ffi::c_void,
            (alphas_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_z_h_inv = device.new_buffer_with_data(
            z_h_inv.as_ptr() as *const std::ffi::c_void,
            (z_h_inv.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_l0 = device.new_buffer_with_data(
            l_0_values.as_ptr() as *const std::ffi::c_void,
            (l_0_values.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_points = device.new_buffer_with_data(
            shifted_points_u64.as_ptr() as *const std::ffi::c_void,
            (shifted_points_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_pub_hash = device.new_buffer_with_data(
            pub_hash_u64.as_ptr() as *const std::ffi::c_void,
            (pub_hash_u64.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_coset = device.new_buffer_with_data(
            coset_data.as_ptr() as *const std::ffi::c_void,
            (coset_data.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let uniforms_slice = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const QuotientUniforms as *const u8,
                std::mem::size_of::<QuotientUniforms>(),
            )
        };
        let buf_uniforms = device.new_buffer_with_data(
            uniforms_slice.as_ptr() as *const std::ffi::c_void,
            uniforms_slice.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offset_zero: u32 = 0;
        let buf_offset = device.new_buffer_with_data(
            &offset_zero as *const u32 as *const std::ffi::c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer: pre-zero from CPU so pages are faulted in
        let output_byte_len = output_len * 8;
        let buf_output = device.new_buffer(
            output_byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = buf_output.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, output_byte_len);
        }

        // Debug buffer (shader expects buffer(16), minimal size)
        let buf_debug = device.new_buffer(8 as u64, MTLResourceOptions::StorageModeShared);

        // Release device lock before GPU dispatch
        drop(device);

        // Encode and dispatch using RUNTIME's shared command queue and PSO
        let pso = runtime.get_quotient_poly_pipeline_state();
        let command_buffer = runtime.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(&buf_wires), 0);
        encoder.set_buffer(1, Some(&buf_cs), 0);
        encoder.set_buffer(2, Some(&buf_zs), 0);
        encoder.set_buffer(3, Some(&buf_zs_next), 0);
        encoder.set_buffer(4, Some(&buf_gates), 0);
        encoder.set_buffer(5, Some(&buf_k_is), 0);
        encoder.set_buffer(6, Some(&buf_betas), 0);
        encoder.set_buffer(7, Some(&buf_gammas), 0);
        encoder.set_buffer(8, Some(&buf_alphas), 0);
        encoder.set_buffer(9, Some(&buf_z_h_inv), 0);
        encoder.set_buffer(10, Some(&buf_l0), 0);
        encoder.set_buffer(11, Some(&buf_points), 0);
        encoder.set_buffer(12, Some(&buf_pub_hash), 0);
        encoder.set_buffer(13, Some(&buf_coset), 0);
        encoder.set_buffer(14, Some(&buf_uniforms), 0);
        encoder.set_buffer(15, Some(&buf_output), 0);
        encoder.set_buffer(16, Some(&buf_debug), 0);
        encoder.set_buffer(17, Some(&buf_offset), 0);

        let max_tg = pso.max_total_threads_per_threadgroup();
        let tg_width = std::cmp::min(256, max_tg);
        let threadgroup_size = MTLSize::new(tg_width, 1, 1);
        let grid_size = MTLSize::new(lde_size_cap as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy output data from GPU
        let output_ptr = buf_output.contents() as *const u64;
        let mut output = Vec::with_capacity(output_len);
        unsafe {
            std::ptr::copy(output_ptr, output.as_mut_ptr(), output_len);
            output.set_len(output_len);
        }
        output
    });

    let output_data = &output_data_tuple[..output_len];

    log::info!(
        "GPU quotient: gpu={:.0}ms, total={:.0}ms",
        gpu_t.elapsed().as_millis(),
        start.elapsed().as_millis(),
    );

    // Convert GPU quotient values into polynomial coefficients via coset iFFT
    let quotient_polys: Vec<PolynomialCoeffs<F>> = (0..num_challenges)
        .into_par_iter()
        .map(|ch| {
            let values: Vec<F> = (0..lde_size)
                .map(|i| {
                    let raw = output_data[i * num_challenges + ch];
                    F::from_noncanonical_u64(raw)
                })
                .collect();
            PolynomialValues::new(values).coset_ifft(F::coset_shift())
        })
        .collect();

    quotient_polys
}

//===================================================================
// CPU fallback (calls the original function)
//===================================================================
fn cpu_fallback(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_commitment: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    // Re-implement the CPU path here since we can't easily call the original
    // generic function with concrete types
    use crate::plonk::vanishing_poly::eval_vanishing_poly_base_batch;
    use crate::plonk::vars::EvaluationVarsBaseBatch;

    let _num_challenges = common_data.config.num_challenges;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();
    let z_h_on_coset = ZeroPolyOnCoset::<F>::new(common_data.degree_bits(), quotient_degree_bits);

    const BATCH_SIZE: usize = 32;
    let points_batches = points.par_chunks(BATCH_SIZE);
    let _num_batches = points.len().div_ceil(BATCH_SIZE);

    let quotient_values: Vec<Vec<F>> = points_batches
        .enumerate()
        .flat_map(|(batch_i, xs_batch)| {
            let indices_batch: Vec<usize> =
                (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + xs_batch.len()).collect();

            let mut shifted_xs_batch = Vec::with_capacity(xs_batch.len());
            let mut local_zs_batch = Vec::with_capacity(xs_batch.len());
            let mut next_zs_batch = Vec::with_capacity(xs_batch.len());
            let mut partial_products_batch = Vec::with_capacity(xs_batch.len());
            let mut s_sigmas_batch = Vec::with_capacity(xs_batch.len());
            let mut local_constants_batch_refs = Vec::with_capacity(xs_batch.len());
            let mut local_wires_batch_refs = Vec::with_capacity(xs_batch.len());

            for (&i, &x) in indices_batch.iter().zip(xs_batch) {
                let shifted_x = F::coset_shift() * x;
                let i_next = (i + next_step) % lde_size;
                let local_cs = prover_data
                    .constants_sigmas_commitment
                    .get_lde_values(i, step);
                let local_constants = &local_cs[common_data.constants_range()];
                let s_sigmas = &local_cs[common_data.sigmas_range()];
                let local_wires = wires_commitment.get_lde_values(i, step);
                let local_zs_pp = zs_partial_products_commitment.get_lde_values(i, step);
                let next_zs_pp = zs_partial_products_commitment.get_lde_values(i_next, step);
                let local_zs = &local_zs_pp[common_data.zs_range()];
                let next_zs = &next_zs_pp[common_data.zs_range()];
                let partial_products = &local_zs_pp[common_data.partial_products_range()];

                local_constants_batch_refs.push(local_constants);
                local_wires_batch_refs.push(local_wires);
                shifted_xs_batch.push(shifted_x);
                local_zs_batch.push(local_zs);
                next_zs_batch.push(next_zs);
                partial_products_batch.push(partial_products);
                s_sigmas_batch.push(s_sigmas);
            }

            let mut local_constants_batch =
                vec![F::ZERO; xs_batch.len() * local_constants_batch_refs[0].len()];
            for i in 0..local_constants_batch_refs[0].len() {
                for (j, constants) in local_constants_batch_refs.iter().enumerate() {
                    local_constants_batch[i * xs_batch.len() + j] = constants[i];
                }
            }
            let mut local_wires_batch =
                vec![F::ZERO; xs_batch.len() * local_wires_batch_refs[0].len()];
            for i in 0..local_wires_batch_refs[0].len() {
                for (j, wires) in local_wires_batch_refs.iter().enumerate() {
                    local_wires_batch[i * xs_batch.len() + j] = wires[i];
                }
            }

            let vars_batch = EvaluationVarsBaseBatch::new(
                xs_batch.len(),
                &local_constants_batch,
                &local_wires_batch,
                public_inputs_hash,
            );

            let empty_lookup: Vec<&[F]> = vec![];
            let empty_lut: Vec<&[F]> = vec![];
            let empty_deltas: &[F] = &[];

            let mut quotient_values_batch = eval_vanishing_poly_base_batch::<F, D>(
                common_data,
                &indices_batch,
                &shifted_xs_batch,
                vars_batch,
                &local_zs_batch,
                &next_zs_batch,
                &empty_lookup,
                &empty_lookup,
                &partial_products_batch,
                &s_sigmas_batch,
                betas,
                gammas,
                empty_deltas,
                alphas,
                &z_h_on_coset,
                &empty_lut,
            );

            for (&i, quotient_values) in indices_batch.iter().zip(quotient_values_batch.iter_mut())
            {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                quotient_values
                    .iter_mut()
                    .for_each(|v| *v *= denominator_inv);
            }
            quotient_values_batch
        })
        .collect();

    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}
