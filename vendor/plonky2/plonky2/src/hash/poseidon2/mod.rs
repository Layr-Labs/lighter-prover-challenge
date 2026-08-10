pub mod config;
pub mod hash;

/// Canonical barycentric weights for the one K6 coset-interpolation shape.
/// Kept outside the macOS module so Linux contract tests can pin both the gate
/// metadata guard and the runtime MSL table to the same values.
#[cfg(any(
    test,
    all(feature = "std", target_arch = "aarch64", target_os = "macos")
))]
pub(crate) const COSET_16_WEIGHTS_U64: [u64; 16] = [
    0xefff_ffff_1000_0001,
    0x100,
    0x10_0000,
    0x1_0000_0000,
    0x1000_0000_0000,
    0x100_0000_0000_0000,
    0xf_ffff_fff0,
    0xffff_ffff_0000,
    0x0fff_ffff_f000_0000,
    0xffff_fffe_ffff_ff01,
    0xffff_fffe_fff0_0001,
    0xffff_fffe_0000_0001,
    0xffff_efff_0000_0001,
    0xfeff_ffff_0000_0001,
    0xffff_ffef_0000_0011,
    0xfffe_ffff_0001_0001,
];

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    is_exclusive_gpu_phase, prewarm as prewarm_gpu, set_exclusive_gpu_phase,
};

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

/// False on platforms without the Metal-backed exclusive proving phases.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn is_exclusive_gpu_phase() -> bool {
    false
}

/// No-op fallback so a process entry point can request GPU pre-warming
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu() {}

#[cfg(test)]
pub mod p3;

/// Platform-independent contract for the runtime-only K6 library. The actual
/// GPU differential lives in `metal.rs`; these checks still run on Linux and
/// catch source slicing, ABI, or old/new kind ownership drift before a Mac run.
#[cfg(test)]
mod k6_contract_tests {
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, PrimeField64};
    use crate::gates::coset_interpolation::CosetInterpolationGate;

    use super::COSET_16_WEIGHTS_U64;

    const MONOLITHIC: &str = include_str!("poseidon2.metal");
    const K6_BODY: &str = include_str!("k6_residual.metal");

    fn between(start_marker: &str, end_marker: &str) -> &'static str {
        let start = MONOLITHIC
            .find(start_marker)
            .unwrap_or_else(|| panic!("missing K6 start marker {start_marker}"));
        let tail = &MONOLITHIC[start..];
        let end = tail
            .find(end_marker)
            .unwrap_or_else(|| panic!("missing K6 end marker {end_marker}"));
        &tail[..end]
    }

    fn assembled_source() -> String {
        let header_end = MONOLITHIC
            .find("// Compile-time Poseidon2 round constants")
            .expect("missing K6 header marker");
        let segments = [
            &MONOLITHIC[..header_end],
            between(
                "inline void add_epsilon_u32",
                "// Final step of the 128-bit Goldilocks reduction",
            ),
            between("inline ulong reduce_top", "// A lazy value is"),
            between(
                "inline ulong gl_mul_add",
                "// x^7 by the addition chain",
            ),
        ];
        let mut source = String::new();
        for segment in segments {
            source.push_str(segment);
            source.push('\n');
        }
        source.push_str(K6_BODY);
        source
    }

    fn msl_u64_array(name: &str) -> Vec<u64> {
        let marker = format!("constant ulong {name}[16] = {{");
        let tail = K6_BODY
            .split_once(&marker)
            .unwrap_or_else(|| panic!("missing MSL array {name}"))
            .1;
        let body = tail
            .split_once("};")
            .unwrap_or_else(|| panic!("unterminated MSL array {name}"))
            .0;
        body.split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                let hex = value
                    .strip_prefix("0x")
                    .and_then(|value| value.strip_suffix("UL"))
                    .unwrap_or_else(|| panic!("invalid MSL u64 literal {value}"));
                u64::from_str_radix(hex, 16)
                    .unwrap_or_else(|_| panic!("invalid MSL u64 hex {value}"))
            })
            .collect()
    }

    #[test]
    fn runtime_source_and_metadata_ownership_are_disjoint() {
        let source = assembled_source();
        assert!(source.len() <= 23 * 1024);
        assert_eq!(source.matches("kernel void ").count(), 1);
        assert!(source.contains("kernel void k6_residual_quotient"));
        assert!(MONOLITHIC.contains("kernel void range_check_gate_quotient"));
        assert!(MONOLITHIC.contains("kind == 12u"));
        for kind in 13..=18 {
            let branch = format!("kind == {kind}u");
            assert_eq!(K6_BODY.matches(&branch).count(), 1, "bad K6 tag {kind}");
            assert!(
                !MONOLITHIC.contains(&branch),
                "kind {kind} is owned by both libraries"
            );
        }
        for binding in [
            "wires [[buffer(0)]]",
            "constants [[buffer(1)]]",
            "output [[buffer(2)]]",
            "alpha_powers [[buffer(3)]]",
            "metadata [[buffer(4)]]",
            "lde_rows [[buffer(5)]]",
            "quotient_rows [[buffer(6)]]",
            "step [[buffer(7)]]",
            "alpha_stride [[buffer(8)]]",
            "k6_count [[buffer(9)]]",
            "public_inputs_hash [[buffer(10)]]",
        ] {
            assert!(K6_BODY.contains(binding), "missing K6 ABI binding {binding}");
        }
        assert!(K6_BODY.contains("metadata + k6_index * 10u"));
        assert!(K6_BODY.contains("gl_add(output[(ulong)gid * 2u], total[0])"));

        type F = GoldilocksField;
        let gate = CosetInterpolationGate::<F, 2>::with_max_degree(4, 6);
        assert_eq!(gate.subgroup_bits, 4);
        assert_eq!(gate.degree, 6);
        assert_eq!(
            gate.barycentric_weights
                .iter()
                .map(PrimeField64::to_canonical_u64)
                .collect::<Vec<_>>(),
            COSET_16_WEIGHTS_U64
        );
        assert_eq!(
            msl_u64_array("COSET_16_WEIGHTS"),
            COSET_16_WEIGHTS_U64
        );
        assert_eq!(
            msl_u64_array("COSET_16_DOMAIN"),
            F::two_adic_subgroup(4)
                .iter()
                .map(PrimeField64::to_canonical_u64)
                .collect::<Vec<_>>()
        );
        for excluded in [
            "POSEIDON2_EXTERNAL_RC",
            "kernel void poseidon2_",
            "kernel void ntt_",
            "kernel void permutation_quotient",
            "kernel void range_check_gate_quotient",
        ] {
            assert!(!source.contains(excluded), "unexpected K6 item {excluded}");
        }
    }
}
