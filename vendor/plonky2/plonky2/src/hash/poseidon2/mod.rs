pub mod config;
pub mod hash;

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

#[cfg(test)]
mod tests {
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64, PrimeField64};

    #[test]
    fn wire_gamma_factor_fusion_preserves_goldilocks_values() {
        type F = GoldilocksField;

        let check = |wire: F, multiplier: F, value: F, gamma: F| {
            let separate = wire.multiply_accumulate(multiplier, value) + gamma;
            let fused = (wire + gamma).multiply_accumulate(multiplier, value);
            assert_eq!(separate.to_canonical_u64(), fused.to_canonical_u64());
        };

        // Include every exceptional raw-representation region, not only
        // canonical field values. Metal buffers can legally carry a lazy
        // representative until the quotient output is canonicalized.
        let boundary = [
            0,
            1,
            (1u64 << 32) - 1,
            1u64 << 32,
            F::ORDER - 1,
            F::ORDER,
            u64::MAX,
        ];
        for &wire in &boundary {
            for &multiplier in &boundary {
                for &value in &boundary {
                    for &gamma in &boundary {
                        check(
                            GoldilocksField(wire),
                            GoldilocksField(multiplier),
                            GoldilocksField(value),
                            GoldilocksField(gamma),
                        );
                    }
                }
            }
        }

        let mut state = 0x2f29_51e1_4d07_7d76u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            GoldilocksField(state)
        };
        for _ in 0..20_000 {
            check(next(), next(), next(), next());
        }
    }
}
