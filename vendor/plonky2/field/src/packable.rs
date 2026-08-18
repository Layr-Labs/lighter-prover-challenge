use crate::packed::PackedField;
use crate::types::Field;

/// Points us to the default packing for a particular field. There may me multiple choices of
/// PackedField for a particular Field (e.g. every Field is also a PackedField), but this is the
/// recommended one. The recommended packing varies by target_arch and target_feature.
pub trait Packable: Field {
    type Packing: PackedField<Scalar = Self>;

    /// A (possibly) wider packing for latency-bound accumulate loops. Evaluating a constraint
    /// expression at a single packed vector forms one serial dependency chain through the packed
    /// multiply kernels; a 2x-wide lockstep packing gives the out-of-order core two independent
    /// chains to overlap (Plonky3 PR #1977). This is intentionally distinct from `Packing`, which
    /// stays at the width the FFT and hashing kernels are tuned for.
    type PackingX2: PackedField<Scalar = Self>;
}

impl<F: Field> Packable for F {
    default type Packing = Self;
    default type PackingX2 = Self;
}

#[cfg(target_arch = "aarch64")]
impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
    type PackingX2 = crate::arch::aarch64::paired_goldilocks_field::PairedGoldilocksField;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    ))
))]
impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = crate::arch::x86_64::avx2_goldilocks_field::Avx2GoldilocksField;
    type PackingX2 = crate::arch::x86_64::avx2_goldilocks_field::Avx2GoldilocksField;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl"
))]
impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = crate::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField;
    type PackingX2 = crate::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField;
}
