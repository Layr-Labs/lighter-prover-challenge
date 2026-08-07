//! LOCAL LAB ONLY — never part of a submission.
//!
//! Kernel-rewrite lab for the Goldilocks reduce128 / multiply primitive family:
//! the 9-instruction pair multiply-reduce (`mul_reduce_pair`), the scalar
//! `mul_acc_reduce`, the pair `mul_acc_reduce_pair`, and `from_noncanonical_u128`
//! (portable `reduce128`).
//!
//! Method (per lab charter):
//!  * exercises the EXACT production code paths (public API: `GoldilocksField`
//!    ops, `Field::multiply_accumulate`, `<GoldilocksField as Packable>::Packing`)
//!    on production shapes: long dependent chains (Horner folds) AND
//!    independent-stream throughput (batch MACs) — both regimes;
//!  * every variant is bit-identity checked against the production path on
//!    1M+ random inputs of every production shape plus full edge-case cross
//!    products (and reduction tails additionally on adversarial (lo, hi) pairs
//!    that force the rare borrow path) BEFORE any timing;
//!  * timing is interleaved ABAB in-process, median of >= 12 reps, and the
//!    timed sequences' final checksums are asserted equal across impls.

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    panic!("lab_reduce128 is an aarch64-only lab bench");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    lab::main();
}

#[cfg(target_arch = "aarch64")]
mod lab {
    use core::arch::asm;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::goldilocks_field::GoldilocksField as F;
    use plonky2::field::packable::Packable;
    use plonky2::field::packed::PackedField;
    use plonky2::field::types::{Field, Field64};
    use std::hint::black_box;
    use std::time::Instant;

    type P = <GoldilocksField as Packable>::Packing; // WideGoldilocksField, 4 lanes

    const EPSILON: u64 = (1 << 32) - 1;

    // Keep the bench thread on a P-core.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

    // ---------------------------------------------------------------- RNG --

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // ------------------------------------------------- production oracles --

    #[inline(always)]
    fn prod_mul(a: u64, b: u64) -> u64 {
        (F(a) * F(b)).0
    }

    #[inline(always)]
    fn prod_mac(acc: u64, a: u64, b: u64) -> u64 {
        Field::multiply_accumulate(&F(acc), F(a), F(b)).0
    }

    #[inline(always)]
    fn prod_reduce128(lo: u64, hi: u64) -> u64 {
        F::from_noncanonical_u128(((hi as u128) << 64) | lo as u128).0
    }

    // ------------------------------------------------------------ variants --
    //
    // Every variant computes bit-for-bit the same u64 representative as the
    // production kernel it shadows. The reduction keeps the exact two-step
    // structure (subtract x_hi_hi with EPSILON correction on borrow, then add
    // x_hi_lo * EPSILON with EPSILON correction on carry) — reordering the
    // steps provably changes the representative (e.g. lo=0, hi=(EPSILON<<32)|1
    // yields ORDER under the production order but 0 under add-first).

    /// V1 `mul_asm9`: exact single-lane scalarization of the promoted
    /// 9-instruction `mul_reduce_pair` kernel (branchless csetm corrections).
    #[inline(always)]
    fn v_mul_asm9(a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "umull {scratch}, {hi:w}, {epsilon:w}",
                "subs  {result}, {result}, {hi}, lsr #32",
                "csetm {hi:w}, cc",
                "sub   {result}, {result}, {hi}",
                "adds  {result}, {result}, {scratch}",
                "csetm {scratch:w}, cs",
                "add   {result}, {result}, {scratch}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V2 `mul_branchy_umull`: latency-specialized. The borrow in
    /// `lo - x_hi_hi` has probability ~2^-34 on random field data, so a
    /// predicted-taken `b.hs` replaces the csetm/sub pair on the critical path.
    #[inline(always)]
    fn v_mul_branchy_umull(a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "umull {scratch}, {hi:w}, {epsilon:w}",
                "subs  {result}, {result}, {hi}, lsr #32",
                "b.hs  2f",
                "sub   {result}, {result}, {epsilon}",
                "2:",
                "adds  {result}, {result}, {scratch}",
                "csetm {hi:w}, cs",
                "add   {result}, {result}, {hi}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V3 `mul_branchy_shift`: V2 plus the epsilon-product computed as
    /// `(hi << 32) - (hi & 2^32-1)` (lsl + extended-register sub, latency 2
    /// from `hi`, zero mul-port pressure) instead of the 3-cycle `umull`.
    #[inline(always)]
    fn v_mul_branchy_shift(a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "lsl   {scratch}, {hi}, #32",
                "sub   {scratch}, {scratch}, {hi:w}, uxtw",
                "subs  {result}, {result}, {hi}, lsr #32",
                "b.hs  2f",
                "sub   {result}, {result}, {epsilon}",
                "2:",
                "adds  {result}, {result}, {scratch}",
                "csetm {hi:w}, cs",
                "add   {result}, {result}, {hi}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V4 `mul_asm9_shift`: branchless like V1 but with the shift-pair epsilon
    /// product — throughput-specialized (2 mul-port ops per multiply, not 3).
    #[inline(always)]
    fn v_mul_asm9_shift(a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "lsl   {scratch}, {hi}, #32",
                "sub   {scratch}, {scratch}, {hi:w}, uxtw",
                "subs  {result}, {result}, {hi}, lsr #32",
                "csetm {hi:w}, cc",
                "sub   {result}, {result}, {hi}",
                "adds  {result}, {result}, {scratch}",
                "csetm {scratch:w}, cs",
                "add   {result}, {result}, {scratch}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V5 `mac_asm11_clone`: exact copy of the production scalar
    /// `mul_acc_reduce` — sanity anchor (must time identical to production).
    #[inline(always)]
    fn v_mac_asm11_clone(acc: u64, a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "adds  {result}, {result}, {acc}",
                "adc   {hi}, {hi}, xzr",
                "umull {scratch}, {hi:w}, {epsilon:w}",
                "subs  {result}, {result}, {hi}, lsr #32",
                "csetm {hi:w}, cc",
                "sub   {result}, {result}, {hi}",
                "adds  {result}, {result}, {scratch}",
                "csetm {scratch:w}, cs",
                "add   {result}, {result}, {scratch}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                acc = in(reg) acc,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V6 `mac_branchy_shift`: latency-specialized MAC — branchy borrow plus
    /// shift-pair epsilon product.
    #[inline(always)]
    fn v_mac_branchy_shift(acc: u64, a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "adds  {result}, {result}, {acc}",
                "adc   {hi}, {hi}, xzr",
                "lsl   {scratch}, {hi}, #32",
                "sub   {scratch}, {scratch}, {hi:w}, uxtw",
                "subs  {result}, {result}, {hi}, lsr #32",
                "b.hs  2f",
                "sub   {result}, {result}, {epsilon}",
                "2:",
                "adds  {result}, {result}, {scratch}",
                "csetm {hi:w}, cs",
                "add   {result}, {result}, {hi}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                acc = in(reg) acc,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V7 `mac_portable`: the portable expression the asm replaced (production
    /// non-aarch64 path) — checks the claimed win of the asm empirically.
    #[inline(always)]
    fn v_mac_portable(acc: u64, a: u64, b: u64) -> u64 {
        F::from_noncanonical_u128((acc as u128) + (a as u128) * (b as u128)).0
    }

    // Pair kernels (packed path). Base clones replicate the production asm
    // blocks exactly; shift variants swap the two umulls for lsl+sub pairs.

    /// V8 `pair_mul_base_clone`: exact copy of production `mul_reduce_pair`.
    #[inline(always)]
    fn v_pair_mul_base(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "umull {scratch0}, {hi0:w}, {epsilon:w}",
                "umull {scratch1}, {hi1:w}, {epsilon:w}",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "csetm {hi0:w}, cc",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "csetm {hi1:w}, cc",
                "sub   {result0}, {result0}, {hi0}",
                "sub   {result1}, {result1}, {hi1}",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    /// V9 `pair_mul_shift`: pair kernel with shift-pair epsilon products
    /// (10 instructions per lane but 2 mul-port ops instead of 3).
    #[inline(always)]
    fn v_pair_mul_shift(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "lsl   {scratch0}, {hi0}, #32",
                "lsl   {scratch1}, {hi1}, #32",
                "sub   {scratch0}, {scratch0}, {hi0:w}, uxtw",
                "sub   {scratch1}, {scratch1}, {hi1:w}, uxtw",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "csetm {hi0:w}, cc",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "csetm {hi1:w}, cc",
                "sub   {result0}, {result0}, {hi0}",
                "sub   {result1}, {result1}, {hi1}",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    /// V10 `pair_mac_base_clone`: exact copy of production `mul_acc_reduce_pair`.
    #[inline(always)]
    fn v_pair_mac_base(
        acc0: u64,
        a0: u64,
        b0: u64,
        acc1: u64,
        a1: u64,
        b1: u64,
    ) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "adds  {result0}, {result0}, {acc0}",
                "adc   {hi0}, {hi0}, xzr",
                "adds  {result1}, {result1}, {acc1}",
                "adc   {hi1}, {hi1}, xzr",
                "umull {scratch0}, {hi0:w}, {epsilon:w}",
                "umull {scratch1}, {hi1:w}, {epsilon:w}",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "csetm {hi0:w}, cc",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "csetm {hi1:w}, cc",
                "sub   {result0}, {result0}, {hi0}",
                "sub   {result1}, {result1}, {hi1}",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                acc0 = in(reg) acc0,
                acc1 = in(reg) acc1,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    /// V11 `pair_mac_shift`: pair MAC with shift-pair epsilon products.
    #[inline(always)]
    fn v_pair_mac_shift(
        acc0: u64,
        a0: u64,
        b0: u64,
        acc1: u64,
        a1: u64,
        b1: u64,
    ) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "adds  {result0}, {result0}, {acc0}",
                "adc   {hi0}, {hi0}, xzr",
                "adds  {result1}, {result1}, {acc1}",
                "adc   {hi1}, {hi1}, xzr",
                "lsl   {scratch0}, {hi0}, #32",
                "lsl   {scratch1}, {hi1}, #32",
                "sub   {scratch0}, {scratch0}, {hi0:w}, uxtw",
                "sub   {scratch1}, {scratch1}, {hi1:w}, uxtw",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "csetm {hi0:w}, cc",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "csetm {hi1:w}, cc",
                "sub   {result0}, {result0}, {hi0}",
                "sub   {result1}, {result1}, {hi1}",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                acc0 = in(reg) acc0,
                acc1 = in(reg) acc1,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    /// V6b `mac_branchy_umull`: branchy borrow, umull epsilon product
    /// (scalar MAC throughput candidate — umull won over shift for plain mul).
    #[inline(always)]
    fn v_mac_branchy_umull(acc: u64, a: u64, b: u64) -> u64 {
        let mut result = a;
        let scratch = b;
        unsafe {
            asm!(
                "umulh {hi}, {result}, {scratch}",
                "mul   {result}, {result}, {scratch}",
                "adds  {result}, {result}, {acc}",
                "adc   {hi}, {hi}, xzr",
                "umull {scratch}, {hi:w}, {epsilon:w}",
                "subs  {result}, {result}, {hi}, lsr #32",
                "b.hs  2f",
                "sub   {result}, {result}, {epsilon}",
                "2:",
                "adds  {result}, {result}, {scratch}",
                "csetm {hi:w}, cs",
                "add   {result}, {result}, {hi}",
                result = inout(reg) result,
                scratch = inout(reg) scratch => _,
                hi = out(reg) _,
                acc = in(reg) acc,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// V12 `pair_mul_branchy`: pair kernel, umull epsilon products, and the
    /// borrow corrections replaced by two rare predicted branches (7 instr
    /// per lane on the common path instead of 9).
    #[inline(always)]
    fn v_pair_mul_branchy(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "umull {scratch0}, {hi0:w}, {epsilon:w}",
                "umull {scratch1}, {hi1:w}, {epsilon:w}",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "b.hs  2f",
                "sub   {result0}, {result0}, {epsilon}",
                "2:",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "b.hs  3f",
                "sub   {result1}, {result1}, {epsilon}",
                "3:",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    /// V13 `pair_mac_branchy`: pair MAC with branchy borrow corrections
    /// (9 instr per lane on the common path instead of 11).
    #[inline(always)]
    fn v_pair_mac_branchy(
        acc0: u64,
        a0: u64,
        b0: u64,
        acc1: u64,
        a1: u64,
        b1: u64,
    ) -> (u64, u64) {
        let mut result0 = a0;
        let mut result1 = a1;
        let scratch0 = b0;
        let scratch1 = b1;
        unsafe {
            asm!(
                "umulh {hi0}, {result0}, {scratch0}",
                "umulh {hi1}, {result1}, {scratch1}",
                "mul   {result0}, {result0}, {scratch0}",
                "mul   {result1}, {result1}, {scratch1}",
                "adds  {result0}, {result0}, {acc0}",
                "adc   {hi0}, {hi0}, xzr",
                "adds  {result1}, {result1}, {acc1}",
                "adc   {hi1}, {hi1}, xzr",
                "umull {scratch0}, {hi0:w}, {epsilon:w}",
                "umull {scratch1}, {hi1:w}, {epsilon:w}",
                "subs  {result0}, {result0}, {hi0}, lsr #32",
                "b.hs  2f",
                "sub   {result0}, {result0}, {epsilon}",
                "2:",
                "subs  {result1}, {result1}, {hi1}, lsr #32",
                "b.hs  3f",
                "sub   {result1}, {result1}, {epsilon}",
                "3:",
                "adds  {result0}, {result0}, {scratch0}",
                "csetm {scratch0:w}, cs",
                "adds  {result1}, {result1}, {scratch1}",
                "csetm {scratch1:w}, cs",
                "add   {result0}, {result0}, {scratch0}",
                "add   {result1}, {result1}, {scratch1}",
                result0 = inout(reg) result0,
                result1 = inout(reg) result1,
                scratch0 = inout(reg) scratch0 => _,
                scratch1 = inout(reg) scratch1 => _,
                hi0 = out(reg) _,
                hi1 = out(reg) _,
                acc0 = in(reg) acc0,
                acc1 = in(reg) acc1,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        (result0, result1)
    }

    // Reduction tails (input = (lo, hi) of the 128-bit value). Used both as
    // from_noncanonical_u128 variants and to bit-identity-test the reduction
    // logic on adversarial (lo, hi) pairs that random a*b cannot reach
    // (P(borrow) ~ 2^-34).

    /// T1: branchless csetm tail — the reduction half of the 9-instr kernel.
    #[inline(always)]
    fn v_red_tail_asm9(lo: u64, hi: u64) -> u64 {
        let mut result = lo;
        let hi = hi;
        unsafe {
            asm!(
                "umull {scratch}, {hi:w}, {epsilon:w}",
                "subs  {result}, {result}, {hi}, lsr #32",
                "csetm {hi:w}, cc",
                "sub   {result}, {result}, {hi}",
                "adds  {result}, {result}, {scratch}",
                "csetm {scratch:w}, cs",
                "add   {result}, {result}, {scratch}",
                result = inout(reg) result,
                hi = inout(reg) hi => _,
                scratch = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        let _ = hi;
        result
    }

    /// T2: branchy borrow + shift-pair epsilon product tail.
    #[inline(always)]
    fn v_red_tail_branchy_shift(lo: u64, hi: u64) -> u64 {
        let mut result = lo;
        let hi = hi;
        unsafe {
            asm!(
                "lsl   {scratch}, {hi}, #32",
                "sub   {scratch}, {scratch}, {hi:w}, uxtw",
                "subs  {result}, {result}, {hi}, lsr #32",
                "b.hs  2f",
                "sub   {result}, {result}, {epsilon}",
                "2:",
                "adds  {result}, {result}, {scratch}",
                "csetm {hi:w}, cs",
                "add   {result}, {result}, {hi}",
                result = inout(reg) result,
                hi = inout(reg) hi => _,
                scratch = out(reg) _,
                epsilon = in(reg) EPSILON,
                options(pure, nomem, nostack),
            );
        }
        let _ = hi;
        result
    }

    // ------------------------------------------------------ identity checks --

    const EDGES: &[u64] = &[
        0,
        1,
        2,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x1_0000_0001,
        0xFFFF_FFFE_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0xFFFF_FFFF_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0xFFFF_FFFF_FFFF_FFFE,
        u64::MAX,
    ];

    fn check_mul_variant(name: &str, f: impl Fn(u64, u64) -> u64) {
        // Full edge cross product.
        for &a in EDGES {
            for &b in EDGES {
                assert_eq!(f(a, b), prod_mul(a, b), "{name}({a:#x}, {b:#x})");
            }
        }
        // 1M random pairs, full u64 range (covers non-canonical inputs).
        let mut s = 0xC0FF_EE00_D15E_A5E5u64;
        for _ in 0..1_000_000 {
            let a = splitmix64(&mut s);
            let b = splitmix64(&mut s);
            assert_eq!(f(a, b), prod_mul(a, b), "{name}({a:#x}, {b:#x})");
        }
        // Mixed random/edge.
        for &e in EDGES {
            for _ in 0..20_000 {
                let r = splitmix64(&mut s);
                assert_eq!(f(e, r), prod_mul(e, r), "{name}({e:#x}, {r:#x})");
                assert_eq!(f(r, e), prod_mul(r, e), "{name}({r:#x}, {e:#x})");
            }
        }
        println!("identity OK: {name} (edge cross + 1M random + mixed)");
    }

    fn check_mac_variant(name: &str, f: impl Fn(u64, u64, u64) -> u64) {
        for &acc in EDGES {
            for &a in EDGES {
                for &b in EDGES {
                    assert_eq!(
                        f(acc, a, b),
                        prod_mac(acc, a, b),
                        "{name}({acc:#x}, {a:#x}, {b:#x})"
                    );
                }
            }
        }
        let mut s = 0xBADC_0DE0_1234_5678u64;
        for _ in 0..1_000_000 {
            let (acc, a, b) = (splitmix64(&mut s), splitmix64(&mut s), splitmix64(&mut s));
            assert_eq!(
                f(acc, a, b),
                prod_mac(acc, a, b),
                "{name}({acc:#x}, {a:#x}, {b:#x})"
            );
        }
        for &e in EDGES {
            for _ in 0..10_000 {
                let (r1, r2) = (splitmix64(&mut s), splitmix64(&mut s));
                assert_eq!(f(e, r1, r2), prod_mac(e, r1, r2), "{name} mixed");
                assert_eq!(f(r1, e, r2), prod_mac(r1, e, r2), "{name} mixed");
                assert_eq!(f(r1, r2, e), prod_mac(r1, r2, e), "{name} mixed");
            }
        }
        println!("identity OK: {name} (edge cross + 1M random + mixed)");
    }

    fn check_pair_mul_variant(name: &str, f: impl Fn(u64, u64, u64, u64) -> (u64, u64)) {
        // Against the production packed path (P = WideGoldilocksField).
        let mut s = 0x1122_3344_5566_7788u64;
        let mut check = |a0: u64, b0: u64, a1: u64, b1: u64| {
            let scalars_a = [F(a0), F(a1), F(a0), F(a1)];
            let scalars_b = [F(b0), F(b1), F(b0), F(b1)];
            let pa = P::pack_slice(&scalars_a);
            let pb = P::pack_slice(&scalars_b);
            let prod = pa[0] * pb[0];
            let lanes = prod.as_slice();
            let (r0, r1) = f(a0, b0, a1, b1);
            assert_eq!(
                (r0, r1),
                (lanes[0].0, lanes[1].0),
                "{name}({a0:#x},{b0:#x},{a1:#x},{b1:#x}) vs packed lanes 0/1"
            );
            assert_eq!(
                (lanes[2].0, lanes[3].0),
                (r0, r1),
                "{name} packed lanes 2/3 disagree"
            );
            // And against the scalar production representative.
            assert_eq!(r0, prod_mul(a0, b0), "{name} lane0 vs scalar");
            assert_eq!(r1, prod_mul(a1, b1), "{name} lane1 vs scalar");
        };
        for &a in EDGES {
            for &b in EDGES {
                check(a, b, b, a);
            }
        }
        for _ in 0..500_000 {
            let (a0, b0) = (splitmix64(&mut s), splitmix64(&mut s));
            let (a1, b1) = (splitmix64(&mut s), splitmix64(&mut s));
            check(a0, b0, a1, b1);
        }
        println!("identity OK: {name} (edge cross + 500k random pairs x 2 lanes)");
    }

    fn check_pair_mac_variant(
        name: &str,
        f: impl Fn(u64, u64, u64, u64, u64, u64) -> (u64, u64),
    ) {
        let mut s = 0xA5A5_5A5A_DEAD_BEEFu64;
        let mut check = |acc0: u64, a0: u64, b0: u64, acc1: u64, a1: u64, b1: u64| {
            let sacc = [F(acc0), F(acc1), F(acc0), F(acc1)];
            let sa = [F(a0), F(a1), F(a0), F(a1)];
            let sb = [F(b0), F(b1), F(b0), F(b1)];
            let pacc = P::pack_slice(&sacc);
            let pa = P::pack_slice(&sa);
            let pb = P::pack_slice(&sb);
            let out = pacc[0].multiply_accumulate(pa[0], pb[0]);
            let lanes = out.as_slice();
            let (r0, r1) = f(acc0, a0, b0, acc1, a1, b1);
            assert_eq!((r0, r1), (lanes[0].0, lanes[1].0), "{name} vs packed");
            assert_eq!(r0, prod_mac(acc0, a0, b0), "{name} lane0 vs scalar");
            assert_eq!(r1, prod_mac(acc1, a1, b1), "{name} lane1 vs scalar");
        };
        for &acc in EDGES {
            for &a in EDGES {
                for &b in EDGES {
                    check(acc, a, b, b, acc, a);
                }
            }
        }
        for _ in 0..500_000 {
            let (acc0, a0, b0) = (splitmix64(&mut s), splitmix64(&mut s), splitmix64(&mut s));
            let (acc1, a1, b1) = (splitmix64(&mut s), splitmix64(&mut s), splitmix64(&mut s));
            check(acc0, a0, b0, acc1, a1, b1);
        }
        println!("identity OK: {name} (edge cross + 500k random triples x 2 lanes)");
    }

    fn check_tail_variant(name: &str, f: impl Fn(u64, u64) -> u64) {
        // Adversarial (lo, hi): full edge cross forces the borrow path
        // (lo < hi >> 32) and both carry outcomes, including the
        // two-representative window W in [0, EPSILON).
        for &lo in EDGES {
            for &hi in EDGES {
                assert_eq!(f(lo, hi), prod_reduce128(lo, hi), "{name}({lo:#x}, {hi:#x})");
            }
        }
        // Explicit two-representative window probes: W = lo - hi_hi + hi_lo*E
        // small; production must pick the same representative.
        for w in 0..64u64 {
            // lo = 0, hi = (E << 32) | 1 gives W = 1*E - E = 0; shift W by w.
            let lo = w;
            let hi = (EPSILON << 32) | 1;
            assert_eq!(f(lo, hi), prod_reduce128(lo, hi), "{name} W-window {w}");
            // borrow + carry combined corner: lo tiny, hi_hi max, hi_lo max.
            let hi2 = u64::MAX;
            assert_eq!(f(w, hi2), prod_reduce128(w, hi2), "{name} borrow corner {w}");
        }
        let mut s = 0xFEED_FACE_CAFE_BEEFu64;
        for _ in 0..1_000_000 {
            let lo = splitmix64(&mut s);
            let hi = splitmix64(&mut s);
            assert_eq!(f(lo, hi), prod_reduce128(lo, hi), "{name}({lo:#x}, {hi:#x})");
            // hi skewed small (extension-sum shape) and hi skewed to force borrows.
            let hi_small = hi >> 60;
            assert_eq!(f(lo, hi_small), prod_reduce128(lo, hi_small), "{name} small-hi");
            let lo_small = lo >> 40;
            assert_eq!(
                f(lo_small, hi),
                prod_reduce128(lo_small, hi),
                "{name} borrow-forcing"
            );
        }
        println!("identity OK: {name} (edge cross + W-window + 3M skewed random)");
    }

    // ------------------------------------------------------------- timing --

    const CHAIN_OPS: usize = 8_000_000;
    const STREAM_SCALARS: usize = 4096;
    const STREAM_PASSES: usize = 2048; // 8.4M lane-ops
    const REPS: usize = 16;
    const MASK: usize = 1023;

    struct Timed {
        name: &'static str,
        ns_per_op: Vec<f64>,
        checksum: u64,
    }

    fn report(shape: &str, timed: &[Timed]) {
        let base_med = median(&timed[0].ns_per_op);
        println!("\n== {shape} (median of {REPS}, ns/op) ==");
        for t in timed {
            let mut v = t.ns_per_op.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = median(&t.ns_per_op);
            let delta = (base_med - med) / base_med * 100.0;
            println!(
                "  {:<28} med {:>7.3}  min {:>7.3}  max {:>7.3}  spread {:>5.1}%  vs-base {:>+6.2}%",
                t.name,
                med,
                v[0],
                v[v.len() - 1],
                (v[v.len() - 1] - v[0]) / med * 100.0,
                delta,
            );
        }
        // Checksums across impls must agree (identical sequences).
        for t in &timed[1..] {
            assert_eq!(
                t.checksum, timed[0].checksum,
                "{shape}: checksum mismatch {} vs {}",
                t.name, timed[0].name
            );
        }
    }

    fn median(v: &[f64]) -> f64 {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        }
    }

    /// Interleaved ABAB timing over closures returning (checksum, ops).
    fn run_shape(shape: &str, impls: Vec<(&'static str, Box<dyn FnMut() -> (u64, usize)>)>) {
        let mut timed: Vec<Timed> = impls
            .iter()
            .map(|(n, _)| Timed {
                name: n,
                ns_per_op: Vec::new(),
                checksum: 0,
            })
            .collect();
        let mut impls = impls;
        // Warmup: one pass each.
        for (_, f) in impls.iter_mut() {
            black_box(f());
        }
        for _rep in 0..REPS {
            for (i, (_, f)) in impls.iter_mut().enumerate() {
                let start = Instant::now();
                let (checksum, ops) = f();
                let el = start.elapsed();
                timed[i].ns_per_op.push(el.as_secs_f64() * 1e9 / ops as f64);
                timed[i].checksum = checksum;
            }
        }
        report(shape, &timed);
    }

    fn gen_field_vec(n: usize, seed: u64) -> Vec<F> {
        let mut s = seed;
        (0..n)
            .map(|_| F(splitmix64(&mut s) % GoldilocksField::ORDER))
            .collect()
    }

    // Shape builders ------------------------------------------------------

    fn chain_mul_prod(m: &[F]) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut x = black_box(F(0x1234_5678_9ABC_DEF0 % GoldilocksField::ORDER));
        for i in 0..CHAIN_OPS {
            x = x * m[i & MASK];
        }
        (x.0, CHAIN_OPS)
    }

    fn chain_mul_var(m: &[F], f: impl Fn(u64, u64) -> u64) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut x = black_box(0x1234_5678_9ABC_DEF0u64 % GoldilocksField::ORDER);
        for i in 0..CHAIN_OPS {
            x = f(x, m[i & MASK].0);
        }
        (x, CHAIN_OPS)
    }

    fn chain_mac_prod(m: &[F], alpha: F) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        // Horner fold: acc = term + acc * alpha (production vanishing-poly shape).
        let mut acc = black_box(F::ZERO);
        for i in 0..CHAIN_OPS {
            acc = Field::multiply_accumulate(&m[i & MASK], acc, alpha);
        }
        (acc.0, CHAIN_OPS)
    }

    fn chain_mac_var(m: &[F], alpha: F, f: impl Fn(u64, u64, u64) -> u64) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut acc = black_box(0u64);
        for i in 0..CHAIN_OPS {
            acc = f(m[i & MASK].0, acc, alpha.0);
        }
        (acc, CHAIN_OPS)
    }

    fn stream_mul8_prod(m: &[F]) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut acc = [
            F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8),
        ];
        for a in acc.iter_mut() {
            *a = black_box(*a);
        }
        for i in 0..CHAIN_OPS / 8 {
            for j in 0..8 {
                acc[j] = acc[j] * m[(i + 37 * j) & MASK];
            }
        }
        (
            acc.iter().fold(0u64, |x, f| x ^ f.0),
            CHAIN_OPS / 8 * 8,
        )
    }

    fn stream_mul8_var(m: &[F], f: impl Fn(u64, u64) -> u64) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut acc = [1u64, 2, 3, 4, 5, 6, 7, 8];
        for a in acc.iter_mut() {
            *a = black_box(*a);
        }
        for i in 0..CHAIN_OPS / 8 {
            for j in 0..8 {
                acc[j] = f(acc[j], m[(i + 37 * j) & MASK].0);
            }
        }
        (acc.iter().fold(0u64, |x, v| x ^ v), CHAIN_OPS / 8 * 8)
    }

    fn stream_mac8_prod(m: &[F], alphas: &[F; 8]) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        // 8 independent Horner folds (vanishing-poly: many independent columns).
        let mut acc = [F::ZERO; 8];
        for a in acc.iter_mut() {
            *a = black_box(*a);
        }
        for i in 0..CHAIN_OPS / 8 {
            for j in 0..8 {
                acc[j] = Field::multiply_accumulate(&m[(i + 41 * j) & MASK], acc[j], alphas[j]);
            }
        }
        (acc.iter().fold(0u64, |x, f| x ^ f.0), CHAIN_OPS / 8 * 8)
    }

    fn stream_mac8_var(
        m: &[F],
        alphas: &[F; 8],
        f: impl Fn(u64, u64, u64) -> u64,
    ) -> (u64, usize) {
        let m: &[F; MASK + 1] = m.try_into().unwrap();
        let mut acc = [0u64; 8];
        for a in acc.iter_mut() {
            *a = black_box(*a);
        }
        for i in 0..CHAIN_OPS / 8 {
            for j in 0..8 {
                acc[j] = f(m[(i + 41 * j) & MASK].0, acc[j], alphas[j].0);
            }
        }
        (acc.iter().fold(0u64, |x, v| x ^ v), CHAIN_OPS / 8 * 8)
    }

    fn stream_packed_mul_prod(a: &[P], b: &[P], out: &mut [P]) -> (u64, usize) {
        for _ in 0..STREAM_PASSES {
            for ((x, y), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                *o = *x * *y;
            }
            black_box(&mut *out);
        }
        let sum = out
            .iter()
            .flat_map(|p| p.as_slice())
            .fold(0u64, |x, f| x ^ f.0);
        (sum, STREAM_PASSES * a.len() * P::WIDTH)
    }

    fn stream_packed_mul_var(
        a: &[u64],
        b: &[u64],
        out: &mut [u64],
        f: impl Fn(u64, u64, u64, u64) -> (u64, u64),
    ) -> (u64, usize) {
        let n = a.len();
        for _ in 0..STREAM_PASSES {
            for ((av, bv), ov) in a
                .chunks_exact(4)
                .zip(b.chunks_exact(4))
                .zip(out.chunks_exact_mut(4))
            {
                let (r0, r1) = f(av[0], bv[0], av[1], bv[1]);
                let (r2, r3) = f(av[2], bv[2], av[3], bv[3]);
                ov[0] = r0;
                ov[1] = r1;
                ov[2] = r2;
                ov[3] = r3;
            }
            black_box(&mut *out);
        }
        (out.iter().fold(0u64, |x, v| x ^ v), STREAM_PASSES * n)
    }

    fn stream_packed_mac_prod(acc: &[P], a: &[P], b: &[P], out: &mut [P]) -> (u64, usize) {
        for _ in 0..STREAM_PASSES {
            for (((c, x), y), o) in acc.iter().zip(a.iter()).zip(b.iter()).zip(out.iter_mut()) {
                *o = c.multiply_accumulate(*x, *y);
            }
            black_box(&mut *out);
        }
        let sum = out
            .iter()
            .flat_map(|p| p.as_slice())
            .fold(0u64, |x, f| x ^ f.0);
        (sum, STREAM_PASSES * a.len() * P::WIDTH)
    }

    fn stream_packed_mac_var(
        acc: &[u64],
        a: &[u64],
        b: &[u64],
        out: &mut [u64],
        f: impl Fn(u64, u64, u64, u64, u64, u64) -> (u64, u64),
    ) -> (u64, usize) {
        let n = a.len();
        for _ in 0..STREAM_PASSES {
            for (((cv, av), bv), ov) in acc
                .chunks_exact(4)
                .zip(a.chunks_exact(4))
                .zip(b.chunks_exact(4))
                .zip(out.chunks_exact_mut(4))
            {
                let (r0, r1) = f(cv[0], av[0], bv[0], cv[1], av[1], bv[1]);
                let (r2, r3) = f(cv[2], av[2], bv[2], cv[3], av[3], bv[3]);
                ov[0] = r0;
                ov[1] = r1;
                ov[2] = r2;
                ov[3] = r3;
            }
            black_box(&mut *out);
        }
        (out.iter().fold(0u64, |x, v| x ^ v), STREAM_PASSES * n)
    }

    fn stream_reduce128_prod(los: &[u64], his: &[u64], out: &mut [u64]) -> (u64, usize) {
        for _ in 0..STREAM_PASSES {
            for ((l, h), o) in los.iter().zip(his.iter()).zip(out.iter_mut()) {
                *o = prod_reduce128(*l, *h);
            }
            black_box(&mut *out);
        }
        (out.iter().fold(0u64, |x, v| x ^ v), STREAM_PASSES * los.len())
    }

    fn stream_reduce128_var(
        los: &[u64],
        his: &[u64],
        out: &mut [u64],
        f: impl Fn(u64, u64) -> u64,
    ) -> (u64, usize) {
        for _ in 0..STREAM_PASSES {
            for ((l, h), o) in los.iter().zip(his.iter()).zip(out.iter_mut()) {
                *o = f(*l, *h);
            }
            black_box(&mut *out);
        }
        (out.iter().fold(0u64, |x, v| x ^ v), STREAM_PASSES * los.len())
    }

    // --------------------------------------------------------------- main --

    pub fn main() {
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
        let args: Vec<String> = std::env::args().collect();
        let skip_identity = args.iter().any(|a| a == "--skip-identity");

        if !skip_identity {
            println!("--- bit-identity checks (before any timing) ---");
            check_mul_variant("V1 mul_asm9", v_mul_asm9);
            check_mul_variant("V2 mul_branchy_umull", v_mul_branchy_umull);
            check_mul_variant("V3 mul_branchy_shift", v_mul_branchy_shift);
            check_mul_variant("V4 mul_asm9_shift", v_mul_asm9_shift);
            check_mac_variant("V5 mac_asm11_clone", v_mac_asm11_clone);
            check_mac_variant("V6 mac_branchy_shift", v_mac_branchy_shift);
            check_mac_variant("V6b mac_branchy_umull", v_mac_branchy_umull);
            check_mac_variant("V7 mac_portable", v_mac_portable);
            check_pair_mul_variant("V8 pair_mul_base_clone", v_pair_mul_base);
            check_pair_mul_variant("V9 pair_mul_shift", v_pair_mul_shift);
            check_pair_mul_variant("V12 pair_mul_branchy", v_pair_mul_branchy);
            check_pair_mac_variant("V10 pair_mac_base_clone", v_pair_mac_base);
            check_pair_mac_variant("V13 pair_mac_branchy", v_pair_mac_branchy);
            check_pair_mac_variant("V11 pair_mac_shift", v_pair_mac_shift);
            check_tail_variant("T1 red_tail_asm9", v_red_tail_asm9);
            check_tail_variant("T2 red_tail_branchy_shift", v_red_tail_branchy_shift);
            println!("--- all identity checks passed ---");
        }

        // Shared data.
        let m = gen_field_vec(MASK + 1, 0x5EED_0001);
        let mut alpha_seed = 0x5EED_0002u64;
        let alpha = F(splitmix64(&mut alpha_seed) % GoldilocksField::ORDER);
        let mut aseed = 0x5EED_0003u64;
        let alphas: [F; 8] =
            core::array::from_fn(|_| F(splitmix64(&mut aseed) % GoldilocksField::ORDER));

        let sa = gen_field_vec(STREAM_SCALARS, 0x5EED_0010);
        let sb = gen_field_vec(STREAM_SCALARS, 0x5EED_0011);
        let sacc = gen_field_vec(STREAM_SCALARS, 0x5EED_0012);
        let pa: Vec<P> = P::pack_slice(&sa).to_vec();
        let pb: Vec<P> = P::pack_slice(&sb).to_vec();
        let pacc: Vec<P> = P::pack_slice(&sacc).to_vec();
        let ra: Vec<u64> = sa.iter().map(|f| f.0).collect();
        let rb: Vec<u64> = sb.iter().map(|f| f.0).collect();
        let racc: Vec<u64> = sacc.iter().map(|f| f.0).collect();

        // reduce128 inputs: product-shaped (full-range hi) and sum-of-4-products
        // shaped (hi up to ~2^66 → hi mostly small).
        let mut rs = 0x5EED_0020u64;
        let mut los_full = vec![0u64; STREAM_SCALARS];
        let mut his_full = vec![0u64; STREAM_SCALARS];
        let mut los_sum = vec![0u64; STREAM_SCALARS];
        let mut his_sum = vec![0u64; STREAM_SCALARS];
        for i in 0..STREAM_SCALARS {
            let a = splitmix64(&mut rs) % GoldilocksField::ORDER;
            let b = splitmix64(&mut rs) % GoldilocksField::ORDER;
            let p = (a as u128) * (b as u128);
            los_full[i] = p as u64;
            his_full[i] = (p >> 64) as u64;
            let mut sum = 0u128;
            for _ in 0..4 {
                let x = splitmix64(&mut rs) % GoldilocksField::ORDER;
                let y = splitmix64(&mut rs) % GoldilocksField::ORDER;
                sum = sum.wrapping_add((x as u128) * (y as u128));
            }
            los_sum[i] = sum as u64;
            his_sum[i] = (sum >> 64) as u64;
        }

        println!("\n--- timing (interleaved ABAB, {REPS} reps) ---");

        // 1. Dependent-chain scalar mul.
        {
            let m1 = m.clone();
            let m2 = m.clone();
            let m3 = m.clone();
            let m4 = m.clone();
            let m5 = m.clone();
            run_shape(
                "chain/scalar-mul (dependent)",
                vec![
                    ("production (portable)", Box::new(move || chain_mul_prod(&m1))),
                    ("V1 mul_asm9", Box::new(move || chain_mul_var(&m2, v_mul_asm9))),
                    (
                        "V2 mul_branchy_umull",
                        Box::new(move || chain_mul_var(&m3, v_mul_branchy_umull)),
                    ),
                    (
                        "V3 mul_branchy_shift",
                        Box::new(move || chain_mul_var(&m4, v_mul_branchy_shift)),
                    ),
                    (
                        "V4 mul_asm9_shift",
                        Box::new(move || chain_mul_var(&m5, v_mul_asm9_shift)),
                    ),
                ],
            );
        }

        // 2. Dependent-chain scalar MAC (Horner fold).
        {
            let m1 = m.clone();
            let m2 = m.clone();
            let m3 = m.clone();
            let m4 = m.clone();
            let m5 = m.clone();
            run_shape(
                "chain/scalar-mac Horner (dependent)",
                vec![
                    (
                        "production (asm11)",
                        Box::new(move || chain_mac_prod(&m1, alpha)),
                    ),
                    (
                        "V5 mac_asm11_clone",
                        Box::new(move || chain_mac_var(&m2, alpha, v_mac_asm11_clone)),
                    ),
                    (
                        "V6 mac_branchy_shift",
                        Box::new(move || chain_mac_var(&m3, alpha, v_mac_branchy_shift)),
                    ),
                    (
                        "V6b mac_branchy_umull",
                        Box::new(move || chain_mac_var(&m5, alpha, v_mac_branchy_umull)),
                    ),
                    (
                        "V7 mac_portable",
                        Box::new(move || chain_mac_var(&m4, alpha, v_mac_portable)),
                    ),
                ],
            );
        }

        // 3. Independent scalar mul streams (8-way).
        {
            let m1 = m.clone();
            let m2 = m.clone();
            let m3 = m.clone();
            let m4 = m.clone();
            let m5 = m.clone();
            run_shape(
                "stream/scalar-mul-8way (independent)",
                vec![
                    ("production (portable)", Box::new(move || stream_mul8_prod(&m1))),
                    ("V1 mul_asm9", Box::new(move || stream_mul8_var(&m2, v_mul_asm9))),
                    (
                        "V2 mul_branchy_umull",
                        Box::new(move || stream_mul8_var(&m3, v_mul_branchy_umull)),
                    ),
                    (
                        "V3 mul_branchy_shift",
                        Box::new(move || stream_mul8_var(&m4, v_mul_branchy_shift)),
                    ),
                    (
                        "V4 mul_asm9_shift",
                        Box::new(move || stream_mul8_var(&m5, v_mul_asm9_shift)),
                    ),
                ],
            );
        }

        // 4. Independent scalar MAC streams (8-way Horner columns).
        {
            let m1 = m.clone();
            let m2 = m.clone();
            let m3 = m.clone();
            let m4 = m.clone();
            let m5 = m.clone();
            let al = alphas;
            run_shape(
                "stream/scalar-mac-8way (independent)",
                vec![
                    (
                        "production (asm11)",
                        Box::new(move || stream_mac8_prod(&m1, &al)),
                    ),
                    (
                        "V5 mac_asm11_clone",
                        Box::new(move || stream_mac8_var(&m2, &al, v_mac_asm11_clone)),
                    ),
                    (
                        "V6 mac_branchy_shift",
                        Box::new(move || stream_mac8_var(&m3, &al, v_mac_branchy_shift)),
                    ),
                    (
                        "V6b mac_branchy_umull",
                        Box::new(move || stream_mac8_var(&m5, &al, v_mac_branchy_umull)),
                    ),
                    (
                        "V7 mac_portable",
                        Box::new(move || stream_mac8_var(&m4, &al, v_mac_portable)),
                    ),
                ],
            );
        }

        // 5. Packed mul stream (batch throughput).
        {
            let (pa1, pb1) = (pa.clone(), pb.clone());
            let mut pout1 = vec![P::ZEROS; pa.len()];
            let (ra1, rb1) = (ra.clone(), rb.clone());
            let mut rout1 = vec![0u64; ra.len()];
            let (ra2, rb2) = (ra.clone(), rb.clone());
            let mut rout2 = vec![0u64; ra.len()];
            let (ra3, rb3) = (ra.clone(), rb.clone());
            let mut rout3 = vec![0u64; ra.len()];
            run_shape(
                "stream/packed-mul (independent)",
                vec![
                    (
                        "production (pair asm9)",
                        Box::new(move || stream_packed_mul_prod(&pa1, &pb1, &mut pout1)),
                    ),
                    (
                        "V8 pair_mul_base_clone",
                        Box::new(move || {
                            stream_packed_mul_var(&ra1, &rb1, &mut rout1, v_pair_mul_base)
                        }),
                    ),
                    (
                        "V9 pair_mul_shift",
                        Box::new(move || {
                            stream_packed_mul_var(&ra2, &rb2, &mut rout2, v_pair_mul_shift)
                        }),
                    ),
                    (
                        "V12 pair_mul_branchy",
                        Box::new(move || {
                            stream_packed_mul_var(&ra3, &rb3, &mut rout3, v_pair_mul_branchy)
                        }),
                    ),
                ],
            );
        }

        // 6. Packed MAC stream (batch throughput).
        {
            let (pacc1, pa1, pb1) = (pacc.clone(), pa.clone(), pb.clone());
            let mut pout1 = vec![P::ZEROS; pa.len()];
            let (racc1, ra1, rb1) = (racc.clone(), ra.clone(), rb.clone());
            let mut rout1 = vec![0u64; ra.len()];
            let (racc2, ra2, rb2) = (racc.clone(), ra.clone(), rb.clone());
            let mut rout2 = vec![0u64; ra.len()];
            let (racc3, ra3, rb3) = (racc.clone(), ra.clone(), rb.clone());
            let mut rout3 = vec![0u64; ra.len()];
            run_shape(
                "stream/packed-mac (independent)",
                vec![
                    (
                        "production (pair asm11)",
                        Box::new(move || {
                            stream_packed_mac_prod(&pacc1, &pa1, &pb1, &mut pout1)
                        }),
                    ),
                    (
                        "V10 pair_mac_base_clone",
                        Box::new(move || {
                            stream_packed_mac_var(&racc1, &ra1, &rb1, &mut rout1, v_pair_mac_base)
                        }),
                    ),
                    (
                        "V11 pair_mac_shift",
                        Box::new(move || {
                            stream_packed_mac_var(&racc2, &ra2, &rb2, &mut rout2, v_pair_mac_shift)
                        }),
                    ),
                    (
                        "V13 pair_mac_branchy",
                        Box::new(move || {
                            stream_packed_mac_var(&racc3, &ra3, &rb3, &mut rout3, v_pair_mac_branchy)
                        }),
                    ),
                ],
            );
        }

        // 7. from_noncanonical_u128 streams, product-shaped hi.
        {
            let (l1, h1) = (los_full.clone(), his_full.clone());
            let mut o1 = vec![0u64; STREAM_SCALARS];
            let (l2, h2) = (los_full.clone(), his_full.clone());
            let mut o2 = vec![0u64; STREAM_SCALARS];
            let (l3, h3) = (los_full.clone(), his_full.clone());
            let mut o3 = vec![0u64; STREAM_SCALARS];
            run_shape(
                "stream/reduce128 full-hi (independent)",
                vec![
                    (
                        "production (portable)",
                        Box::new(move || stream_reduce128_prod(&l1, &h1, &mut o1)),
                    ),
                    (
                        "T1 red_tail_asm9",
                        Box::new(move || stream_reduce128_var(&l2, &h2, &mut o2, v_red_tail_asm9)),
                    ),
                    (
                        "T2 red_tail_branchy_shift",
                        Box::new(move || {
                            stream_reduce128_var(&l3, &h3, &mut o3, v_red_tail_branchy_shift)
                        }),
                    ),
                ],
            );
        }

        // 8. from_noncanonical_u128 streams, sum-of-products-shaped hi.
        {
            let (l1, h1) = (los_sum.clone(), his_sum.clone());
            let mut o1 = vec![0u64; STREAM_SCALARS];
            let (l2, h2) = (los_sum.clone(), his_sum.clone());
            let mut o2 = vec![0u64; STREAM_SCALARS];
            let (l3, h3) = (los_sum.clone(), his_sum.clone());
            let mut o3 = vec![0u64; STREAM_SCALARS];
            run_shape(
                "stream/reduce128 sum-hi (independent)",
                vec![
                    (
                        "production (portable)",
                        Box::new(move || stream_reduce128_prod(&l1, &h1, &mut o1)),
                    ),
                    (
                        "T1 red_tail_asm9",
                        Box::new(move || stream_reduce128_var(&l2, &h2, &mut o2, v_red_tail_asm9)),
                    ),
                    (
                        "T2 red_tail_branchy_shift",
                        Box::new(move || {
                            stream_reduce128_var(&l3, &h3, &mut o3, v_red_tail_branchy_shift)
                        }),
                    ),
                ],
            );
        }

        println!("\nlab_reduce128 done.");
    }
}
