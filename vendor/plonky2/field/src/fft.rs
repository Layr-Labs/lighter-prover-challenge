use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
#[cfg(target_arch = "aarch64")]
use crate::goldilocks_field::{mul_16th_root_powers, mul_fourth_root};

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

/// Static butterfly twiddle dispatch. The marker type is chosen once per FFT
/// transform, so there is no type test or dynamic branch inside a butterfly.
trait FftTwiddleMul<P: PackedField> {
    fn mul(twiddle: P, value: P) -> P;
}

struct GeneralTwiddle;
struct BaseSubfieldTwiddle;

impl<P: PackedField> FftTwiddleMul<P> for GeneralTwiddle {
    #[inline(always)]
    fn mul(twiddle: P, value: P) -> P {
        twiddle * value
    }
}

impl<P: PackedField> FftTwiddleMul<P> for BaseSubfieldTwiddle {
    #[inline(always)]
    fn mul(twiddle: P, value: P) -> P {
        P::mul_fft_base_twiddle(twiddle, value)
    }
}

pub type FftRootTable<F> = Vec<Vec<F>>;

pub fn fft_root_table<F: Field>(n: usize) -> FftRootTable<F> {
    let lg_n = log2_strict(n);
    // bases[i] = g^2^i, for i = 0, ..., lg_n - 1
    let mut bases = Vec::with_capacity(lg_n);
    let mut base = F::primitive_root_of_unity(lg_n);
    bases.push(base);
    for _ in 1..lg_n {
        base = base.square(); // base = g^2^_
        bases.push(base);
    }

    let mut root_table = Vec::with_capacity(lg_n);
    for lg_m in 1..=lg_n {
        let half_m = 1 << (lg_m - 1);
        let base = bases[lg_n - lg_m];
        let root_row = base.powers().take(half_m.max(2)).collect();
        root_table.push(root_row);
    }
    root_table
}

/// Process-wide cache of FFT root tables for the prover's hot field types,
/// contention-free on the steady-state path: one static fixed-size array of
/// `OnceLock` slots per cached field type, indexed by log2(size). A hit is a
/// `TypeId` compare (const-folded per monomorphization) plus one atomic
/// acquire load — no mutex, no hashing, no shared cache-line writes after a
/// slot initializes. `fft_root_table` is deterministic per (field, size), so
/// a cache hit returns exactly the values a fresh computation would; the
/// cache only avoids recomputing the table on every table-less FFT (FRI fold
/// rounds, the final-polynomial coset FFT, IFFT/LDE calls without a
/// precomputed table). Field types without a dedicated table (and sizes past
/// `MAX_LG_N`) fall back to a fresh, value-identical computation.
#[cfg(feature = "std")]
mod root_table_cache {
    use alloc::vec::Vec;
    use core::any::{Any, TypeId};
    use std::sync::{Arc, OnceLock};

    use super::{FftRootTable, fft_root_table};
    use crate::extension::quadratic::QuadraticExtension;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field;

    /// Slots for sizes up to `1 << 32` elements, far past any FFT here.
    const MAX_LG_N: usize = 33;

    /// A slot holds the type-erased `Arc<FftRootTable<F>>` for its static's
    /// fixed field type; erasure keeps the generic accessor safe (no
    /// transmute) while each static's writer only ever stores its own type.
    type Slot = OnceLock<Arc<dyn Any + Send + Sync>>;

    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY_SLOT: Slot = OnceLock::new();

    static GOLDILOCKS_TABLES: [Slot; MAX_LG_N] = [EMPTY_SLOT; MAX_LG_N];
    static GOLDILOCKS_EXT2_TABLES: [Slot; MAX_LG_N] = [EMPTY_SLOT; MAX_LG_N];

    /// The dedicated slot array for `F`, if `F` is one of the cached types.
    fn per_type_tables<F: Field>() -> Option<&'static [Slot; MAX_LG_N]> {
        let f = TypeId::of::<F>();
        if f == TypeId::of::<GoldilocksField>() {
            Some(&GOLDILOCKS_TABLES)
        } else if f == TypeId::of::<QuadraticExtension<GoldilocksField>>() {
            Some(&GOLDILOCKS_EXT2_TABLES)
        } else {
            None
        }
    }

    pub(super) fn get<F: Field>(lg_n: usize) -> Arc<FftRootTable<F>> {
        if let Some(tables) = per_type_tables::<F>() {
            if let Some(slot) = tables.get(lg_n) {
                // First caller for this (type, size) computes the table; racers
                // block only during that one-time construction. Afterwards this
                // is a single atomic load of the initialized slot.
                let erased = slot.get_or_init(|| Arc::new(fft_root_table::<F>(1 << lg_n)));
                if let Ok(table) = Arc::clone(erased).downcast::<FftRootTable<F>>() {
                    return table;
                }
                // Unreachable in practice: each static stores only its own
                // type. Fall through to a fresh (value-identical) computation.
            }
        }
        Arc::new(fft_root_table::<F>(1 << lg_n))
    }

    static GOLDILOCKS_SUBGROUPS: [Slot; MAX_LG_N] = [EMPTY_SLOT; MAX_LG_N];

    /// The dedicated subgroup slot array for `F`, if `F` is a cached type.
    /// Only the base field is cached: subgroup elements of an extension's
    /// two-adic subgroup are not needed on any hot path.
    fn per_type_subgroups<F: Field>() -> Option<&'static [Slot; MAX_LG_N]> {
        (TypeId::of::<F>() == TypeId::of::<GoldilocksField>()).then_some(&GOLDILOCKS_SUBGROUPS)
    }

    /// `F::two_adic_subgroup(lg_n)` through the same process-wide cache
    /// discipline as the root tables: deterministic per (field, size), so a
    /// hit returns exactly the values a fresh computation would.
    pub(super) fn get_subgroup<F: Field>(lg_n: usize) -> Arc<Vec<F>> {
        if let Some(tables) = per_type_subgroups::<F>() {
            if let Some(slot) = tables.get(lg_n) {
                let erased = slot.get_or_init(|| Arc::new(F::two_adic_subgroup(lg_n)));
                if let Ok(subgroup) = Arc::clone(erased).downcast::<Vec<F>>() {
                    return subgroup;
                }
            }
        }
        Arc::new(F::two_adic_subgroup(lg_n))
    }
}

/// Process-wide cached `F::two_adic_subgroup(lg_n)`, value-identical to a
/// fresh computation. Avoids the per-call primitive-root exponentiation and
/// power chain on hot paths that need a small fixed subgroup repeatedly.
#[cfg(feature = "std")]
pub fn cached_two_adic_subgroup<F: Field>(lg_n: usize) -> alloc::sync::Arc<Vec<F>> {
    root_table_cache::get_subgroup::<F>(lg_n)
}

#[cfg(not(feature = "std"))]
pub fn cached_two_adic_subgroup<F: Field>(lg_n: usize) -> alloc::sync::Arc<Vec<F>> {
    alloc::sync::Arc::new(F::two_adic_subgroup(lg_n))
}

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    if let Some(table) = root_table {
        fft_classic(input, zero_factor.unwrap_or(0), table);
        return;
    }
    #[cfg(feature = "std")]
    let computed_root_table = root_table_cache::get::<F>(log2_strict(input.len()));
    #[cfg(not(feature = "std"))]
    let computed_root_table = fft_root_table::<F>(input.len());

    fft_classic(input, zero_factor.unwrap_or(0), &computed_root_table);
}

/// Computes an FFT in the caller-provided buffer.
///
/// This is equivalent to [`fft_with_options`], but permits buffers backed by
/// shared CPU/GPU memory to remain in place.
#[inline]
pub fn fft_in_place_with_options<F: Field>(
    buffer: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    fft_dispatch(buffer, zero_factor, root_table);
}

#[inline]
pub fn fft<F: Field>(poly: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    fft_with_options(poly, None, None)
}

#[inline]
pub fn fft_with_options<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);
    PolynomialValues::new(buffer)
}

#[inline]
pub fn ifft<F: Field>(poly: PolynomialValues<F>) -> PolynomialCoeffs<F> {
    ifft_with_options(poly, None, None)
}

pub fn ifft_with_options<F: Field>(
    poly: PolynomialValues<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialCoeffs<F> {
    ifft_with_options_and_postscale(poly, zero_factor, root_table, None)
}

pub(crate) fn ifft_with_options_and_postscale<F: Field>(
    poly: PolynomialValues<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
    postscale: Option<&[F]>,
) -> PolynomialCoeffs<F> {
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);
    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);

    match postscale {
        None => {
            // We reverse all values except the first, and divide each by n.
            buffer[0] *= n_inv;
            buffer[n / 2] *= n_inv;
            for i in 1..(n / 2) {
                let j = n - i;
                let coeffs_i = buffer[j] * n_inv;
                let coeffs_j = buffer[i] * n_inv;
                buffer[i] = coeffs_i;
                buffer[j] = coeffs_j;
            }
        }
        Some(scales) => {
            assert_eq!(scales.len(), n);
            // Fuse the caller's coefficient scaling into the same writes as
            // IFFT reversal and normalization, preserving multiplication order.
            buffer[0] *= n_inv;
            buffer[n / 2] *= n_inv;
            buffer[0] *= scales[0];
            if n > 1 {
                buffer[n / 2] *= scales[n / 2];
            }
            for i in 1..(n / 2) {
                let j = n - i;
                let mut coeffs_i = buffer[j] * n_inv;
                let mut coeffs_j = buffer[i] * n_inv;
                coeffs_i *= scales[i];
                coeffs_j *= scales[j];
                buffer[i] = coeffs_i;
                buffer[j] = coeffs_j;
            }
        }
    }
    PolynomialCoeffs { coeffs: buffer }
}

/// `ifft` of a borrowed column without the caller-side copy: the initial
/// bit-reversal permutation is applied as an out-of-place gather from
/// `values` into the fresh buffer (the same permutation `fft_classic`'s
/// in-place pass would apply to a clone), after which the identical
/// butterfly layers and coefficient reversal/scaling run. Value-identical
/// to `ifft(PolynomialValues::new(values.to_vec()))`.
pub fn ifft_borrowed<F: Field>(values: &[F]) -> PolynomialCoeffs<F> {
    let n = values.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let mut buffer = plonky2_util::reverse_index_bits(values);

    #[cfg(feature = "std")]
    let root_table = root_table_cache::get::<F>(lg_n);
    #[cfg(not(feature = "std"))]
    let root_table = fft_root_table::<F>(n);

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    if lg_n <= lg_packed_width {
        fft_classic_simd::<F>(&mut buffer, 0, lg_n, &root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(&mut buffer, 0, lg_n, &root_table);
    }

    // Identical post-pass to `ifft_with_options`: reverse all values except
    // the first, dividing each by n.
    buffer[0] *= n_inv;
    buffer[n / 2] *= n_inv;
    for i in 1..(n / 2) {
        let j = n - i;
        let coeffs_i = buffer[j] * n_inv;
        let coeffs_j = buffer[i] * n_inv;
        buffer[i] = coeffs_i;
        buffer[j] = coeffs_j;
    }
    PolynomialCoeffs { coeffs: buffer }
}

/// Generic FFT implementation that works with both scalar and packed inputs.
#[unroll_for_loops]
fn fft_classic_simd_with<P, M>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    let lg_packed_width = log2_strict(P::WIDTH); // 0 when P is a scalar.
    let packed_values = P::pack_slice_mut(values);
    let packed_n = packed_values.len();
    debug_assert!(packed_n == 1 << (lg_n - lg_packed_width));

    // Want the below for loop to unroll, hence the need for a literal.
    // This loop will not run when P is a scalar.
    assert!(lg_packed_width <= 4);
    for lg_half_m in 0..4 {
        if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
            // Intuitively, we split values into m slices: subarr[0], ..., subarr[m - 1]. Each of
            // those slices is split into two halves: subarr[j].left, subarr[j].right. We do
            // (subarr[j].left[k], subarr[j].right[k])
            //   := f(subarr[j].left[k], subarr[j].right[k], omega[k]),
            // where f(u, v, omega) = (u + omega * v, u - omega * v).
            let half_m = 1 << lg_half_m;

            // Set omega to root_table[lg_half_m][0..half_m] but repeated.
            let mut omega = P::default();
            for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                *omega_j = root_table[lg_half_m][j % half_m];
            }

            for k in (0..packed_n).step_by(2) {
                // We have two vectors and want to do math on pairs of adjacent elements (or for
                // lg_half_m > 0, pairs of adjacent blocks of elements). .interleave does the
                // appropriate shuffling and is its own inverse.
                let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                let t = M::mul(omega, v);
                (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
            }
        }
    }

    // We've already done the first lg_packed_width (if they were required) iterations.
    let s = max(r, lg_packed_width);
    fft_classic_simd_optimized_layers::<P, M>(packed_values, s, lg_n, root_table);
}

#[inline(always)]
fn fft_classic_simd<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    if lg_n == P::Scalar::TWO_ADICITY {
        // A quadratic extension's final two-adic level has extension-only
        // roots. Keep general multiplication for that entire (impractically
        // large) transform; all smaller transforms use base-subfield dispatch.
        fft_classic_simd_with::<P, GeneralTwiddle>(values, r, lg_n, root_table);
    } else {
        fft_classic_simd_with::<P, BaseSubfieldTwiddle>(values, r, lg_n, root_table);
    }
}

/// Goldilocks `x + y` on two lanes, reproducing `impl Add for GoldilocksField`
/// word for word:
///     (sum, over)  = x.overflowing_add(y)
///     (sum, over2) = sum.overflowing_add(over as u64 * EPSILON)
///     if over2 { sum += EPSILON }
/// The scalar form branch-hints that second correction as rare; here it is an
/// unconditional masked add, so the vector form *removes* a branch. Because
/// `Add` is pure arithmetic -- no assembly, no lookup -- the result is a pure
/// function of the inputs and these words are identical, not merely congruent.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gl_add_neon(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
    eps: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    let sum = vaddq_u64(x, y);
    // Unsigned carry: the wrapped sum is below the addend exactly on overflow.
    let over = vcltq_u64(sum, x);
    let sum2 = vaddq_u64(sum, vandq_u64(over, eps));
    let over2 = vcltq_u64(sum2, sum);
    vaddq_u64(sum2, vandq_u64(over2, eps))
}

/// Goldilocks `x - y` on two lanes, reproducing `impl Sub for GoldilocksField`.
/// Borrow is `x < y`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gl_sub_neon(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
    eps: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    let diff = vsubq_u64(x, y);
    let under = vcltq_u64(x, y);
    let adj = vandq_u64(under, eps);
    let diff2 = vsubq_u64(diff, adj);
    let under2 = vcltq_u64(diff, adj);
    vsubq_u64(diff2, vandq_u64(under2, eps))
}

/// One butterfly layer over base-field scalars, with the modular reduction in
/// vector registers.
///
/// Same blocks, same pairing, same twiddles, same order as the generic body:
/// sub-blocks of `m = 2^(lg_half_m+1)` elements, pairing `j` with `half + j`.
/// The multiply stays scalar -- aarch64 has no 64x64->128 widening multiply, so
/// vectorising it would cost more than it saves -- and goes through the same
/// paired `NeonGoldilocksField` assembly the generic path uses. Only `t`
/// crosses the register files.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fft_classic_simd_single_layer_neon(
    values: &mut [crate::goldilocks_field::GoldilocksField],
    lg_half_m: usize,
    omega_row: &[crate::goldilocks_field::GoldilocksField],
) {
    use core::arch::aarch64::*;
    use crate::arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;

    const EPSILON: u64 = (1 << 32) - 1;
    let half = 1usize << lg_half_m;
    let m = half << 1;
    debug_assert!(omega_row.len() >= half);
    let base = values.as_mut_ptr().cast::<u64>();
    unsafe {
        let eps = vdupq_n_u64(EPSILON);
        let mut k = 0;
        while k + m <= values.len() {
            let mut j = 0;
            while j + 2 <= half {
                let v = NeonGoldilocksField([
                    *values.get_unchecked(k + half + j),
                    *values.get_unchecked(k + half + j + 1),
                ]);
                let w = NeonGoldilocksField([
                    *omega_row.get_unchecked(j),
                    *omega_row.get_unchecked(j + 1),
                ]);
                let t = w * v;
                // The only register-file crossing in the loop: two fmovs.
                let tv = vcombine_u64(vcreate_u64(t.0[0].0), vcreate_u64(t.0[1].0));
                let u = vld1q_u64(base.add(k + j));
                vst1q_u64(base.add(k + j), gl_add_neon(u, tv, eps));
                vst1q_u64(base.add(k + half + j), gl_sub_neon(u, tv, eps));
                j += 2;
            }
            // `half` is a power of two and at least 2 whenever this path is
            // taken, so this tail never runs; kept so the kernel is correct for
            // any shape rather than only the ones production uses.
            while j < half {
                let t = omega_row[j] * values[k + half + j];
                let u = values[k + j];
                values[k + j] = u + t;
                values[k + half + j] = u - t;
                j += 1;
            }
            k += m;
        }
    }
}

/// One FRI butterfly layer over quadratic-extension elements, with the
/// modular reduction in vector registers.
///
/// `QuadraticExtension<GoldilocksField>` is `X^2 - 7`, so
/// `(a0 + a1*u) * (b0 + b1*u) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*u`.
/// Production FRI twiddle rows are almost entirely base-subfield values
/// `[w, 0]`, for which the product is `[w*a0, w*a1]` — two base multiplications
/// instead of four. A single row-level scan picks that fast path (one paired
/// `NeonGoldilocksField` mul hides the scalar latency); otherwise the general
/// four-product form runs. The butterfly `u + t` / `u - t` reductions always
/// run as two-lane vector adds/subs reproducing `impl Add/Sub for
/// GoldilocksField` word for word. Same blocks, same pairing, same twiddles,
/// same order as the generic body.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fft_classic_simd_single_layer_neon_ext(
    values: &mut [crate::extension::quadratic::QuadraticExtension<
        crate::goldilocks_field::GoldilocksField,
    >],
    lg_half_m: usize,
    omega_row: &[crate::extension::quadratic::QuadraticExtension<
        crate::goldilocks_field::GoldilocksField,
    >],
) {
    use core::arch::aarch64::*;
    use crate::arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;
    use crate::extension::quadratic::QuadraticExtension;
    use crate::goldilocks_field::GoldilocksField;

    const EPSILON: u64 = (1 << 32) - 1;
    const W: u64 = 7;
    let half = 1usize << lg_half_m;
    let m = half << 1;
    debug_assert!(omega_row.len() >= half);
    let base_subfield = omega_row[..half]
        .iter()
        .all(|w| w.0[1].0 == 0);
    unsafe {
        let eps = vdupq_n_u64(EPSILON);
        let mut k = 0;
        while k + m <= values.len() {
            let mut j = 0;
            while j + 1 <= half {
                let v = *values.get_unchecked(k + half + j);
                let w = *omega_row.get_unchecked(j);
                let (c0, c1) = if base_subfield {
                    // [w0, 0] * [v0, v1] = [w0*v0, w0*v1]: one paired mul.
                    let p = NeonGoldilocksField([w.0[0], w.0[0]])
                        * NeonGoldilocksField([v.0[0], v.0[1]]);
                    (p.0[0].0, p.0[1].0)
                } else {
                    // General extension product with the W = 7 conjugation.
                    let p = NeonGoldilocksField([w.0[0], w.0[1]])
                        * NeonGoldilocksField([v.0[0], v.0[1]]);
                    let q = NeonGoldilocksField([w.0[0], w.0[1]])
                        * NeonGoldilocksField([v.0[1], v.0[0]]);
                    let c0 = (p.0[0] + GoldilocksField(W) * p.0[1]).0;
                    let c1 = (q.0[0] + q.0[1]).0;
                    (c0, c1)
                };
                let tv = vcombine_u64(vcreate_u64(c0), vcreate_u64(c1));

                let u = *values.get_unchecked(k + j);
                let uv = vcombine_u64(vcreate_u64(u.0[0].0), vcreate_u64(u.0[1].0));
                let sum = gl_add_neon(uv, tv, eps);
                let diff = gl_sub_neon(uv, tv, eps);
                *values.get_unchecked_mut(k + j) = QuadraticExtension([
                    GoldilocksField(vgetq_lane_u64(sum, 0)),
                    GoldilocksField(vgetq_lane_u64(sum, 1)),
                ]);
                *values.get_unchecked_mut(k + half + j) = QuadraticExtension([
                    GoldilocksField(vgetq_lane_u64(diff, 0)),
                    GoldilocksField(vgetq_lane_u64(diff, 1)),
                ]);
                j += 1;
            }
            k += m;
        }
    }
}

#[inline(always)]
fn fft_classic_simd_single_layer_with<P, M>(
    packed_values: &mut [P],
    lg_half_m: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    // Base-field fast path: the reduction runs in vector registers. Guarded on
    // exact type identity, the same way `fft_zero_padded_cache_blocks` guards
    // its rate-8 specialisation. Every other instantiation -- scalar
    // `GoldilocksField`, and `QuadraticExtension` for FRI, both of which are
    // `WIDTH == 1` -- falls through to the generic body below.
    //
    // The twiddle marker `M` is deliberately not referenced by this arm.
    // Reaching it proves `P::Scalar` is exactly `GoldilocksField`, whose
    // `mul_fft_base_twiddle` is the default full multiplication, so both
    // markers denote the same product here. The vector-reduction kernel is
    // therefore untouched, bit for bit, by the twiddle specialization; only the
    // generic body dispatches on `M` -- and that is the body every extension
    // transform takes, since `QuadraticExtension` never matches this `TypeId`.
    #[cfg(target_arch = "aarch64")]
    if lg_half_m >= lg_packed_width
        && core::any::TypeId::of::<P>()
            == core::any::TypeId::of::<
                crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField,
            >()
    {
        // SAFETY: the `TypeId` compare proves `P` is exactly
        // `WideGoldilocksField`, hence `P::Scalar` is exactly
        // `GoldilocksField`. `WideGoldilocksField` is `#[repr(transparent)]`
        // over `[NeonGoldilocksField; 2]`, itself `#[repr(transparent)]` over
        // `[GoldilocksField; 2]`, so the packed slice is exactly `4 * len`
        // contiguous scalars with the same alignment. Only the generic spelling
        // of the types differs at this point.
        let scalars = unsafe {
            core::slice::from_raw_parts_mut(
                packed_values
                    .as_mut_ptr()
                    .cast::<crate::goldilocks_field::GoldilocksField>(),
                packed_values.len() * P::WIDTH,
            )
        };
        let row = &root_table[lg_half_m];
        let omega_row = unsafe {
            core::slice::from_raw_parts(
                row.as_ptr().cast::<crate::goldilocks_field::GoldilocksField>(),
                row.len(),
            )
        };
        fft_classic_simd_single_layer_neon(scalars, lg_half_m, omega_row);
        return;
    }
    // Quadratic-extension (FRI) fast path: the butterfly add/sub reductions
    // run as two-lane vectors over each element's Goldilocks components, with
    // a row-level base-subfield twiddle check (two products instead of four).
    #[cfg(target_arch = "aarch64")]
    if lg_half_m >= lg_packed_width
        && core::any::TypeId::of::<P>()
            == core::any::TypeId::of::<
                crate::extension::quadratic::QuadraticExtension<
                    crate::goldilocks_field::GoldilocksField,
                >,
            >()
    {
        // SAFETY: the `TypeId` compare proves `P` is exactly
        // `QuadraticExtension<GoldilocksField>`, whose `PackedField` `WIDTH`
        // is 1, so `packed_values` is exactly `len` contiguous extension
        // elements with the same alignment (a single `[GoldilocksField; 2]`
        // field, no padding). The omega row is already the correct type.
        let ext_values = unsafe {
            core::slice::from_raw_parts_mut(
                packed_values
                    .as_mut_ptr()
                    .cast::<crate::extension::quadratic::QuadraticExtension<
                        crate::goldilocks_field::GoldilocksField,
                    >>(),
                packed_values.len() * P::WIDTH,
            )
        };
        let omega_row = unsafe {
            let row = &root_table[lg_half_m];
            core::slice::from_raw_parts(
                row.as_ptr()
                    .cast::<crate::extension::quadratic::QuadraticExtension<
                        crate::goldilocks_field::GoldilocksField,
                    >>(),
                row.len(),
            )
        };
        fft_classic_simd_single_layer_neon_ext(ext_values, lg_half_m, omega_row);
        return;
    }

    let lg_m = lg_half_m + 1;
    let m = 1 << lg_m; // Subarray size (in field elements).
    let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
    let half_packed_m = packed_m / 2;
    debug_assert!(half_packed_m != 0);

    // Omega values for this iteration, as a slice of vectors.
    //
    // Indexing this loop by `k + half_packed_m + j` costs a panic branch per
    // access -- ten of the eleven branches in the compiled inner loop -- because
    // the compiler cannot relate those indices to `packed_values.len()`.
    // Walking the same elements through `chunks_exact_mut` and `split_at_mut`
    // discharges the bounds by construction: identical accesses in identical
    // order, so the raw `GoldilocksField.0` words are unchanged, with no
    // `unsafe` and no per-butterfly branch.
    //
    // `omega_table` is truncated first on purpose: `zip` stops at the shortest
    // iterator, so a short twiddle row would silently perform fewer butterflies
    // where the indexed form panicked. Slicing keeps that failure loud and pays
    // one check per layer instead of per butterfly.
    let omega_table = &P::pack_slice(&root_table[lg_half_m])[..half_packed_m];
    for block in packed_values.chunks_exact_mut(packed_m) {
        let (lows, highs) = block.split_at_mut(half_packed_m);
        for ((u, v), &omega) in lows.iter_mut().zip(highs.iter_mut()).zip(omega_table) {
            let t = M::mul(omega, *v);
            let u_value = *u;
            *u = u_value + t;
            *v = u_value - t;
        }
    }
}

/// Two consecutive stages fused into one radix-4-style traversal: the exact
/// same butterflies with the exact same `root_table` twiddles, but each
/// quarter-block element is loaded and stored once per stage *pair* instead
/// of once per stage, halving whole-array memory passes for these layers.
#[inline(always)]
fn fft_classic_simd_fused_two_layers_with<P, M>(
    packed_values: &mut [P],
    lg_half_m: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    // Quarter size in vectors for the size-2^(lg_half_m + 2) fused block.
    let q = (1usize << lg_half_m) >> lg_packed_width;
    debug_assert!(q != 0);
    let stage1_omegas = P::pack_slice(&root_table[lg_half_m]);
    let stage2_omegas = P::pack_slice(&root_table[lg_half_m + 1]);

    for k in (0..packed_values.len()).step_by(4 * q) {
        for j in 0..q {
            let w1 = stage1_omegas[j];
            let a = packed_values[k + j];
            let b = packed_values[k + q + j];
            let c = packed_values[k + 2 * q + j];
            let d = packed_values[k + 3 * q + j];

            // First stage: butterflies within [a,b] and within [c,d].
            let t = M::mul(w1, b);
            let (ab0, ab1) = (a + t, a - t);
            let t = M::mul(w1, d);
            let (cd0, cd1) = (c + t, c - t);

            // Second stage: butterflies pairing positions j and j + 2q.
            let t = M::mul(stage2_omegas[j], cd0);
            packed_values[k + j] = ab0 + t;
            packed_values[k + 2 * q + j] = ab0 - t;
            let t = M::mul(stage2_omegas[q + j], cd1);
            packed_values[k + q + j] = ab1 + t;
            packed_values[k + 3 * q + j] = ab1 - t;
        }
    }
}

/// Two FFT stages evaluated as a true radix-4 butterfly over Goldilocks
/// scalars. If `p = q^2`, the historical stages
///
/// ```text
/// (a, b), (c, d) --p--> ... --q,iq--> outputs
/// ```
///
/// are algebraically identical to three general products
/// `B = q*c`, `C = p*b`, `D = q^3*d`, followed by additions and one
/// multiplication by `i = 2^48`. The latter is a shift plus reduction. The
/// general multiplication count therefore drops from four to three per
/// scalar butterfly. Two adjacent scalar butterflies share each paired NEON
/// multiply and vector add/sub sequence.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fft_classic_simd_true_radix4_neon(
    values: &mut [crate::goldilocks_field::GoldilocksField],
    lg_half_m: usize,
    root_table: &FftRootTable<crate::goldilocks_field::GoldilocksField>,
) {
    use core::arch::aarch64::*;
    use crate::arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;

    const EPSILON: u64 = (1 << 32) - 1;
    let h = 1usize << lg_half_m;
    let block_len = h << 2;
    let twice_h = h << 1;
    let p_row = &root_table[lg_half_m];
    let q_row = &root_table[lg_half_m + 1];
    debug_assert!(p_row.len() >= h);
    debug_assert!(q_row.len() >= twice_h);
    debug_assert_eq!(values.len() % block_len, 0);

    let base = values.as_mut_ptr().cast::<u64>();
    unsafe {
        let eps = vdupq_n_u64(EPSILON);
        let mut k = 0;
        while k < values.len() {
            let mut j = 0;
            while j + 2 <= h {
                let p = NeonGoldilocksField([
                    *p_row.get_unchecked(j),
                    *p_row.get_unchecked(j + 1),
                ]);
                let q = NeonGoldilocksField([
                    *q_row.get_unchecked(j),
                    *q_row.get_unchecked(j + 1),
                ]);
                let q3_at = |lane_j: usize| {
                    let exponent = 3 * lane_j;
                    if exponent < twice_h {
                        *q_row.get_unchecked(exponent)
                    } else {
                        -*q_row.get_unchecked(exponent - twice_h)
                    }
                };
                let q3 = NeonGoldilocksField([q3_at(j), q3_at(j + 1)]);

                let b = NeonGoldilocksField([
                    *values.get_unchecked(k + h + j),
                    *values.get_unchecked(k + h + j + 1),
                ]);
                let c = NeonGoldilocksField([
                    *values.get_unchecked(k + 2 * h + j),
                    *values.get_unchecked(k + 2 * h + j + 1),
                ]);
                let d = NeonGoldilocksField([
                    *values.get_unchecked(k + 3 * h + j),
                    *values.get_unchecked(k + 3 * h + j + 1),
                ]);

                // Exactly three paired general multiplications.
                let b_twiddled = q * c;
                let c_twiddled = p * b;
                let d_twiddled = q3 * d;
                let bv = vcombine_u64(
                    vcreate_u64(b_twiddled.0[0].0),
                    vcreate_u64(b_twiddled.0[1].0),
                );
                let cv = vcombine_u64(
                    vcreate_u64(c_twiddled.0[0].0),
                    vcreate_u64(c_twiddled.0[1].0),
                );
                let dv = vcombine_u64(
                    vcreate_u64(d_twiddled.0[0].0),
                    vcreate_u64(d_twiddled.0[1].0),
                );
                let av = vld1q_u64(base.add(k + j));

                let t0 = gl_add_neon(av, cv, eps);
                let t1 = gl_sub_neon(av, cv, eps);
                let t2 = gl_add_neon(bv, dv, eps);
                let b_minus_d = gl_sub_neon(bv, dv, eps);
                let t3 = vcombine_u64(
                    vcreate_u64(
                        mul_fourth_root(GoldilocksField(vgetq_lane_u64(b_minus_d, 0))).0,
                    ),
                    vcreate_u64(
                        mul_fourth_root(GoldilocksField(vgetq_lane_u64(b_minus_d, 1))).0,
                    ),
                );

                vst1q_u64(base.add(k + j), gl_add_neon(t0, t2, eps));
                vst1q_u64(base.add(k + h + j), gl_add_neon(t1, t3, eps));
                vst1q_u64(base.add(k + 2 * h + j), gl_sub_neon(t0, t2, eps));
                vst1q_u64(base.add(k + 3 * h + j), gl_sub_neon(t1, t3, eps));
                j += 2;
            }

            // Only the scalar `P = GoldilocksField`, `lg_half_m = 0` case has
            // an odd quarter. It is tiny but keeps the exact-type scalar
            // dispatch complete rather than adding a transform-level special
            // case.
            while j < h {
                let exponent = 3 * j;
                let q3 = if exponent < twice_h {
                    q_row[exponent]
                } else {
                    -q_row[exponent - twice_h]
                };
                let a = values[k + j];
                let b = values[k + h + j];
                let c = values[k + 2 * h + j];
                let d = values[k + 3 * h + j];
                let b_twiddled = q_row[j] * c;
                let c_twiddled = p_row[j] * b;
                let d_twiddled = q3 * d;
                let t0 = a + c_twiddled;
                let t1 = a - c_twiddled;
                let t2 = b_twiddled + d_twiddled;
                let t3 = mul_fourth_root(b_twiddled - d_twiddled);
                values[k + j] = t0 + t2;
                values[k + h + j] = t1 + t3;
                values[k + 2 * h + j] = t0 - t2;
                values[k + 3 * h + j] = t1 - t3;
                j += 1;
            }
            k += block_len;
        }
    }
}

/// Production layer driver. Exact AArch64 Goldilocks scalar/packed
/// instantiations use true radix-4 stage pairs; all other field and packing
/// types retain the historical generic driver unchanged. Type identity is
/// tested once per layer range, never in the pair or butterfly loops.
#[inline(always)]
fn fft_classic_simd_optimized_layers<P, M>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    #[cfg(target_arch = "aarch64")]
    {
        use crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
        use crate::goldilocks_field::GoldilocksField;

        let concrete = core::any::TypeId::of::<P>();
        if concrete == core::any::TypeId::of::<GoldilocksField>()
            || concrete == core::any::TypeId::of::<WideGoldilocksField>()
        {
            // SAFETY: either exact-type comparison proves the scalar storage is
            // contiguous Goldilocks words. The packed case is transparent over
            // four Goldilocks scalars; the scalar case has width one.
            let scalars = unsafe {
                core::slice::from_raw_parts_mut(
                    packed_values.as_mut_ptr().cast::<GoldilocksField>(),
                    packed_values.len() * P::WIDTH,
                )
            };
            let roots = unsafe {
                // The exact-type checks above also prove `P::Scalar` is
                // Goldilocks for both accepted instantiations.
                &*(root_table as *const FftRootTable<P::Scalar>
                    as *const FftRootTable<GoldilocksField>)
            };
            let mut layer = start;
            // Leave the most expensive (largest) layers paired when the count
            // is odd by consuming one leading radix-2 layer.
            if (end - start) & 1 == 1 {
                fft_classic_simd_single_layer_neon(scalars, layer, &roots[layer]);
                layer += 1;
            }
            while layer + 1 < end {
                fft_classic_simd_true_radix4_neon(scalars, layer, roots);
                layer += 2;
            }
            return;
        }
    }

    fft_classic_simd_layers::<P, M>(packed_values, start, end, root_table);
}

/// Run FFT stages `start..end`, one whole-buffer pass each.
///
/// A radix-4 traversal fusing stage pairs used to drive these layers, on
/// the reasoning that it halved whole-array memory passes. Memory passes
/// are not what this kernel is short of: removing the 2^13 cache blocking
/// entirely changes the time by under 1% at every thread count from 1 to
/// 12. What the fused form does cost is registers -- four packed values
/// plus three twiddle vectors live, against two and one here, while the
/// Goldilocks multiply is scalar GPR assembly. Measured on both production
/// LDE shapes (2^19 and 2^17, rate 8) with interleaved arms and per-thread
/// minimums, single layers are 3-7% faster at every thread count, and
/// bit-identical: same butterflies, same twiddles, same per-element order,
/// same raw `GoldilocksField.0` words.
///
/// The twiddle marker `M` rides on this structure: it is chosen once per
/// transform by the dispatch entry and resolved statically inside each layer,
/// so it adds no live register and no per-butterfly branch to either the
/// vector-reduction kernel or the generic body.
#[inline(always)]
fn fft_classic_simd_layers<P, M>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    let lg_packed_width = log2_strict(P::WIDTH);
    for lg_half_m in start..end {
        fft_classic_simd_single_layer_with::<P, M>(
            packed_values,
            lg_half_m,
            lg_packed_width,
            root_table,
        );
    }
}

#[inline(always)]
fn fft_zero_padded_first_layer_block_with<P, M>(
    packed_values: &mut [P],
    source_start: usize,
    nonzero_len: usize,
    destination: usize,
    packed_repeat: usize,
    omega_table: &[P],
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    debug_assert!(nonzero_len >= 2);

    // Expanding backwards ensures that every source pair is read before an earlier destination
    // can overwrite it. Destinations are vector-aligned because repeat >= the packed width.
    for pair in (0..nonzero_len / 2).rev() {
        let source = source_start + pair * 2;
        let u = packed_values[source / P::WIDTH].as_slice()[source % P::WIDTH];
        let v = packed_values[(source + 1) / P::WIDTH].as_slice()[(source + 1) % P::WIDTH];
        let u = P::from(u);
        let v = P::from(v);
        let pair_destination = destination + pair * 2 * packed_repeat;

        for j in 0..packed_repeat {
            let t = M::mul(omega_table[j], v);
            packed_values[pair_destination + j] = u + t;
            packed_values[pair_destination + packed_repeat + j] = u - t;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fft_zero_padded_rate_8_first_layer_block(
    packed_values: &mut [WideGoldilocksField],
    source_start: usize,
    nonzero_len: usize,
    destination: usize,
) {
    debug_assert!(nonzero_len >= 2);

    for pair in (0..nonzero_len / 2).rev() {
        let source = source_start + pair * 2;
        let u = packed_values[source / 4].as_slice()[source % 4];
        let v = packed_values[(source + 1) / 4].as_slice()[(source + 1) % 4];
        let products = mul_16th_root_powers(v);
        let low = *WideGoldilocksField::from_slice(&products[..4]);
        let high = *WideGoldilocksField::from_slice(&products[4..]);
        let u = WideGoldilocksField::from(u);
        let pair_destination = destination + pair * 4;

        packed_values[pair_destination] = u + low;
        packed_values[pair_destination + 1] = u + high;
        packed_values[pair_destination + 2] = u - low;
        packed_values[pair_destination + 3] = u - high;
    }
}

/// Expand a bit-reversed nonzero prefix and perform its first nontrivial FFT layer in one pass.
///
/// This is called only when each repeated run contains at least one packed vector.
fn fft_zero_padded_first_layer<P, M>(
    values: &mut [P::Scalar],
    r: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;
    debug_assert!(repeat >= P::WIDTH);

    let packed_repeat = repeat / P::WIDTH;
    let packed_values = P::pack_slice_mut(values);
    let omega_table = P::pack_slice(&root_table[r]);
    fft_zero_padded_first_layer_block_with::<P, M>(
        packed_values,
        0,
        nonzero_len,
        0,
        packed_repeat,
        omega_table,
    );
}

fn fft_zero_padded_cache_blocks<P, M>(
    values: &mut [P::Scalar],
    r: usize,
    lg_block_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) where
    P: PackedField,
    M: FftTwiddleMul<P>,
{
    let repeat = 1 << r;
    let block_len = 1 << lg_block_n;
    let nonzero_per_block = block_len >> r;
    let packed_repeat = repeat / P::WIDTH;
    let packed_block_len = block_len / P::WIDTH;
    let num_blocks = values.len() / block_len;
    let packed_values = P::pack_slice_mut(values);
    let omega_table = P::pack_slice(&root_table[r]);

    // Expand blocks from the end of the buffer so their output cannot clobber unread prefix
    // coefficients. Complete every block-local layer immediately while the block is still hot.
    for block in (0..num_blocks).rev() {
        let source_start = block * nonzero_per_block;
        let destination = block * packed_block_len;
        #[cfg(target_arch = "aarch64")]
        if r == 3 && core::any::TypeId::of::<P>() == core::any::TypeId::of::<WideGoldilocksField>()
        {
            let wide_values = unsafe {
                // SAFETY: The TypeId check proves this is the exact concrete packed type;
                // only the generic spelling of the slice differs at this point.
                core::slice::from_raw_parts_mut(
                    packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                    packed_values.len(),
                )
            };
            fft_zero_padded_rate_8_first_layer_block(
                wide_values,
                source_start,
                nonzero_per_block,
                destination,
            );
        } else {
            fft_zero_padded_first_layer_block_with::<P, M>(
                packed_values,
                source_start,
                nonzero_per_block,
                destination,
                packed_repeat,
                omega_table,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        fft_zero_padded_first_layer_block_with::<P, M>(
            packed_values,
            source_start,
            nonzero_per_block,
            destination,
            packed_repeat,
            omega_table,
        );
        fft_classic_simd_optimized_layers::<P, M>(
            &mut packed_values[destination..destination + packed_block_len],
            r + 1,
            lg_block_n,
            root_table,
        );
    }
}

#[inline(never)]
fn prepare_zero_padded_fft<F, M>(
    values: &mut [F],
    r: usize,
    lg_n: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<F>,
) -> usize
where
    F: Field,
    M: FftTwiddleMul<<F as Packable>::Packing>,
{
    debug_assert!(r > 0 && r <= lg_n);

    // A zero-padded input only has n/2^r live coefficients. Bit-reversing the full buffer would
    // place those coefficients at multiples of 2^r, after which the skipped FFT layers copy each
    // one across its following 2^r-element run. Produce that exact state directly by reversing
    // just the live prefix.
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;
    reverse_index_bits_in_place(&mut values[..nonzero_len]);

    if r >= lg_packed_width && r < lg_n {
        // Keep values plus the largest local twiddle row within Apple Silicon's 128 KiB L1D.
        // Both 2^13 base-field and 2^12 quadratic-extension blocks use about 96 KiB.
        let lg_block_n = match core::mem::size_of::<F>() {
            0..=8 => 13,
            9..=16 => 12,
            _ => 11,
        };
        if r + 1 < lg_block_n && lg_block_n <= lg_n {
            fft_zero_padded_cache_blocks::<<F as Packable>::Packing, M>(
                values, r, lg_block_n, root_table,
            );
            lg_block_n
        } else {
            // Fuse the expansion with the first nontrivial layer, eliminating one full-buffer
            // write/read cycle while retaining the existing skipped-layer semantics.
            fft_zero_padded_first_layer::<<F as Packable>::Packing, M>(values, r, root_table);
            r + 1
        }
    } else {
        for i in (0..nonzero_len).rev() {
            let value = values[i];
            values[i * repeat..(i + 1) * repeat].fill(value);
        }
        r
    }
}

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
#[inline(always)]
fn fft_classic_with<F, M>(values: &mut [F], r: usize, root_table: &FftRootTable<F>)
where
    F: Field,
    M: FftTwiddleMul<F> + FftTwiddleMul<<F as Packable>::Packing>,
{
    let n = values.len();
    let lg_n = log2_strict(n);
    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    let first_layer = if r == 0 {
        reverse_index_bits_in_place(values);
        0
    } else {
        prepare_zero_padded_fft::<F, M>(values, r, lg_n, lg_packed_width, root_table)
    };

    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd_with::<F, M>(values, first_layer, lg_n, root_table);
    } else {
        fft_classic_simd_with::<<F as Packable>::Packing, M>(values, first_layer, lg_n, root_table);
    }
}

pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    let lg_n = log2_strict(values.len());
    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    if lg_n == F::TWO_ADICITY {
        // The final quadratic-extension root is not in its base field. Keep
        // the full multiplication fallback for the entire transform. This
        // branch is once per FFT, never once per butterfly.
        fft_classic_with::<F, GeneralTwiddle>(values, r, root_table);
    } else {
        fft_classic_with::<F, BaseSubfieldTwiddle>(values, r, root_table);
    }
}

// =====================================================================
// LAB VARIANTS — lab-zero-pad-fft branch, kernel-rewrite experiments.
// Each variant is a separate function (no flags). All are exercised by
// the local-only bench target `benches/lab_zero_pad_fft.rs`; only
// variants proven bit-identical and faster get wired into the
// production path above.
// =====================================================================
pub mod lab {
    use plonky2_util::{log2_strict, reverse_index_bits_in_place};

    #[cfg(target_arch = "aarch64")]
    use crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
    #[cfg(target_arch = "aarch64")]
    use crate::goldilocks_field::mul_16th_root_powers;

    // The lab variants are inactive experiments; they keep the pre-existing
    // general twiddle multiplication so their behaviour is unchanged by the
    // base-subfield twiddle specialization used on the production path. Some
    // of them fold a coset shift into the root table, which is not guaranteed
    // to stay in a base subfield, so `GeneralTwiddle` is also the only sound
    // marker for them.
    use super::{
        FftRootTable, GeneralTwiddle, fft_classic, fft_classic_simd_fused_two_layers_with,
        fft_classic_simd_layers, fft_classic_simd_single_layer_with, fft_classic_simd_with,
        fft_root_table, fft_zero_padded_first_layer_block_with,
    };
    #[cfg(target_arch = "aarch64")]
    use super::fft_zero_padded_rate_8_first_layer_block;
    use crate::packable::Packable;
    use crate::packed::PackedField;
    use crate::types::Field;

    /// Baseline entry point: the exact production `fft_classic`, re-exported
    /// so the lab bench can call all arms through the same kind of shim.
    pub fn fft_classic_baseline<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        fft_classic(values, r, root_table);
    }

    /// Production `lg_block_n` selection, kept in sync with
    /// `prepare_zero_padded_fft`.
    fn production_lg_block_n<F: Field>() -> usize {
        match core::mem::size_of::<F>() {
            0..=8 => 13,
            9..=16 => 12,
            _ => 11,
        }
    }

    // -----------------------------------------------------------------
    // Variant A1: fuse the zero-run expansion with the first TWO
    // nontrivial layers (r and r+1) in a single in-register pass,
    // deleting one full read+write pass over each cache block.
    // -----------------------------------------------------------------

    /// Generic fused expansion + layer `r` + layer `r+1` for one block.
    /// Requires `nonzero_len % 4 == 0` and `packed_repeat >= 1`.
    #[inline(always)]
    fn fused_expand_two_layers_block<P: PackedField>(
        packed_values: &mut [P],
        source_start: usize,
        nonzero_len: usize,
        destination: usize,
        packed_repeat: usize,
        omega_r: &[P],
        omega_r1: &[P],
    ) {
        debug_assert!(nonzero_len >= 4 && nonzero_len % 4 == 0);
        let pr = packed_repeat;
        for quad in (0..nonzero_len / 4).rev() {
            let source = source_start + quad * 4;
            let s0 = packed_values[source / P::WIDTH].as_slice()[source % P::WIDTH];
            let s1 = packed_values[(source + 1) / P::WIDTH].as_slice()[(source + 1) % P::WIDTH];
            let s2 = packed_values[(source + 2) / P::WIDTH].as_slice()[(source + 2) % P::WIDTH];
            let s3 = packed_values[(source + 3) / P::WIDTH].as_slice()[(source + 3) % P::WIDTH];
            let u0 = P::from(s0);
            let v0 = P::from(s1);
            let u1 = P::from(s2);
            let v1 = P::from(s3);
            let dest = destination + quad * 4 * pr;
            for j in 0..pr {
                // Layer r butterflies (exactly the expansion pass's math).
                let t = omega_r[j] * v0;
                let a0 = u0 + t; // would-be store at dest + j
                let a1 = u0 - t; // would-be store at dest + pr + j
                let t = omega_r[j] * v1;
                let c0 = u1 + t; // would-be store at dest + 2*pr + j
                let c1 = u1 - t; // would-be store at dest + 3*pr + j
                // Layer r+1 butterflies on the in-register intermediates,
                // with the identical twiddles the unfused single-layer pass
                // would have used.
                let t = omega_r1[j] * c0;
                packed_values[dest + j] = a0 + t;
                packed_values[dest + 2 * pr + j] = a0 - t;
                let t = omega_r1[pr + j] * c1;
                packed_values[dest + pr + j] = a1 + t;
                packed_values[dest + 3 * pr + j] = a1 - t;
            }
        }
    }

    /// aarch64 rate-8 specialization of the fused expand+two-layers block:
    /// stage r=3 twiddles via the cheap 16th-root shift multiply, stage 4
    /// with the packed `root_table[4]` row.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn fused_rate_8_expand_two_layers_block(
        packed_values: &mut [WideGoldilocksField],
        source_start: usize,
        nonzero_len: usize,
        destination: usize,
        omega4: &[WideGoldilocksField],
    ) {
        debug_assert!(nonzero_len >= 4 && nonzero_len % 4 == 0);
        debug_assert!(omega4.len() >= 4);
        for quad in (0..nonzero_len / 4).rev() {
            let source = source_start + quad * 4;
            let s0 = packed_values[source / 4].as_slice()[source % 4];
            let s1 = packed_values[(source + 1) / 4].as_slice()[(source + 1) % 4];
            let s2 = packed_values[(source + 2) / 4].as_slice()[(source + 2) % 4];
            let s3 = packed_values[(source + 3) / 4].as_slice()[(source + 3) % 4];

            let p0 = mul_16th_root_powers(s1);
            let low0 = *WideGoldilocksField::from_slice(&p0[..4]);
            let high0 = *WideGoldilocksField::from_slice(&p0[4..]);
            let p1 = mul_16th_root_powers(s3);
            let low1 = *WideGoldilocksField::from_slice(&p1[..4]);
            let high1 = *WideGoldilocksField::from_slice(&p1[4..]);
            let u0 = WideGoldilocksField::from(s0);
            let u1 = WideGoldilocksField::from(s2);

            let a = [u0 + low0, u0 + high0, u0 - low0, u0 - high0];
            let c = [u1 + low1, u1 + high1, u1 - low1, u1 - high1];
            let dest = destination + quad * 8;
            for k in 0..4 {
                let t = omega4[k] * c[k];
                packed_values[dest + k] = a[k] + t;
                packed_values[dest + 4 + k] = a[k] - t;
            }
        }
    }

    /// Variant-A1 cache-blocked expansion: fused expand+2 layers, then the
    /// production pair-fused layer schedule for the remaining block layers.
    fn cache_blocks_a1<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_block_n: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let repeat = 1 << r;
        let block_len = 1 << lg_block_n;
        let nonzero_per_block = block_len >> r;
        let packed_repeat = repeat / P::WIDTH;
        let packed_block_len = block_len / P::WIDTH;
        let num_blocks = values.len() / block_len;
        let packed_values = P::pack_slice_mut(values);
        let omega_r = P::pack_slice(&root_table[r]);
        let omega_r1 = P::pack_slice(&root_table[r + 1]);

        for block in (0..num_blocks).rev() {
            let source_start = block * nonzero_per_block;
            let destination = block * packed_block_len;
            #[cfg(target_arch = "aarch64")]
            let specialized = r == 3
                && core::any::TypeId::of::<P>() == core::any::TypeId::of::<WideGoldilocksField>();
            #[cfg(not(target_arch = "aarch64"))]
            let specialized = false;
            if specialized {
                #[cfg(target_arch = "aarch64")]
                {
                    let wide_values = unsafe {
                        core::slice::from_raw_parts_mut(
                            packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                            packed_values.len(),
                        )
                    };
                    let wide_omega4 = unsafe {
                        core::slice::from_raw_parts(
                            omega_r1.as_ptr().cast::<WideGoldilocksField>(),
                            omega_r1.len(),
                        )
                    };
                    fused_rate_8_expand_two_layers_block(
                        wide_values,
                        source_start,
                        nonzero_per_block,
                        destination,
                        wide_omega4,
                    );
                }
            } else {
                fused_expand_two_layers_block(
                    packed_values,
                    source_start,
                    nonzero_per_block,
                    destination,
                    packed_repeat,
                    omega_r,
                    omega_r1,
                );
            }
            fft_classic_simd_layers::<_, GeneralTwiddle>(
                &mut packed_values[destination..destination + packed_block_len],
                r + 2,
                lg_block_n,
                root_table,
            );
        }
    }

    /// Variant A1 entry point: production structure with the fused
    /// expand+two-layers block. Falls back to the untouched production path
    /// when the shape is outside the fused kernel's domain.
    pub fn fft_classic_a1<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>();
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        if !eligible {
            fft_classic(values, r, root_table);
            return;
        }
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        cache_blocks_a1::<<F as Packable>::Packing>(values, r, lg_block_n, root_table);
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            root_table,
        );
    }

    // -----------------------------------------------------------------
    // Variant A2: three consecutive stages fused into one radix-8-style
    // traversal, cutting whole-array passes for the post-block layers
    // (and optionally in-block layers) from ceil(k/2) to ~k/3.
    // -----------------------------------------------------------------

    /// Three consecutive stages fused: identical butterflies with identical
    /// `root_table` twiddles, each element loaded/stored once per stage
    /// TRIPLE. Requires `lg_half_m >= lg_packed_width`.
    #[inline(always)]
    fn fused_three_layers<P: PackedField>(
        packed_values: &mut [P],
        lg_half_m: usize,
        lg_packed_width: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let q = (1usize << lg_half_m) >> lg_packed_width;
        debug_assert!(q != 0);
        let w1 = P::pack_slice(&root_table[lg_half_m]);
        let w2 = P::pack_slice(&root_table[lg_half_m + 1]);
        let w3 = P::pack_slice(&root_table[lg_half_m + 2]);

        for k in (0..packed_values.len()).step_by(8 * q) {
            for j in 0..q {
                let x0 = packed_values[k + j];
                let x1 = packed_values[k + q + j];
                let x2 = packed_values[k + 2 * q + j];
                let x3 = packed_values[k + 3 * q + j];
                let x4 = packed_values[k + 4 * q + j];
                let x5 = packed_values[k + 5 * q + j];
                let x6 = packed_values[k + 6 * q + j];
                let x7 = packed_values[k + 7 * q + j];

                // Stage 1: pairs (0,1) (2,3) (4,5) (6,7), twiddle w1[j].
                let w1j = w1[j];
                let t = w1j * x1;
                let y0 = x0 + t;
                let y1 = x0 - t;
                let t = w1j * x3;
                let y2 = x2 + t;
                let y3 = x2 - t;
                let t = w1j * x5;
                let y4 = x4 + t;
                let y5 = x4 - t;
                let t = w1j * x7;
                let y6 = x6 + t;
                let y7 = x6 - t;

                // Stage 2: pairs (0,2) (1,3) (4,6) (5,7), twiddles w2[j], w2[q+j].
                let w2a = w2[j];
                let w2b = w2[q + j];
                let t = w2a * y2;
                let z0 = y0 + t;
                let z2 = y0 - t;
                let t = w2b * y3;
                let z1 = y1 + t;
                let z3 = y1 - t;
                let t = w2a * y6;
                let z4 = y4 + t;
                let z6 = y4 - t;
                let t = w2b * y7;
                let z5 = y5 + t;
                let z7 = y5 - t;

                // Stage 3: pairs (i, i+4), twiddles w3[i*q + j].
                let t = w3[j] * z4;
                packed_values[k + j] = z0 + t;
                packed_values[k + 4 * q + j] = z0 - t;
                let t = w3[q + j] * z5;
                packed_values[k + q + j] = z1 + t;
                packed_values[k + 5 * q + j] = z1 - t;
                let t = w3[2 * q + j] * z6;
                packed_values[k + 2 * q + j] = z2 + t;
                packed_values[k + 6 * q + j] = z2 - t;
                let t = w3[3 * q + j] * z7;
                packed_values[k + 3 * q + j] = z3 + t;
                packed_values[k + 7 * q + j] = z3 - t;
            }
        }
    }

    /// Variant-A2 layer scheduler: prefer triples, then pairs, avoiding
    /// single-layer passes wherever the count allows. Applies the exact same
    /// butterflies in the exact same layer order as the production scheduler.
    fn simd_layers_v2<P: PackedField>(
        packed_values: &mut [P],
        start: usize,
        end: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let lg_packed_width = log2_strict(P::WIDTH);
        let mut l = start;
        let count = end - start;
        match count % 3 {
            1 if count >= 4 => {
                fft_classic_simd_fused_two_layers_with::<_, GeneralTwiddle>(
                    packed_values,
                    l,
                    lg_packed_width,
                    root_table,
                );
                l += 2;
                fft_classic_simd_fused_two_layers_with::<_, GeneralTwiddle>(
                    packed_values,
                    l,
                    lg_packed_width,
                    root_table,
                );
                l += 2;
            }
            1 => {
                fft_classic_simd_single_layer_with::<_, GeneralTwiddle>(
                    packed_values,
                    l,
                    lg_packed_width,
                    root_table,
                );
                l += 1;
            }
            2 => {
                fft_classic_simd_fused_two_layers_with::<_, GeneralTwiddle>(
                    packed_values,
                    l,
                    lg_packed_width,
                    root_table,
                );
                l += 2;
            }
            _ => {}
        }
        while l < end {
            fused_three_layers(packed_values, l, lg_packed_width, root_table);
            l += 3;
        }
    }

    /// Variant A2 entry point: production expansion/blocking, triple-fused
    /// schedule for both the in-block layers and the post-block layers.
    pub fn fft_classic_a2<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>();
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        if !eligible {
            fft_classic(values, r, root_table);
            return;
        }
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        cache_blocks_v2::<<F as Packable>::Packing>(values, r, lg_block_n, root_table, false);
        let packed_values = <F as Packable>::Packing::pack_slice_mut(values);
        simd_layers_v2(packed_values, lg_block_n, lg_n, root_table);
    }

    /// Cache-blocked expansion shared by A2/A12: `fused_expand` selects the
    /// production single-layer expansion (false) or the A1 fused
    /// expand+two-layers kernel (true); remaining block layers run under the
    /// triple-fused schedule.
    fn cache_blocks_v2<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_block_n: usize,
        root_table: &FftRootTable<P::Scalar>,
        fused_expand: bool,
    ) {
        let repeat = 1 << r;
        let block_len = 1 << lg_block_n;
        let nonzero_per_block = block_len >> r;
        let packed_repeat = repeat / P::WIDTH;
        let packed_block_len = block_len / P::WIDTH;
        let num_blocks = values.len() / block_len;
        let packed_values = P::pack_slice_mut(values);
        let omega_r = P::pack_slice(&root_table[r]);
        let omega_r1 = P::pack_slice(&root_table[r + 1]);

        for block in (0..num_blocks).rev() {
            let source_start = block * nonzero_per_block;
            let destination = block * packed_block_len;
            #[cfg(target_arch = "aarch64")]
            let specialized = r == 3
                && core::any::TypeId::of::<P>() == core::any::TypeId::of::<WideGoldilocksField>();
            #[cfg(not(target_arch = "aarch64"))]
            let specialized = false;
            let first_remaining = if fused_expand {
                if specialized {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let wide_values = unsafe {
                            core::slice::from_raw_parts_mut(
                                packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                                packed_values.len(),
                            )
                        };
                        let wide_omega4 = unsafe {
                            core::slice::from_raw_parts(
                                omega_r1.as_ptr().cast::<WideGoldilocksField>(),
                                omega_r1.len(),
                            )
                        };
                        fused_rate_8_expand_two_layers_block(
                            wide_values,
                            source_start,
                            nonzero_per_block,
                            destination,
                            wide_omega4,
                        );
                    }
                } else {
                    fused_expand_two_layers_block(
                        packed_values,
                        source_start,
                        nonzero_per_block,
                        destination,
                        packed_repeat,
                        omega_r,
                        omega_r1,
                    );
                }
                r + 2
            } else {
                if specialized {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let wide_values = unsafe {
                            core::slice::from_raw_parts_mut(
                                packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                                packed_values.len(),
                            )
                        };
                        fft_zero_padded_rate_8_first_layer_block(
                            wide_values,
                            source_start,
                            nonzero_per_block,
                            destination,
                        );
                    }
                } else {
                    fft_zero_padded_first_layer_block_with::<_, GeneralTwiddle>(
                        packed_values,
                        source_start,
                        nonzero_per_block,
                        destination,
                        packed_repeat,
                        omega_r,
                    );
                }
                r + 1
            };
            simd_layers_v2(
                &mut packed_values[destination..destination + packed_block_len],
                first_remaining,
                lg_block_n,
                root_table,
            );
        }
    }

    /// Variant A12 entry point: A1's fused expand+two-layers block plus A2's
    /// triple-fused schedule for all remaining layers.
    pub fn fft_classic_a12<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>();
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        if !eligible {
            fft_classic(values, r, root_table);
            return;
        }
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        cache_blocks_v2::<<F as Packable>::Packing>(values, r, lg_block_n, root_table, true);
        let packed_values = <F as Packable>::Packing::pack_slice_mut(values);
        simd_layers_v2(packed_values, lg_block_n, lg_n, root_table);
    }

    // -----------------------------------------------------------------
    // Variant C: block-size retune. Identical mechanics to production,
    // `lg_block_n` supplied by the caller.
    // -----------------------------------------------------------------

    /// Production zero-padded FFT with a caller-chosen cache-block size.
    pub fn fft_classic_block_size<F: Field>(
        values: &mut [F],
        r: usize,
        root_table: &FftRootTable<F>,
        lg_block_n: usize,
    ) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        if !eligible {
            fft_classic(values, r, root_table);
            return;
        }
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        super::fft_zero_padded_cache_blocks::<<F as Packable>::Packing, GeneralTwiddle>(
            values, r, lg_block_n, root_table,
        );
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            root_table,
        );
    }

    // -----------------------------------------------------------------
    // Variant B: coset-folded twiddles. Row `l` of the root table is
    // premultiplied by shift^(2^(lg_n-1-l)); running the standard
    // zero-padded FFT with this table on RAW coefficients computes the
    // coset FFT directly, deleting the whole coset-power multiply pass.
    // Exactly equivalent (field arithmetic is exact); raw representation
    // may differ where multiplication order differs.
    // -----------------------------------------------------------------

    /// Build the coset-folded root table for size `n` and shift `shift`.
    pub fn coset_folded_root_table<F: Field>(n: usize, shift: F) -> FftRootTable<F> {
        let lg_n = log2_strict(n);
        let mut table = fft_root_table::<F>(n);
        for (l, row) in table.iter_mut().enumerate() {
            let s = shift.exp_power_of_2(lg_n - 1 - l);
            for x in row.iter_mut() {
                *x = s * *x;
            }
        }
        table
    }

    /// aarch64 rate-8 expansion block under coset folding: the folded row-3
    /// twiddles are sigma * omega16^j, and `mul_16th_root_powers` is linear,
    /// so premultiplying v by sigma keeps the cheap shift-multiply kernel.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn rate_8_first_layer_block_sigma(
        packed_values: &mut [WideGoldilocksField],
        source_start: usize,
        nonzero_len: usize,
        destination: usize,
        sigma: crate::goldilocks_field::GoldilocksField,
    ) {
        debug_assert!(nonzero_len >= 2);
        for pair in (0..nonzero_len / 2).rev() {
            let source = source_start + pair * 2;
            let u = packed_values[source / 4].as_slice()[source % 4];
            let v = packed_values[(source + 1) / 4].as_slice()[(source + 1) % 4];
            let products = mul_16th_root_powers(sigma * v);
            let low = *WideGoldilocksField::from_slice(&products[..4]);
            let high = *WideGoldilocksField::from_slice(&products[4..]);
            let u = WideGoldilocksField::from(u);
            let pair_destination = destination + pair * 4;

            packed_values[pair_destination] = u + low;
            packed_values[pair_destination + 1] = u + high;
            packed_values[pair_destination + 2] = u - low;
            packed_values[pair_destination + 3] = u - high;
        }
    }

    /// Variant-B cache-blocked expansion: production structure, but the
    /// aarch64 rate-8 fast path premultiplies by sigma (the folded row-r
    /// shift factor) instead of reading the (already folded) row.
    fn cache_blocks_coset<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_block_n: usize,
        folded_table: &FftRootTable<P::Scalar>,
        sigma: P::Scalar,
    ) {
        let repeat = 1 << r;
        let block_len = 1 << lg_block_n;
        let nonzero_per_block = block_len >> r;
        let packed_repeat = repeat / P::WIDTH;
        let packed_block_len = block_len / P::WIDTH;
        let num_blocks = values.len() / block_len;
        let packed_values = P::pack_slice_mut(values);
        let omega_table = P::pack_slice(&folded_table[r]);

        for block in (0..num_blocks).rev() {
            let source_start = block * nonzero_per_block;
            let destination = block * packed_block_len;
            #[cfg(target_arch = "aarch64")]
            let specialized = r == 3
                && core::any::TypeId::of::<P>() == core::any::TypeId::of::<WideGoldilocksField>();
            #[cfg(not(target_arch = "aarch64"))]
            let specialized = false;
            if specialized {
                #[cfg(target_arch = "aarch64")]
                {
                    let wide_values = unsafe {
                        core::slice::from_raw_parts_mut(
                            packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                            packed_values.len(),
                        )
                    };
                    let sigma_gl = unsafe {
                        *(&sigma as *const P::Scalar)
                            .cast::<crate::goldilocks_field::GoldilocksField>()
                    };
                    rate_8_first_layer_block_sigma(
                        wide_values,
                        source_start,
                        nonzero_per_block,
                        destination,
                        sigma_gl,
                    );
                }
            } else {
                fft_zero_padded_first_layer_block_with::<_, GeneralTwiddle>(
                    packed_values,
                    source_start,
                    nonzero_per_block,
                    destination,
                    packed_repeat,
                    omega_table,
                );
            }
            fft_classic_simd_layers::<_, GeneralTwiddle>(
                &mut packed_values[destination..destination + packed_block_len],
                r + 1,
                lg_block_n,
                folded_table,
            );
        }
    }

    /// Variant B entry point: zero-padded COSET FFT of raw (unscaled)
    /// coefficients using a coset-folded root table. Replaces
    /// `batch_multiply(prefix, coset_powers); fft_classic(...)`.
    pub fn fft_classic_coset_folded<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
    ) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>();
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        assert!(
            eligible,
            "coset-folded lab variant only covers the production cache-block shapes"
        );
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        let sigma = shift.exp_power_of_2(lg_n - 1 - r);
        cache_blocks_coset::<<F as Packable>::Packing>(values, r, lg_block_n, folded_table, sigma);
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            folded_table,
        );
    }

    /// Variant B+A12 entry point: coset folding plus the fused kernels.
    /// The fused expand+two-layers block reads both folded rows directly
    /// (the 16th-root shift trick does not apply to folded row 4, whose
    /// entries are sigma4 * omega32^j with sigma4 a general element — but
    /// row 4 was always a general packed multiply, so nothing is lost).
    pub fn fft_classic_coset_folded_a12<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        _shift: F,
    ) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>();
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        assert!(
            eligible,
            "coset-folded lab variant only covers the production cache-block shapes"
        );
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        cache_blocks_v2_generic_expand::<<F as Packable>::Packing>(
            values,
            r,
            lg_block_n,
            folded_table,
        );
        let packed_values = <F as Packable>::Packing::pack_slice_mut(values);
        simd_layers_v2(packed_values, lg_block_n, lg_n, folded_table);
    }

    // -----------------------------------------------------------------
    // Combo variants: compose the three individually-positive mechanisms
    // (B coset folding, A1 fused expand+2, block-size retune) without the
    // negative triple-layer fusion.
    // -----------------------------------------------------------------

    /// aarch64 rate-8 fused expand+two-layers with the folded row-r shift
    /// factor premultiplied into v (keeps the cheap 16th-root kernel), and
    /// the folded row r+1 for the second stage.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn fused_rate_8_expand_two_layers_block_sigma(
        packed_values: &mut [WideGoldilocksField],
        source_start: usize,
        nonzero_len: usize,
        destination: usize,
        omega4: &[WideGoldilocksField],
        sigma: crate::goldilocks_field::GoldilocksField,
    ) {
        debug_assert!(nonzero_len >= 4 && nonzero_len % 4 == 0);
        debug_assert!(omega4.len() >= 4);
        for quad in (0..nonzero_len / 4).rev() {
            let source = source_start + quad * 4;
            let s0 = packed_values[source / 4].as_slice()[source % 4];
            let s1 = packed_values[(source + 1) / 4].as_slice()[(source + 1) % 4];
            let s2 = packed_values[(source + 2) / 4].as_slice()[(source + 2) % 4];
            let s3 = packed_values[(source + 3) / 4].as_slice()[(source + 3) % 4];

            let p0 = mul_16th_root_powers(sigma * s1);
            let low0 = *WideGoldilocksField::from_slice(&p0[..4]);
            let high0 = *WideGoldilocksField::from_slice(&p0[4..]);
            let p1 = mul_16th_root_powers(sigma * s3);
            let low1 = *WideGoldilocksField::from_slice(&p1[..4]);
            let high1 = *WideGoldilocksField::from_slice(&p1[4..]);
            let u0 = WideGoldilocksField::from(s0);
            let u1 = WideGoldilocksField::from(s2);

            let a = [u0 + low0, u0 + high0, u0 - low0, u0 - high0];
            let c = [u1 + low1, u1 + high1, u1 - low1, u1 - high1];
            let dest = destination + quad * 8;
            for k in 0..4 {
                let t = omega4[k] * c[k];
                packed_values[dest + k] = a[k] + t;
                packed_values[dest + 4 + k] = a[k] - t;
            }
        }
    }

    /// Folded-table cache blocks with the A1 fused expand+two-layers kernel
    /// and the production pair-fused schedule for the remaining layers.
    fn cache_blocks_folded_a1<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_block_n: usize,
        folded_table: &FftRootTable<P::Scalar>,
        sigma: P::Scalar,
    ) {
        let repeat = 1 << r;
        let block_len = 1 << lg_block_n;
        let nonzero_per_block = block_len >> r;
        let packed_repeat = repeat / P::WIDTH;
        let packed_block_len = block_len / P::WIDTH;
        let num_blocks = values.len() / block_len;
        let packed_values = P::pack_slice_mut(values);
        let omega_r = P::pack_slice(&folded_table[r]);
        let omega_r1 = P::pack_slice(&folded_table[r + 1]);

        for block in (0..num_blocks).rev() {
            let source_start = block * nonzero_per_block;
            let destination = block * packed_block_len;
            #[cfg(target_arch = "aarch64")]
            let specialized = r == 3
                && core::any::TypeId::of::<P>() == core::any::TypeId::of::<WideGoldilocksField>();
            #[cfg(not(target_arch = "aarch64"))]
            let specialized = false;
            if specialized {
                #[cfg(target_arch = "aarch64")]
                {
                    let wide_values = unsafe {
                        core::slice::from_raw_parts_mut(
                            packed_values.as_mut_ptr().cast::<WideGoldilocksField>(),
                            packed_values.len(),
                        )
                    };
                    let wide_omega4 = unsafe {
                        core::slice::from_raw_parts(
                            omega_r1.as_ptr().cast::<WideGoldilocksField>(),
                            omega_r1.len(),
                        )
                    };
                    let sigma_gl = unsafe {
                        *(&sigma as *const P::Scalar)
                            .cast::<crate::goldilocks_field::GoldilocksField>()
                    };
                    fused_rate_8_expand_two_layers_block_sigma(
                        wide_values,
                        source_start,
                        nonzero_per_block,
                        destination,
                        wide_omega4,
                        sigma_gl,
                    );
                }
            } else {
                fused_expand_two_layers_block(
                    packed_values,
                    source_start,
                    nonzero_per_block,
                    destination,
                    packed_repeat,
                    omega_r,
                    omega_r1,
                );
            }
            fft_classic_simd_layers::<_, GeneralTwiddle>(
                &mut packed_values[destination..destination + packed_block_len],
                r + 2,
                lg_block_n,
                folded_table,
            );
        }
    }

    /// Shared shell for the folded+A1 combos: `lg_block_n` chosen by caller.
    fn coset_a1_shell<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
        lg_block_n: usize,
    ) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        assert!(
            eligible,
            "coset-folded lab combo only covers the production cache-block shapes"
        );
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        let sigma = shift.exp_power_of_2(lg_n - 1 - r);
        cache_blocks_folded_a1::<<F as Packable>::Packing>(values, r, lg_block_n, folded_table, sigma);
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            folded_table,
        );
    }

    /// Combo: coset folding + A1 fused expansion, production block size.
    pub fn fft_classic_coset_a1<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
    ) {
        coset_a1_shell(values, r, folded_table, shift, production_lg_block_n::<F>());
    }

    /// Combo: coset folding + A1 fused expansion, block size production-1.
    pub fn fft_classic_coset_a1_blkm1<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
    ) {
        coset_a1_shell(values, r, folded_table, shift, production_lg_block_n::<F>() - 1);
    }

    /// Combo: coset folding + A1 fused expansion, block size production+1.
    pub fn fft_classic_coset_a1_blkp1<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
    ) {
        coset_a1_shell(values, r, folded_table, shift, production_lg_block_n::<F>() + 1);
    }

    /// Combo: A1 fused expansion + block size production-1, NO folding
    /// (standard table, caller does the coset-power multiply pass).
    pub fn fft_classic_a1_blkm1<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>() - 1;
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        if !eligible {
            fft_classic(values, r, root_table);
            return;
        }
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        cache_blocks_a1::<<F as Packable>::Packing>(values, r, lg_block_n, root_table);
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            root_table,
        );
    }

    /// Combo: coset folding + production expansion (rate-8 sigma trick),
    /// block size production+1 (the ext field's best block in run 1).
    pub fn fft_classic_coset_blkp1<F: Field>(
        values: &mut [F],
        r: usize,
        folded_table: &FftRootTable<F>,
        shift: F,
    ) {
        let n = values.len();
        let lg_n = log2_strict(n);
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        let lg_block_n = production_lg_block_n::<F>() + 1;
        let eligible = r > 0
            && r >= lg_packed_width
            && r < lg_n
            && r + 1 < lg_block_n
            && lg_block_n <= lg_n;
        assert!(
            eligible,
            "coset-folded lab combo only covers the production cache-block shapes"
        );
        let nonzero_len = n >> r;
        reverse_index_bits_in_place(&mut values[..nonzero_len]);
        let sigma = shift.exp_power_of_2(lg_n - 1 - r);
        cache_blocks_coset::<<F as Packable>::Packing>(values, r, lg_block_n, folded_table, sigma);
        fft_classic_simd_with::<<F as Packable>::Packing, GeneralTwiddle>(
            values,
            lg_block_n,
            lg_n,
            folded_table,
        );
    }

    /// A12-style cache blocks that always use the GENERIC fused expansion
    /// (reading twiddle rows from the table), usable with folded tables.
    fn cache_blocks_v2_generic_expand<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_block_n: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let repeat = 1 << r;
        let block_len = 1 << lg_block_n;
        let nonzero_per_block = block_len >> r;
        let packed_repeat = repeat / P::WIDTH;
        let packed_block_len = block_len / P::WIDTH;
        let num_blocks = values.len() / block_len;
        let packed_values = P::pack_slice_mut(values);
        let omega_r = P::pack_slice(&root_table[r]);
        let omega_r1 = P::pack_slice(&root_table[r + 1]);

        for block in (0..num_blocks).rev() {
            let source_start = block * nonzero_per_block;
            let destination = block * packed_block_len;
            fused_expand_two_layers_block(
                packed_values,
                source_start,
                nonzero_per_block,
                destination,
                packed_repeat,
                omega_r,
                omega_r1,
            );
            simd_layers_v2(
                &mut packed_values[destination..destination + packed_block_len],
                r + 2,
                lg_block_n,
                root_table,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cmp::{max, min};

    use plonky2_util::{log2_ceil, log2_strict, reverse_index_bits_in_place};
    use unroll::unroll_for_loops;

    use super::{BaseSubfieldTwiddle, FftTwiddleMul, GeneralTwiddle};
    use crate::extension::quadratic::QuadraticExtension;
    use crate::fft::{FftRootTable, fft, fft_classic, fft_root_table, fft_with_options, ifft};
    use crate::goldilocks_field::GoldilocksField;

    /// Portable copy of the general extension-butterfly product used by the
    /// NEON path: `(a0 + a1*u) * (b0 + b1*u)` with `u^2 = 7`, reduced
    /// component-wise exactly like the generic `QuadraticExtension` mul.
    fn ext_product_portable(
        w: &QuadraticExtension<GoldilocksField>,
        v: &QuadraticExtension<GoldilocksField>,
    ) -> [u64; 2] {
        let a = w.0[0] * v.0[0];
        let b = w.0[1] * v.0[1];
        let c = w.0[0] * v.0[1];
        let d = w.0[1] * v.0[0];
        let c0 = (a + GoldilocksField(7) * b).0;
        let c1 = (c + d).0;
        [c0, c1]
    }

    #[test]
    fn extension_neon_products_match_generic_mul() {
        use crate::types::Sample;
        for _ in 0..1000 {
            let w = QuadraticExtension::<GoldilocksField>::rand();
            let v = QuadraticExtension::<GoldilocksField>::rand();
            let expected = (w * v).0;
            let got = ext_product_portable(&w, &v);
            assert_eq!(
                [got[0], got[1]],
                [expected[0].0, expected[1].0],
                "general extension product formula diverges"
            );

            // Base-subfield twiddle fast path: [w0, 0] * [v0, v1] = [w0*v0, w0*v1].
            let ws = QuadraticExtension::<GoldilocksField>([
                w.0[0],
                GoldilocksField::ZERO,
            ]);
            let expected_s = (ws * v).0;
            let p0 = (ws.0[0] * v.0[0]).0;
            let p1 = (ws.0[0] * v.0[1]).0;
            assert_eq!(
                [p0, p1],
                [expected_s[0].0, expected_s[1].0],
                "base-subfield twiddle product diverges"
            );
        }
    }
    use crate::packable::Packable;
    use crate::packed::PackedField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;

    /// `ifft_borrowed` must be bit-identical to `ifft` of a copy across
    /// sizes straddling the packed width and the small/large bit-reversal
    /// strategies, for both a cached-table field type and an uncached one.
    #[test]
    fn ifft_borrowed_matches_ifft() {
        use crate::extension::quartic::QuarticExtension;
        use crate::fft::ifft_borrowed;
        use crate::types::Sample;

        fn check<F: Field + Sample>() {
            for lg_n in [1usize, 2, 4, 6, 7, 10, 13] {
                let n = 1 << lg_n;
                let values = F::rand_vec(n);
                let expected = ifft(PolynomialValues::new(values.clone()));
                let actual = ifft_borrowed(&values);
                assert_eq!(expected.coeffs, actual.coeffs);
            }
        }

        check::<GoldilocksField>();
        check::<QuadraticExtension<GoldilocksField>>();
        check::<QuarticExtension<GoldilocksField>>();
    }

    /// The cached-table dispatch path (no caller-supplied table) must return
    /// bit-identical results to an explicitly computed fresh table, on both
    /// cold and warm cache, for the cached field types (base and quadratic
    /// extension, each with a dedicated `OnceLock` slot array) and for an
    /// uncached field type (quartic extension), which takes the fresh-compute
    /// fallback on every call.
    #[cfg(feature = "std")]
    #[test]
    fn cached_root_table_matches_fresh_table() {
        use crate::extension::quartic::QuarticExtension;
        use crate::types::Sample;

        fn check<F: Field + Sample>() {
            for lg_n in [1usize, 3, 7] {
                let n = 1 << lg_n;
                let table = fft_root_table::<F>(n);
                let poly = PolynomialCoeffs::new(F::rand_vec(n));
                let expected = fft_with_options(poly.clone(), None, Some(&table));
                // First call may populate the cache, second one must hit it.
                // (For an uncached type both calls take the fallback.)
                let cold = fft_with_options(poly.clone(), None, None);
                let warm = fft_with_options(poly, None, None);
                assert_eq!(expected.values, cold.values);
                assert_eq!(expected.values, warm.values);
            }
        }

        // Cached types: dedicated per-type slot arrays.
        check::<GoldilocksField>();
        check::<QuadraticExtension<GoldilocksField>>();
        // Uncached type: exercises the value-identical fallback path.
        check::<QuarticExtension<GoldilocksField>>();
    }


    /// Driving the layers as single stages must be **bit**-identical to the
    /// radix-4 fused traversal it replaced on the production path, not
    /// merely congruent: the two perform the same butterflies on the same
    /// values with the same twiddles in the same per-element order, so the
    /// raw `GoldilocksField.0` words must match exactly. Every start stage
    /// at each size is covered, so both stage-count parities are exercised.
    #[test]
    fn single_layer_driver_matches_fused_reference_raw_words() {
        use crate::fft::{
            fft_classic_simd_fused_two_layers_with, fft_classic_simd_layers,
            fft_classic_simd_single_layer_with,
        };

        /// The pre-change driver, verbatim, as the oracle. It is pinned to
        /// `GeneralTwiddle`, i.e. the plain `twiddle * value` this code used
        /// before the twiddle specialization, while the arm under test runs the
        /// marker production selects. The comparison therefore still isolates
        /// the defusion and additionally pins the base-field marker choice to
        /// the same raw words.
        fn drive_fused<P: PackedField>(
            packed_values: &mut [P],
            start: usize,
            end: usize,
            root_table: &FftRootTable<P::Scalar>,
        ) {
            let lg_packed_width = log2_strict(P::WIDTH);
            let mut lg_half_m = start;
            if (end - start) % 2 == 1 {
                fft_classic_simd_single_layer_with::<P, GeneralTwiddle>(
                    packed_values,
                    lg_half_m,
                    lg_packed_width,
                    root_table,
                );
                lg_half_m += 1;
            }
            while lg_half_m < end {
                fft_classic_simd_fused_two_layers_with::<P, GeneralTwiddle>(
                    packed_values,
                    lg_half_m,
                    lg_packed_width,
                    root_table,
                );
                lg_half_m += 2;
            }
        }

        type F = GoldilocksField;
        type P = <F as Packable>::Packing;
        for lg_n in [4usize, 5, 8, 11, 13, 16, 17] {
            let n = 1 << lg_n;
            let roots = fft_root_table::<F>(n);
            for start in 2..lg_n {
                let mut expected = deterministic_values(n);
                let mut actual = expected.clone();
                drive_fused::<P>(P::pack_slice_mut(&mut expected), start, lg_n, &roots);
                fft_classic_simd_layers::<P, BaseSubfieldTwiddle>(
                    P::pack_slice_mut(&mut actual),
                    start,
                    lg_n,
                    &roots,
                );
                for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        a.0, e.0,
                        "raw word mismatch at 2^{lg_n} start {start} index {i}"
                    );
                }
            }
        }
    }

    /// The true radix-4 identity replaces four general twiddle products with
    /// three plus the shift-only fourth-root product. Reassociation can select
    /// a different non-canonical representative, so the contract here is field
    /// equality against the two historical stages at every paired layer.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn true_radix4_pair_matches_two_layers_canonically() {
        use crate::fft::{
            fft_classic_simd_fused_two_layers_with, fft_classic_simd_true_radix4_neon,
        };

        type F = GoldilocksField;
        type P = <F as Packable>::Packing;
        let lg_packed_width = log2_strict(P::WIDTH);
        for lg_n in [4usize, 5, 8, 11, 13, 16] {
            let n = 1usize << lg_n;
            let roots = fft_root_table::<F>(n);
            for lg_half_m in lg_packed_width..lg_n - 1 {
                let mut expected = deterministic_values(n);
                let mut actual = expected.clone();
                fft_classic_simd_fused_two_layers_with::<P, GeneralTwiddle>(
                    P::pack_slice_mut(&mut expected),
                    lg_half_m,
                    lg_packed_width,
                    &roots,
                );
                fft_classic_simd_true_radix4_neon(&mut actual, lg_half_m, &roots);
                assert_eq!(
                    actual, expected,
                    "true radix-4 mismatch at 2^{lg_n}, layers {lg_half_m}/{}",
                    lg_half_m + 1
                );
            }
        }
    }


    /// The NEON base-field layer must be **bit**-identical to the generic body
    /// it specialises: same butterflies, same twiddles, same order, so the raw
    /// `GoldilocksField.0` words must match exactly rather than merely be
    /// congruent. Seeded with values above ORDER and within 97 of 2^64 so the
    /// rare double-overflow and double-underflow corrections actually fire --
    /// a differential that never exercises its corrections cannot catch them.
    ///
    /// The quadratic extension is included deliberately: it is `WIDTH == 1` and
    /// must take the generic fallback, so the fallback being exercised is as
    /// much a correctness requirement as the fast path.
    ///
    /// The oracle keeps the plain `twiddle * value` product, while the arm under
    /// test runs `BaseSubfieldTwiddle`, the marker production selects for these
    /// sizes. The extension half is therefore simultaneously the fallback
    /// differential it always was and a twiddle-specialization differential.
    #[test]
    fn neon_single_layer_matches_generic_raw_words() {
        use crate::fft::fft_classic_simd_single_layer_with;
        use crate::types::{Field64, PrimeField64};

        /// Verbatim copy of the generic body, as the oracle.
        fn generic<P: PackedField>(
            packed_values: &mut [P],
            lg_half_m: usize,
            lg_packed_width: usize,
            root_table: &FftRootTable<P::Scalar>,
        ) {
            let lg_m = lg_half_m + 1;
            let m = 1 << lg_m;
            let packed_m = m >> lg_packed_width;
            let half_packed_m = packed_m / 2;
            let omega_table = &P::pack_slice(&root_table[lg_half_m])[..half_packed_m];
            for block in packed_values.chunks_exact_mut(packed_m) {
                let (lows, highs) = block.split_at_mut(half_packed_m);
                for ((u, v), &omega) in lows.iter_mut().zip(highs.iter_mut()).zip(omega_table) {
                    let t = omega * *v;
                    let u_value = *u;
                    *u = u_value + t;
                    *v = u_value - t;
                }
            }
        }

        fn adversarial<F: Field + Field64>(n: usize) -> Vec<F> {
            (0..n)
                .map(|i| {
                    F::from_noncanonical_u64(match i % 5 {
                        0 => (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        1 => F::ORDER.wrapping_add(i as u64 % 1000),
                        2 => u64::MAX - (i as u64 % 97),
                        3 => (i as u64).wrapping_mul(0xdead_beef) | (1 << 63),
                        _ => i as u64,
                    })
                })
                .collect()
        }

        type F = GoldilocksField;
        type P = <F as Packable>::Packing;
        let lg_packed_width = log2_strict(P::WIDTH);
        for lg_n in [6usize, 8, 11, 13, 16, 17] {
            let n = 1usize << lg_n;
            let roots = fft_root_table::<F>(n);
            for lg_half_m in lg_packed_width..lg_n {
                let mut expected = adversarial::<F>(n);
                let mut actual = expected.clone();
                generic::<P>(P::pack_slice_mut(&mut expected), lg_half_m, lg_packed_width, &roots);
                fft_classic_simd_single_layer_with::<P, BaseSubfieldTwiddle>(
                    P::pack_slice_mut(&mut actual),
                    lg_half_m,
                    lg_packed_width,
                    &roots,
                );
                for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        a.0, e.0,
                        "raw word mismatch at 2^{lg_n} layer {lg_half_m} index {i}"
                    );
                }
            }
        }

        // Fallback instantiations: both are WIDTH == 1 and must be unaffected.
        type FE = QuadraticExtension<GoldilocksField>;
        for lg_n in [6usize, 9] {
            let n = 1usize << lg_n;
            let roots_ext = fft_root_table::<FE>(n);
            for lg_half_m in 0..lg_n {
                let mut expected: Vec<FE> = (0..n)
                    .map(|i| QuadraticExtension([deterministic_value(i), deterministic_value(i + n)]))
                    .collect();
                let mut actual = expected.clone();
                generic::<FE>(&mut expected, lg_half_m, 0, &roots_ext);
                fft_classic_simd_single_layer_with::<FE, BaseSubfieldTwiddle>(&mut actual, lg_half_m, 0, &roots_ext);
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert_eq!(a.0[0].to_canonical_u64(), e.0[0].to_canonical_u64());
                    assert_eq!(a.0[1].to_canonical_u64(), e.0[1].to_canonical_u64());
                }
            }
        }
    }

    #[test]
    fn fft_and_ifft() {
        type F = GoldilocksField;
        let degree = 200usize;
        let degree_padded = degree.next_power_of_two();

        // Create a vector of coeffs; the first degree of them are
        // "random", the last degree_padded-degree of them are zero.
        let coeffs = (0..degree)
            .map(|i| F::from_canonical_usize(i * 1337 % 100))
            .chain(core::iter::repeat_n(F::ZERO, degree_padded - degree))
            .collect::<Vec<_>>();
        assert_eq!(coeffs.len(), degree_padded);
        let coefficients = PolynomialCoeffs { coeffs };

        let points = fft(coefficients.clone());
        assert_eq!(points, evaluate_naive(&coefficients));

        let interpolated_coefficients = ifft(points);
        for i in 0..degree {
            assert_eq!(interpolated_coefficients.coeffs[i], coefficients.coeffs[i]);
        }
        for i in degree..degree_padded {
            assert_eq!(interpolated_coefficients.coeffs[i], F::ZERO);
        }

        for r in 0..4 {
            // expand coefficients by factor 2^r by filling with zeros
            let zero_tail = coefficients.lde(r);
            assert_eq!(
                fft(zero_tail.clone()),
                fft_with_options(zero_tail, Some(r), None)
            );
        }
    }

    fn evaluate_naive<F: Field>(coefficients: &PolynomialCoeffs<F>) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_padded = 1 << log2_ceil(degree);

        let coefficients_padded = coefficients.padded(degree_padded);
        evaluate_naive_power_of_2(&coefficients_padded)
    }

    fn evaluate_naive_power_of_2<F: Field>(
        coefficients: &PolynomialCoeffs<F>,
    ) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_log = log2_strict(degree);

        let subgroup = F::two_adic_subgroup(degree_log);

        let values = subgroup
            .into_iter()
            .map(|x| evaluate_at_naive(coefficients, x))
            .collect();
        PolynomialValues::new(values)
    }

    fn evaluate_at_naive<F: Field>(coefficients: &PolynomialCoeffs<F>, point: F) -> F {
        let mut sum = F::ZERO;
        let mut point_power = F::ONE;
        for &c in &coefficients.coeffs {
            sum += c * point_power;
            point_power *= point;
        }
        sum
    }

    /// The original stage-major SIMD kernel, retained only as a differential-test oracle.
    #[unroll_for_loops]
    fn fft_classic_simd_reference<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_n: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let lg_packed_width = log2_strict(P::WIDTH);
        let packed_values = P::pack_slice_mut(values);
        let packed_n = packed_values.len();

        assert!(lg_packed_width <= 4);
        for lg_half_m in 0..4 {
            if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
                let half_m = 1 << lg_half_m;
                let mut omega = P::default();
                for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                    *omega_j = root_table[lg_half_m][j % half_m];
                }

                for k in (0..packed_n).step_by(2) {
                    let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                    let t = omega * v;
                    (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
                }
            }
        }

        let s = max(r, lg_packed_width);
        for lg_half_m in s..lg_n {
            let packed_m = 1 << (lg_half_m + 1 - lg_packed_width);
            let half_packed_m = packed_m / 2;
            let omega_table = P::pack_slice(&root_table[lg_half_m]);

            for k in (0..packed_n).step_by(packed_m) {
                for j in 0..half_packed_m {
                    let omega = omega_table[j];
                    let t = omega * packed_values[k + half_packed_m + j];
                    let u = packed_values[k + j];
                    packed_values[k + j] = u + t;
                    packed_values[k + half_packed_m + j] = u - t;
                }
            }
        }
    }

    fn fft_classic_reference<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        reverse_index_bits_in_place(values);
        let n = values.len();
        let lg_n = log2_strict(n);

        if r > 0 {
            let mask = !((1 << r) - 1);
            for i in 0..n {
                values[i] = values[i & mask];
            }
        }

        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        if lg_n <= lg_packed_width {
            fft_classic_simd_reference::<F>(values, r, lg_n, root_table);
        } else {
            fft_classic_simd_reference::<<F as Packable>::Packing>(values, r, lg_n, root_table);
        }
    }

    fn deterministic_value(i: usize) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(
            (i as u64).wrapping_mul(0x9e37_79b9).rotate_left(13) % 1_000_000_007,
        )
    }

    fn deterministic_values(n: usize) -> Vec<GoldilocksField> {
        (0..n).map(deterministic_value).collect()
    }

    fn ifft_reference(
        mut values: Vec<GoldilocksField>,
        root_table: &FftRootTable<GoldilocksField>,
    ) -> Vec<GoldilocksField> {
        let n = values.len();
        let lg_n = log2_strict(n);
        let n_inv = GoldilocksField::inverse_2exp(lg_n);
        fft_classic_reference(&mut values, 0, root_table);

        values[0] *= n_inv;
        values[n / 2] *= n_inv;
        for i in 1..n / 2 {
            let j = n - i;
            let coeff_i = values[j] * n_inv;
            let coeff_j = values[i] * n_inv;
            values[i] = coeff_i;
            values[j] = coeff_j;
        }
        values
    }

    #[test]
    fn fft_matches_stage_major_reference_across_sizes() {
        for lg_n in [0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 16, 18] {
            let n = 1 << lg_n;
            let roots = fft_root_table(n);
            let mut expected = deterministic_values(n);
            let mut actual = expected.clone();

            fft_classic_reference(&mut expected, 0, &roots);
            fft_classic(&mut actual, 0, &roots);
            assert_eq!(actual, expected, "FFT mismatch at 2^{lg_n}");
        }
    }

    #[test]
    fn coset_fft_and_ifft_match_stage_major_reference() {
        type F = GoldilocksField;
        let shift = F::coset_shift();

        for lg_n in [3, 8, 12, 15] {
            let n = 1 << lg_n;
            let roots = fft_root_table(n);
            let coeffs = deterministic_values(n);

            let mut expected_fft = coeffs
                .iter()
                .copied()
                .zip(shift.powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            fft_classic_reference(&mut expected_fft, 0, &roots);
            let actual_fft = PolynomialCoeffs::new(coeffs).coset_fft(shift).values;
            assert_eq!(actual_fft, expected_fft, "coset FFT mismatch at 2^{lg_n}");

            let values = deterministic_values(n);
            let expected_ifft = ifft_reference(values.clone(), &roots)
                .into_iter()
                .zip(shift.inverse().powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            let actual_ifft = PolynomialValues::new(values).coset_ifft(shift).coeffs;
            assert_eq!(
                actual_ifft, expected_ifft,
                "coset IFFT mismatch at 2^{lg_n}"
            );
        }
    }

    #[test]
    fn zero_padded_lde_shapes_match_full_and_shortcut_references() {
        type F = GoldilocksField;
        let shift = F::coset_shift();

        for (base_lg_n, r) in [(3, 1), (8, 2), (10, 3), (12, 3), (13, 3), (15, 3)] {
            let lg_n = base_lg_n + r;
            let n = 1 << lg_n;
            let nonzero_len = 1 << base_lg_n;
            let roots = fft_root_table(n);
            let padded = deterministic_values(nonzero_len)
                .into_iter()
                .chain(core::iter::repeat_n(F::ZERO, n - nonzero_len))
                .collect::<Vec<_>>();

            let mut expected_shortcut = padded.clone();
            fft_classic_reference(&mut expected_shortcut, r, &roots);
            let mut expected_full = padded.clone();
            fft_classic_reference(&mut expected_full, 0, &roots);
            assert_eq!(
                expected_shortcut, expected_full,
                "reference shortcut mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );

            let mut actual = padded.clone();
            fft_classic(&mut actual, r, &roots);
            assert_eq!(
                actual, expected_full,
                "zero-padded FFT mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );

            let mut expected_coset = padded
                .iter()
                .copied()
                .zip(shift.powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            fft_classic_reference(&mut expected_coset, 0, &roots);
            let actual_coset = PolynomialCoeffs::new(padded)
                .coset_fft_with_options(shift, Some(r), Some(&roots))
                .values;
            assert_eq!(
                actual_coset, expected_coset,
                "zero-padded coset FFT mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );
        }
    }

    #[test]
    fn quadratic_base_twiddle_mul_matches_general_raw_limbs() {
        type F = GoldilocksField;
        type FE = QuadraticExtension<F>;

        let check = |twiddle: FE, value: FE| {
            assert_eq!(
                twiddle.0[1].0, 0,
                "test twiddle is not in the base subfield"
            );
            let expected = twiddle * value;
            let actual = <BaseSubfieldTwiddle as FftTwiddleMul<FE>>::mul(twiddle, value);
            assert_eq!(
                [actual.0[0].0, actual.0[1].0],
                [expected.0[0].0, expected.0[1].0],
                "raw limb mismatch for twiddle={twiddle:?}, value={value:?}"
            );
        };

        // Include canonical boundaries and every useful non-canonical u64
        // boundary. Equality is deliberately on raw limbs, not field values.
        let specials = [
            0,
            1,
            2,
            0xFFFF_FFFE_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0xFFFF_FFFF_0000_0001,
            0xFFFF_FFFF_0000_0002,
            u64::MAX,
        ];
        for &w in &specials {
            for &a0 in &specials {
                for &a1 in &specials {
                    check(
                        QuadraticExtension([GoldilocksField(w), F::ZERO]),
                        QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)]),
                    );
                }
            }
        }

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            check(
                QuadraticExtension([GoldilocksField(next()), F::ZERO]),
                QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]),
            );
        }

        // Every usable Goldilocks-extension FFT root row is base-subfield.
        // The one extra two-adic root is extension-only and must retain the
        // general marker path.
        let root_value = QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
        for lg_n in 1..FE::TWO_ADICITY {
            let root = FE::primitive_root_of_unity(lg_n);
            assert_eq!(root.0[1].0, 0, "2^{lg_n} root escaped the base subfield");
            check(root, root_value);
        }
        let extension_root = FE::primitive_root_of_unity(FE::TWO_ADICITY);
        assert_ne!(extension_root.0[1].0, 0);
        let expected = extension_root * root_value;
        let actual = <GeneralTwiddle as FftTwiddleMul<FE>>::mul(extension_root, root_value);
        assert_eq!(
            [actual.0[0].0, actual.0[1].0],
            [expected.0[0].0, expected.0[1].0]
        );
    }

    #[test]
    fn zero_padded_extension_fft_matches_reference() {
        type F = GoldilocksField;
        type FE = QuadraticExtension<F>;

        let r = 3;
        for lg_n in [9usize, 11, 12, 13, 15, 17] {
            let n = 1 << lg_n;
            let nonzero_len = n >> r;
            let roots = fft_root_table(n);
            let raw_value = |i: usize, salt: u64| {
                let mut x = (i as u64).wrapping_add(salt);
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                GoldilocksField(x)
            };
            let mut actual = (0..nonzero_len)
                .map(|i| {
                    QuadraticExtension([
                        raw_value(i, 0x9E37_79B9_7F4A_7C15),
                        raw_value(i, 0xD1B5_4A32_D192_ED03),
                    ])
                })
                .chain(core::iter::repeat_n(FE::ZERO, n - nonzero_len))
                .collect::<Vec<_>>();
            let mut expected = actual.clone();

            fft_classic_reference(&mut expected, r, &roots);
            fft_classic(&mut actual, r, &roots);
            assert_eq!(
                actual, expected,
                "extension FFT mismatch at 2^{lg_n}, r={r}"
            );
        }
    }

}
