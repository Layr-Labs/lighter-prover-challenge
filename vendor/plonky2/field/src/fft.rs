use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
#[cfg(target_arch = "aarch64")]
use crate::goldilocks_field::mul_16th_root_powers;

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

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
fn fft_classic_simd<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
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
                let t = omega * v;
                (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
            }
        }
    }

    // We've already done the first lg_packed_width (if they were required) iterations.
    let s = max(r, lg_packed_width);
    fft_classic_simd_layers(packed_values, s, lg_n, root_table);
}

#[inline(always)]
fn fft_classic_simd_single_layer<P: PackedField>(
    packed_values: &mut [P],
    lg_half_m: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_m = lg_half_m + 1;
    let m = 1 << lg_m; // Subarray size (in field elements).
    let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
    let half_packed_m = packed_m / 2;
    debug_assert!(half_packed_m != 0);

    // Omega values for this iteration, as a slice of vectors.
    let omega_table = P::pack_slice(&root_table[lg_half_m]);
    for k in (0..packed_values.len()).step_by(packed_m) {
        for j in 0..half_packed_m {
            let omega = omega_table[j];
            let t = omega * packed_values[k + half_packed_m + j];
            let u = packed_values[k + j];
            packed_values[k + j] = u + t;
            packed_values[k + half_packed_m + j] = u - t;
        }
    }
}

/// Two consecutive stages fused into one radix-4-style traversal: the exact
/// same butterflies with the exact same `root_table` twiddles, but each
/// quarter-block element is loaded and stored once per stage *pair* instead
/// of once per stage, halving whole-array memory passes for these layers.
#[inline(always)]
fn fft_classic_simd_fused_two_layers<P: PackedField>(
    packed_values: &mut [P],
    lg_half_m: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
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
            let t = w1 * b;
            let (ab0, ab1) = (a + t, a - t);
            let t = w1 * d;
            let (cd0, cd1) = (c + t, c - t);

            // Second stage: butterflies pairing positions j and j + 2q.
            let t = stage2_omegas[j] * cd0;
            packed_values[k + j] = ab0 + t;
            packed_values[k + 2 * q + j] = ab0 - t;
            let t = stage2_omegas[q + j] * cd1;
            packed_values[k + q + j] = ab1 + t;
            packed_values[k + 3 * q + j] = ab1 - t;
        }
    }
}

#[inline(always)]
fn fft_classic_simd_layers<P: PackedField>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH);
    let mut lg_half_m = start;
    // Odd stage count: run the first stage unfused so the remainder pairs up.
    if (end - start) % 2 == 1 {
        fft_classic_simd_single_layer(packed_values, lg_half_m, lg_packed_width, root_table);
        lg_half_m += 1;
    }
    while lg_half_m < end {
        fft_classic_simd_fused_two_layers(
            packed_values,
            lg_half_m,
            lg_packed_width,
            root_table,
        );
        lg_half_m += 2;
    }
}

#[inline(always)]
fn fft_zero_padded_first_layer_block<P: PackedField>(
    packed_values: &mut [P],
    source_start: usize,
    nonzero_len: usize,
    destination: usize,
    packed_repeat: usize,
    omega_table: &[P],
) {
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
            let t = omega_table[j] * v;
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
fn fft_zero_padded_first_layer<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;
    debug_assert!(repeat >= P::WIDTH);

    let packed_repeat = repeat / P::WIDTH;
    let packed_values = P::pack_slice_mut(values);
    let omega_table = P::pack_slice(&root_table[r]);
    fft_zero_padded_first_layer_block(packed_values, 0, nonzero_len, 0, packed_repeat, omega_table);
}

fn fft_zero_padded_cache_blocks<P: PackedField>(
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
            fft_zero_padded_first_layer_block(
                packed_values,
                source_start,
                nonzero_per_block,
                destination,
                packed_repeat,
                omega_table,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        fft_zero_padded_first_layer_block(
            packed_values,
            source_start,
            nonzero_per_block,
            destination,
            packed_repeat,
            omega_table,
        );
        fft_classic_simd_layers(
            &mut packed_values[destination..destination + packed_block_len],
            r + 1,
            lg_block_n,
            root_table,
        );
    }
}

#[inline(never)]
fn prepare_zero_padded_fft<F: Field>(
    values: &mut [F],
    r: usize,
    lg_n: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<F>,
) -> usize {
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
            fft_zero_padded_cache_blocks::<<F as Packable>::Packing>(
                values, r, lg_block_n, root_table,
            );
            lg_block_n
        } else {
            // Fuse the expansion with the first nontrivial layer, eliminating one full-buffer
            // write/read cycle while retaining the existing skipped-layer semantics.
            fft_zero_padded_first_layer::<<F as Packable>::Packing>(values, r, root_table);
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
pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    let first_layer = if r == 0 {
        reverse_index_bits_in_place(values);
        0
    } else {
        prepare_zero_padded_fft(values, r, lg_n, lg_packed_width, root_table)
    };

    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd::<F>(values, first_layer, lg_n, root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(values, first_layer, lg_n, root_table);
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

    use super::{
        FftRootTable, fft_classic, fft_classic_simd, fft_classic_simd_fused_two_layers,
        fft_classic_simd_layers, fft_classic_simd_single_layer, fft_root_table,
        fft_zero_padded_first_layer_block,
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
            fft_classic_simd_layers(
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
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, root_table);
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
                fft_classic_simd_fused_two_layers(packed_values, l, lg_packed_width, root_table);
                l += 2;
                fft_classic_simd_fused_two_layers(packed_values, l, lg_packed_width, root_table);
                l += 2;
            }
            1 => {
                fft_classic_simd_single_layer(packed_values, l, lg_packed_width, root_table);
                l += 1;
            }
            2 => {
                fft_classic_simd_fused_two_layers(packed_values, l, lg_packed_width, root_table);
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
                    fft_zero_padded_first_layer_block(
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
        super::fft_zero_padded_cache_blocks::<<F as Packable>::Packing>(
            values, r, lg_block_n, root_table,
        );
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, root_table);
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
                fft_zero_padded_first_layer_block(
                    packed_values,
                    source_start,
                    nonzero_per_block,
                    destination,
                    packed_repeat,
                    omega_table,
                );
            }
            fft_classic_simd_layers(
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
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, folded_table);
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
            fft_classic_simd_layers(
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
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, folded_table);
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
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, root_table);
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
        fft_classic_simd::<<F as Packable>::Packing>(values, lg_block_n, lg_n, folded_table);
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

    use crate::extension::quadratic::QuadraticExtension;
    use crate::fft::{FftRootTable, fft, fft_classic, fft_root_table, fft_with_options, ifft};
    use crate::goldilocks_field::GoldilocksField;
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
    fn zero_padded_extension_fft_matches_reference() {
        type F = GoldilocksField;
        type FE = QuadraticExtension<F>;

        let base_lg_n = 10;
        let r = 3;
        let n = 1 << (base_lg_n + r);
        let nonzero_len = 1 << base_lg_n;
        let roots = fft_root_table(n);
        let mut actual = (0..nonzero_len)
            .map(|i| {
                QuadraticExtension([deterministic_value(i), deterministic_value(i + nonzero_len)])
            })
            .chain(core::iter::repeat_n(FE::ZERO, n - nonzero_len))
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        fft_classic_reference(&mut expected, r, &roots);
        fft_classic(&mut actual, r, &roots);
        assert_eq!(actual, expected);
    }

}
