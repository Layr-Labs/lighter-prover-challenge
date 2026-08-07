use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

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
}

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
    live_prefix_bit_reversed: bool,
) {
    // `live_prefix_bit_reversed` is a literal at every caller and this function
    // is `#[inline]`, so the branch is const-folded away per entry point.
    if let Some(table) = root_table {
        if live_prefix_bit_reversed {
            fft_classic_maybe_prereversed(input, zero_factor.unwrap_or(0), table, true);
        } else {
            fft_classic(input, zero_factor.unwrap_or(0), table);
        }
        return;
    }
    #[cfg(feature = "std")]
    let computed_root_table = root_table_cache::get::<F>(log2_strict(input.len()));
    #[cfg(not(feature = "std"))]
    let computed_root_table = fft_root_table::<F>(input.len());

    if live_prefix_bit_reversed {
        fft_classic_maybe_prereversed(input, zero_factor.unwrap_or(0), &computed_root_table, true);
    } else {
        fft_classic(input, zero_factor.unwrap_or(0), &computed_root_table);
    }
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
    fft_dispatch(buffer, zero_factor, root_table, false);
}

/// [`fft_in_place_with_options`] for a caller that has already bit-reversed the
/// live coefficient prefix. See [`fft_with_options_prefix_bit_reversed`].
#[inline]
pub fn fft_in_place_with_options_prefix_bit_reversed<F: Field>(
    buffer: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    fft_dispatch(buffer, zero_factor, root_table, true);
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
    fft_dispatch(&mut buffer, zero_factor, root_table, false);
    PolynomialValues::new(buffer)
}

/// [`fft_with_options`] for a caller that has already bit-reversed the live
/// coefficient prefix — `coeffs[..len >> zero_factor]`, the exact range this
/// FFT would otherwise reverse itself (the whole buffer when `zero_factor` is
/// `None` or `Some(0)`).
///
/// A producer that writes that prefix can perform the permutation as it writes
/// (see [`plonky2_util::fill_bit_reversed`]) instead of sweeping the buffer
/// again afterwards. Everything downstream of the permutation is unchanged, so
/// the result is bit-identical to [`fft_with_options`] on the unpermuted
/// coefficients.
#[inline]
pub fn fft_with_options_prefix_bit_reversed<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table, true);
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
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table, false);

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

/// Stage-major traversal of layers `start..end`: every layer (or fused layer
/// pair) sweeps the whole buffer once.
#[inline(always)]
fn fft_classic_simd_layers_flat<P: PackedField>(
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

/// Byte budget for one second-level-cache-resident block of upper-layer work.
///
/// Apple Silicon's performance clusters share one L2 between the cores in the
/// cluster (16 MiB across five cores on an M4 Pro), and the prover runs one
/// LDE column per worker, so a block has to stay well inside a single core's
/// share with room for the twiddle rows those layers stream alongside the
/// data. Swept over 2^14..2^18 elements at the production shape; 512 KiB was
/// the best or tied-best at every site and leaves the most headroom under
/// cluster sharing.
const L2_BLOCK_BYTES: usize = 1 << 19;

/// `log2` of how many `elem_bytes`-sized elements fit in `budget` bytes.
const fn lg_block_elems(elem_bytes: usize, budget: usize) -> usize {
    let mut count = budget / if elem_bytes == 0 { 1 } else { elem_bytes };
    let mut lg = 0;
    while count > 1 {
        count >>= 1;
        lg += 1;
    }
    lg
}

/// `log2` of the L1-resident block length used by the low layers: the block
/// plus the largest twiddle row it reads stays inside Apple Silicon's 128 KiB
/// L1D (about 96 KiB for both the base field and the quadratic extension).
const fn lg_l1_block_elems(elem_bytes: usize) -> usize {
    match elem_bytes {
        0..=8 => 13,
        9..=16 => 12,
        _ => 11,
    }
}

#[inline(always)]
fn fft_classic_simd_layers<P: PackedField>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    fft_classic_simd_layers_tuned(packed_values, start, end, root_table, L2_BLOCK_BYTES);
}

/// Smallest column window worth streaming, in bytes.
///
/// The column-blocked traversal replaces one sequential sweep per layer pair
/// with `2^(end - lg_row)` interleaved runs of this length. Below roughly a
/// 16 KiB run the extra stream starts and the lost prefetch depth cost more
/// than the deleted passes save, so the decomposition lifts `lg_row` (adding
/// a contiguous-block phase) rather than narrowing the window past it.
const MIN_COLUMN_RUN_BYTES: usize = 1 << 14;

/// Layers `start..end` of a length-`2^end` transform, cache-blocked when the
/// buffer is larger than one L2 block.
///
/// Above layer `lg_row` the buffer is `2^(end - lg_row)` completed transforms
/// ("rows") of length `2^lg_row` laid out contiguously, and every remaining
/// butterfly pairs two rows at the *same* offset within the row. Columns are
/// therefore independent across the entire remaining layer list, so a window
/// of consecutive columns can be carried through all of them in one residency
/// instead of every fused layer pair sweeping the whole buffer.
///
/// The butterflies, their operands and their twiddles are exactly those of
/// [`fft_classic_simd_layers_flat`]; only the order in which independent
/// columns are visited changes, so the buffer ends bit-identical.
fn fft_classic_simd_layers_tuned<P: PackedField>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
    l2_block_bytes: usize,
) {
    debug_assert_eq!(packed_values.len() * P::WIDTH, 1usize << end);
    let elem_bytes = core::mem::size_of::<P::Scalar>();
    let lg_block = lg_block_elems(elem_bytes, l2_block_bytes);
    let lg_min_cols = lg_block_elems(elem_bytes, MIN_COLUMN_RUN_BYTES);
    let lg_packed_width = log2_strict(P::WIDTH);
    // Nothing to gain while the whole transform already fits one block, and
    // nothing to work with unless a minimum window is at least one vector.
    //
    // A field with no packed kernel (`WIDTH == 1`) is skipped as well: its
    // per-element arithmetic is expensive enough — the quadratic extension
    // costs about 3.3x a base-field butterfly for 2x the bytes — that these
    // layers sit far below the memory roofline, and the per-window setup the
    // blocking adds is then pure overhead (measured 5-9% slower on the
    // extension-field FRI coset FFT at 2^19).
    if P::WIDTH == 1 || end <= lg_block || lg_min_cols < lg_packed_width || lg_min_cols >= lg_block
    {
        fft_classic_simd_layers_flat(packed_values, start, end, root_table);
        return;
    }

    // Rows are independent below layer `lg_row`; finish each one while it is
    // still L1-resident, which is also what gives the column phase a row long
    // enough to carve wide windows out of.
    let lg_row = max(start, min(lg_l1_block_elems(elem_bytes), end));
    if lg_row > start {
        let packed_row = (1usize << lg_row) >> lg_packed_width;
        for row in packed_values.chunks_mut(packed_row) {
            fft_classic_simd_layers_flat(row, start, lg_row, root_table);
        }
    }
    fft_upper_layers_blocked(
        packed_values,
        lg_row,
        end,
        lg_block,
        lg_min_cols,
        root_table,
    );
}

/// Layers `lg_row..end` over a buffer already grouped into completed rows of
/// length `2^lg_row`.
///
/// A single column-blocked phase can only span so many layers before the
/// window it can afford — `2^(lg_block + lg_row - end)` columns — drops below
/// [`MIN_COLUMN_RUN_BYTES`]. When it does, the layers are split: contiguous
/// blocks of `2^mid` elements are independent below layer `mid`, so those
/// layers are finished block by block (recursively, so each block is itself
/// blocked if it is still too large), and the column phase then starts from
/// the wider row `mid` with exactly the minimum window.
fn fft_upper_layers_blocked<P: PackedField>(
    packed_values: &mut [P],
    lg_row: usize,
    end: usize,
    lg_block: usize,
    lg_min_cols: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    debug_assert!(lg_row <= end);
    if end <= lg_block || lg_row >= end {
        fft_classic_simd_layers_flat(packed_values, lg_row, end, root_table);
        return;
    }

    // Narrowest row the column phase can start from and still stream well.
    let mid = end + lg_min_cols - lg_block;
    if mid <= lg_row {
        fft_upper_layers_column_blocked(
            packed_values,
            lg_row,
            end,
            lg_block + lg_row - end,
            root_table,
        );
        return;
    }
    if mid >= end {
        // The layer span is too wide for even one split to help.
        fft_classic_simd_layers_flat(packed_values, lg_row, end, root_table);
        return;
    }

    let packed_mid = (1usize << mid) >> log2_strict(P::WIDTH);
    for block in packed_values.chunks_mut(packed_mid) {
        fft_upper_layers_blocked(block, lg_row, mid, lg_block, lg_min_cols, root_table);
    }
    fft_upper_layers_column_blocked(packed_values, mid, end, lg_min_cols, root_table);
}

/// One upper layer restricted to the column window `[col, col + 2^lg_cols)` of
/// every row. Layer `lg_row + stage` pairs rows `q` and `q + 2^stage` at each
/// column `p` with twiddle `root_table[lg_row + stage][(q % 2^stage) * 2^lg_row + p]`,
/// which is the same butterfly the stage-major kernel performs at flat index
/// `q * 2^lg_row + p`.
#[inline(always)]
fn fft_upper_single_layer_cols<P: PackedField>(
    packed_values: &mut [P],
    lg_row: usize,
    stage: usize,
    col: usize,
    lg_cols: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let row_len = 1usize << lg_row;
    let cols = 1usize << lg_cols;
    let packed_row = row_len >> lg_packed_width;
    let packed_col = col >> lg_packed_width;
    let packed_cols = cols >> lg_packed_width;
    let rows = packed_values.len() / packed_row;
    let half = 1usize << stage;
    let omegas = &root_table[lg_row + stage];

    for base in (0..rows).step_by(2 * half) {
        for a in 0..half {
            let offset = a * row_len + col;
            let omega = P::pack_slice(&omegas[offset..offset + cols]);
            let lo_row = base + a;
            let (lo_side, hi_side) = packed_values.split_at_mut((lo_row + half) * packed_row);
            let lo = &mut lo_side[lo_row * packed_row + packed_col..][..packed_cols];
            let hi = &mut hi_side[packed_col..][..packed_cols];
            for j in 0..packed_cols {
                let t = omega[j] * hi[j];
                let u = lo[j];
                lo[j] = u + t;
                hi[j] = u - t;
            }
        }
    }
}

/// [`fft_upper_single_layer_cols`] for two consecutive upper layers fused into
/// one radix-4 traversal of four row segments, mirroring
/// [`fft_classic_simd_fused_two_layers`] with rows in place of packed offsets.
#[inline(always)]
fn fft_upper_fused_two_layers_cols<P: PackedField>(
    packed_values: &mut [P],
    lg_row: usize,
    stage: usize,
    col: usize,
    lg_cols: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let row_len = 1usize << lg_row;
    let cols = 1usize << lg_cols;
    let packed_row = row_len >> lg_packed_width;
    let packed_col = col >> lg_packed_width;
    let packed_cols = cols >> lg_packed_width;
    let rows = packed_values.len() / packed_row;
    let half = 1usize << stage;
    let stage1_omegas = &root_table[lg_row + stage];
    let stage2_omegas = &root_table[lg_row + stage + 1];

    for base in (0..rows).step_by(4 * half) {
        for a in 0..half {
            // Rows `r0 < r0 + half < r0 + 2 * half < r0 + 3 * half`. The first
            // layer pairs them as (0, 1) and (2, 3) with one twiddle row; the
            // second pairs (0, 2) and (1, 3), the latter half a block further
            // into the next twiddle row.
            let offset = a * row_len + col;
            let w1 = P::pack_slice(&stage1_omegas[offset..offset + cols]);
            let w2_lo = P::pack_slice(&stage2_omegas[offset..offset + cols]);
            let offset_hi = offset + half * row_len;
            let w2_hi = P::pack_slice(&stage2_omegas[offset_hi..offset_hi + cols]);

            let r0 = base + a;
            let (s0, rest) = packed_values.split_at_mut((r0 + half) * packed_row);
            let (s1, rest) = rest.split_at_mut(half * packed_row);
            let (s2, s3) = rest.split_at_mut(half * packed_row);
            let va = &mut s0[r0 * packed_row + packed_col..][..packed_cols];
            let vb = &mut s1[packed_col..][..packed_cols];
            let vc = &mut s2[packed_col..][..packed_cols];
            let vd = &mut s3[packed_col..][..packed_cols];

            for j in 0..packed_cols {
                let w = w1[j];
                let a_val = va[j];
                let b_val = vb[j];
                let c_val = vc[j];
                let d_val = vd[j];

                let t = w * b_val;
                let (ab0, ab1) = (a_val + t, a_val - t);
                let t = w * d_val;
                let (cd0, cd1) = (c_val + t, c_val - t);

                let t = w2_lo[j] * cd0;
                va[j] = ab0 + t;
                vc[j] = ab0 - t;
                let t = w2_hi[j] * cd1;
                vb[j] = ab1 + t;
                vd[j] = ab1 - t;
            }
        }
    }
}

/// Layers `lg_row..end` run one column window at a time, each window carried
/// through every remaining layer while it is L2-resident.
fn fft_upper_layers_column_blocked<P: PackedField>(
    packed_values: &mut [P],
    lg_row: usize,
    end: usize,
    lg_cols: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    for col in (0..1usize << lg_row).step_by(1 << lg_cols) {
        fft_upper_layers_one_column_window(packed_values, lg_row, end, lg_cols, col, root_table);
    }
}

/// Every layer in `lg_row..end`, for the single column window starting at
/// `col`. This is the unit of residency: the window's `2^(end - lg_row)` row
/// segments are read in for the first layer and written back after the last.
#[inline(always)]
fn fft_upper_layers_one_column_window<P: PackedField>(
    packed_values: &mut [P],
    lg_row: usize,
    end: usize,
    lg_cols: usize,
    col: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH);
    let layers = end - lg_row;
    let mut stage = 0;
    // Odd layer count: run the first layer unfused so the remainder pairs up.
    if layers % 2 == 1 {
        fft_upper_single_layer_cols(
            packed_values,
            lg_row,
            stage,
            col,
            lg_cols,
            lg_packed_width,
            root_table,
        );
        stage += 1;
    }
    while stage < layers {
        fft_upper_fused_two_layers_cols(
            packed_values,
            lg_row,
            stage,
            col,
            lg_cols,
            lg_packed_width,
            root_table,
        );
        stage += 2;
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

    // A zero-padded input only has n/2^r live coefficients, and the caller has
    // already bit-reversed exactly that prefix (bit-reversing the full buffer
    // would place those coefficients at multiples of 2^r, after which the
    // skipped FFT layers copy each one across its following 2^r-element run;
    // reversing just the live prefix produces that state directly).
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;

    if r >= lg_packed_width && r < lg_n {
        // Keep values plus the largest local twiddle row within Apple Silicon's 128 KiB L1D.
        // Both 2^13 base-field and 2^12 quadratic-extension blocks use about 96 KiB.
        let lg_block_n = lg_l1_block_elems(core::mem::size_of::<F>());
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
    fft_classic_maybe_prereversed(values, r, root_table, false);
}

/// [`fft_classic`], optionally skipping the initial bit-reversal because the
/// caller already produced `values[..n >> r]` in bit-reversed order. That
/// prefix is the only range the reversal touches (the whole buffer when
/// `r == 0`), so skipping it is the sole difference; every later stage is
/// reached in an identical state.
fn fft_classic_maybe_prereversed<F: Field>(
    values: &mut [F],
    r: usize,
    root_table: &FftRootTable<F>,
    live_prefix_bit_reversed: bool,
) {
    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    if !live_prefix_bit_reversed {
        // `n >> 0 == n`, so this is the full-buffer reversal when `r == 0`.
        reverse_index_bits_in_place(&mut values[..n >> r]);
    }

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    let first_layer = if r == 0 {
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

    /// The two-level cache blocking of the upper layers must be a pure
    /// reordering: `fft_classic_simd_layers_tuned` has to leave the buffer
    /// bit-identical to the stage-major `fft_classic_simd_layers_flat` for
    /// every block budget the tuning sweep considers, at every layer boundary
    /// the FFT entry points can hand it (the packed-width floor, an L1 block
    /// boundary, and both sides of it), on raw limbs.
    #[test]
    fn column_blocked_layers_match_flat_layers() {
        use crate::extension::FieldExtension;
        use crate::fft::{fft_classic_simd_layers_flat, fft_classic_simd_layers_tuned};
        use crate::types::{PrimeField64, Sample};

        fn check<F: Field + Sample + Packable, const D: usize>(
            starts: &[usize],
            raw: fn(&[F]) -> Vec<u64>,
        ) where
            F: FieldExtension<D>,
        {
            const BUDGETS: [usize; 3] = [1 << 19, 1 << 20, 1 << 21];

            for lg_n in [16usize, 17, 18, 19] {
                let n = 1usize << lg_n;
                let roots = fft_root_table::<F>(n);
                let input = F::rand_vec(n);
                for &start in starts {
                    let mut expected = input.clone();
                    fft_classic_simd_layers_flat(
                        <F as Packable>::Packing::pack_slice_mut(&mut expected),
                        start,
                        lg_n,
                        &roots,
                    );
                    let expected = raw(&expected);
                    for budget in BUDGETS {
                        let mut actual = input.clone();
                        fft_classic_simd_layers_tuned(
                            <F as Packable>::Packing::pack_slice_mut(&mut actual),
                            start,
                            lg_n,
                            &roots,
                            budget,
                        );
                        assert_eq!(
                            raw(&actual),
                            expected,
                            "2^{lg_n}, start {start}, budget {budget}"
                        );
                    }
                }
            }
        }

        // `start` must be at least the packing width for the packed kernels.
        check::<GoldilocksField, 1>(&[2, 5, 11, 12, 13, 14], |values| {
            values.iter().map(|x| x.to_noncanonical_u64()).collect()
        });
        check::<QuadraticExtension<GoldilocksField>, 2>(&[0, 5, 11, 12, 13], |values| {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        });
    }

    /// End-to-end differential for the cache-blocked FFT against the original
    /// stage-major implementation, over the LDE sizes where the second-level
    /// blocking engages, every production `rate_bits` and both prover field
    /// types. Raw limbs.
    #[test]
    fn cache_blocked_fft_matches_stage_major_reference_at_lde_sizes() {
        use crate::extension::FieldExtension;
        use crate::types::{PrimeField64, Sample};

        fn check<F: Field + Sample, const D: usize>(raw: fn(&[F]) -> Vec<u64>)
        where
            F: FieldExtension<D>,
        {
            for lg_n in 14usize..=19 {
                let n = 1usize << lg_n;
                let roots = fft_root_table::<F>(n);
                for rate_bits in 0..=3usize {
                    let live = n >> rate_bits;
                    let mut coeffs = F::rand_vec(live);
                    coeffs.resize(n, F::ZERO);

                    let mut expected = coeffs.clone();
                    fft_classic_reference(&mut expected, rate_bits, &roots);
                    let expected = raw(&expected);

                    let mut actual = coeffs.clone();
                    fft_classic(&mut actual, rate_bits, &roots);
                    assert_eq!(
                        raw(&actual),
                        expected,
                        "classic entry, 2^{lg_n}, rate_bits {rate_bits}"
                    );
                }
            }
        }

        check::<GoldilocksField, 1>(|values| {
            values.iter().map(|x| x.to_noncanonical_u64()).collect()
        });
        check::<QuadraticExtension<GoldilocksField>, 2>(|values| {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        });
    }

    /// Micro-benchmark for the upper-layer cache blocking (ignored by default).
    /// `cargo test --release -p plonky2_field --lib micro_fft_cache_blocking -- --ignored --nocapture`
    ///
    /// Both parts run three arms rotated through the slot order on every
    /// repetition: the stage-major kernel, a byte-identical *second call to the
    /// same stage-major kernel* (the null, whose spread against the first is
    /// the measurement floor on this box), and the blocked kernel. Reported as
    /// min-of-reps plus the paired win rate over the per-repetition ratios.
    #[cfg(feature = "std")]
    #[test]
    #[ignore]
    fn micro_fft_cache_blocking() {
        use std::time::Instant;

        use crate::fft::{
            L2_BLOCK_BYTES, fft_classic_simd_layers_flat, fft_classic_simd_layers_tuned,
            prepare_zero_padded_fft,
        };
        use crate::types::Sample;

        type F = GoldilocksField;
        type P = <F as Packable>::Packing;
        type Ext2 = QuadraticExtension<GoldilocksField>;

        const REPS: usize = 41;

        /// (A) The layers the blocking rewrites, in isolation, at each layer
        /// boundary the production FFT sites hand over at.
        fn sweep<F: Field + Sample + Packable>(label: &str, start: usize, lg_n: usize) {
            let n = 1usize << lg_n;
            let roots = fft_root_table::<F>(n);
            let mut buf = F::rand_vec(n);
            let iters = ((1usize << 22) / n).max(4);

            let mut best = [f64::MAX; 3];
            let mut ratios: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for rep in 0..REPS {
                let mut t = [0.0f64; 3];
                for slot in 0..3 {
                    let arm = (slot + rep) % 3;
                    let start_time = Instant::now();
                    for _ in 0..iters {
                        if arm == 2 {
                            fft_classic_simd_layers_tuned(
                                <F as Packable>::Packing::pack_slice_mut(&mut buf),
                                start,
                                lg_n,
                                &roots,
                                L2_BLOCK_BYTES,
                            );
                        } else {
                            fft_classic_simd_layers_flat(
                                <F as Packable>::Packing::pack_slice_mut(&mut buf),
                                start,
                                lg_n,
                                &roots,
                            );
                        }
                        core::hint::black_box(&buf);
                    }
                    t[arm] = start_time.elapsed().as_secs_f64() / iters as f64;
                }
                if rep > 1 {
                    for k in 0..3 {
                        best[k] = best[k].min(t[k]);
                    }
                    for k in 0..2 {
                        ratios[k].push(t[0] / t[k + 1]);
                    }
                }
            }
            let labels = ["null(flat)", "blocked"];
            print!("{label:<30} flat={:9.1}us", best[0] * 1e6);
            for k in 0..2 {
                ratios[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let wins = ratios[k].iter().filter(|r| **r > 1.0).count();
                print!(
                    "  {}={:9.1}us min={:5.3}x med-paired={:5.3}x wins {wins}/{}",
                    labels[k],
                    best[k + 1] * 1e6,
                    best[0] / best[k + 1],
                    ratios[k][ratios[k].len() / 2],
                    ratios[k].len()
                );
            }
            println!();
        }

        println!("-- rewritten layers only --");
        // LDE FFT, 172 per proof: zero-padded expansion stops at layer 13.
        sweep::<F>("lde fft   base 2^19 from 13", 13, 19);
        // Quotient coset IFFT, 2 per proof: no zero padding, starts at the
        // packing width.
        sweep::<F>("quotient  base 2^19 from  2", 2, 19);
        // FRI final-polynomial coset FFT over the extension field, which the
        // `P::WIDTH == 1` gate keeps on the flat path.
        sweep::<Ext2>("fri coset ext2 2^19 from 12", 12, 19);

        // (B) The full LDE column pipeline at the shipped block size: prefix
        // copy, zero-padded expansion with L1-blocked low layers, then the
        // upper layers.
        println!("-- full lde column pipeline (rate_bits = 3) --");
        let lg_packed_width = log2_strict(P::WIDTH);
        for lg_degree in [14usize, 16] {
            let rate_bits = 3usize;
            let lg_n = lg_degree + rate_bits;
            let n = 1usize << lg_n;
            let degree = 1usize << lg_degree;
            let roots = fft_root_table::<F>(n);
            let coeffs = F::rand_vec(degree);
            let mut buf: Vec<F> = Vec::with_capacity(n);
            unsafe { buf.set_len(n) };
            let iters = ((1usize << 22) / n).max(4);

            let mut best = [f64::MAX; 3];
            let mut ratios: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut sink = 0u64;
            for rep in 0..REPS {
                let mut t = [0.0f64; 3];
                for slot in 0..3 {
                    let arm = (slot + rep) % 3;
                    let start = Instant::now();
                    for _ in 0..iters {
                        buf[..degree].copy_from_slice(&coeffs);
                        let first = prepare_zero_padded_fft(
                            &mut buf,
                            rate_bits,
                            lg_n,
                            lg_packed_width,
                            &roots,
                        );
                        if arm == 2 {
                            fft_classic_simd_layers_tuned(
                                P::pack_slice_mut(&mut buf),
                                first,
                                lg_n,
                                &roots,
                                L2_BLOCK_BYTES,
                            );
                        } else {
                            fft_classic_simd_layers_flat(
                                P::pack_slice_mut(&mut buf),
                                first,
                                lg_n,
                                &roots,
                            );
                        }
                        core::hint::black_box(&buf);
                    }
                    t[arm] = start.elapsed().as_secs_f64() / iters as f64;
                    sink ^= buf[0].0;
                }

                if rep > 1 {
                    for k in 0..3 {
                        best[k] = best[k].min(t[k]);
                    }
                    for k in 0..2 {
                        ratios[k].push(t[0] / t[k + 1]);
                    }
                }
            }
            let labels = ["null(flat)", "blocked"];
            print!("deg=2^{lg_degree} lde=2^{lg_n}  flat={:9.1}us", best[0] * 1e6);
            for k in 0..2 {
                ratios[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let wins = ratios[k].iter().filter(|r| **r > 1.0).count();
                print!(
                    "  {}={:9.1}us min={:5.3}x med-paired={:5.3}x wins {wins}/{}",
                    labels[k],
                    best[k + 1] * 1e6,
                    best[0] / best[k + 1],
                    ratios[k][ratios[k].len() / 2],
                    ratios[k].len()
                );
            }
            println!("  sink={}", sink & 1);
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

    /// `fft_with_options_prefix_bit_reversed` must return exactly what
    /// `fft_with_options` returns on the same (unpermuted) coefficients, once
    /// the caller has applied the permutation the FFT would have applied
    /// itself. Swept over both dispatch paths of `reverse_index_bits_in_place`
    /// (2^11..2^13 small, 2^14..2^17 chunked, odd exponents hitting the
    /// two-transpose case), every production `rate_bits`, and both the base
    /// field and the quadratic extension. Compared on raw limbs. The
    /// in-place entry point is checked alongside, against the same reference.
    #[test]
    fn prefix_bit_reversed_fft_matches_classic() {
        use plonky2_util::fill_bit_reversed;

        use crate::extension::FieldExtension;
        use crate::fft::fft_in_place_with_options_prefix_bit_reversed;
        use crate::types::{PrimeField64, Sample};

        fn check<F: Field + Sample, const D: usize>(raw: fn(&[F]) -> Vec<u64>)
        where
            F: FieldExtension<D>,
        {
            for lg_n in [11usize, 12, 13, 14, 15, 16, 17] {
                let n = 1usize << lg_n;
                for rate_bits in 0..=3usize {
                    // The FFT reverses `coeffs[..n >> rate_bits]`; the rest is
                    // the zero tail it never reads.
                    let live = n >> rate_bits;
                    let mut coeffs = F::rand_vec(live);
                    coeffs.resize(n, F::ZERO);

                    let expected = PolynomialCoeffs::new(coeffs.clone())
                        .fft_with_options(Some(rate_bits), None)
                        .values;

                    let mut prereversed = coeffs.clone();
                    fill_bit_reversed(&mut prereversed[..live], |out, start| {
                        out.copy_from_slice(&coeffs[start..start + out.len()]);
                    });

                    let actual = PolynomialCoeffs::new(prereversed.clone())
                        .fft_with_options_prefix_bit_reversed(Some(rate_bits), None)
                        .values;
                    assert_eq!(
                        raw(&actual),
                        raw(&expected),
                        "owned entry, 2^{lg_n}, rate_bits {rate_bits}"
                    );

                    let mut in_place = prereversed;
                    fft_in_place_with_options_prefix_bit_reversed(
                        &mut in_place,
                        Some(rate_bits),
                        None,
                    );
                    assert_eq!(
                        raw(&in_place),
                        raw(&expected),
                        "in-place entry, 2^{lg_n}, rate_bits {rate_bits}"
                    );
                }
            }
        }

        check::<GoldilocksField, 1>(|values| {
            values.iter().map(|x| x.to_noncanonical_u64()).collect()
        });
        check::<QuadraticExtension<GoldilocksField>, 2>(|values| {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        });
    }
}
