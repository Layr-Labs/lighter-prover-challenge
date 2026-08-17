use core::mem::size_of;

use plonky2_util::log2_strict;

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::types::Field;

const fn pack_with_leftovers_split_point<P: PackedField>(slice: &[P::Scalar]) -> usize {
    let n = slice.len();
    let n_leftover = n % P::WIDTH;
    n - n_leftover
}

fn pack_slice_with_leftovers<P: PackedField>(slice: &[P::Scalar]) -> (&[P], &[P::Scalar]) {
    let split_point = pack_with_leftovers_split_point::<P>(slice);
    let (slice_packable, slice_leftovers) = slice.split_at(split_point);
    let slice_packed = P::pack_slice(slice_packable);
    (slice_packed, slice_leftovers)
}

fn pack_slice_with_leftovers_mut<P: PackedField>(
    slice: &mut [P::Scalar],
) -> (&mut [P], &mut [P::Scalar]) {
    let split_point = pack_with_leftovers_split_point::<P>(slice);
    let (slice_packable, slice_leftovers) = slice.split_at_mut(split_point);
    let slice_packed = P::pack_slice_mut(slice_packable);
    (slice_packed, slice_leftovers)
}

/// Elementwise inplace multiplication of two slices of field elements.
/// Implementation be faster than the trivial for loop.
pub fn batch_multiply_inplace<F: Field>(out: &mut [F], a: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "both arrays must have the same length");

    // Split out slice of vectors, leaving leftovers as scalars
    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);

    // Multiply packed and the leftovers
    for (x_out, x_a) in out_packed.iter_mut().zip(a_packed) {
        *x_out *= *x_a;
    }
    for (x_out, x_a) in out_leftovers.iter_mut().zip(a_leftovers) {
        *x_out *= *x_a;
    }
}
/// Elementwise `out[i] = a[i] * b[i]`, writing a destination that is not also an
/// input.
///
/// The LDE fill previously reached its coset-scaled state in two passes over
/// `degree` words: `copy_from_slice` (read `a`, write `out`) followed by
/// `batch_multiply_inplace` (read `out`, read `b`, write `out`). The
/// intermediate unscaled copy is never observed — the FFT only ever sees the
/// scaled values — so the copy is a materialization that can be deleted by
/// folding the multiply into the same pass.
///
/// The packed/scalar split is identical to `batch_multiply_inplace`'s: the
/// maximal `P::WIDTH` prefix uses packed multiplication and the ragged tail uses
/// the same scalar operation, so every produced word is bit-identical to the
/// two-pass form.
pub fn batch_multiply_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "output and first input must have the same length");
    assert_eq!(n, b.len(), "output and second input must have the same length");

    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);
    let (b_packed, b_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(b);

    for ((x_out, x_a), x_b) in out_packed.iter_mut().zip(a_packed).zip(b_packed) {
        *x_out = *x_a * *x_b;
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out = *x_a * *x_b;
    }
}

/// Reverse the low `lb` bits of `i`, discarding the rest.
#[inline(always)]
fn reverse_low_bits(i: usize, lb: usize) -> usize {
    // `wrapping_shr` handles `lb == 0`: there `i == 0`, so the answer is `0`,
    // and a shift by `usize::BITS` is a no-op that happens to give it.
    i.reverse_bits().wrapping_shr(usize::BITS - lb as u32)
}

/// At or below this many bytes the trivial gather wins outright and the
/// blocked one only pays for its buffer round trip.
///
/// `plonky2_util::reverse_index_bits_in_place` puts the same boundary at
/// 64 KiB, but it is choosing between a trivial loop and an *in-place*
/// transpose, which moves each element once. The blocked gather moves each
/// element twice -- into the tile buffer and out of it -- so it needs a bigger
/// array before the locality it buys covers the extra pass. Measured on the
/// M3 Max, blocked/naive is 1.54x at 128 KiB, 1.03x at 256 KiB and 0.66x at
/// 512 KiB; see `gatherblk-notes.md`.
const BLOCKED_GATHER_MIN_BYTES: usize = 1 << 18;

/// Byte budget for the tile buffer the blocked gather transposes through. It
/// is read and written `TILE` times per tile, so it has to stay in L1; 32 KiB
/// fits the smallest L1 data cache on the cores this runs on. At 8-byte
/// elements this admits `TILE = 64`, which is also where the measured sweep
/// peaks -- `TILE = 128` overflows L1 and gives back half the win.
const GATHER_TILE_BYTES: usize = 1 << 15;

/// Tile width the dispatch uses, as a log2.
const GATHER_LB_TILE: usize = 6;

/// `out[i] = <the run-filler's value at rev(i)>` in `TILE x TILE` tiles, where
/// `rev` reverses the low `lb_n` bits of an index.
///
/// The naive form of this gather -- walk `out` in order, read `a[rev(i)]` --
/// writes sequentially but reads a new cache line on *every* element, because
/// consecutive destinations differ in the low index bit and so their sources
/// differ in the high one. Each source line is eventually used for all eight
/// of its words, but the reuse distance is `n/8` elements, so at a production
/// LDE prefix (512 KiB at `degree = 2^16`, 2 MiB at `2^18`) the line has long
/// been evicted from L1 and the pass pulls 8x its own size out of L2.
///
/// The blocking is the same decomposition `plonky2_util`'s cache-oblivious
/// arm uses, adapted from an in-place swap to a copy. Split the index into
/// three fields, `i = (h, m, l)` with `h` and `l` `LB_TILE` bits wide; then
///
/// ```text
/// rev(h, m, l) = (rev(l), rev(m), rev(h))
/// ```
///
/// so for a fixed `m` the sources form `TILE` runs of `TILE` contiguous
/// elements (one run per `l`, walked by `rev(h)`) and the destinations form
/// `TILE` runs of `TILE` contiguous elements (one run per `h`, walked by `l`).
/// A `TILE x TILE` buffer absorbs the transpose between them, so both sides of
/// the permutation move in whole cache lines and every line fetched is fully
/// consumed before it is dropped.
///
/// The traversal advances `m` in address order, which walks the destination
/// bases forward; running the outer loop over `rev(m)` instead -- walking the
/// *source* bases forward -- measured identical at `TILE = 64`, where a run is
/// already eight cache lines and the run, not the order of the runs, is what
/// the prefetcher sees.
///
/// This reorders independent `(src, dst)` pairs and nothing else: the same
/// `fill_run` produces the same value for the same source index, and each
/// destination is still written exactly once. Every word is bit-identical to
/// the naive loop's, not merely congruent.
///
/// `fill_run(dst, src)` must fill all of `dst` from `src..src + dst.len()`.
#[inline]
fn blocked_bit_reversed_gather<F: Field, const LB_TILE: usize, const BUF: usize>(
    out: &mut [F],
    lb_n: usize,
    fill_run: impl Fn(&mut [F], usize),
) {
    debug_assert_eq!(BUF, 1 << (2 * LB_TILE));
    debug_assert!(lb_n >= 2 * LB_TILE);
    debug_assert_eq!(out.len(), 1 << lb_n);

    let tile = 1usize << LB_TILE;
    let lb_mid = lb_n - 2 * LB_TILE;
    let row = 1usize << (lb_n - LB_TILE);
    let mut buf = [F::ZERO; BUF];

    for m in 0..1usize << lb_mid {
        let rm = reverse_low_bits(m, lb_mid);
        for l in 0..tile {
            let src = (reverse_low_bits(l, LB_TILE) << (lb_n - LB_TILE)) + (rm << LB_TILE);
            fill_run(&mut buf[l << LB_TILE..(l + 1) << LB_TILE], src);
        }
        for h in 0..tile {
            let dst = h * row + (m << LB_TILE);
            let rh = reverse_low_bits(h, LB_TILE);
            for l in 0..tile {
                // SAFETY: `dst + l` is `(h, m, l)` reassembled, which is an
                // `lb_n`-bit value and so below `out.len()`; `(l << LB_TILE) +
                // rh` is below `1 << 2 * LB_TILE == BUF`.
                unsafe {
                    *out.get_unchecked_mut(dst + l) = *buf.get_unchecked((l << LB_TILE) + rh);
                }
            }
        }
    }
}

/// `out[i] = <the run-filler's value at rev(i)>`, blocked when the array is
/// big enough for the blocking to pay, and a trivial reversed-index loop
/// otherwise.
///
/// An element too wide for a `GATHER_LB_TILE` tile to fit the L1 budget also
/// keeps the trivial loop. That case has no caller -- both fused writers are
/// reached only from the base-field LDE fill -- and the tile width that would
/// suit it was not measured, so it is left on the arm that was.
#[inline(always)]
fn bit_reversed_gather_into<F: Field>(out: &mut [F], fill_run: impl Fn(&mut [F], usize)) {
    let n = out.len();
    if n <= 1 {
        if n == 1 {
            fill_run(out, 0);
        }
        return;
    }
    let lb_n = log2_strict(n);

    if size_of::<F>() << lb_n > BLOCKED_GATHER_MIN_BYTES
        && size_of::<F>() << (2 * GATHER_LB_TILE) <= GATHER_TILE_BYTES
        && lb_n >= 2 * GATHER_LB_TILE
    {
        blocked_bit_reversed_gather::<F, GATHER_LB_TILE, { 1 << (2 * GATHER_LB_TILE) }>(
            out, lb_n, fill_run,
        );
    } else {
        // AArch64 reverses bits in one instruction, so the index can be
        // computed per element; the chunked form in `plonky2_util` exists to
        // keep x86 from re-reversing per element, and buys nothing here.
        let shift = usize::BITS - lb_n as u32;
        for (i, x_out) in out.iter_mut().enumerate() {
            let src = i.reverse_bits() >> shift;
            fill_run(core::slice::from_mut(x_out), src);
        }
    }
}

/// Elementwise `out[i] = a[rev(i)] * b[rev(i)]`, where `rev` reverses the low
/// `log2(out.len())` bits of an index.
///
/// This is `batch_multiply_into` with the bit-reversal permutation that
/// `fft_classic`'s zero-padded prologue would otherwise apply to the product in
/// a second read+write pass over the live prefix. The permutation is applied by
/// [`bit_reversed_gather_into`], so at a production prefix it is the blocked
/// traversal and not a naive gather -- the pass it replaces was itself the
/// blocked, cache-oblivious arm of `reverse_index_bits_in_place`, and a naive
/// gather gives back more locality than the deleted read+write pass was worth.
///
/// Every produced word is the word the two-pass form produced, at the same
/// index: `a[j] * b[j]` is the identical scalar multiplication and only its
/// destination moved. `batch_multiply_into`'s packed prefix is not reproduced
/// -- permuted destinations are not contiguous -- but that costs no
/// representative: `WideGoldilocksField`'s multiplication is two independent
/// `NeonGoldilocksField` lane pairs over `mul_reduce_pair`, whose assembly is
/// documented to compute the same intermediates, and therefore the same
/// non-canonical `u64`, as the scalar `reduce128` that `Mul for
/// GoldilocksField` reduces through.
pub fn bit_reversed_multiply_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "output and first input must have the same length");
    assert_eq!(n, b.len(), "output and second input must have the same length");

    bit_reversed_gather_into(out, |run, src| {
        // SAFETY: every source run the gather asks for lies in `0..n`, and `a`
        // and `b` were asserted to have length `n` above.
        let (x_a, x_b) = unsafe {
            (
                a.get_unchecked(src..src + run.len()),
                b.get_unchecked(src..src + run.len()),
            )
        };
        for ((x_out, x_a), x_b) in run.iter_mut().zip(x_a).zip(x_b) {
            *x_out = *x_a * *x_b;
        }
    });
}

/// Elementwise `out[i] = a[rev(i)]`: [`bit_reversed_multiply_into`] with no
/// second factor, for callers whose scaling has been folded into the twiddles
/// and who only need the permutation the FFT prologue would have applied.
pub fn bit_reversed_copy_into<F: Field>(out: &mut [F], a: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "output and input must have the same length");

    bit_reversed_gather_into(out, |run, src| {
        // SAFETY: as in `bit_reversed_multiply_into`.
        run.copy_from_slice(unsafe { a.get_unchecked(src..src + run.len()) });
    });
}

/// Elementwise multiply two slices and add the products to an output slice.
pub fn batch_multiply_add_inplace<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(
        n,
        a.len(),
        "output and first input must have the same length"
    );
    assert_eq!(
        n,
        b.len(),
        "output and second input must have the same length"
    );

    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);
    let (b_packed, b_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(b);

    for ((x_out, x_a), x_b) in out_packed.iter_mut().zip(a_packed).zip(b_packed) {
        *x_out = x_out.multiply_accumulate(*x_a, *x_b);
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out += *x_a * *x_b;
    }
}

/// Elementwise inplace addition of two slices of field elements.
/// Implementation be faster than the trivial for loop.
pub fn batch_add_inplace<F: Field>(out: &mut [F], a: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "both arrays must have the same length");

    // Split out slice of vectors, leaving leftovers as scalars
    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);

    // Add packed and the leftovers
    for (x_out, x_a) in out_packed.iter_mut().zip(a_packed) {
        *x_out += *x_a;
    }
    for (x_out, x_a) in out_leftovers.iter_mut().zip(a_leftovers) {
        *x_out += *x_a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks_field::GoldilocksField;

    /// Q5: the fused `copy + scale` must be raw-`u64`-identical to the
    /// two-pass `copy_from_slice` + `batch_multiply_inplace` it replaces, at
    /// every length across the packed/leftover split, and it must assign
    /// every destination slot (the destination is uninitialized).
    #[test]
    fn batch_multiply_into_matches_copy_then_inplace() {
        const POISON: GoldilocksField = GoldilocksField(u64::MAX);
        for n in 0..40usize {
            let a = (0..n)
                .map(|i| GoldilocksField::from_canonical_usize(2 * i + 3))
                .collect::<Vec<_>>();
            let b = (0..n)
                .map(|i| GoldilocksField::from_canonical_usize(3 * i + 5))
                .collect::<Vec<_>>();
            let mut reference = vec![POISON; n];
            reference.copy_from_slice(&a);
            batch_multiply_inplace(&mut reference, &b);

            let mut fused = vec![POISON; n];
            batch_multiply_into(&mut fused, &a, &b);

            for i in 0..n {
                assert_ne!(fused[i].0, POISON.0, "slot {i} of {n} never written");
                assert_eq!(fused[i].0, reference[i].0, "slot {i} of {n} differs");
            }
        }
    }

    /// Sabotage control: a fused writer that stopped at the packed prefix —
    /// i.e. dropped the ragged tail — must be caught by the sweep above.
    #[test]
    fn fused_multiply_differential_catches_a_dropped_tail() {
        const POISON: GoldilocksField = GoldilocksField(u64::MAX);
        let n = 11usize;
        let a = (0..n)
            .map(|i| GoldilocksField::from_canonical_usize(2 * i + 3))
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|i| GoldilocksField::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let mut sabotaged = vec![POISON; n];
        let split = n - n % <GoldilocksField as Packable>::Packing::WIDTH;
        batch_multiply_into(&mut sabotaged[..split], &a[..split], &b[..split]);
        assert!(
            sabotaged.iter().any(|x| x.0 == POISON.0),
            "sweep failed to notice a dropped ragged tail"
        );
    }

    /// Non-canonical raw words, so the differentials below are comparing
    /// representatives and not just values.
    fn raw_words(n: usize, salt: u64) -> Vec<GoldilocksField> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
                x ^= x >> 29;
                GoldilocksField(x)
            })
            .collect()
    }

    /// Every length the gather's size dispatch can route differently.
    ///
    /// `0..12` is the packed/leftover and `n <= 1` ground the fused writers
    /// always had to cover. `15` is exactly `BLOCKED_GATHER_MIN_BYTES`
    /// (8 bytes x 2^15 = 256 KiB), the last shape on the trivial arm, and `16`
    /// is the first on the blocked arm -- and both are production LDE
    /// prefixes, `16` being the ranked circuit's `degree`. `18` is the block
    /// circuit's, and `14` is a shape the blocked arm deliberately declines.
    const GATHER_SHAPES: &[usize] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    ];

    /// `bit_reversed_multiply_into` produces exactly the words
    /// `batch_multiply_into` produces, permuted by the index bit reversal that
    /// the zero-padded FFT prologue would have applied in a second pass.
    /// Compared raw, not canonically: the point of the fusion is that no
    /// representative moves.
    ///
    /// The blocked traversal only reorders independent `(src, dst)` pairs, so
    /// it has to hold at exactly the same raw-word standard as the naive one
    /// did; the sweep straddles the size dispatch so both arms are pinned.
    #[test]
    fn bit_reversed_multiply_matches_permuted_batch_multiply() {
        for &lb_n in GATHER_SHAPES {
            let n = 1usize << lb_n;
            let a = raw_words(n, 0);
            let b = raw_words(n, 0xD1B5_4A32_D192_ED03);

            let mut two_pass = (0..n).map(|_| GoldilocksField::ZERO).collect::<Vec<_>>();
            batch_multiply_into(&mut two_pass, &a, &b);
            plonky2_util::reverse_index_bits_in_place(&mut two_pass);

            let mut fused = (0..n).map(|_| GoldilocksField::ZERO).collect::<Vec<_>>();
            bit_reversed_multiply_into(&mut fused, &a, &b);

            assert_eq!(
                two_pass.iter().map(|x| x.0).collect::<Vec<_>>(),
                fused.iter().map(|x| x.0).collect::<Vec<_>>(),
                "fused bit-reversed multiply diverged at 2^{lb_n}"
            );
        }
    }

    /// `bit_reversed_copy_into` is the same permutation as
    /// `bit_reversed_multiply_into`, without the second factor.
    #[test]
    fn bit_reversed_copy_matches_permuted_copy() {
        for &lb_n in GATHER_SHAPES {
            let n = 1usize << lb_n;
            let a = raw_words(n, 0);

            let mut two_pass = a.clone();
            plonky2_util::reverse_index_bits_in_place(&mut two_pass);

            let mut fused = (0..n).map(|_| GoldilocksField::ZERO).collect::<Vec<_>>();
            bit_reversed_copy_into(&mut fused, &a);

            assert_eq!(
                two_pass.iter().map(|x| x.0).collect::<Vec<_>>(),
                fused.iter().map(|x| x.0).collect::<Vec<_>>(),
                "fused bit-reversed copy diverged at 2^{lb_n}"
            );
        }
    }

    /// The blocking is a bit reversal at every tile width, not only at the one
    /// the dispatch selects.
    ///
    /// `GATHER_LB_TILE` is a locality knob picked by measurement, so it is the
    /// kind of constant that gets retuned; pinning three widths against the
    /// naive gather says a retune cannot change what the function computes.
    /// It also drives the odd `lb_n` cases, where the middle field has an odd
    /// number of bits, at every width.
    #[test]
    fn every_tile_width_matches_the_naive_gather() {
        for lb_n in 8..17usize {
            let n = 1usize << lb_n;
            let a = raw_words(n, 0x5DEE_CE66_D1B5_4A32);
            let copy_run = |run: &mut [GoldilocksField], src: usize| {
                run.copy_from_slice(&a[src..src + run.len()]);
            };

            let mut naive = a.clone();
            plonky2_util::reverse_index_bits_in_place(&mut naive);
            let naive = naive.iter().map(|x| x.0).collect::<Vec<_>>();

            let mut tile16 = vec![GoldilocksField::ZERO; n];
            blocked_bit_reversed_gather::<_, 4, 256>(&mut tile16, lb_n, copy_run);
            assert_eq!(
                naive,
                tile16.iter().map(|x| x.0).collect::<Vec<_>>(),
                "tile 16 gather diverged at 2^{lb_n}"
            );

            if lb_n >= 10 {
                let mut tile32 = vec![GoldilocksField::ZERO; n];
                blocked_bit_reversed_gather::<_, 5, 1024>(&mut tile32, lb_n, copy_run);
                assert_eq!(
                    naive,
                    tile32.iter().map(|x| x.0).collect::<Vec<_>>(),
                    "tile 32 gather diverged at 2^{lb_n}"
                );
            }

            if lb_n >= 12 {
                let mut tile64 = vec![GoldilocksField::ZERO; n];
                blocked_bit_reversed_gather::<_, 6, 4096>(&mut tile64, lb_n, copy_run);
                assert_eq!(
                    naive,
                    tile64.iter().map(|x| x.0).collect::<Vec<_>>(),
                    "tile 64 gather diverged at 2^{lb_n}"
                );
            }
        }
    }

    /// Sabotage control for the blocking: the tile transpose is what makes the
    /// traversal a bit reversal rather than a shuffle, so dropping the tile's
    /// own index reversal — reading `buf[l][h]` where the gather reads
    /// `buf[l][rev(h)]` — must be caught by the sweeps above.
    ///
    /// Without this, a blocked gather that happened to be the identity on the
    /// tile (which it is at `TILE = 1` and `TILE = 2`) would look verified.
    #[test]
    fn blocked_gather_differential_catches_an_unreversed_tile() {
        const LB_N: usize = 14;
        const LB_TILE: usize = 4;
        let n = 1usize << LB_N;
        let a = raw_words(n, 0x2545_F491_4F6C_DD1D);

        let mut want = a.clone();
        plonky2_util::reverse_index_bits_in_place(&mut want);

        // `blocked_bit_reversed_gather` with `rev(h)` replaced by `h`.
        let tile = 1usize << LB_TILE;
        let lb_mid = LB_N - 2 * LB_TILE;
        let row = 1usize << (LB_N - LB_TILE);
        let mut sabotaged = vec![GoldilocksField::ZERO; n];
        let mut buf = vec![GoldilocksField::ZERO; tile * tile];
        for m in 0..1usize << lb_mid {
            let rm = reverse_low_bits(m, lb_mid);
            for l in 0..tile {
                let src = (reverse_low_bits(l, LB_TILE) << (LB_N - LB_TILE)) + (rm << LB_TILE);
                buf[l << LB_TILE..(l + 1) << LB_TILE].copy_from_slice(&a[src..src + tile]);
            }
            for h in 0..tile {
                let dst = h * row + (m << LB_TILE);
                for l in 0..tile {
                    sabotaged[dst + l] = buf[(l << LB_TILE) + h];
                }
            }
        }

        assert_ne!(
            want.iter().map(|x| x.0).collect::<Vec<_>>(),
            sabotaged.iter().map(|x| x.0).collect::<Vec<_>>(),
            "an unreversed tile produced the bit-reversal anyway"
        );
    }

    #[test]
    fn batch_multiply_add_matches_scalar_with_packed_leftovers() {
        let mut out = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(i + 1))
            .collect::<Vec<_>>();
        let a = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(2 * i + 3))
            .collect::<Vec<_>>();
        let b = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let expected = out
            .iter()
            .zip(&a)
            .zip(&b)
            .map(|((&x_out, &x_a), &x_b)| x_out + x_a * x_b)
            .collect::<Vec<_>>();

        batch_multiply_add_inplace(&mut out, &a, &b);

        assert_eq!(out, expected);
    }
}

