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

/// Elementwise `out[i] = a[i] * b[i]`, writing (never reading) `out`.
///
/// This is the write-into form of [`batch_multiply_inplace`]: a caller that
/// would otherwise copy `a` into `out` and then multiply in place gets the same
/// values from a single pass, deleting one full read+write sweep of `out`.
///
/// Value-exactness: the packed/leftover split point is a function of the shared
/// length alone, so it is the same one [`batch_multiply_inplace`] would pick,
/// and each element is produced by the same `Mul` (packed or scalar) applied to
/// the same two operands. Results are bit-identical to copy-then-multiply.
///
/// `out` may point at uninitialized memory: every element is written and none
/// is read.
pub fn batch_multiply_to<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "output and first input must have the same length");
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
        *x_out = *x_a * *x_b;
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out = *x_a * *x_b;
    }
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

    /// `batch_multiply_to` is the fused form of "copy `a`, then multiply in
    /// place by `b`". It must be bit-identical (raw u64 limbs) to that pair of
    /// passes at every length, in particular across the packed/leftover split
    /// where one implementation could use a packed `Mul` and the other a
    /// scalar one. Covered for the base field and the quadratic extension,
    /// with lengths spanning zero, sub-vector, exact multiples, and the
    /// production LDE prefix sizes 2^11..2^17.
    #[test]
    fn batch_multiply_to_matches_copy_then_inplace() {
        use crate::extension::FieldExtension;
        use crate::extension::quadratic::QuadraticExtension;
        use crate::types::{PrimeField64, Sample};

        fn raw_base(values: &[GoldilocksField]) -> Vec<u64> {
            values.iter().map(|x| x.to_noncanonical_u64()).collect()
        }
        fn raw_ext(values: &[QuadraticExtension<GoldilocksField>]) -> Vec<u64> {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        }

        let mut lengths = vec![0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 255, 257];
        lengths.extend((11..=17).map(|lg_n| 1usize << lg_n));
        lengths.extend((11..=13).map(|lg_n| (1usize << lg_n) + 1));

        for n in lengths {
            let a = GoldilocksField::rand_vec(n);
            let b = GoldilocksField::rand_vec(n);
            let mut expected = a.clone();
            batch_multiply_inplace(&mut expected, &b);
            let mut actual = vec![GoldilocksField::ZERO; n];
            batch_multiply_to(&mut actual, &a, &b);
            assert_eq!(raw_base(&actual), raw_base(&expected), "base field, n = {n}");

            let a = QuadraticExtension::<GoldilocksField>::rand_vec(n);
            let b = QuadraticExtension::<GoldilocksField>::rand_vec(n);
            let mut expected = a.clone();
            batch_multiply_inplace(&mut expected, &b);
            let mut actual = vec![QuadraticExtension::<GoldilocksField>::ZERO; n];
            batch_multiply_to(&mut actual, &a, &b);
            assert_eq!(raw_ext(&actual), raw_ext(&expected), "extension, n = {n}");
        }
    }

    /// Micro-benchmark for the LDE prefix pipeline (ignored by default).
    /// `cargo test --release -p plonky2_field --lib micro_lde_prefix -- --ignored --nocapture`
    ///
    /// Four arms rotated through the slot order every repetition: (A) the
    /// pre-fusion `copy -> multiply in place -> bit-reverse`, (A') a
    /// byte-identical second copy of A (the null, whose spread against A is the
    /// measurement floor on this box), (B) `batch_multiply_to` then
    /// bit-reverse, and (C) the shipped `fill_bit_reversed` + `batch_multiply_to`.
    #[cfg(feature = "std")]
    #[test]
    #[ignore]
    fn micro_lde_prefix() {
        use std::time::Instant;

        use plonky2_util::reverse_index_bits_in_place;

        use crate::types::Sample;

        type F = GoldilocksField;
        const REPS: usize = 41;

        for lg_n in [14usize, 16] {
            let n = 1usize << lg_n;
            let coeffs = F::rand_vec(n);
            let powers = F::rand_vec(n);
            let lde_len = n << 3;

            // One reused LDE-sized buffer: the allocation is not what is under
            // test, and a fresh 4 MiB mapping per iteration would swamp the
            // pass costs with page faults.
            let mut buf: Vec<F> = Vec::with_capacity(lde_len);
            unsafe { buf.set_len(lde_len) };

            let iters = ((1usize << 22) / n).max(4);

            let mut best = [f64::MAX; 4];
            let mut ratios: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut sink = 0u64;

            for rep in 0..REPS {
                let mut t = [0.0f64; 4];
                for slot in 0..4 {
                    let arm = (slot + rep) % 4;
                    let start = Instant::now();
                    for _ in 0..iters {
                        match arm {
                            2 => {
                                batch_multiply_to(&mut buf[..n], &coeffs, &powers);
                                reverse_index_bits_in_place(&mut buf[..n]);
                            }
                            3 => {
                                plonky2_util::fill_bit_reversed(&mut buf[..n], |out, start| {
                                    let len = out.len();
                                    batch_multiply_to(
                                        out,
                                        &coeffs[start..start + len],
                                        &powers[start..start + len],
                                    );
                                });
                            }
                            // Arms 0 and 1 are the same pre-fusion body.
                            _ => {
                                buf[..n].copy_from_slice(&coeffs);
                                batch_multiply_inplace(&mut buf[..n], &powers);
                                reverse_index_bits_in_place(&mut buf[..n]);
                            }
                        }
                        core::hint::black_box(&buf);
                    }
                    t[arm] = start.elapsed().as_secs_f64() / iters as f64;
                    sink = sink.wrapping_add(buf[0].0 ^ buf[n - 1].0);
                }

                if rep > 1 {
                    for k in 0..4 {
                        best[k] = best[k].min(t[k]);
                    }
                    for k in 0..3 {
                        ratios[k].push(t[0] / t[k + 1]);
                    }
                }
            }
            let labels = ["null(pre)", "B(mul_to)", "C(shipped)"];
            print!("n=2^{lg_n:<2}  A(pre)={:8.2}us", best[0] * 1e6);
            for k in 0..3 {
                ratios[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let wins = ratios[k].iter().filter(|r| **r > 1.0).count();
                print!(
                    "  {}={:8.2}us min={:5.3}x med-paired={:5.3}x wins {wins}/{}",
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
