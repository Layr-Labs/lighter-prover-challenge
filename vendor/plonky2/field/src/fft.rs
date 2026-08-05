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

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    let computed_root_table = root_table.is_none().then(|| fft_root_table(input.len()));
    let used_root_table = root_table.or(computed_root_table.as_ref()).unwrap();

    fft_classic(input, zero_factor.unwrap_or(0), used_root_table);
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
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);

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

    for lg_half_m in s..lg_n {
        let lg_m = lg_half_m + 1;
        let m = 1 << lg_m; // Subarray size (in field elements).
        let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
        let half_packed_m = packed_m / 2;
        debug_assert!(half_packed_m != 0);

        // omega values for this iteration, as slice of vectors
        let omega_table = P::pack_slice(&root_table[lg_half_m][..]);
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

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
fn reverse_index_bits_and_expand_zero_tail<T: Copy>(values: &mut [T], r: usize, lg_n: usize) {
    if r > 0 && r <= lg_n {
        let active_len = values.len() >> r;
        let block_len = 1 << r;

        // For every active index i, rev_lg_n(i) = rev_(lg_n-r)(i) << r because its
        // top r bits are zero. Bit-reverse only that active prefix, then expand it
        // backwards so no unread prefix element can be overwritten.
        reverse_index_bits_in_place(&mut values[..active_len]);
        for i in (0..active_len).rev() {
            let value = values[i];
            values[i * block_len..(i + 1) * block_len].fill(value);
        }
    } else {
        // Preserve the original path for r == 0 and invalid hints.
        reverse_index_bits_in_place(values);
        if r > 0 {
            let mask = !((1 << r) - 1);
            for i in 0..values.len() {
                values[i] = values[i & mask];
            }
        }
    }
}

pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    let n = values.len();
    let lg_n = log2_strict(n);
    reverse_index_bits_and_expand_zero_tail(values, r, lg_n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd::<F>(values, r, lg_n, root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(values, r, lg_n, root_table);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use plonky2_util::{log2_ceil, log2_strict, reverse_index_bits_in_place};

    use super::{fft_classic, fft_classic_simd, reverse_index_bits_and_expand_zero_tail};
    use crate::fft::{fft, fft_root_table, fft_with_options, ifft, FftRootTable};
    use crate::goldilocks_field::GoldilocksField;
    use crate::packable::Packable;
    use crate::packed::PackedField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;

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

    fn full_bit_reversal_reference<T: Copy>(values: &mut [T], r: usize) {
        reverse_index_bits_in_place(values);
        if r > 0 {
            let mask = !((1 << r) - 1);
            for i in 0..values.len() {
                values[i] = values[i & mask];
            }
        }
    }

    fn fft_classic_full_bit_reversal_reference<F: Field>(
        values: &mut [F],
        r: usize,
        root_table: &FftRootTable<F>,
    ) {
        full_bit_reversal_reference(values, r);
        let lg_n = log2_strict(values.len());
        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        if lg_n <= lg_packed_width {
            fft_classic_simd::<F>(values, r, lg_n, root_table);
        } else {
            fft_classic_simd::<<F as Packable>::Packing>(values, r, lg_n, root_table);
        }
    }

    fn deterministic_goldilocks_values(len: usize) -> Vec<GoldilocksField> {
        (0..len)
            .map(|i| {
                GoldilocksField::from_noncanonical_u64(
                    (i as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left((i % 64) as u32)
                        .wrapping_add(0xffff_ffff_0000_0000),
                )
            })
            .collect()
    }

    fn raw_goldilocks(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|value| value.0).collect()
    }

    #[test]
    fn sparse_bit_reversal_preparation_matches_full_path_raw() {
        type F = GoldilocksField;

        for log_n in 1..=15 {
            let len = 1 << log_n;
            for r in 1..=log_n.min(5) {
                let mut input = deterministic_goldilocks_values(len >> r);
                input.resize(len, F::ZERO);

                let mut expected = input.clone();
                full_bit_reversal_reference(&mut expected, r);
                let mut actual = input;
                reverse_index_bits_and_expand_zero_tail(&mut actual, r, log_n);

                assert_eq!(
                    raw_goldilocks(&actual),
                    raw_goldilocks(&expected),
                    "prepared buffer mismatch at log_n={log_n}, r={r}"
                );
            }
        }
    }

    #[test]
    fn sparse_bit_reversal_fft_matches_full_path_raw_across_shifts() {
        type F = GoldilocksField;

        let shifts = [
            F::ZERO,
            F::ONE,
            F::from_canonical_u64(7),
            F::MULTIPLICATIVE_GROUP_GENERATOR,
            GoldilocksField(u64::MAX),
        ];
        for log_n in [1, 2, 5, 8, 12, 13, 14, 15] {
            let len = 1 << log_n;
            let root_table = fft_root_table::<F>(len);
            for r in 1..=log_n.min(5) {
                let active = deterministic_goldilocks_values(len >> r);
                for shift in shifts {
                    let mut twisted = Vec::with_capacity(len);
                    twisted.extend(
                        shift
                            .powers()
                            .zip(&active)
                            .map(|(power, &coefficient)| power * coefficient),
                    );
                    twisted.resize(len, F::ZERO);

                    let mut expected = twisted.clone();
                    fft_classic_full_bit_reversal_reference(&mut expected, r, &root_table);
                    let mut actual = twisted;
                    fft_classic(&mut actual, r, &root_table);

                    assert_eq!(
                        raw_goldilocks(&actual),
                        raw_goldilocks(&expected),
                        "FFT mismatch at log_n={log_n}, r={r}, shift={shift:?}"
                    );
                }
            }
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
}
