use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `rate = |K|/|H|`.
    rate: usize,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
    /// The periodic numerators `Z_H(g * w^i) / |H|` used by `L_0` evaluation.
    l_0_numerators: Vec<F>,
}

impl<F: Field> ZeroPolyOnCoset<F> {
    pub fn new(n_log: usize, rate_bits: usize) -> Self {
        let g_pow_n = F::coset_shift().exp_power_of_2(n_log);
        let evals = F::two_adic_subgroup(rate_bits)
            .into_iter()
            .map(|x| g_pow_n * x - F::ONE)
            .collect::<Vec<_>>();
        let inverses = F::batch_multiplicative_inverse(&evals);
        let n_inv = F::inverse_2exp(n_log);
        let l_0_numerators = evals.iter().map(|&eval| eval * n_inv).collect();
        Self {
            rate: 1 << rate_bits,
            evals,
            inverses,
            l_0_numerators,
        }
    }

    /// Returns `Z_H(g * w^i)`.
    pub fn eval(&self, i: usize) -> F {
        self.evals[i % self.rate]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i % self.rate]
    }

    /// Like `eval_inverse`, but for a range of indices starting with `i_start`.
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let mut packed = P::ZEROS;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// Returns `L_0(x) = Z_H(x)/(n * (x - 1))` with `x = w^i`.
    pub fn eval_l_0(&self, i: usize, x: F) -> F {
        self.l_0_numerators[i % self.rate] * (x - F::ONE).inverse()
    }

    /// Batched version of [`Self::eval_l_0`] which uses one inversion for the whole batch.
    ///
    /// Both output buffers retain their allocations for reuse by the caller. `values` has the
    /// same order as `indices` and `xs`; `denominators` is scratch after this call.
    pub fn eval_l_0_batch_into(
        &self,
        indices: &[usize],
        xs: &[F],
        denominators: &mut Vec<F>,
        values: &mut Vec<F>,
    ) {
        assert_eq!(indices.len(), xs.len());
        denominators.clear();
        denominators.extend(xs.iter().map(|&x| x - F::ONE));
        F::batch_multiplicative_inverse_into(denominators, values);
        for (&index, value) in indices.iter().zip(values) {
            *value *= self.l_0_numerators[index % self.rate];
        }
    }

    /// Batched `L_0` evaluation specialized for consecutive domain indices. Strip-mining the
    /// periodic numerators avoids an index load and remainder operation for every point.
    pub fn eval_l_0_batch_contiguous_into(
        &self,
        index_start: usize,
        xs: &[F],
        denominators: &mut Vec<F>,
        values: &mut Vec<F>,
    ) {
        denominators.clear();
        denominators.extend(xs.iter().map(|&x| x - F::ONE));
        F::batch_multiplicative_inverse_into(denominators, values);

        let offset = index_start & (self.rate - 1);
        let first_len = (self.rate - offset).min(values.len());
        let (first, rest) = values.split_at_mut(first_len);
        for (value, &numerator) in first
            .iter_mut()
            .zip(&self.l_0_numerators[offset..offset + first_len])
        {
            *value *= numerator;
        }
        for chunk in rest.chunks_mut(self.rate) {
            for (value, &numerator) in chunk.iter_mut().zip(&self.l_0_numerators) {
                *value *= numerator;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::ZeroPolyOnCoset;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field;

    #[test]
    fn eval_l_0_batch_matches_scalar() {
        type F = GoldilocksField;

        for n_log in [1, 8, 16] {
            for rate_bits in [0, 1, 3] {
                let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
                let root = F::primitive_root_of_unity(n_log + rate_bits);
                for start in [0, 5, 31, (1 << rate_bits) + 7] {
                    for len in [0, 1, 2, 3, 4, 7, 31, 32] {
                        let indices = (start..start + len).collect::<Vec<_>>();
                        let xs = indices
                            .iter()
                            .map(|&i| F::coset_shift() * root.exp_u64(i as u64))
                            .collect::<Vec<_>>();
                        let n = F::from_canonical_usize(1 << n_log);
                        let expected = indices
                            .iter()
                            .zip(&xs)
                            .map(|(&i, &x)| zero_poly.eval(i) * (n * (x - F::ONE)).inverse())
                            .collect::<Vec<_>>();
                        let scalar = indices
                            .iter()
                            .zip(&xs)
                            .map(|(&i, &x)| zero_poly.eval_l_0(i, x))
                            .collect::<Vec<_>>();
                        assert_eq!(scalar, expected);
                        let mut denominators = Vec::new();
                        let mut actual = Vec::new();
                        zero_poly.eval_l_0_batch_into(
                            &indices,
                            &xs,
                            &mut denominators,
                            &mut actual,
                        );
                        assert_eq!(
                            actual, expected,
                            "n_log={n_log}, rate_bits={rate_bits}, start={start}, len={len}"
                        );
                        assert_eq!(
                            actual.iter().map(|x| x.0).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.0).collect::<Vec<_>>(),
                            "raw representations differ"
                        );

                        let mut contiguous_denominators = Vec::new();
                        let mut contiguous = Vec::new();
                        zero_poly.eval_l_0_batch_contiguous_into(
                            start,
                            &xs,
                            &mut contiguous_denominators,
                            &mut contiguous,
                        );
                        assert_eq!(contiguous, expected);
                        assert_eq!(
                            contiguous.iter().map(|x| x.0).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.0).collect::<Vec<_>>(),
                            "contiguous raw representations differ"
                        );
                    }
                }
            }
        }
    }

    /// Manual release-mode microbenchmark with the production quotient batch size and a
    /// representative degree-2^16/rate-8 quotient domain.
    #[test]
    #[ignore]
    fn benchmark_eval_l_0_batch() {
        type F = GoldilocksField;
        const N_LOG: usize = 16;
        const RATE_BITS: usize = 3;
        const BATCH_SIZE: usize = 32;
        const REPEATS: usize = 5;

        let zero_poly = ZeroPolyOnCoset::<F>::new(N_LOG, RATE_BITS);
        let points = F::two_adic_subgroup(N_LOG + RATE_BITS);
        let xs = points
            .into_iter()
            .map(|x| F::coset_shift() * x)
            .collect::<Vec<_>>();
        let indices = (0..xs.len()).collect::<Vec<_>>();

        let start = Instant::now();
        let mut scalar = Vec::with_capacity(xs.len());
        for _ in 0..REPEATS {
            scalar.clear();
            scalar.extend(
                indices
                    .iter()
                    .zip(&xs)
                    .map(|(&i, &x)| zero_poly.eval_l_0(i, x)),
            );
            black_box(&scalar);
        }
        let scalar_elapsed = start.elapsed();

        let start = Instant::now();
        let mut denominators = Vec::with_capacity(BATCH_SIZE);
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        let mut batched = Vec::with_capacity(xs.len());
        for _ in 0..REPEATS {
            batched.clear();
            for (indices, xs) in indices.chunks(BATCH_SIZE).zip(xs.chunks(BATCH_SIZE)) {
                zero_poly.eval_l_0_batch_into(indices, xs, &mut denominators, &mut batch);
                batched.extend_from_slice(&batch);
            }
            black_box(&batched);
        }
        let batch_elapsed = start.elapsed();

        assert_eq!(batched, scalar);
        eprintln!(
            "eval_l_0 degree=2^{N_LOG} rate=2^{RATE_BITS}, {} points x {REPEATS}: scalar={scalar_elapsed:?}, batch32={batch_elapsed:?}, speedup={:.2}x",
            xs.len(),
            scalar_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64()
        );
    }
}
