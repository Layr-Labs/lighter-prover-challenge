use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `n = |H|`.
    n: F,
    /// `rate = |K|/|H|`.
    rate: usize,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
}

impl<F: Field> ZeroPolyOnCoset<F> {
    pub fn new(n_log: usize, rate_bits: usize) -> Self {
        let g_pow_n = F::coset_shift().exp_power_of_2(n_log);
        let evals = F::two_adic_subgroup(rate_bits)
            .into_iter()
            .map(|x| g_pow_n * x - F::ONE)
            .collect::<Vec<_>>();
        let inverses = F::batch_multiplicative_inverse(&evals);
        Self {
            n: F::from_canonical_usize(1 << n_log),
            rate: 1 << rate_bits,
            evals,
            inverses,
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
        // Could also precompute the inverses using Montgomery.
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }

    /// Evaluates `L_0` for matching batches of coset indices and points, reusing caller-owned
    /// storage for the denominators, their inverses, and the results.
    pub fn eval_l_0_batch_into(
        &self,
        indices: &[usize],
        xs: &[F],
        denominators: &mut Vec<F>,
        inverses: &mut Vec<F>,
        out: &mut Vec<F>,
    ) {
        assert_eq!(indices.len(), xs.len());

        denominators.clear();
        denominators.extend(xs.iter().map(|&x| self.n * (x - F::ONE)));
        F::batch_multiplicative_inverse_into(denominators, inverses);

        out.clear();
        out.reserve(indices.len());
        out.extend(
            indices
                .iter()
                .zip(inverses.iter())
                .map(|(&index, &inverse)| self.eval(index) * inverse),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::ZeroPolyOnCoset;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Field64, PrimeField64};

    #[test]
    fn eval_l_0_batch_matches_scalar_for_full_and_tail_batches() {
        type F = GoldilocksField;

        for rate_bits in [0, 1, 3] {
            let n_log = 5;
            let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
            let subgroup = F::two_adic_subgroup(n_log + rate_bits);

            for len in [0usize, 1, 2, 3, 5, 31, 32] {
                let indices = (0..len).collect::<Vec<_>>();
                let mut xs = indices
                    .iter()
                    .map(|&i| F::coset_shift() * subgroup[i])
                    .collect::<Vec<_>>();
                if let Some(first) = xs.first_mut() {
                    *first = GoldilocksField(F::ORDER + 2);
                }
                let expected = indices
                    .iter()
                    .zip(&xs)
                    .map(|(&i, &x)| zero_poly.eval_l_0(i, x).to_canonical_u64())
                    .collect::<Vec<_>>();
                let mut denominators = Vec::new();
                let mut inverses = Vec::new();
                let mut actual = Vec::new();

                zero_poly.eval_l_0_batch_into(
                    &indices,
                    &xs,
                    &mut denominators,
                    &mut inverses,
                    &mut actual,
                );

                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_canonical_u64())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }

    #[test]
    fn eval_l_0_batch_raw_representations_are_characterized() {
        type F = GoldilocksField;

        let zero_poly = ZeroPolyOnCoset::<F>::new(5, 3);
        let indices = (0..32).collect::<Vec<_>>();
        let subgroup = F::two_adic_subgroup(8);
        let mut xs = indices
            .iter()
            .map(|&i| F::coset_shift() * subgroup[i])
            .collect::<Vec<_>>();
        xs[0] = GoldilocksField(F::ORDER + 2);
        let expected_raw = indices
            .iter()
            .zip(&xs)
            .map(|(&i, &x)| zero_poly.eval_l_0(i, x).to_noncanonical_u64())
            .collect::<Vec<_>>();
        let mut denominators = Vec::new();
        let mut inverses = Vec::new();
        let mut actual = Vec::new();

        zero_poly.eval_l_0_batch_into(&indices, &xs, &mut denominators, &mut inverses, &mut actual);

        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_noncanonical_u64())
                .collect::<Vec<_>>(),
            expected_raw
        );
    }

    #[test]
    fn eval_l_0_batch_reuses_or_grows_prefilled_buffers() {
        type F = GoldilocksField;

        let zero_poly = ZeroPolyOnCoset::<F>::new(5, 3);
        let indices = (0..32).collect::<Vec<_>>();
        let subgroup = F::two_adic_subgroup(8);
        let xs = indices
            .iter()
            .map(|&i| F::coset_shift() * subgroup[i])
            .collect::<Vec<_>>();

        for initial_capacity in [1usize, 64] {
            let mut denominators = Vec::with_capacity(initial_capacity);
            let mut inverses = Vec::with_capacity(initial_capacity);
            let mut out = Vec::with_capacity(initial_capacity);
            denominators.push(F::TWO);
            inverses.push(F::TWO);
            out.push(F::TWO);
            let capacities = (denominators.capacity(), inverses.capacity(), out.capacity());

            zero_poly.eval_l_0_batch_into(
                &indices,
                &xs,
                &mut denominators,
                &mut inverses,
                &mut out,
            );

            assert_eq!(denominators.len(), 32);
            assert_eq!(inverses.len(), 32);
            assert_eq!(out.len(), 32);
            if initial_capacity >= 32 {
                assert_eq!(denominators.capacity(), capacities.0);
                assert_eq!(inverses.capacity(), capacities.1);
                assert_eq!(out.capacity(), capacities.2);
            } else {
                assert!(denominators.capacity() >= 32);
                assert!(inverses.capacity() >= 32);
                assert!(out.capacity() >= 32);
            }
        }
    }

    #[test]
    fn eval_l_0_batch_panics_on_zero_denominator() {
        type F = GoldilocksField;

        let zero_poly = ZeroPolyOnCoset::<F>::new(5, 3);
        let mut denominators = vec![F::TWO];
        let mut inverses = vec![F::TWO];
        let mut out = vec![F::TWO];

        let result = catch_unwind(AssertUnwindSafe(|| {
            zero_poly.eval_l_0_batch_into(
                &[0],
                &[F::ONE],
                &mut denominators,
                &mut inverses,
                &mut out,
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn eval_l_0_batch_validates_matching_lengths_before_mutating_buffers() {
        type F = GoldilocksField;

        let zero_poly = ZeroPolyOnCoset::<F>::new(5, 3);
        let mut denominators = vec![F::TWO];
        let mut inverses = vec![F::TWO];
        let mut out = vec![F::TWO];

        let result = catch_unwind(AssertUnwindSafe(|| {
            zero_poly.eval_l_0_batch_into(
                &[0, 1],
                &[F::TWO],
                &mut denominators,
                &mut inverses,
                &mut out,
            );
        }));

        assert!(result.is_err());
        assert_eq!(denominators, vec![F::TWO]);
        assert_eq!(inverses, vec![F::TWO]);
        assert_eq!(out, vec![F::TWO]);
    }
}
