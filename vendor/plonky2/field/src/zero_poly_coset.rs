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

    /// Batched [`Self::eval_l_0`]: computes `L_0(x_k)` for aligned `indices`/`xs` using a single
    /// Montgomery batch inversion for all denominators `n * (x_k - 1)` instead of one inversion
    /// per point. `denominators` and `out` are caller-owned scratch; both retain capacity across
    /// calls. On return `out[k] == self.eval_l_0(indices[k], xs[k])` exactly: the batch inversion
    /// produces the same individual inverse each scalar call computes, and the final product
    /// multiplies the same precomputed `Z_H` evaluation.
    pub fn eval_l_0_batch_into(
        &self,
        indices: &[usize],
        xs: &[F],
        denominators: &mut Vec<F>,
        out: &mut Vec<F>,
    ) {
        debug_assert_eq!(indices.len(), xs.len());
        denominators.clear();
        denominators.extend(xs.iter().map(|&x| self.n * (x - F::ONE)));
        F::batch_multiplicative_inverse_into(denominators, out);
        for (value, &i) in out.iter_mut().zip(indices) {
            *value *= self.eval(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::ZeroPolyOnCoset;
    use crate::goldilocks_field::GoldilocksField as F;
    use crate::types::{Field, PrimeField64, Sample};

    /// The batched helper must reproduce the scalar `eval_l_0` bit for bit, for every batch
    /// length including the 0/1/2/3 special cases of the Montgomery batch inversion, and the
    /// caller-owned buffers must be reusable across calls without affecting results.
    #[test]
    fn eval_l_0_batch_matches_scalar() {
        let n_log = 5;
        let rate_bits = 3;
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
        let subgroup = F::two_adic_subgroup(n_log + rate_bits);

        let mut denominators = Vec::new();
        let mut batched = Vec::new();
        for batch_len in [0usize, 1, 2, 3, 4, 5, 31, 32] {
            let indices: Vec<usize> = (0..batch_len).map(|k| (k * 7 + 3) % subgroup.len()).collect();
            // Quotient-domain points are coset elements g * w^i; use the same form here so the
            // denominators are guaranteed nonzero, as in production.
            let xs: Vec<F> = indices
                .iter()
                .map(|&i| F::coset_shift() * subgroup[i])
                .collect();
            zero_poly.eval_l_0_batch_into(&indices, &xs, &mut denominators, &mut batched);
            assert_eq!(batched.len(), batch_len);
            for k in 0..batch_len {
                let scalar = zero_poly.eval_l_0(indices[k], xs[k]);
                assert_eq!(
                    batched[k].to_canonical_u64(),
                    scalar.to_canonical_u64(),
                    "batch length {batch_len}, point {k}"
                );
            }
        }

        // Random nonzero inputs through the reusable-buffer inverse as well.
        let random: Vec<F> = (0..40).map(|_| F::rand()).filter(|x| !x.is_zero()).collect();
        let mut out = Vec::new();
        F::batch_multiplicative_inverse_into(&random, &mut out);
        for (inv, x) in out.iter().zip(&random) {
            assert_eq!(*inv * *x, F::ONE);
        }
    }
}
