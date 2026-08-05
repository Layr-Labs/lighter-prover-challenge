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

    /// Returns `L_0(x)` for corresponding batches of evaluation indices and points.
    pub fn eval_l_0_batch(&self, indices: &[usize], xs: &[F]) -> Vec<F> {
        assert_eq!(indices.len(), xs.len());
        let denominators = xs
            .iter()
            .map(|&x| self.n * (x - F::ONE))
            .collect::<Vec<_>>();
        let mut values = F::batch_multiplicative_inverse(&denominators);
        for (value, &i) in values.iter_mut().zip(indices) {
            *value *= self.eval(i);
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks_field::GoldilocksField;

    #[test]
    fn batch_l_0_matches_scalar_evaluations() {
        type F = GoldilocksField;
        let zero_poly = ZeroPolyOnCoset::<F>::new(4, 3);
        let points = F::two_adic_subgroup(7)
            .into_iter()
            .map(|x| F::coset_shift() * x)
            .collect::<Vec<_>>();
        let indices = (7..39).collect::<Vec<_>>();
        let xs = indices.iter().map(|&i| points[i]).collect::<Vec<_>>();
        let expected = indices
            .iter()
            .zip(&xs)
            .map(|(&i, &x)| zero_poly.eval_l_0(i, x))
            .collect::<Vec<_>>();

        assert_eq!(zero_poly.eval_l_0_batch(&indices, &xs), expected);
    }
}
