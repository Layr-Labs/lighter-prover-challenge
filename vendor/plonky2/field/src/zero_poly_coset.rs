use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug, Eq, PartialEq)]
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

    /// Evaluates `L_0` over an entire coset using one batched field inversion.
    pub fn l_0_evals(&self, xs: &[F]) -> Vec<F> {
        let denominators = xs
            .iter()
            .map(|&x| self.n * (x - F::ONE))
            .collect::<Vec<_>>();
        let denominator_inverses = F::batch_multiplicative_inverse(&denominators);

        denominator_inverses
            .into_iter()
            .enumerate()
            .map(|(i, denominator_inverse)| self.eval(i) * denominator_inverse)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field;
    use crate::zero_poly_coset::ZeroPolyOnCoset;

    #[test]
    fn batched_l_0_evals_match_definition() {
        type F = GoldilocksField;
        const N_LOG: usize = 5;
        const RATE_BITS: usize = 3;

        let points = F::two_adic_subgroup(N_LOG + RATE_BITS)
            .into_iter()
            .map(|x| F::coset_shift() * x)
            .collect::<Vec<_>>();
        let zero_poly = ZeroPolyOnCoset::<F>::new(N_LOG, RATE_BITS);
        let actual = zero_poly.l_0_evals(&points);
        let n = F::from_canonical_usize(1 << N_LOG);
        let expected = points
            .iter()
            .map(|&x| (x.exp_power_of_2(N_LOG) - F::ONE) * (n * (x - F::ONE)).inverse())
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }
}
