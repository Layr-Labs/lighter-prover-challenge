use alloc::sync::Arc;
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
    /// Optional precomputed inverses of the `L_0` denominator `n * (x - 1)` for every point
    /// `x = g * w^i` of the coset, indexed by `i in 0..n * rate`. These depend only on
    /// `(n, rate, g)` — not on any challenge — so callers may attach a table shared across
    /// proofs via [`Self::with_l_0_denominator_inverses`]. Each entry must be bit-identical to
    /// `(self.n * (x - F::ONE)).inverse()`, the value [`Self::eval_l_0`] computes without the
    /// table.
    l_0_denominator_inverses: Option<Arc<Vec<F>>>,
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
            l_0_denominator_inverses: None,
        }
    }

    /// Attaches a precomputed table of `L_0` denominator inverses (see the field docs for the
    /// exact contract). With a table attached, [`Self::eval_l_0`] reads entry `i` instead of
    /// computing the per-point field inversion.
    pub fn with_l_0_denominator_inverses(mut self, table: Arc<Vec<F>>) -> Self {
        self.l_0_denominator_inverses = Some(table);
        self
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
        if let Some(table) = &self.l_0_denominator_inverses {
            // The table entry is bit-identical to the expression below, so the product is too.
            return self.eval(i) * table[i];
        }
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }
}
