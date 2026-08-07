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
    /// Optional precomputed `L_0(x) = Z_H(x)/(n * (x - 1))` values for every point
    /// `x = g * w^i` of the coset, indexed by `i in 0..n * rate`. These depend only on
    /// `(n, rate, g)` — not on any challenge — so callers may attach a table shared across
    /// proofs via [`Self::with_l_0_values`].
    l_0_values: Option<Arc<Vec<F>>>,
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
            l_0_values: None,
        }
    }

    /// Attaches a precomputed table of full `L_0` values (see the field docs for the exact
    /// contract). With a table attached, quotient evaluation copies values directly, deleting
    /// both the per-point inversion and the remaining per-proof multiplication.
    pub fn with_l_0_values(mut self, table: Arc<Vec<F>>) -> Self {
        self.l_0_values = Some(table);
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
        if let Some(table) = &self.l_0_values {
            return table[i];
        }
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }

    /// Evaluates `L_0` at a consecutive batch of coset points using Montgomery's trick.
    ///
    /// Quotient evaluation consumes points in contiguous batches. Grouping their denominator
    /// inversions replaces one exponentiation per point with one inversion for the whole batch;
    /// both caller-owned buffers retain their allocations across batches.
    pub fn eval_l_0_batch_contiguous_into(
        &self,
        index_start: usize,
        xs: &[F],
        denominators: &mut Vec<F>,
        values: &mut Vec<F>,
    ) {
        if let Some(table) = &self.l_0_values {
            values.clear();
            values.extend_from_slice(&table[index_start..index_start + xs.len()]);
            return;
        }

        denominators.clear();
        denominators.extend(xs.iter().map(|&x| self.n * (x - F::ONE)));
        F::batch_multiplicative_inverse_into(denominators, values);
        for (offset, value) in values.iter_mut().enumerate() {
            *value *= self.eval(index_start + offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::ZeroPolyOnCoset;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field;

    #[test]
    fn contiguous_l0_batch_and_cached_table_match_scalar() {
        type F = GoldilocksField;

        for n_log in [2usize, 6] {
            for rate_bits in [0usize, 1, 3] {
                let domain_log = n_log + rate_bits;
                let xs = F::two_adic_subgroup(domain_log)
                    .into_iter()
                    .map(|x| F::coset_shift() * x)
                    .collect::<Vec<_>>();
                let direct = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
                let expected = xs
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| direct.eval_l_0(i, x))
                    .collect::<Vec<_>>();

                for batch_size in [1usize, 2, 3, 7, 32] {
                    let mut denominators = vec![F::ONE; batch_size + 3];
                    let mut actual = vec![F::ONE; batch_size + 5];
                    for (batch, xs_batch) in xs.chunks(batch_size).enumerate() {
                        let start = batch * batch_size;
                        direct.eval_l_0_batch_contiguous_into(
                            start,
                            xs_batch,
                            &mut denominators,
                            &mut actual,
                        );
                        assert_eq!(actual, expected[start..start + xs_batch.len()]);
                    }
                }

                let n = F::from_canonical_usize(1 << n_log);
                let denominator_inverses = F::batch_multiplicative_inverse(
                    &xs.iter().map(|&x| n * (x - F::ONE)).collect::<Vec<_>>(),
                );
                let table = denominator_inverses
                    .into_iter()
                    .enumerate()
                    .map(|(i, inverse)| direct.eval(i) * inverse)
                    .collect();
                let cached = ZeroPolyOnCoset::<F>::new(n_log, rate_bits)
                    .with_l_0_values(Arc::new(table));
                let cached_scalar = xs
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| cached.eval_l_0(i, x))
                    .collect::<Vec<_>>();
                assert_eq!(cached_scalar, expected);

                let mut denominators = Vec::new();
                let mut actual = Vec::new();
                cached.eval_l_0_batch_contiguous_into(0, &xs, &mut denominators, &mut actual);
                assert_eq!(actual, expected);
            }
        }
    }
}
