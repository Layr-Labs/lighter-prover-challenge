use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `n = |H|`.
    n: F,
    /// `rate - 1`. `rate = |K|/|H| = 1 << rate_bits` is a power of two by
    /// construction, so the wrap-around index `i % rate` used by [`Self::eval`]
    /// and [`Self::eval_inverse`] is exactly `i & rate_mask` — same value, but a
    /// single-cycle AND instead of the hardware integer division the modulus
    /// compiles to (`rate` is a runtime field, so the power-of-two form is not
    /// visible to the optimizer at the use sites).
    rate_mask: usize,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
    /// Optional process-shared `L_0` table. The legacy cache stores only denominator
    /// inverses, while the full cache stores the result of the inherited
    /// `self.eval(i) * denominator_inverse[i]` operation. Keeping the representation in the
    /// table kind makes [`Self::eval_l_0_evaluations`] a safe one-time discriminator for hot
    /// point loops, while [`Self::eval_l_0`] retains the generic/no-cache fallback.
    l_0_table: Option<L0Table<F>>,
}

#[derive(Debug)]
enum L0Table<F: Field> {
    DenominatorInverses(Arc<Vec<F>>),
    Evaluations(Arc<Vec<F>>),
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
            rate_mask: (1 << rate_bits) - 1,
            evals,
            inverses,
            l_0_table: None,
        }
    }

    /// Attaches a precomputed table of `L_0` denominator inverses (see the field docs for the
    /// exact contract). With a table attached, [`Self::eval_l_0`] reads entry `i` instead of
    /// computing the per-point field inversion.
    pub fn with_l_0_denominator_inverses(mut self, table: Arc<Vec<F>>) -> Self {
        self.l_0_table = Some(L0Table::DenominatorInverses(table));
        self
    }

    /// Attaches full precomputed `L_0` evaluations. Entry `i` must be raw-identical to
    /// `self.eval(i) * (self.n * (x_i - F::ONE)).inverse()`, including that multiplication's
    /// operand order. Hot loops can select this representation once through
    /// [`Self::eval_l_0_evaluations`] and then load entries without a per-point tag test.
    pub fn with_l_0_evaluations(mut self, table: Arc<Vec<F>>) -> Self {
        self.l_0_table = Some(L0Table::Evaluations(table));
        self
    }

    /// Returns the full `L_0` table only when that representation is attached. This method is
    /// intentionally separate from [`Self::eval_l_0`]: callers with a point loop can branch
    /// once on the table kind and remove both the old per-point `Option` test and multiply.
    pub fn eval_l_0_evaluations(&self) -> Option<&[F]> {
        match &self.l_0_table {
            Some(L0Table::Evaluations(table)) => Some(table),
            _ => None,
        }
    }

    /// Returns `Z_H(g * w^i)`.
    pub fn eval(&self, i: usize) -> F {
        self.evals[i & self.rate_mask]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i & self.rate_mask]
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
        match &self.l_0_table {
            Some(L0Table::Evaluations(table)) => table[i],
            Some(L0Table::DenominatorInverses(table)) => {
                // Preserve the inherited operand order and raw representative exactly.
                self.eval(i) * table[i]
            }
            None => self.eval(i) * (self.n * (x - F::ONE)).inverse(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field64;

    /// `eval` / `eval_inverse` index with `i & rate_mask`; that must be
    /// raw-`u64` identical to the `i % rate` it replaced, over every index the
    /// quotient pass can reach (`0..n * rate`) and then some.
    #[test]
    fn masked_index_matches_modulus() {
        type F = GoldilocksField;

        for n_log in [0usize, 1, 4, 8] {
            for rate_bits in [0usize, 1, 3, 4] {
                let z = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
                let rate = 1usize << rate_bits;
                assert_eq!(z.rate_mask, rate - 1);
                for i in 0..(1usize << n_log) * rate + 2 * rate {
                    assert_eq!(z.eval(i).0, z.evals[i % rate].0, "eval({i})");
                    assert_eq!(
                        z.eval_inverse(i).0,
                        z.inverses[i % rate].0,
                        "eval_inverse({i})"
                    );
                }
            }
        }
    }

    /// A full table is already the final field representation. Accessing it must not
    /// canonicalize dirty/noncanonical words or accidentally fall back to arithmetic using `x`.
    #[test]
    fn full_l_0_table_preserves_raw_words() {
        type F = GoldilocksField;

        let order = F::ORDER;
        let raw = (0..32)
            .map(|i| match i % 4 {
                0 => F(order),
                1 => F(order + i as u64),
                2 => F(u64::MAX - i as u64),
                _ => F(i as u64),
            })
            .collect::<Vec<_>>();
        let z = ZeroPolyOnCoset::<F>::new(2, 3)
            .with_l_0_evaluations(Arc::new(raw.clone()));
        let attached = z.eval_l_0_evaluations().expect("full table must be visible");
        assert_eq!(attached.len(), raw.len());
        for (i, expected) in raw.iter().enumerate() {
            assert_eq!(attached[i].0, expected.0, "attached table entry {i}");
            assert_eq!(
                z.eval_l_0(i, F(u64::MAX - i as u64)).0,
                expected.0,
                "eval_l_0 table entry {i}"
            );
        }

        let denominators = Arc::new(vec![F::ONE; raw.len()]);
        let legacy = ZeroPolyOnCoset::<F>::new(2, 3)
            .with_l_0_denominator_inverses(denominators);
        assert!(legacy.eval_l_0_evaluations().is_none());
    }
}
