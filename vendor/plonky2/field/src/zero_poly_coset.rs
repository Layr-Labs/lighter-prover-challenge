use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::packable::Packable;
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
            rate_mask: (1 << rate_bits) - 1,
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
        self.evals[i & self.rate_mask]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i & self.rate_mask]
    }

    /// Like `eval_inverse`, but for a range of indices starting with `i_start`.
    /// Consecutive table entries load with `from_slice`; the wrap at `rate`
    /// still gathers. Bit-identical to the scalar loop.
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let start = i_start & self.rate_mask;
        if start + P::WIDTH <= self.inverses.len() {
            return *P::from_slice(&self.inverses[start..start + P::WIDTH]);
        }
        let mut packed = P::ZEROS;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// Calls `visit(offset, inverses[start..start+run])` for each contiguous
    /// run of `eval_inverse(i_start + offset + ·)` that does not wrap `rate`.
    /// Covers `offset in 0..n`. Bit-identical to a scalar `eval_inverse` loop.
    pub fn for_each_inverse_run(&self, i_start: usize, n: usize, mut visit: impl FnMut(usize, &[F])) {
        let rate = self.rate_mask + 1;
        debug_assert_eq!(self.inverses.len(), rate);
        let mut k = 0;
        while k < n {
            let start = (i_start + k) & self.rate_mask;
            let run = (rate - start).min(n - k);
            visit(k, &self.inverses[start..start + run]);
            k += run;
        }
    }

    /// Returns `L_0(x) = Z_H(x)/(n * (x - 1))` with `x = w^i`.
    pub fn eval_l_0(&self, i: usize, x: F) -> F {
        if let Some(table) = &self.l_0_denominator_inverses {
            // The table entry is bit-identical to the expression below, so the product is too.
            return self.eval(i) * table[i];
        }
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }

    /// Fills `out[k] = eval_l_0(i_start + k, xs[k])` for consecutive indices.
    /// With the denominator table attached (production), this is
    /// `evals[(i_start+k) & mask] * table[i_start+k]` via wrap-free eval
    /// runs and a contiguous table slice. Bit-identical to a scalar loop.
    pub fn eval_l_0_into(&self, i_start: usize, xs: &[F], out: &mut [F])
    where
        F: Packable,
    {
        debug_assert_eq!(xs.len(), out.len());
        if let Some(table) = &self.l_0_denominator_inverses {
            let table = table.as_slice();
            debug_assert!(i_start + out.len() <= table.len());
            let rate = self.rate_mask + 1;
            let width = F::Packing::WIDTH;
            let mut k = 0;
            while k < out.len() {
                let start = (i_start + k) & self.rate_mask;
                let run = (rate - start).min(out.len() - k);
                let evals = &self.evals[start..start + run];
                let dens = &table[i_start + k..i_start + k + run];
                let mut off = 0;
                while off + width <= run {
                    let e = *F::Packing::from_slice(&evals[off..off + width]);
                    let d = *F::Packing::from_slice(&dens[off..off + width]);
                    out[k + off..k + off + width].copy_from_slice((e * d).as_slice());
                    off += width;
                }
                for i in off..run {
                    out[k + i] = evals[i] * dens[i];
                }
                k += run;
            }
            return;
        }
        for (k, (slot, &x)) in out.iter_mut().zip(xs.iter()).enumerate() {
            *slot = self.eval_l_0(i_start + k, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks_field::GoldilocksField;

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
                let span = (1usize << n_log) * rate + 2 * rate;
                let mut got = vec![F::ZERO; span.min(64)];
                if !got.is_empty() {
                    let mut k = 0;
                    z.for_each_inverse_run(0, got.len(), |off, run| {
                        got[off..off + run.len()].copy_from_slice(run);
                        k += run.len();
                    });
                    assert_eq!(k, got.len());
                    for (i, &v) in got.iter().enumerate() {
                        assert_eq!(v.0, z.eval_inverse(i).0, "inverse_run({i})");
                    }
                }
            }
        }
    }

    #[test]
    fn eval_l_0_into_matches_scalar_with_table() {
        use crate::types::Sample;
        type F = GoldilocksField;

        for n_log in [1usize, 4] {
            for rate_bits in [1usize, 3] {
                let span = (1usize << n_log) * (1usize << rate_bits);
                let table = alloc::sync::Arc::new(F::rand_vec(span));
                let z = ZeroPolyOnCoset::<F>::new(n_log, rate_bits)
                    .with_l_0_denominator_inverses(table.clone());
                let n = span.min(32);
                let xs = F::rand_vec(n);
                let mut got = vec![F::ZERO; n];
                z.eval_l_0_into(0, &xs, &mut got);
                for i in 0..n {
                    assert_eq!(got[i].0, z.eval_l_0(i, xs[i]).0, "l0_into({i})");
                }
            }
        }
    }
}
