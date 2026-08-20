use alloc::vec;
use alloc::vec::Vec;
use core::any::TypeId;

use plonky2_util::log2_ceil;

use crate::extension::quadratic::QuadraticExtension;
use crate::goldilocks_extensions::ext2_mul_add;
use crate::goldilocks_field::GoldilocksField;
use crate::polynomial::PolynomialCoeffs;
use crate::types::Field;

/// Env-gated switch for the fused Horner fast path in
/// `divide_by_linear_padded_in_place`. The optimization is ON by default;
/// setting `LIGHTER_DISABLE_FRI_HORNER_FUSION=1` rolls back to the generic
/// scalar `acc * z + a` spelling. Mirrors the `LIGHTER_DISABLE_POW_QUAD`
/// convention: only the exact value `1` disables, missing/empty/other values
/// keep the fast path.
#[cfg(feature = "std")]
#[inline]
fn horner_fusion_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIGHTER_DISABLE_FRI_HORNER_FUSION").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    })
}

#[cfg(not(feature = "std"))]
#[inline(always)]
const fn horner_fusion_enabled() -> bool {
    true
}

impl<F: Field> PolynomialCoeffs<F> {
    /// Polynomial division.
    /// Returns `(q, r)`, the quotient and remainder of the polynomial division of `a` by `b`.
    pub fn div_rem(&self, b: &Self) -> (Self, Self) {
        let (a_degree_plug_1, b_degree_plus_1) = (self.degree_plus_one(), b.degree_plus_one());
        if a_degree_plug_1 == 0 {
            (Self::zero(1), Self::empty())
        } else if b_degree_plus_1 == 0 {
            panic!("Division by zero polynomial");
        } else if a_degree_plug_1 < b_degree_plus_1 {
            (Self::zero(1), self.clone())
        } else if b_degree_plus_1 == 1 {
            (self * b.coeffs[0].inverse(), Self::empty())
        } else {
            let rev_b = b.rev();
            let rev_b_inv = rev_b.inv_mod_xn(a_degree_plug_1 - b_degree_plus_1 + 1);
            let rhs: Self = self.rev().coeffs[..=a_degree_plug_1 - b_degree_plus_1]
                .to_vec()
                .into();
            let rev_q: Self = (&rev_b_inv * &rhs).coeffs[..=a_degree_plug_1 - b_degree_plus_1]
                .to_vec()
                .into();
            let mut q = rev_q.rev();
            let qb = &q * b;
            let mut r = self - &qb;
            q.trim();
            r.trim();
            (q, r)
        }
    }

    /// Polynomial long division.
    /// Returns `(q, r)`, the quotient and remainder of the polynomial division of `a` by `b`.
    /// Generally slower that the equivalent function `Polynomial::polynomial_division`.
    pub fn div_rem_long_division(&self, b: &Self) -> (Self, Self) {
        let b = b.trimmed();

        let (a_degree_plus_1, b_degree_plus_1) = (self.degree_plus_one(), b.degree_plus_one());
        if a_degree_plus_1 == 0 {
            (Self::zero(1), Self::empty())
        } else if b_degree_plus_1 == 0 {
            panic!("Division by zero polynomial");
        } else if a_degree_plus_1 < b_degree_plus_1 {
            (Self::zero(1), self.clone())
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = Self::zero(a_degree_plus_1 - b_degree_plus_1 + 1);
            let mut remainder = self.clone();
            // Can unwrap here because we know self is not zero.
            let divisor_leading_inv = b.lead().inverse();
            while !remainder.is_zero() && remainder.degree_plus_one() >= b_degree_plus_1 {
                let cur_q_coeff = remainder.lead() * divisor_leading_inv;
                let cur_q_degree = remainder.degree_plus_one() - b_degree_plus_1;
                quotient.coeffs[cur_q_degree] = cur_q_coeff;

                for (i, &div_coeff) in b.coeffs.iter().enumerate() {
                    remainder.coeffs[cur_q_degree + i] -= cur_q_coeff * div_coeff;
                }
                remainder.trim();
            }
            (quotient, remainder)
        }
    }

    /// Like `divide_by_linear`, but consumes `self`, reuses its buffer, and
    /// leaves the top coefficient zero (the power-of-two pad the FRI opening
    /// path would otherwise push). Same Horner recurrence in the same order,
    /// so the quotient coefficients are bit-identical: slot `i` receives
    /// `b_{i+1}` while the accumulator carries `b_i` downward, and the first
    /// step writes the zero seed into the top slot.
    pub fn divide_by_linear_padded_in_place(mut self, z: F) -> PolynomialCoeffs<F> {
        let len = self.coeffs.len();
        if len == 0 {
            return self;
        }
        // Production Goldilocks-quadratic fast path. The Horner recurrence
        // `acc = acc * z + a` is exactly `ext2_mul_add(acc, z, a)`, which
        // fuses the addend into the multiply's accumulators for two
        // `reduce160` per step instead of the separate spelling's four (two
        // for the delayed extension multiply, two for the canonicalizing
        // extension add). The stored value is the previous accumulator,
        // identical to the generic path; only the representative of the
        // running accumulator may differ by a multiple of `p`, and every
        // downstream consumer is congruence-preserving. This mirrors
        // `accumulate_linear_quotient` in `fri/oracle.rs` and operates under
        // the same field-value-exact (not raw-exact) license documented by
        // `ext2_mul_add_matches_mul_then_add_as_field_values`.
        if horner_fusion_enabled()
            && TypeId::of::<F>() == TypeId::of::<QuadraticExtension<GoldilocksField>>()
        {
            // SAFETY: the `TypeId` comparison proves `F` is exactly
            // `QuadraticExtension<GoldilocksField>`, so the casts below
            // preserve layout, length and alignment, and the reads/writes
            // are of an initialized `Copy` value of that same type.
            let coeffs_q = unsafe {
                core::slice::from_raw_parts_mut(
                    self.coeffs.as_mut_ptr().cast::<QuadraticExtension<GoldilocksField>>(),
                    len,
                )
            };
            let z_q = unsafe { *(&z as *const F).cast::<QuadraticExtension<GoldilocksField>>() };
            let mut acc = QuadraticExtension::<GoldilocksField>::ZERO;
            for i in (0..len).rev() {
                let prev = acc;
                acc = ext2_mul_add(acc, z_q, coeffs_q[i]);
                coeffs_q[i] = prev;
            }
            return self;
        }
        let mut acc = F::ZERO;
        for i in (0..self.coeffs.len()).rev() {
            let a = self.coeffs[i];
            let prev = acc;
            acc = acc * z + a;
            self.coeffs[i] = prev;
        }
        self
    }

    /// Let `self=p(X)`, this returns `(p(X)-p(z))/(X-z)`.
    /// See <https://en.wikipedia.org/wiki/Horner%27s_method>
    pub fn divide_by_linear(&self, z: F) -> PolynomialCoeffs<F> {
        let mut bs = self
            .coeffs
            .iter()
            .rev()
            .scan(F::ZERO, |acc, &c| {
                *acc = *acc * z + c;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        bs.pop();
        bs.reverse();
        Self { coeffs: bs }
    }

    /// Computes the inverse of `self` modulo `x^n`.
    pub fn inv_mod_xn(&self, n: usize) -> Self {
        assert!(n > 0, "`n` needs to be nonzero");
        assert!(self.coeffs[0].is_nonzero(), "Inverse doesn't exist.");

        // If polynomial is constant, return the inverse of the constant.
        if self.degree_plus_one() == 1 {
            return Self::new(vec![self.coeffs[0].inverse()]);
        }

        let h = if self.len() < n {
            self.padded(n)
        } else {
            self.clone()
        };

        let mut a = Self::empty();
        a.coeffs.push(h.coeffs[0].inverse());
        for i in 0..log2_ceil(n) {
            let l = 1 << i;
            let h0 = h.coeffs[..l].to_vec().into();
            let mut h1: Self = h.coeffs[l..].to_vec().into();
            let mut c = &a * &h0;
            if l == c.len() {
                c = Self::zero(1);
            } else {
                c.coeffs.drain(0..l);
            }
            h1.trim();
            let mut tmp = &a * &h1;
            tmp = &tmp + &c;
            tmp.coeffs.iter_mut().for_each(|x| *x = -(*x));
            tmp.trim();
            let mut b = &a * &tmp;
            b.trim();
            if b.len() > l {
                b.coeffs.drain(l..);
            }
            a.coeffs.extend_from_slice(&b.coeffs);
        }
        a.coeffs.drain(n..);
        a
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::rngs::OsRng;

    use crate::extension::quadratic::QuadraticExtension;
    use crate::extension::quartic::QuarticExtension;
    use crate::goldilocks_field::GoldilocksField;
    use crate::polynomial::PolynomialCoeffs;
    use crate::types::{Field, Field64, PrimeField64, Sample};

    #[test]
    fn test_division_by_linear() {
        type F = QuarticExtension<GoldilocksField>;
        let n = OsRng.gen_range(1..1000);
        let poly = PolynomialCoeffs::new(F::rand_vec(n));
        let z = F::rand();
        let ev = poly.eval(z);

        let quotient = poly.divide_by_linear(z);
        assert_eq!(
            poly,
            &(&quotient * &vec![-z, F::ONE].into()) + &vec![ev].into() // `quotient * (X-z) + ev`
        );
    }

    /// Differential for the fused Horner fast path in
    /// `divide_by_linear_padded_in_place`. The fused `ext2_mul_add` recurrence
    /// is field-value-exact, not raw-identical, to the separate
    /// `acc * z + a` spelling: the extension `Add` can hand back a
    /// representative in `[p, 2^64)` where the fused form's single `reduce160`
    /// returns the canonical one (see
    /// `ext2_mul_add_matches_mul_then_add_as_field_values`). The contract here
    /// is therefore canonical equality plus the structural bound that the two
    /// representatives of every quotient limb never differ by more than a
    /// single `p`, mirroring the established fused-multiply-accumulate
    /// contract. Inputs mix canonical, `ORDER + limb`, and `u64::MAX`
    /// noncanonical representatives.
    #[test]
    fn divide_by_linear_padded_fused_matches_scalar_as_field_values() {
        type F = QuadraticExtension<GoldilocksField>;
        let p = GoldilocksField::ORDER;
        let raw_specials = [0u64, 1, 2, p - 2, p - 1, p, p + 1, u64::MAX - 1, u64::MAX];

        let check = |fused: &PolynomialCoeffs<F>, scalar: &PolynomialCoeffs<F>, what: &str| {
            assert_eq!(fused.coeffs.len(), scalar.coeffs.len(), "length for {what}");
            for i in 0..fused.coeffs.len() {
                for limb in 0..2 {
                    let (a, e) = (fused.coeffs[i].0[limb].0, scalar.coeffs[i].0[limb].0);
                    assert_eq!(
                        fused.coeffs[i].0[limb].to_canonical_u64(),
                        scalar.coeffs[i].0[limb].to_canonical_u64(),
                        "canonical quotient limb {limb} at slot {i} mismatch for {what}"
                    );
                    let spread = a.max(e) - a.min(e);
                    assert!(
                        spread == 0 || spread == p,
                        "quotient limb {limb} at slot {i} reps differ by {spread}, not 0 or p, for {what}"
                    );
                }
            }
        };

        // Reference: the generic scalar Horner recurrence, identical to the
        // pre-optimization body, run on a pristine copy so the fused path's
        // env gate and `TypeId` reinterpretation are not on the critical path.
        let scalar_horner = |poly: PolynomialCoeffs<F>, z: F| -> PolynomialCoeffs<F> {
            let mut acc = F::ZERO;
            let mut coeffs = poly.coeffs;
            for i in (0..coeffs.len()).rev() {
                let a = coeffs[i];
                let prev = acc;
                acc = acc * z + a;
                coeffs[i] = prev;
            }
            PolynomialCoeffs { coeffs }
        };

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut case = 0usize;
        for len in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 256, 257] {
            for &zx in &raw_specials {
                for &zy in &raw_specials {
                    let z = QuadraticExtension([GoldilocksField(zx), GoldilocksField(zy)]);
                    let coeffs: Vec<F> = (0..len)
                        .map(|i| {
                            let a0 = if i < raw_specials.len() {
                                raw_specials[i]
                            } else {
                                next()
                            };
                            let a1 = if i < raw_specials.len() {
                                raw_specials[raw_specials.len() - 1 - i]
                            } else {
                                next()
                            };
                            QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)])
                        })
                        .collect();
                    let poly = PolynomialCoeffs::new(coeffs.clone());
                    let fused = poly.divide_by_linear_padded_in_place(z);
                    let scalar = scalar_horner(PolynomialCoeffs::new(coeffs), z);
                    check(&fused, &scalar, &format!("case {case} len {len} z={z:?}"));
                    case += 1;
                }
            }
        }

        // Random sweep with noncanonical representatives.
        for _ in 0..20_000 {
            let len = (next() as usize % 64) + 1;
            let z = QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
            let coeffs: Vec<F> = (0..len)
                .map(|_| QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]))
                .collect();
            let poly = PolynomialCoeffs::new(coeffs.clone());
            let fused = poly.divide_by_linear_padded_in_place(z);
            let scalar = scalar_horner(PolynomialCoeffs::new(coeffs), z);
            check(&fused, &scalar, &format!("random case {case}"));
            case += 1;
        }
    }
}
