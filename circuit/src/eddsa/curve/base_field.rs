// Portions of this file are derived from ecgfp5
// Copyright (c) 2022 Thomas Pornin
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::extension::{Extendable, FieldExtension, Frobenius};
use plonky2::field::ops::Square;
use plonky2::field::types::{Field, PrimeField};
use plonky2::hash::hash_types::RichField;

use crate::types::config::F;

pub trait Legendre<F: Field> {
    fn legendre(&self) -> F;
}

impl Legendre<F> for QuinticExtension<F> {
    fn legendre(&self) -> F {
        let frob1 = self.frobenius();
        let frob2 = frob1.frobenius();

        let frob1_times_frob2 = frob1 * frob2;
        let frob2_frob1_times_frob2 = frob1_times_frob2.repeated_frobenius(2);

        let xr_ext = *self * frob1_times_frob2 * frob2_frob1_times_frob2;
        let xr: F = <QuinticExtension<F> as FieldExtension<5>>::to_basefield_array(&xr_ext)[0];

        legendre_symbol_goldilocks(xr)
    }
}

/// Computes `x^((p - 1) / 2)` for the Goldilocks prime
/// `p = 2^64 - 2^32 + 1` without a field inversion.
#[inline]
fn legendre_symbol_goldilocks(x: F) -> F {
    // Build x^(2^32 - 1) with the same short addition chain used by the
    // Goldilocks inverse, then square 31 times:
    // (2^32 - 1) * 2^31 = (p - 1) / 2.
    let t2 = x.square() * x;
    let t3 = t2.square() * x;
    let t6 = t3.exp_power_of_2(3) * t3;
    let t12 = t6.exp_power_of_2(6) * t6;
    let t24 = t12.exp_power_of_2(12) * t12;
    let t30 = t24.exp_power_of_2(6) * t6;
    let t31 = t30.square() * x;
    let t32 = t31.square() * x;
    t32.exp_power_of_2(31)
}

pub trait SquareRoot: Sized {
    fn sqrt(&self) -> Option<Self>;
    fn canonical_sqrt(&self) -> Option<Self>;
}

impl SquareRoot for QuinticExtension<F> {
    fn sqrt(&self) -> Option<Self> {
        sqrt_quintic_ext_goldilocks(*self)
    }

    fn canonical_sqrt(&self) -> Option<Self> {
        canonical_sqrt_quintic_ext_goldilocks(*self)
    }
}

pub trait InverseOrZero: Sized {
    fn inverse_or_zero(&self) -> Self;
}

impl InverseOrZero for F {
    fn inverse_or_zero(&self) -> Self {
        self.try_inverse().unwrap_or(F::ZERO)
    }
}

impl InverseOrZero for QuinticExtension<F> {
    fn inverse_or_zero(&self) -> Self {
        self.try_inverse().unwrap_or(QuinticExtension::<F>::ZERO)
    }
}

pub trait Sgn0 {
    fn sgn0(&self) -> bool;
}

impl Sgn0 for QuinticExtension<F> {
    fn sgn0(&self) -> bool {
        quintic_ext_sgn0(*self)
    }
}

/// returns true or false indicating a notion of "sign" for quintic_ext.
/// This is used to canonicalize the square root
/// This is an implementation of the function sgn0 from the IRTF's hash-to-curve document
/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-hash-to-curve-07#name-the-sgn0-function
pub(crate) fn quintic_ext_sgn0<F: RichField + Extendable<5>>(x: QuinticExtension<F>) -> bool {
    let mut sign = false;
    let mut zero = true;
    for &limb in x.0.iter() {
        let sign_i = limb.to_canonical_u64() & 1 == 0;
        let zero_i = limb == F::ZERO;
        sign = sign || (zero && sign_i);
        zero = zero && zero_i;
    }
    sign
}

// returns the "canoncal" square root of x, if it exists
// the "canonical" square root is the one such that `sgn0(sqrt(x)) == true`
pub(crate) fn canonical_sqrt_quintic_ext_goldilocks(
    x: QuinticExtension<F>,
) -> Option<QuinticExtension<F>> {
    match sqrt_quintic_ext_goldilocks(x) {
        Some(root_x) => {
            if quintic_ext_sgn0(root_x) {
                Some(-root_x)
            } else {
                Some(root_x)
            }
        }
        None => None,
    }
}

/// returns `Some(sqrt(x))` if `x` is a square in the field, and `None` otherwise
/// basically copied from here: https://github.com/pornin/ecquintic_ext/blob/ce059c6d1e1662db437aecbf3db6bb67fe63c716/python/ecGFp5.py#L879
pub(crate) fn sqrt_quintic_ext_goldilocks(x: QuinticExtension<F>) -> Option<QuinticExtension<F>> {
    let v = x.exp_power_of_2(31);
    let d = x * v.exp_power_of_2(32) * v.try_inverse().unwrap_or(QuinticExtension::<F>::ZERO);
    let e = (d * d.repeated_frobenius(2)).frobenius();
    let f = e.square();

    let [x0, x1, x2, x3, x4] = x.0;
    let [f0, f1, f2, f3, f4] = f.0;
    let g = x0 * f0 + F::from_canonical_u64(3) * (x1 * f4 + x2 * f3 + x3 * f2 + x4 * f1);

    g.sqrt().map(|s| e.inverse_or_zero() * s.into())
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;
    use rand::thread_rng;

    use super::*;
    use crate::eddsa::curve::test_utils::gfp5_random_non_square;

    #[test]
    fn test_legendre() {
        // test zero
        assert_eq!(F::ZERO, QuinticExtension::<F>::ZERO.legendre());

        // test non-squares
        for _ in 0..32 {
            let x = gfp5_random_non_square();
            let legendre_sym = x.legendre();

            assert_eq!(legendre_sym, -F::ONE);
        }

        // test squares
        for _ in 0..32 {
            let x = QuinticExtension::<F>::sample(&mut thread_rng());
            let square = x * x;
            let legendre_sym = square.legendre();

            assert_eq!(legendre_sym, F::ONE);
        }

        // test zero
        let x = QuinticExtension::<F>::ZERO;
        let square = x * x;
        let legendre_sym = square.legendre();
        assert_eq!(legendre_sym, F::ZERO);
    }

    #[test]
    fn legendre_symbol_matches_inverse_reference() {
        let reference = |x: F| {
            let x_31 = x.exp_power_of_2(31);
            x_31.exp_power_of_2(32) * x_31.inverse_or_zero()
        };
        let check = |x| assert_eq!(legendre_symbol_goldilocks(x), reference(x));

        for x in [F::ZERO, F::ONE, F::TWO, F::NEG_ONE] {
            check(x);
        }

        let mut state = 0x6A09_E667_F3BC_C909u64;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            check(F::from_noncanonical_u64(state));
        }
    }

    #[test]
    fn test_sqrt_quintic_ext_outside_circuit() {
        let mut rng = thread_rng();

        for _ in 0..30 {
            let x = QuinticExtension::<F>::sample(&mut rng);
            let square = x * x;
            let sqrt = square.sqrt().unwrap();

            assert_eq!(sqrt * sqrt, square);
        }
    }

    #[test]
    fn test_canonical_sqrt_quintic_ext_outside_circuit() {
        let mut rng = thread_rng();

        for _ in 0..30 {
            let x = QuinticExtension::<F>::sample(&mut rng);
            let square = x * x;
            let sqrt = square.canonical_sqrt().unwrap();

            assert_eq!(sqrt * sqrt, square);
            assert!(!sqrt.sgn0())
        }
    }
}
