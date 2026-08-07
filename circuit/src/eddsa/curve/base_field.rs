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

        goldilocks_legendre(xr)
    }
}

/// Evaluates the Goldilocks quadratic character `xr^((p - 1) / 2)` for
/// `p = 2^64 - 2^32 + 1` with a direct fixed addition chain.
///
/// For nonzero `xr`, the previous tail `xr^(2^63) * inverse(xr^(2^31))` equals
/// `xr^(2^63 - 2^31) = xr^((p - 1) / 2)`; zero maps to zero under both
/// formulas. Since `(p - 1) / 2 = (2^32 - 1) * 2^31`, the chain first builds
/// `xr^(2^32 - 1)` and then performs 31 final squarings: 62 squarings plus 8
/// multiplies in total, with no inversion, replacing 125 squarings, 9
/// multiplies, and a full Fermat inversion chain.
#[inline]
pub(crate) fn goldilocks_legendre(xr: F) -> F {
    // tK denotes xr^(2^K - 1).
    let t2 = xr.square() * xr;
    let t3 = t2.square() * xr;
    let t6 = t3.exp_power_of_2(3) * t3;
    let t12 = t6.exp_power_of_2(6) * t6;
    let t24 = t12.exp_power_of_2(12) * t12;
    let t30 = t24.exp_power_of_2(6) * t6;
    let t31 = t30.square() * xr;
    let t32 = t31.square() * xr;
    // xr^((2^32 - 1) * 2^31) = xr^((p - 1) / 2).
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
    use plonky2::field::types::{Field64, PrimeField64, Sample};
    use rand::thread_rng;

    use super::*;
    use crate::eddsa::curve::test_utils::gfp5_random_non_square;

    /// The exact pre-optimization Legendre tail: `xr^(2^63) * inverse(xr^(2^31))`,
    /// kept as the reference for the differential test below.
    fn legacy_goldilocks_legendre(xr: F) -> F {
        let xr_31 = xr.exp_power_of_2(31);
        let xr_63 = xr_31.exp_power_of_2(32);

        // only way `xr_31` can be zero is if `xr` is zero, in which case `self` is zero, in which case we want to return zero.
        let xr_31_inv_or_zero = xr_31.inverse_or_zero();
        xr_63 * xr_31_inv_or_zero
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn test_goldilocks_legendre_chain_matches_legacy_expression() {
        let check = |raw: u64| {
            let xr = F::from_noncanonical_u64(raw);
            let expected = legacy_goldilocks_legendre(xr);
            let got = goldilocks_legendre(xr);
            assert_eq!(
                got.to_canonical_u64(),
                expected.to_canonical_u64(),
                "canonical mismatch for raw input {raw:#x}"
            );
            if xr.to_canonical_u64() != 0 {
                // Raw (pre-canonicalization) u64 equality for every nonzero
                // input. For a *noncanonical encoding of zero* (raw >= ORDER)
                // the legacy tail returns the literal `F::ZERO` constant (raw
                // 0) via `inverse_or_zero`, while the chain propagates
                // `reduce128`'s noncanonical zero (raw ORDER): the identical
                // field value under this field's canonical `PartialEq`, which
                // is all any `legendre()` consumer compares. Canonical
                // equality is asserted above for those inputs.
                assert_eq!(got.0, expected.0, "raw u64 mismatch for raw input {raw:#x}");
            }
        };

        // Edge inputs, including every interesting boundary around the modulus
        // and the u64 range.
        for raw in [
            0,
            1,
            2,
            3,
            (1u64 << 31) - 1,
            1u64 << 31,
            (1u64 << 32) - 1,
            1u64 << 32,
            (1u64 << 32) + 1,
            (1u64 << 63) - 1,
            1u64 << 63,
            F::ORDER - 2,
            F::ORDER - 1,
        ] {
            check(raw);
        }

        // Noncanonical representations: every raw value in [ORDER, u64::MAX]
        // encodes a small field element noncanonically.
        for k in 0..1_000u64 {
            check(F::ORDER + k);
            check(u64::MAX - k);
        }

        // 10,000 deterministic full-width u64 inputs.
        let mut state = 0x0123_4567_89AB_CDEFu64;
        for _ in 0..10_000 {
            check(splitmix64(&mut state));
        }
    }

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
