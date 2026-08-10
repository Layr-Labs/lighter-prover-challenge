//! F7a / M6 — "ECDSA/Schnorr sqrt-hint sign binding, point decompression".
//! **Refuted for the decode path**, by a property of the curve constant.
//!
//! Paired Lean proof: `review/lean/F7a_DecodeSignBinding.lean`.
//!
//! Revision 1 of the finding database cited lines 709, 723, 955 and 960 for both
//! F7a (`eddsa/schnorr.rs`, 272 lines) and M6 (`eddsa/curve/base_field.rs`,
//! 560 lines) — the same four numbers for two files, neither of which is long
//! enough to contain them. The concern was never located, so this is a fresh
//! audit rather than a re-check.
//!
//! THE REAL MECHANISM is `ecgfp5_point_decode`
//! (circuit/src/eddsa/gadgets/curve.rs:348-388):
//!
//! ```text
//! e     = w^2 - A
//! delta = e^2 - 4B
//! (r, delta_is_sqrt) = try_any_sqrt_quintic_ext(delta)   // ANY square root
//! x1 = (e + r) / 2
//! x2 = (e - r) / 2
//! x  = select(legendre(x1) == 1, x1, x2)
//! ```
//!
//! The hint returns *any* root, so `r` and `-r` are both admissible — and
//! flipping the sign swaps `x1` and `x2`. The decode is therefore well defined
//! only if the `legendre(x1) == 1` test picks the same field element either way,
//! i.e. only if exactly one of `x1`, `x2` is a quadratic residue.
//!
//! It is, and the reason is arithmetic rather than a constraint:
//!
//! ```text
//! x1 * x2 = (e^2 - r^2)/4 = (e^2 - delta)/4 = (e^2 - (e^2 - 4B))/4 = B
//! ```
//!
//! so `legendre(x1) * legendre(x2) = legendre(B)`. With `B` a NON-residue,
//! exactly one factor is `+1` and the selection is sign-invariant. The tests
//! below check that `B` really is a non-residue, using the crate's own field
//! code, and that the product identity holds.
//!
//! RESIDUAL, not covered here: this settles the sign of the decode's sqrt hint.
//! It does not audit `legendre_sym_quintic_ext` itself, nor the scalar bit
//! decomposition that F7a also mentions. Those remain open.

use circuit::eddsa::curve::base_field::{Legendre, SquareRoot};
use circuit::types::config::F;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::Field;

/// `ECgFp5Point::B = 263 * X` (circuit/src/eddsa/curve/curve.rs:154), where `X`
/// generates `GF(p^5) = GF(p)[X]/(X^5 - 3)`.
fn curve_b() -> QuinticExtension<F> {
    QuinticExtension([
        F::ZERO,
        F::from_canonical_u64(263),
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ])
}

/// **The refutation.** `B` is a non-residue in `GF(p^5)`, so `x1 * x2 = B`
/// forces exactly one of the two candidates to be a residue and the decode's
/// `legendre(x1) == 1` selection cannot depend on which square root the hint
/// supplied.
#[test]
fn curve_constant_b_is_a_non_residue() {
    let legendre = curve_b().legendre();
    assert_eq!(
        legendre,
        F::NEG_ONE,
        "B must be a quadratic non-residue in GF(p^5); if it were a residue the \
         ecgfp5 decode would be ambiguous under the sign of the sqrt hint"
    );
}

/// The control: a value that *is* a residue reports `+1`, so the test above is
/// reading a real signal rather than a constant.
#[test]
fn legendre_distinguishes_residues() {
    let one = QuinticExtension::<F>::ONE;
    assert_eq!(one.legendre(), F::ONE, "1 is a residue");

    let b = curve_b();
    let b_squared = b * b;
    assert_eq!(
        b_squared.legendre(),
        F::ONE,
        "B^2 is a residue by construction"
    );
}

/// The product identity the argument rests on, checked over concrete `e`:
/// `x1 * x2 = B` exactly, so their Legendre symbols multiply to `legendre(B)`
/// and cannot both be `+1`.
#[test]
fn x1_times_x2_is_b_and_exactly_one_is_a_residue() {
    let b = curve_b();
    let four_b = b * F::from_canonical_u64(4).into();
    let two_inv = QuinticExtension::<F>::TWO.inverse();
    let mut checked = 0usize;

    for k in 1u64..60 {
        let e = QuinticExtension::<F>([
            F::from_canonical_u64(k),
            F::from_canonical_u64(k * 7 + 1),
            F::from_canonical_u64(k * 13 + 2),
            F::ZERO,
            F::from_canonical_u64(k * 3),
        ]);
        let delta = e * e - four_b;

        // Only e values for which delta is actually a square correspond to a
        // decodable point; skip the rest.
        if delta.legendre() != F::ONE {
            continue;
        }
        let r = delta.canonical_sqrt().expect("delta is a residue");
        assert_eq!(r * r, delta, "sqrt must be a genuine root");

        let x1 = (e + r) * two_inv;
        let x2 = (e - r) * two_inv;

        assert_eq!(x1 * x2, b, "x1 * x2 must equal B for e = {k:?}");

        let l1 = x1.legendre();
        let l2 = x2.legendre();
        assert_ne!(l1, F::ZERO, "x1 nonzero since x1*x2 = B != 0");
        assert_ne!(l2, F::ZERO, "x2 nonzero since x1*x2 = B != 0");
        assert_ne!(
            l1, l2,
            "exactly one of x1, x2 must be a residue — this is what makes the \
             decode independent of the sqrt hint's sign"
        );
        checked += 1;
    }

    assert!(
        checked >= 10,
        "only {checked} decodable e values were exercised; the loop must not be \
         vacuous or this test proves nothing"
    );
}
