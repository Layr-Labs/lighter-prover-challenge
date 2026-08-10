//! F2 — "div_rem quotient non-uniqueness". **REFUTED.**
//!
//! Paired Lean proof: `review/lean/F2_DivRemUniqueness.lean`.
//!
//! The database entry claims:
//!
//! > div_rem_biguint does not uniquely constrain the quotient when a*d >= p
//! > (256-bit wraparound). Multiple (q, r) pairs satisfy q*d + r = a (mod p).
//!
//! The premise is that the gadget checks the division equation *modulo p*. It
//! does not. `div_rem_biguint_trimmed` (circuit/src/bigint/div_rem.rs:84-121)
//! allocates `div` and `rem` with `add_virtual_biguint_target_safe`, which
//! range-checks every limb to 32 bits, then asserts the equation through
//! `mul_biguint` / `add_biguint` — carrying multi-limb routines — and compares
//! with `connect_biguint`, which asserts the *excess high limbs are zero*
//! rather than truncating to the shorter operand.
//!
//! So the constraint is `q*b + r = a` over the integers together with `r < b`,
//! and the division algorithm pins `(q, r)`. These tests show the pinning
//! directly: the honest quotient proves, and every neighbour fails — including
//! the specific `p`-shifted quotient the finding predicts should work.

use anyhow::Result;
use circuit::bigint::biguint::{CircuitBuilderBiguint, WitnessBigUint};
use circuit::bigint::div_rem::CircuitBuilderBiguintDivRem;
use circuit::types::config::{BIG_U96_LIMBS, Builder, C, F};
use num::BigUint;
use plonky2::field::types::Field64;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_data::CircuitConfig;

/// `a = 2^70 + 12345`, comfortably larger than the Goldilocks modulus, so the
/// finding's `a >= p` precondition is satisfied.
fn dividend() -> BigUint {
    (BigUint::from(1u32) << 70) + BigUint::from(12345u32)
}

fn divisor() -> BigUint {
    BigUint::from(1000u32)
}

fn goldilocks_order() -> BigUint {
    BigUint::from(F::ORDER)
}

/// Builds `div_rem(a, b)` and pins the quotient to `claimed_q`. Returns whether
/// a proof exists.
fn quotient_is_provable(claimed_q: &BigUint) -> bool {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

    let a = builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS);
    let b = builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS);

    let (div, _rem) = builder.div_rem_biguint(&a, &b);

    let claimed = builder.constant_biguint(claimed_q);
    builder.connect_biguint(&div, &claimed);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_biguint_target(&a, &dividend()).unwrap();
    pw.set_biguint_target(&b, &divisor()).unwrap();

    // `prove` alone is not a soundness oracle here: it returns Ok for a witness
    // that violates pure gate constraints, and only `verify` rejects those. The
    // acceptance criterion must be that a *verifiable* proof exists.
    match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_ok(),
        Err(_) => false,
    }
}

/// Sanity: the honest quotient proves. Without this the "failure" tests below
/// would be vacuous.
#[test]
fn honest_quotient_proves() -> Result<()> {
    let q = dividend() / divisor();
    assert!(
        quotient_is_provable(&q),
        "the true quotient must be provable, else the harness proves nothing"
    );
    Ok(())
}

/// The quotient is pinned: its immediate neighbours are unsatisfiable.
#[test]
fn neighbouring_quotients_are_unsatisfiable() {
    let q = dividend() / divisor();

    assert!(
        !quotient_is_provable(&(&q + BigUint::from(1u32))),
        "q + 1 must be unsatisfiable: it would force a negative remainder"
    );
    assert!(
        !quotient_is_provable(&(&q - BigUint::from(1u32))),
        "q - 1 must be unsatisfiable: it would force a remainder >= b"
    );
}

/// **The refutation proper.** The finding predicts a second valid quotient once
/// `a >= p`, obtained by shifting through the modulus. If the equation were
/// checked mod p, the quotient of `a - p` would be equally valid. It is not.
#[test]
fn the_p_shifted_quotient_predicted_by_the_finding_does_not_verify() {
    let a = dividend();
    let p = goldilocks_order();
    assert!(a > p, "the finding's precondition a >= p must actually hold");

    // What a mod-p equation would accept: the quotient of the other
    // representative of `a`'s residue class.
    let shifted_q = (&a - &p) / divisor();
    let honest_q = &a / divisor();
    assert_ne!(
        shifted_q, honest_q,
        "the two candidate quotients must differ, else the test is vacuous"
    );

    assert!(
        !quotient_is_provable(&shifted_q),
        "the p-shifted quotient must be unsatisfiable: the gadget asserts \
         q*b + r = a over the integers, never modulo p"
    );
}

/// The mechanism behind the refutation: `connect_biguint` does not stop at the
/// shorter operand, it asserts the excess high limbs are zero — so there is no
/// room for `div*b + rem` to exceed `a` by a multiple of the limb base.
#[test]
fn excess_high_limbs_are_asserted_zero() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

    // A 3-limb target whose value needs only 1 limb, connected to a 1-limb
    // constant: the top two limbs must be forced to zero.
    let wide = builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS);
    let narrow = builder.constant_biguint(&BigUint::from(7u32));
    builder.connect_biguint(&wide, &narrow);

    let data = builder.build::<C>();

    // Value 7 with a nonzero high limb: 7 + 2^32. If `connect_biguint`
    // truncated, this would pass.
    let mut pw = PartialWitness::<F>::new();
    pw.set_biguint_target(&wide, &(BigUint::from(7u32) + (BigUint::from(1u32) << 32)))
        .unwrap();

    assert!(
        match data.prove(pw) { Ok(pr) => data.verify(pr).is_err(), Err(_) => true },
        "connect_biguint must reject a nonzero high limb; this is what closes \
         the wraparound escape the finding assumes"
    );
}
