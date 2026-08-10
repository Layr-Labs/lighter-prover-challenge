//! F3a / F6 — `is_lte` compares field representatives with unpinned operands,
//! so the largest field element passes as `<= 0`.
//!
//! Paired Lean proof: `review/lean/F3a_ComparisonFlip.lean`.
//!
//! `CircuitBuilderSubtractiveComparison`'s own doc comment claims:
//!
//! > These will fail if one of the targets exceed given `num_bits`, hence,
//! > inputs are expected to be range-checked beforehand.
//!
//! They do not fail. The `num_bits <= 16` bucket constrains only
//!
//! ```text
//! output_result = b - a + 2^16 * output_borrow   (in the field)
//! output_result ∈ [0, 2^16)
//! output_borrow ∈ {0, 1}
//! ```
//!
//! (`circuit/src/uint/u16/gates/subtraction_u16.rs:119-140`) and leaves `a` and
//! `b` free. For `a = p - 1, b = 0` the only satisfying assignment is
//! `output_borrow = 0`, i.e. "not greater than" — so the flip is forced, not
//! merely available.

use anyhow::Result;
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::comparison::subtractive_comparison::CircuitBuilderSubtractiveComparison;
use circuit::types::config::{Builder, C, F};
use plonky2::field::types::{Field, Field64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

/// Acceptance criterion for the rejection tests in this file.
///
/// `prove` alone is not a soundness oracle in this vendored plonky2: it returns
/// `Ok` for a witness that violates pure gate constraints, and only `verify`
/// rejects those. (Generator-caught violations do surface as a `prove` error,
/// which is why some rejections show up earlier.) "Rejected" must therefore mean
/// "no verifiable proof exists", not merely "prove failed".
fn proves_and_verifies(
    data: &plonky2::plonk::circuit_data::CircuitData<F, C, 2>,
    pw: PartialWitness<F>,
) -> bool {
    match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_ok(),
        Err(_) => false,
    }
}

/// The counterexample: `is_lte(p - 1, 0, 16)` is satisfied, so the circuit
/// accepts a proof asserting that the largest field element is `<= 0`.
#[test]
fn largest_field_element_compares_as_lte_zero() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    // The only constraint in the circuit is the gadget itself, asserted true.
    let a_lte_b = builder.is_lte(a, b, 16);
    builder.assert_true(a_lte_b);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    // p - 1: a perfectly legal canonical field element, and the largest one.
    pw.set_target(a, F::from_canonical_u64(F::ORDER - 1))?;
    pw.set_target(b, F::ZERO)?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// The same operands under an honest integer comparison disagree: `p - 1 > 0`.
/// Asserting the gadget's answer is *false* is unsatisfiable, which is what
/// "the flip is forced" means — the prover has no honest option available.
#[test]
fn the_honest_answer_is_unsatisfiable_for_those_operands() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    let a_lte_b = builder.is_lte(a, b, 16);
    builder.assert_false(a_lte_b);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(F::ORDER - 1)).unwrap();
    pw.set_target(b, F::ZERO).unwrap();

    assert!(
        !proves_and_verifies(&data, pw),
        "borrow = 1 must be unsatisfiable for (p-1, 0): the 2^16 window admits \
         only output_borrow = 0, which is why the comparison flips rather than \
         failing loudly"
    );
}

/// The dual, and the remediation: with both operands range-checked into the
/// 16-bit window the gadget is exact, so the same assertion fails for genuinely
/// ordered operands.
#[test]
fn in_range_operands_compare_correctly() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    // What the caller is assumed to have done and, at every site the database
    // flags, has not. `register_range_check` only records the target — the
    // constraints appear when `perform_registered_range_checks` runs, which is
    // what every production `define` does at the end.
    builder.register_range_check(a, 16);
    builder.register_range_check(b, 16);
    builder.perform_registered_range_checks();

    let a_lte_b = builder.is_lte(a, b, 16);
    builder.assert_true(a_lte_b);

    let data = builder.build::<C>();

    // 60000 <= 5 is false, and now the range check makes it unprovable.
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(60000)).unwrap();
    pw.set_target(b, F::from_canonical_u64(5)).unwrap();
    assert!(
        !proves_and_verifies(&data, pw),
        "with range-checked operands the gadget must reject 60000 <= 5"
    );
}
