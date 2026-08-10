//! F3b — `is_lte_cache` is keyed on `(a, b)` only, so an earlier narrow
//! comparison answers a later wide one.
//!
//! Paired Lean proof: `review/lean/F3b_CacheWidthBlind.lean`.
//!
//! `circuit/src/builder/types.rs:41`
//!
//! ```text
//! pub(crate) is_lte_cache: HashMap<(Target, Target), (BoolTarget, usize)>,
//! ```
//!
//! The width is stored in the *value*, so it cannot affect lookup;
//! `subtractive_comparison.rs:79-90` reads it back only to format a `warn!` and
//! then returns the stale `BoolTarget`. In a release prover even the warning is
//! gone: the workspace pins `log` with `release_max_level_off`, which compiles
//! `warn!` out entirely.
//!
//! The decisive test is `same_query_two_outcomes_depending_on_call_order`: the
//! identical 64-bit question is answered differently depending only on whether
//! an unrelated 16-bit call happened first.

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

/// The cache hit itself: two different widths, one `BoolTarget`.
#[test]
fn different_widths_return_the_same_target() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    let narrow = builder.is_lte(a, b, 16);
    let wide = builder.is_lte(a, b, 64);

    assert_eq!(
        narrow.target, wide.target,
        "is_lte_cache ignores num_bits, so the 64-bit query must be answered by \
         the 16-bit result that was already cached"
    );
}

/// `cmp` has the identical omission on the identical lines — any fix must cover
/// both caches.
#[test]
fn cmp_cache_has_the_same_omission() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    let narrow = builder.cmp(a, b, 16);
    let wide = builder.cmp(a, b, 64);

    assert_eq!(
        narrow.target, wide.target,
        "cmp_cache is keyed on (a, b) only, exactly as is_lte_cache is"
    );
}

/// A circuit whose only comparison is the honest 64-bit one rejects
/// `p - 1 <= 0`, as it should: that path decomposes both operands into
/// range-checked u32 limbs and compares them as integers.
#[test]
fn wide_comparison_alone_rejects_the_forgery() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    let wide = builder.is_lte(a, b, 64);
    builder.assert_true(wide);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(F::ORDER - 1)).unwrap();
    pw.set_target(b, F::ZERO).unwrap();

    assert!(
        !proves_and_verifies(&data, pw),
        "the 64-bit path is sound on its own: p-1 <= 0 must be rejected"
    );
}

/// **The counterexample.** Same operands, same 64-bit question, opposite answer
/// — decided entirely by a preceding 16-bit call on the same targets.
#[test]
fn same_query_two_outcomes_depending_on_call_order() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    // Some earlier, unrelated part of the circuit compared these two targets in
    // a 16-bit domain. Its result is now cached under the key (a, b).
    let _narrow = builder.is_lte(a, b, 16);

    // This caller believes it is asking a 64-bit question. It receives the
    // 16-bit answer, whose backing constraints are the 2^16 subtraction window.
    let wide = builder.is_lte(a, b, 64);
    builder.assert_true(wide);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(F::ORDER - 1))?;
    pw.set_target(b, F::ZERO)?;

    // Identical to `wide_comparison_alone_rejects_the_forgery` except for the
    // discarded narrow call above — and that one line flips the outcome.
    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// The remediation, mirroring what `split_le_cache` already does: rebuild when
/// the requested width differs. Distinct target pairs miss the cache, so each
/// question is answered by constraints built at its own width — and then the
/// forgery is rejected.
#[test]
fn distinct_targets_are_not_cross_contaminated() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();
    let a2 = builder.add_virtual_target();
    let b2 = builder.add_virtual_target();

    let _narrow = builder.is_lte(a, b, 16);
    let wide = builder.is_lte(a2, b2, 64);
    builder.connect(a, a2);
    builder.connect(b, b2);
    builder.assert_true(wide);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(F::ORDER - 1)).unwrap();
    pw.set_target(b, F::ZERO).unwrap();

    assert!(
        !proves_and_verifies(&data, pw),
        "when the wide query builds its own constraints the forgery is caught; \
         only the shared cache key hides it"
    );
}
