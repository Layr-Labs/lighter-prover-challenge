//! M7 — "JumpStateTarget fields are unchecked witness fields". **Refuted as
//! stated.**
//!
//! Paired Lean proof: `review/lean/M7_JumpStateContinuity.lean`.
//!
//! `last_active_tx_index`, `run_start_prev_index` and `tx_count` are indeed
//! allocated by `add_virtual_target()` and never range-checked, but they are
//! never free either: the base case pins all 27 jump-state elements,
//! `define_jump_step` *computes* each successor with select/add, and
//! `JumpStateTarget::connect` ties chunk boundaries together. The tests below
//! exercise the two ends of that argument.
//!
//! What is genuinely unconstrained is `tx_index` (tx_constraints.rs:360). It is
//! bounded only relationally, by the `register_range_check(gap_minus_one, 32)`
//! in `define_jump_step`. `define_jump_step` is private, so the gap tests here
//! transcribe those four lines verbatim rather than calling it — they establish
//! what that range check does and does not buy, not that the caller wires it up
//! correctly. The base-case tests do use the real gadget.

use anyhow::Result;
use circuit::block_tx::JumpStateTarget;
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::types::config::{Builder, C, F};
use plonky2::field::types::{Field, Field64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

// ---------------------------------------------------------------------------
// The base case, against the real `conditional_assert_initial`.
// ---------------------------------------------------------------------------

/// Acceptance criterion for every test in this file.
///
/// IMPORTANT: `prove` is not a soundness oracle in this vendored plonky2. It
/// returns `Ok` for a witness that violates pure gate constraints — only
/// `verify` rejects those. (Violations caught by a generator do surface as a
/// `prove` error, which is why some rejections show up earlier.) A PoC that
/// tests `prove(..).is_ok()` alone can therefore report a forgery as accepted
/// when no verifiable proof exists. Always require both.
fn proves_and_verifies(data: &plonky2::plonk::circuit_data::CircuitData<F, C, 2>,
                       pw: PartialWitness<F>) -> bool {
    match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_ok(),
        Err(_) => false,
    }
}

/// Build a circuit asserting the initial jump state unconditionally, and try to
/// witness `last_active_tx_index` with `value`.
fn base_case_accepts(value: F) -> bool {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let jump = JumpStateTarget::new(&mut builder);
    let state_root = builder.add_virtual_hash();
    let delta_root = builder.add_virtual_hash();

    let enabled = builder._true();
    jump.conditional_assert_initial(&mut builder, enabled, state_root, delta_root);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(jump.last_active_tx_index, value).unwrap();
    pw.set_target(jump.run_start_prev_index, F::NEG_ONE).unwrap();
    pw.set_target(jump.tx_count, F::ZERO).unwrap();
    for i in 0..4 {
        pw.set_target(state_root.elements[i], F::ZERO).unwrap();
        pw.set_target(delta_root.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.prev_new_state_root.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.prev_new_delta_root.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.run_start_old_state_root.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.run_start_old_delta_root.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.coverage_hash.elements[i], F::ZERO).unwrap();
        pw.set_target(jump.claims_hash.elements[i], F::ZERO).unwrap();
    }

    proves_and_verifies(&data, pw)
}

/// The sentinel `-1` is accepted — the control, without which the rejection
/// tests below would prove nothing.
#[test]
fn base_case_accepts_the_sentinel() {
    assert!(
        base_case_accepts(F::NEG_ONE),
        "JumpState::initial uses F::NEG_ONE, so the base case must accept it"
    );
}

/// **The refutation.** `conditional_assert_initial` asserts
/// `last_active_tx_index + 1 == 0` in the field, and exactly one field element
/// satisfies that. Not being range-checked buys the prover nothing: every other
/// value is rejected, including the ones a range check would have been aimed at.
#[test]
fn base_case_rejects_every_other_value() {
    for v in [
        F::ZERO,
        F::ONE,
        F::from_canonical_u64(500),
        F::from_canonical_u64(1u64 << 32),
        F::from_canonical_u64(F::ORDER - 2),
    ] {
        assert!(
            !base_case_accepts(v),
            "base case must pin last_active_tx_index to the -1 sentinel; \
             accepted {v} instead"
        );
    }
}

// ---------------------------------------------------------------------------
// The gap range check, transcribed from define_jump_step (private).
// ---------------------------------------------------------------------------

/// Transcription of `define_jump_step`'s monotonicity constraint for an active
/// transaction:
///
/// ```text
/// let gap = builder.sub(tx_index, jump.last_active_tx_index);
/// let gap_minus_one = builder.sub(gap, one);
/// builder.register_range_check(gap_minus_one, 32);
/// ```
///
/// Returns whether `tx_index` is acceptable following `last_active`.
fn gap_check_accepts(last_active: F, tx_index: F) -> bool {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let prev = builder.add_virtual_target();
    let idx = builder.add_virtual_target();

    let one = builder.one();
    let gap = builder.sub(idx, prev);
    let gap_minus_one = builder.sub(gap, one);
    builder.register_range_check(gap_minus_one, 32);

    // `register_range_check` only *records* the target; the constraints are
    // emitted here. Every production circuit does this at the end of `define`
    // (block_tx_constraints.rs:172, block_constraints.rs:764, and eight more).
    // See `deferred_range_checks_are_silently_dropped_if_never_performed`.
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(prev, last_active).unwrap();
    pw.set_target(idx, tx_index).unwrap();

    proves_and_verifies(&data, pw)
}

/// From the `-1` sentinel, index 0 is the only sensible first index and it is
/// accepted: `gap = 0 - (-1) = 1`, so `gap_minus_one = 0`.
#[test]
fn sentinel_admits_index_zero() {
    assert!(gap_check_accepts(F::NEG_ONE, F::ZERO));
}

/// Indices must strictly increase: repeating or going backwards drives
/// `gap_minus_one` negative, which is `p - k` in the field and fails the 32-bit
/// range check.
#[test]
fn indices_cannot_repeat_or_decrease() {
    let prev = F::from_canonical_u64(100);
    assert!(
        !gap_check_accepts(prev, prev),
        "a repeated index gives gap = 0, so gap_minus_one = -1: must be rejected"
    );
    assert!(
        !gap_check_accepts(prev, F::from_canonical_u64(99)),
        "a decreasing index must be rejected"
    );
    assert!(
        gap_check_accepts(prev, F::from_canonical_u64(101)),
        "the next index must be accepted, else this harness proves nothing"
    );
}

/// The bound is exactly 2^32: a gap of `2^32` is the largest accepted, and one
/// more is rejected. This is what caps how fast the index can run away, and so
/// how many chain steps it would take to wrap the field.
#[test]
fn gap_is_capped_at_two_to_the_32() {
    let prev = F::from_canonical_u64(0);
    assert!(
        gap_check_accepts(prev, F::from_canonical_u64(1u64 << 32)),
        "gap_minus_one = 2^32 - 1 is the largest value the 32-bit check admits"
    );
    assert!(
        !gap_check_accepts(prev, F::from_canonical_u64((1u64 << 32) + 1)),
        "gap_minus_one = 2^32 must be rejected"
    );
}

/// The residual, made concrete. Without the range check the prover could pick
/// `tx_index = last_active` shifted by `p - 1`, which is the *same* index in the
/// field — indices could repeat forever. The check is what forbids it, and it
/// forbids it only for gaps below 2^32, which is why the no-wraparound argument
/// is bounded by the number of chain steps rather than being unconditional.
#[test]
fn wraparound_value_is_rejected_by_the_range_check() {
    let prev = F::from_canonical_u64(100);
    // prev + 1 + (p - 1) == prev in the field.
    let replay = prev;
    assert!(
        !gap_check_accepts(prev, replay),
        "the field-wrapping replay index must be rejected; it is only the \
         32-bit range check that rejects it"
    );
}

/// A latent hazard found while writing these tests, adjacent to M7 rather than
/// part of it. `register_range_check` is a *deferred* registration: it records
/// the target and emits nothing. The constraints appear only when
/// `perform_registered_range_checks()` runs. A circuit that registers checks and
/// forgets that call has **no range checks at all** and still builds and proves.
///
/// `Builder::build` (builder/types.rs:94-100) does notice — via `log::warn!`.
/// The workspace pins `log` with `release_max_level_off`, so in a release
/// prover that warning is compiled out and the omission is completely silent.
/// This is the same compiled-out-guard shape as F3b.
///
/// Not a live defect: all twelve production call sites do perform their checks
/// (verified by grep). It is a defence-in-depth gap — a build-time `assert!`
/// or a `#[must_use]`-style guard would fail loudly instead.
#[test]
fn deferred_range_checks_are_silently_dropped_if_never_performed() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let prev = builder.add_virtual_target();
    let idx = builder.add_virtual_target();

    let one = builder.one();
    let gap = builder.sub(idx, prev);
    let gap_minus_one = builder.sub(gap, one);
    builder.register_range_check(gap_minus_one, 32);

    // Deliberately NOT calling perform_registered_range_checks().
    let data = builder.build::<C>();

    // The decreasing index that `indices_cannot_repeat_or_decrease` rejects.
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(prev, F::from_canonical_u64(100))?;
    pw.set_target(idx, F::from_canonical_u64(99))?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
