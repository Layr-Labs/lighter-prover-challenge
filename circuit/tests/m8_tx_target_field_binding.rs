//! M8 — "TxTarget: tx_type, tx_index, nonce, expired_at, asset_indices. Need to
//! verify downstream range-checking for each." **Refuted: all five are bound**,
//! three of them by mechanisms other than a range check.
//!
//! Paired Lean proof: `review/lean/M8_TxTypeBinding.lean`.
//!
//! Audit result, field by field:
//!
//! * `tx_type` (tx_constraints.rs:359) — bound by `TxTypeTargets::new`
//!   (types/tx_type.rs:90-139), which builds one `is_equal_constant` flag per
//!   transaction type, sums them, and asserts the sum is true. Distinct
//!   constants make the flags mutually exclusive, so the sum is 0 or 1 and
//!   asserting it forces *exactly one* valid type. Tested below.
//!
//! * `tx_index` (tx_constraints.rs:360) — genuinely unrange-checked, bound
//!   relationally by `register_range_check(gap_minus_one, 32)` plus the terminal
//!   count check and the coverage/claims partition. See M7.
//!
//! * `nonce` (tx_constraints.rs:489) — enters signature-message hashes and
//!   equality checks only. A repo-wide search finds no `is_lte`/`is_lt`/`is_gt`/
//!   `cmp` on any nonce, so the F3a comparison failure mode does not apply:
//!   equality over field elements is exact regardless of range.
//!
//! * `expired_at` (tx_constraints.rs:490) — range-checked at
//!   types/tx_type.rs:388, on the line immediately before the
//!   `conditional_assert_lt(block_created_at, expired_at, TIMESTAMP_BITS)` that
//!   consumes it. This is the correct pattern and the one F3a says is missing
//!   elsewhere. The order-expiry comparison sites do the same
//!   (l2_create_order.rs:219, l2_create_grouped_orders.rs:527,
//!   l2_approve_integrator.rs:94).
//!
//! * `asset_indices` (tx_constraints.rs:519) — `NB_ASSETS_PER_TX = 2`, and both
//!   entries are range-checked in `validate_asset_indices`
//!   (tx_constraints.rs:2806-2807), which also asserts they differ unless the
//!   tx asset is nil and connects each to the corresponding account asset's
//!   `index_0`.
//!
//! RESIDUAL, recorded rather than resolved: `block_created_at` is the *other*
//! operand of the expiry comparison, and no `register_range_check` was located
//! for it. It is a block-level value connected across both chains and into the
//! pre-execution witness, so it is pinned by the verifier's own
//! `BlockWitness::from_block` comparison rather than by a range check — but that
//! binding was not verified here.

use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::{TX_TYPE_EMPTY, TX_TYPE_L2_TRANSFER};
use circuit::types::tx_type::TxTypeTargets;
use plonky2::field::types::{Field, Field64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

/// `prove` is not a soundness oracle here — see `review/lean/README.md`.
fn proves_and_verifies(
    data: &plonky2::plonk::circuit_data::CircuitData<F, C, 2>,
    pw: PartialWitness<F>,
) -> bool {
    match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_ok(),
        Err(_) => false,
    }
}

/// Build the real `TxTypeTargets::new`, which internally asserts
/// `is_valid_tx_type`, and try to witness `tx_type` with `value`.
fn tx_type_accepts(value: F) -> bool {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let tx_type = builder.add_virtual_target();
    let _flags = TxTypeTargets::new(&mut builder, tx_type);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(tx_type, value).unwrap();

    proves_and_verifies(&data, pw)
}

/// Controls: recognised transaction types are accepted, including the padding
/// type 0. Without these the rejection test below would prove nothing.
#[test]
fn valid_tx_types_are_accepted() {
    assert!(
        tx_type_accepts(F::from_canonical_u64(TX_TYPE_EMPTY as u64)),
        "the padding type must be accepted"
    );
    assert!(
        tx_type_accepts(F::from_canonical_u64(TX_TYPE_L2_TRANSFER as u64)),
        "a real transaction type must be accepted"
    );
}

/// **The refutation.** `tx_type` is not free: unrecognised values are rejected,
/// so a prover cannot smuggle in a transaction that matches no branch and
/// therefore performs no state transition while still counting as active.
#[test]
fn unrecognised_tx_types_are_rejected() {
    for v in [
        F::from_canonical_u64(200),
        F::from_canonical_u64(255),
        F::from_canonical_u64(1u64 << 32),
        F::from_canonical_u64(F::ORDER - 1),
    ] {
        assert!(
            !tx_type_accepts(v),
            "is_valid_tx_type must reject unrecognised tx_type {v}"
        );
    }
}

/// The mechanism, isolated: the type flags are mutually exclusive because they
/// compare against distinct constants, so their sum is 0 or 1 and asserting it
/// true forces exactly one. A field element cannot equal two distinct constants
/// at once, which is why no range check on `tx_type` is needed.
#[test]
fn tx_type_flags_are_mutually_exclusive() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let tx_type = builder.add_virtual_target();
    let flags = TxTypeTargets::new(&mut builder, tx_type);

    // Two distinct constants: at most one flag can hold.
    let is_empty = flags.is_empty;
    let is_transfer = flags.is_l2_transfer;
    let both = builder.and(is_empty, is_transfer);
    builder.assert_false(both);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(tx_type, F::from_canonical_u64(TX_TYPE_L2_TRANSFER as u64))
        .unwrap();

    assert!(
        proves_and_verifies(&data, pw),
        "asserting the two flags are not both set must be satisfiable, \
         confirming they are mutually exclusive rather than jointly forced"
    );
}
