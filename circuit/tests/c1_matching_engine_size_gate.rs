//! C1, last link — does `execute_matching` route a forged register through to
//! the order-book write?
//!
//! Two questions, answered separately because only one of them is executable.
//!
//! (A) THE ROUTING GATE — settled by source trace, not by execution.
//!     `tx_state.order` becomes the register-derived order at
//!     matching_engine.rs:1521-1527:
//!
//!         let insert_taker_to_order_book =
//!             builder.and_not(insert_taker_order, is_pending_trigger_status_not_na);
//!         tx_state.order = select_order_target(
//!             builder, insert_taker_to_order_book, &register_order, &tx_state.order);
//!
//!     and `tx_state.order` is exactly the `order_after` handed to
//!     `get_order_book_path_delta` (tx_constraints.rs:2373). The gate resolves to
//!
//!         insert_taker_order          from :398-407, the ordinary
//!                                     "non crossing non ioc limit - put order to
//!                                     orderbook" branch, whose condition is
//!                                     `update_status_flags AND order_leaf_is_empty`
//!                                     with `limit_flag` conditionally asserted;
//!         NOT is_pending_trigger_status_not_na
//!                                     i.e. pending_trigger_status == TRIGGER_STATUS_NA,
//!                                     which is 0.
//!
//!     None of `update_status_flags`, `order_leaf_is_empty`, `limit_flag` or
//!     `pending_trigger_status` is a function of `pending_size`. A forged
//!     register shaped as an ordinary resting limit order on an empty book slot
//!     therefore reaches the write with its size unexamined.
//!
//!     LIMITATION, stated plainly: this is a reading of the branch structure.
//!     `TxState` has 40 fields and only a placeholder `Default`, so calling the
//!     real `execute_matching` from an integration test would mean rebuilding
//!     the transaction circuit's setup — the test would be a transcription, and
//!     transcriptions are what this file set out to avoid. No end-to-end
//!     execution of the routing is claimed.
//!
//! (B) THE SIZE COMPARISON — executable, and tested below.
//!     Before the branch, matching_engine.rs:434-440 does
//!
//!         let mut optimistic_trade_amount = builder.min(
//!             &[ tx_state.account_order.remaining_base_amount,
//!                tx_state.register_stack[0].pending_size, // anything written to register is range-checked
//!             ], ORDER_SIZE_BITS);
//!
//!     That comment is the assumption F1a falsifies. It holds for registers a
//!     transaction handler wrote in this same transaction; it does not hold for
//!     `register_stack_before`, which is a witness input bound only by the
//!     non-injective `RegisterStackTarget::hash`. And the call is a 48-bit
//!     subtractive comparison on that unpinned operand — precisely F3a.

use anyhow::Result;
use circuit::comparison::subtractive_comparison::CircuitBuilderSubtractiveComparison;
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::ORDER_SIZE_BITS;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

/// Run the exact call from matching_engine.rs:434 — `min([remaining, size], 48)`
/// — and report which operand it selects.
fn min_selects(remaining: u64, pending_size: u64) -> Result<u64> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();
    let m = builder.min(&[a, b], ORDER_SIZE_BITS);
    let out = builder.add_virtual_public_input();
    builder.connect(m, out);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(remaining))?;
    pw.set_target(b, F::from_canonical_u64(pending_size))?;

    let proof = data.prove(pw)?;
    data.verify(proof.clone())?;
    Ok(proof.public_inputs[0].to_canonical_u64())
}

/// Control: on honest, in-range operands the gadget picks the true minimum.
#[test]
fn min_is_correct_on_honest_operands() -> Result<()> {
    assert_eq!(min_selects(500, 1000)?, 500);
    assert_eq!(min_selects(1000, 500)?, 500);
    Ok(())
}

/// **The size comparison misbehaves on a forged register.** With
/// `pending_size = p - 1000` — the value C1's forged register carries — the
/// 48-bit `min` at matching_engine.rs:434 does not return the honest minimum.
///
/// Whichever way it resolves, the assumption in the source comment is violated:
/// the gadget is being asked to compare a value that is not a 48-bit integer,
/// and `CircuitBuilderSubtractiveComparison`'s own contract says operands "are
/// expected to be range-checked beforehand".
#[test]
fn min_misbehaves_on_a_negative_pending_size() -> Result<()> {
    let remaining: u64 = 1000;
    let forged_size = F::ORDER - 1000; // p - 1000, i.e. -1000

    let got = min_selects(remaining, forged_size)?;

    // Honest integer semantics would give `remaining`, since p-1000 is the
    // largest field element there is.
    assert_ne!(
        got, remaining,
        "min returned the honest answer; if this ever holds, the 48-bit \
         comparison is sound on out-of-range operands and this part of C1 \
         collapses"
    );
    assert_eq!(
        got, forged_size,
        "the gadget selects the out-of-range operand as the 'minimum' — the \
         negative size wins the comparison it should have lost"
    );
    Ok(())
}

/// The remediation, at the same point: range-check `pending_size` to
/// `ORDER_SIZE_BITS` before it is compared, and the forged value is rejected
/// outright rather than winning the comparison.
#[test]
fn range_checking_the_operand_rejects_it() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();
    builder.register_range_check(b, ORDER_SIZE_BITS);
    let _m = builder.min(&[a, b], ORDER_SIZE_BITS);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(a, F::from_canonical_u64(1000)).unwrap();
    pw.set_target(b, F::from_canonical_u64(F::ORDER - 1000)).unwrap();

    let rejected = match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_err(),
        Err(_) => true,
    };
    assert!(
        rejected,
        "with the operand range-checked, p-1000 must not be admissible"
    );
}
