//! F5 (absorbing M4/M5) — order-book path aggregates are range-checked only at
//! the root, so an intermediate level can carry a negative subtree total.
//!
//! Paired Lean proof: `review/lean/F5_OrderBookAggregates.lean`.
//!
//! `get_order_book_path_delta` (circuit/src/matching_engine.rs:47-135) rebuilds
//! the path bottom-up with plain field `add`/`sub`, then registers exactly four
//! range checks — all on `order_book_path_after_vec[ORDER_BOOK_MERKLE_LEVELS - 1]`,
//! i.e. the root. `ORDER_BOOK_MERKLE_LEVELS = 80` (constants.rs:452) and
//! `BASE_SUM_BITS = QUOTE_SUM_BITS = 63` (constants.rs:73-74), so 79 intermediate
//! levels and every prover-supplied `path_before` node are unconstrained field
//! elements.
//!
//! The recurrence telescopes: `path_after[i] = path_before[i] + (order_after -
//! order_before)`. So the prover picks each `path_before[i]` freely and the
//! matching `path_after[i]` follows. A range check on the root does not bound
//! its summands in a field, because a level carrying `p - k` is `-k`.
//!
//! These tests call the real gadget, not a transcription.

use anyhow::Result;
use circuit::matching_engine::get_order_book_path_delta;
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::{BASE_SUM_BITS, ORDER_BOOK_MERKLE_LEVELS};
use circuit::types::order::OrderTarget;
use circuit::types::order_book_node::OrderBookNodeTarget;
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

/// The level whose aggregate we make negative. Any index below the root works;
/// 40 is chosen to be unambiguously interior.
const FORGED_LEVEL: usize = 40;

/// Build the real gadget. `also_range_check_interior` adds the check the circuit
/// omits, so the same witness can be run against both the shipped constraint set
/// and the repaired one.
fn run(interior_ask_base: F, also_range_check_interior: bool) -> bool {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

    let order_before = OrderTarget::new(&mut builder);
    let order_after = OrderTarget::new(&mut builder);
    let path_before: Vec<OrderBookNodeTarget> = (0..ORDER_BOOK_MERKLE_LEVELS)
        .map(|_| OrderBookNodeTarget::new(&mut builder))
        .collect();
    let path_before: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS] =
        path_before.try_into().expect("path length");

    let path_after =
        get_order_book_path_delta(&mut builder, &order_before, &path_before, &order_after);

    if also_range_check_interior {
        // The remediation: hold every level to the budget, not just the root.
        for node in path_after.iter() {
            builder.register_range_check(node.ask_base_sum, BASE_SUM_BITS);
        }
    }
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    // Keep the order delta zero so `path_after[i] == path_before[i]` exactly.
    // Whatever we witness for a `path_before` level is what that level reports.
    let mut pw = PartialWitness::<F>::new();
    for t in [
        order_before.price_index,
        order_before.nonce_index,
        order_before.ask_base_sum,
        order_before.ask_quote_sum,
        order_before.bid_base_sum,
        order_before.bid_quote_sum,
        order_after.price_index,
        order_after.nonce_index,
        order_after.ask_base_sum,
        order_after.ask_quote_sum,
        order_after.bid_base_sum,
        order_after.bid_quote_sum,
    ] {
        pw.set_target(t, F::ZERO).unwrap();
    }

    for (i, node) in path_before.iter().enumerate() {
        let ask_base = if i == FORGED_LEVEL {
            interior_ask_base
        } else {
            F::ZERO
        };
        pw.set_target(node.ask_base_sum, ask_base).unwrap();
        pw.set_target(node.ask_quote_sum, F::ZERO).unwrap();
        pw.set_target(node.bid_base_sum, F::ZERO).unwrap();
        pw.set_target(node.bid_quote_sum, F::ZERO).unwrap();
        for e in node.sibling_child_hash.elements.iter() {
            pw.set_target(*e, F::ZERO).unwrap();
        }
    }

    proves_and_verifies(&data, pw)
}

/// Control: an honest interior aggregate is accepted. Without this the rejection
/// test below would prove nothing.
#[test]
fn honest_interior_aggregate_is_accepted() {
    assert!(
        run(F::from_canonical_u64(1_000_000), false),
        "an in-range interior aggregate must be accepted"
    );
}

/// **The counterexample.** Level 40 of the *after* path reports `p - 1000` —
/// a subtree holding minus one thousand units of ask size. The root's 63-bit
/// range check does not see it, and the proof verifies.
#[test]
fn interior_level_can_carry_a_negative_aggregate() -> Result<()> {
    let negative_thousand = F::from_canonical_u64(F::ORDER - 1000);
    assert!(
        run(negative_thousand, false),
        "the shipped constraint set range-checks only level {}, so an interior \
         negative aggregate must slip through",
        ORDER_BOOK_MERKLE_LEVELS - 1
    );
    Ok(())
}

/// The value really is outside the budget the root is held to — it is not merely
/// a large-but-legal number. `p - 1000` exceeds `2^63`, so the same value at the
/// root would be rejected.
#[test]
fn the_forged_value_would_fail_the_root_check() {
    let negative_thousand = F::ORDER - 1000;
    assert!(
        negative_thousand >= (1u64 << BASE_SUM_BITS),
        "p - 1000 must exceed the 2^{BASE_SUM_BITS} budget, else the test is vacuous"
    );
}

/// **The remediation.** Register the same check on every level and the identical
/// witness is rejected. This isolates the defect to the missing checks rather
/// than to anything about the recurrence itself.
///
/// The rejection surfaces as a panic rather than a failed verification: a
/// 63-bit check is not one of `CUSTOM_GATE_SIZES` and 63 is not a multiple of 8,
/// so `perform_registered_range_checks` routes it to `split_le`, whose
/// `BaseSumGate` generator asserts the value fits its limbs
/// (vendor/plonky2/plonky2/src/gates/base_sum.rs:247). That is a strictly
/// stronger rejection than a verify failure — the prover cannot even build a
/// witness.
#[test]
#[should_panic(expected = "Integer too large to fit in given number of limbs")]
fn range_checking_every_level_rejects_it() {
    let negative_thousand = F::from_canonical_u64(F::ORDER - 1000);
    run(negative_thousand, true);
}

/// ...and the fix is not vacuous: an honest aggregate still passes with every
/// level checked.
#[test]
fn honest_aggregate_still_passes_with_full_range_checks() {
    assert!(
        run(F::from_canonical_u64(1_000_000), true),
        "range-checking every level must not reject honest aggregates"
    );
}

// ---------------------------------------------------------------------------
// Reachability: what the isolated tests above do and do not establish.
// ---------------------------------------------------------------------------

/// SCOPE CORRECTION. The tests above build `path_before` as free targets. In
/// production (tx_constraints.rs:2354) the before-path is merkle-verified
/// against `market_before.order_book_root` *before* reaching
/// `get_order_book_path_delta` (:2373), and the result is committed via
/// `recalculate_order_book_tree_root` (:2380). So those tests establish a
/// property of the gadget in isolation, not that a prover can reach it.
///
/// This test establishes the thing that *is* reachable, and it is the reason
/// the isolated result matters: the order-book tree never validates its own
/// aggregates. `verify_order_book_tree_merkle_proof` folds each node's four
/// sums into the hash chain verbatim (`internal_hash()` is the four sums, not a
/// Poseidon image of them) and checks only that the chain reproduces the root.
/// Nothing anywhere asserts that a parent's sum equals the sum of its children.
///
/// Consequence: the aggregates are sound only as an *induction* — zero at
/// genesis (`EMPTY_ORDER_BOOK_TREE_ROOT`, set at l1_create_market.rs:392),
/// maintained by the uniform `+delta` update — and that induction is never
/// re-checked. A single transaction that commits one bad delta corrupts every
/// ancestor on its path permanently and invisibly, because every later
/// transaction just adds its own delta to whatever is already there.
#[test]
fn merkle_proof_accepts_internally_inconsistent_aggregates() -> Result<()> {
    use circuit::hash_utils::CircuitBuilderHashUtils;
    use circuit::order_book_tree_helpers::{
        recalculate_order_book_tree_root, verify_order_book_tree_merkle_proof,
    };

    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

    let leaf = builder.add_virtual_hash();
    let path: Vec<OrderBookNodeTarget> = (0..ORDER_BOOK_MERKLE_LEVELS)
        .map(|_| OrderBookNodeTarget::new(&mut builder))
        .collect();
    let path: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS] =
        path.try_into().expect("path length");
    let helpers: [_; ORDER_BOOK_MERKLE_LEVELS] =
        core::array::from_fn(|_| builder._false());

    // The root is whatever this path hashes to...
    let root = recalculate_order_book_tree_root(&mut builder, leaf, &path, helpers);
    // ...and the proof is then checked against it, exactly as production does.
    verify_order_book_tree_merkle_proof(&mut builder, &root, leaf, &path, helpers);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    for e in leaf.elements.iter() {
        pw.set_target(*e, F::ZERO)?;
    }
    // Aggregates that are nonsense as a tree: a leaf-adjacent node claiming a
    // huge total under an empty leaf, and a parent claiming *less* than its
    // child. No node's sum is the sum of anything below it.
    for (i, node) in path.iter().enumerate() {
        let absurd = match i {
            0 => F::from_canonical_u64(1_000_000_000),
            1 => F::from_canonical_u64(7),
            _ => F::ZERO,
        };
        pw.set_target(node.ask_base_sum, absurd)?;
        pw.set_target(node.ask_quote_sum, F::ZERO)?;
        pw.set_target(node.bid_base_sum, F::ZERO)?;
        pw.set_target(node.bid_quote_sum, F::ZERO)?;
        for e in node.sibling_child_hash.elements.iter() {
            pw.set_target(*e, F::ZERO)?;
        }
    }

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
