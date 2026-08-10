//! C1, spot block-level guard — the real reason the negative-order injection
//! does not go through on the spot insert path.
//!
//! Paired with C1's gadget-level PoCs (f1_f5_negative_order_injection.rs) and the
//! perps lock-bypass proof (review/lean/C1_LockBypass.lean).
//!
//! On the spot insert path, `increment_locked_balance_for_order` calls
//! `get_locked_amount_and_ask_asset_index` (matching_engine.rs:2906-2934), which
//! is the real guard:
//!
//! ```text
//! let base_amount_big = builder.target_to_biguint(base_amount);   // UNSIGNED
//! let locked_amount   = builder.mul_biguint(&base_amount_big, &multiplier);
//! let (success, locked_amount) = builder.try_trim_biguint(&locked_amount, BIG_U96_LIMBS);
//! builder.conditional_assert_true(is_enabled, success);
//! ```
//!
//! `target_to_biguint` decomposes the wire as an unsigned integer. Where the
//! order-book path reads a negative `pending_size = p - k` as `-k`, this path
//! reads the very same wire as the full field magnitude `p - k ≈ 1.8e19`, and
//! locks `(p - k) * multiplier`. So placing a negative-size order on spot forces
//! the account to hold an astronomical locked balance — the injection is not
//! "free", and a self-consistent block witness would have to fund it.
//!
//! This test drives the real gadget and reads the locked amount out, quantifying
//! the guard: honest size 10 locks 30; size `p - 10` locks `3*(p-10) ≈ 5.5e19`.

use anyhow::Result;
use circuit::bigint::biguint::CircuitBuilderBiguint;
use circuit::matching_engine::get_locked_amount_and_ask_asset_index;
use circuit::types::config::{Builder, C, F};
use circuit::types::market::MarketTarget;
use num::BigUint;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

/// Multiplier the market applies to an ask's base size.
const MULT: u64 = 3;

/// Run the real `get_locked_amount_and_ask_asset_index` for an ask of
/// `base_amount`, exposing the locked amount, and return it as a BigUint.
fn locked_amount_for(base_amount: F) -> Result<BigUint> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let market = MarketTarget::new(&mut builder);
    let base = builder.add_virtual_target();
    let price = builder.add_virtual_target();
    let is_ask = builder._true();
    let enabled = builder._true();

    let (locked, _asset) =
        get_locked_amount_and_ask_asset_index(&mut builder, enabled, &market, base, price, is_ask);
    builder.register_public_input_biguint(&locked);
    builder.perform_registered_range_checks();

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    // MarketTarget::new builds is_lte(market_index, ..) internally, so its own
    // targets need witnessing even though the lock gadget doesn't read them.
    pw.set_target(market.market_index, F::ZERO)?;
    pw.set_target(market.market_type, F::ZERO)?;
    // Ask branch reads size_extension_multiplier; set it and the asset ids.
    pw.set_target(market.size_extension_multiplier, F::from_canonical_u64(MULT))?;
    pw.set_target(market.quote_extension_multiplier, F::ONE)?;
    pw.set_target(market.base_asset_id, F::ONE)?;
    pw.set_target(market.quote_asset_id, F::from_canonical_u64(2))?;
    pw.set_target(base, base_amount)?;
    pw.set_target(price, F::from_canonical_u64(7))?;

    let proof = data.prove(pw)?;
    data.verify(proof.clone())?;

    // Reconstruct the locked amount from the public-input limbs (base 2^32).
    let limbs: Vec<u64> = proof
        .public_inputs
        .iter()
        .map(|f| f.to_canonical_u64())
        .collect();
    let mut v = BigUint::ZERO;
    for limb in limbs.iter().rev() {
        v = (v << 32) + BigUint::from(*limb);
    }
    Ok(v)
}

/// Control: an honest ask of size 10 locks `10 * 3 = 30`.
#[test]
fn honest_size_locks_the_expected_small_amount() -> Result<()> {
    let locked = locked_amount_for(F::from_canonical_u64(10))?;
    assert_eq!(locked, BigUint::from(30u64), "honest lock must be 10*3 = 30");
    Ok(())
}

/// **The spot block-level guard.** A negative-size order — `pending_size = p-10`,
/// the value C1 injects into the order book as `-10` — is read here as the full
/// unsigned magnitude, and locks `3*(p-10)`. That is ~5.5e19 units the account
/// must genuinely hold: the injection is not free on spot.
#[test]
fn negative_size_forces_an_astronomical_lock() -> Result<()> {
    let neg_ten = F::from_canonical_u64(F::ORDER - 10); // p - 10, i.e. -10
    let locked = locked_amount_for(neg_ten)?;

    let expected = BigUint::from(F::ORDER - 10) * BigUint::from(MULT);
    assert_eq!(
        locked, expected,
        "the unsigned read locks (p-10)*3, not 30"
    );

    // Concretely: about 5.5e19, dwarfing any representable honest balance
    // (i64::MAX is ~9.2e18). Funding this in a consistent block witness is the
    // real obstacle to a spot-path C1 exploit.
    let threshold = BigUint::from(1u64) << 65; // ~3.7e19
    assert!(
        locked > threshold,
        "the forced lock must exceed 2^65; got {locked}"
    );
    Ok(())
}
