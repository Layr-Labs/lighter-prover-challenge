//! F1e — two `is_empty` aliasing sites the finding database does not record.
//!
//! Paired Lean proof: `review/lean/F1_EmptinessTaxonomy.lean`.
//!
//! Enumerating every `is_empty(&self, builder)` in `circuit/src/types/` turns up
//! six gadgets built on one aggregated `add_many` sum. The database records four
//! (F1a base_register_info, F1b account_asset, F1c account_order, M9 market).
//! The two below are not recorded anywhere:
//!
//!   * `circuit/src/types/margined_asset.rs:158` — 16 operands
//!   * `circuit/src/types/asset.rs:115`          —  8 operands
//!
//! The remaining eight emptiness gadgets use one `is_zero` per field combined
//! with `multi_and`, which admits no absorber at all; `sound_pattern_rejects_
//! the_same_trick` pins that contrast down against the real `order.rs` gadget.

use anyhow::Result;
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::types::asset::AssetTarget;
use circuit::types::config::{Builder, C, F};
use circuit::types::margined_asset::MarginedAssetTarget;
use circuit::types::order::OrderTarget;
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

/// `margined_asset.rs:158`. The three BigUints use the *safe* constructor, so
/// their limbs are range-checked and cannot absorb; the seven scalars are bare
/// `add_virtual_target()` and can. A live lending-risk configuration — LTV,
/// liquidation threshold, liquidation factor — reads as empty.
#[test]
fn margined_asset_with_live_risk_parameters_reads_as_empty() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let asset = MarginedAssetTarget::new(&mut builder);

    let is_empty = asset.is_empty(&mut builder);
    builder.assert_true(is_empty);

    let data = builder.build::<C>();

    // Visible content: 3 + 8000 + 9000 + 500000 + 250000 + 1000 + 500 = 768503.
    let visible: u64 = 768503;
    let divider: u64 = (1u64 << 63) - 1;
    let liquidation_fee = F::ORDER - visible - divider;

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(asset.asset_index, F::from_canonical_u64(3))?;
    pw.set_target(asset.loan_to_value, F::from_canonical_u64(8000))?;
    pw.set_target(asset.liquidation_threshold, F::from_canonical_u64(9000))?;
    pw.set_target(asset.liquidation_factor, F::from_canonical_u64(500000))?;
    pw.set_target(asset.index_price, F::from_canonical_u64(250000))?;
    // The two absorbers.
    pw.set_target(asset.index_price_divider, F::from_canonical_u64(divider))?;
    pw.set_target(asset.liquidation_fee, F::from_canonical_u64(liquidation_fee))?;
    // Range-checked limbs stay honest.
    pw.set_target(asset.global_supply_cap.limbs[0].0, F::from_canonical_u64(1000))?;
    pw.set_target(asset.global_supply_cap.limbs[1].0, F::ZERO)?;
    pw.set_target(asset.global_supply_cap.limbs[2].0, F::ZERO)?;
    pw.set_target(asset.user_supply_cap.limbs[0].0, F::from_canonical_u64(500))?;
    pw.set_target(asset.user_supply_cap.limbs[1].0, F::ZERO)?;
    pw.set_target(asset.user_supply_cap.limbs[2].0, F::ZERO)?;
    for limb in &asset.total_supplied_amount.limbs {
        pw.set_target(limb.0, F::ZERO)?;
    }

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// `asset.rs:115`. Here the BigUints use the *unsafe* constructor, so any two of
/// the six limbs absorb.
#[test]
fn asset_with_live_extension_multiplier_reads_as_empty() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let asset = AssetTarget::new(&mut builder);

    let is_empty = asset.is_empty(&mut builder);
    builder.assert_true(is_empty);

    let data = builder.build::<C>();

    // Visible content: 1000000 + 100 + 200 + 1 + 2 = 1000303.
    let visible: u64 = 1000303;
    let absorber_a: u64 = (1u64 << 63) - 1;
    let absorber_b = F::ORDER - visible - absorber_a;

    let mut pw = PartialWitness::<F>::new();
    pw.set_target(
        asset.extension_multiplier.limbs[0].0,
        F::from_canonical_u64(1000000),
    )?;
    pw.set_target(asset.extension_multiplier.limbs[1].0, F::ZERO)?;
    pw.set_target(asset.min_transfer_amount.limbs[0].0, F::from_canonical_u64(100))?;
    pw.set_target(
        asset.min_transfer_amount.limbs[1].0,
        F::from_canonical_u64(absorber_a),
    )?;
    pw.set_target(
        asset.min_withdrawal_amount.limbs[0].0,
        F::from_canonical_u64(200),
    )?;
    pw.set_target(
        asset.min_withdrawal_amount.limbs[1].0,
        F::from_canonical_u64(absorber_b),
    )?;
    pw.set_target(asset.margin_mode, F::ONE)?;
    pw.set_target(asset.margin_index, F::from_canonical_u64(2))?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// The contrast that makes the taxonomy actionable. `order.rs:151` checks each
/// field with its own `is_zero` and combines with `multi_and`. No absorber
/// exists: making the sum vanish is irrelevant when the gadget never forms a
/// sum. This is why the V8.4 audit's order-book-leaf negative was correct.
#[test]
fn sound_pattern_rejects_the_same_trick() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let order = OrderTarget::new(&mut builder);

    let is_empty = order.is_empty(&mut builder);
    builder.assert_true(is_empty);

    let data = builder.build::<C>();

    // The pair that would absorb under an aggregated sum: x and p - x.
    let x: u64 = 1_000_000;
    let mut pw = PartialWitness::<F>::new();
    pw.set_target(order.ask_base_sum, F::from_canonical_u64(x)).unwrap();
    pw.set_target(order.ask_quote_sum, F::from_canonical_u64(F::ORDER - x))
        .unwrap();
    pw.set_target(order.bid_base_sum, F::ZERO).unwrap();
    pw.set_target(order.bid_quote_sum, F::ZERO).unwrap();

    assert!(
        !proves_and_verifies(&data, pw),
        "per-field is_zero + multi_and admits no absorber: each field must be \
         zero on its own, so the aggregated-sum trick cannot apply"
    );
}
