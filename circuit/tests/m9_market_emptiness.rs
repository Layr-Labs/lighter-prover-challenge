//! M9 — `MarketTarget::is_empty` / `hash`: a live market commits to the empty
//! hash, by two independent routes.
//!
//! Paired Lean proof: `review/lean/M9_MarketEmptiness.lean`.
//!
//! The database files M9 as "to-audit: MarketTarget has 15 virtual targets for
//! scalar fields plus order_book_root. Need to verify which are range-checked
//! and which feed leaf hash / is_empty."
//!
//! Answer: none of the 15 are range-checked (`market.rs:157-185` is 15 bare
//! `add_virtual_target()` calls), all 15 feed both `is_empty` and the hash
//! preimage — and `order_book_root` feeds the hash preimage but *not*
//! `is_empty`. That asymmetry is route 1 below and is not in the database at
//! all.

use anyhow::Result;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::types::config::{Builder, C, F};
use circuit::types::market::MarketTarget;
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

/// Set the 15 summed scalars, leaving `order_book_root` to the caller.
fn set_scalars(pw: &mut PartialWitness<F>, m: &MarketTarget, v: &[u64; 15]) -> Result<()> {
    pw.set_target(m.ask_nonce, F::from_canonical_u64(v[0]))?;
    pw.set_target(m.bid_nonce, F::from_canonical_u64(v[1]))?;
    pw.set_target(m.taker_fee, F::from_canonical_u64(v[2]))?;
    pw.set_target(m.maker_fee, F::from_canonical_u64(v[3]))?;
    pw.set_target(m.liquidation_fee, F::from_canonical_u64(v[4]))?;
    pw.set_target(m.min_base_amount, F::from_canonical_u64(v[5]))?;
    pw.set_target(m.min_quote_amount, F::from_canonical_u64(v[6]))?;
    pw.set_target(m.status, F::from_canonical_u64(v[7]))?;
    pw.set_target(m.order_quote_limit, F::from_canonical_u64(v[8]))?;
    pw.set_target(m.total_order_count, F::from_canonical_u64(v[9]))?;
    pw.set_target(m.market_type, F::from_canonical_u64(v[10]))?;
    pw.set_target(m.base_asset_id, F::from_canonical_u64(v[11]))?;
    pw.set_target(m.quote_asset_id, F::from_canonical_u64(v[12]))?;
    pw.set_target(m.size_extension_multiplier, F::from_canonical_u64(v[13]))?;
    pw.set_target(m.quote_extension_multiplier, F::from_canonical_u64(v[14]))?;
    Ok(())
}

/// **Route 1.** `order_book_root` is in the hash preimage and absent from
/// `is_empty`, so a market carrying a live order-book root hashes as the empty
/// market with every scalar honestly zero. No absorber, no out-of-range value —
/// nothing a range check would catch.
#[test]
fn live_order_book_root_hashes_as_empty_market() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let market = MarketTarget::new(&mut builder);

    let hash = market.hash(&mut builder);
    let empty = builder.zero_hash_out();
    builder.connect_hashes(hash, empty);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    set_scalars(&mut pw, &market, &[0; 15])?;
    pw.set_target(market.market_index, F::ZERO)?;
    // A live order book hanging off a market that commits to the empty hash.
    for (i, e) in market.order_book_root.elements.iter().enumerate() {
        pw.set_target(*e, F::from_canonical_u64(123456789 + i as u64))?;
    }

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// **Route 2.** The F1 absorber on the scalars: a fully configured perpetual
/// market — status 1, type 1, assets 2/3, fees 100/50, 7 resting orders — also
/// commits to the empty hash.
#[test]
fn configured_market_hashes_as_empty_via_absorber() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let market = MarketTarget::new(&mut builder);

    let hash = market.hash(&mut builder);
    let empty = builder.zero_hash_out();
    builder.connect_hashes(hash, empty);

    let data = builder.build::<C>();

    // visible = status 1 + base 2 + quote 3 + taker 100 + maker 50
    //         + min_base 1000 + type 1 + order_count 7 = 1164
    let visible: u64 = 1164;
    let size_ext: u64 = (1u64 << 63) - 1;
    let quote_ext = F::ORDER - visible - size_ext;

    let mut pw = PartialWitness::<F>::new();
    set_scalars(
        &mut pw,
        &market,
        &[
            0, 0, 100, 50, 0, 1000, 0, 1, 0, 7, 1, 2, 3, size_ext, quote_ext,
        ],
    )?;
    pw.set_target(market.market_index, F::ZERO)?;
    for e in market.order_book_root.elements.iter() {
        pw.set_target(*e, F::from_canonical_u64(987654321))?;
    }

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}

/// The control: an honest, in-range, non-empty market does *not* hash as empty.
/// Without this the two tests above would not distinguish a real aliasing from
/// a circuit that accepts everything.
#[test]
fn honest_nonempty_market_does_not_hash_as_empty() {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let market = MarketTarget::new(&mut builder);

    let hash = market.hash(&mut builder);
    let empty = builder.zero_hash_out();
    builder.connect_hashes(hash, empty);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::<F>::new();
    set_scalars(&mut pw, &market, &[0, 0, 100, 50, 0, 1000, 0, 1, 0, 7, 1, 2, 3, 0, 0])
        .unwrap();
    pw.set_target(market.market_index, F::ZERO).unwrap();
    for e in market.order_book_root.elements.iter() {
        pw.set_target(*e, F::ZERO).unwrap();
    }

    assert!(
        !proves_and_verifies(&data, pw),
        "a market with nonzero in-range scalars must not hash as empty; if this \
         passes the aliasing tests above prove nothing"
    );
}
