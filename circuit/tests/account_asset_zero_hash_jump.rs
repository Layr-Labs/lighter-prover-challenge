use anyhow::Result;
use circuit::bigint::biguint::CircuitBuilderBiguint;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::types::account_asset::{AccountAsset, AccountAssetTarget, AccountAssetTargetWitness};
use circuit::types::config::{Builder, C, F};
use num::BigUint;
use plonky2::field::types::{Field, Field64};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;

#[test]
fn empty_asset_hash_normalizes_into_a_canonical_nonempty_locked_balance() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let old = AccountAssetTarget::new(&mut builder);
    let old_witness = old.clone();

    let old_hash = old.hash(&mut builder);
    let zero_hash = builder.zero_hash_out();
    builder.connect_hashes(old_hash, zero_hash);

    // Adding D to malformed locked-balance limbs [p-t, t, 0] yields canonical limbs
    // [D-t, t, 0], i.e. D + t*(2^32-1), even though the old hash is zero.
    let delta = builder.constant_biguint(&BigUint::from(1_000u64));
    let new_locked_balance = builder.add_biguint(&old.locked_balance, &delta);

    let expected = AccountAssetTarget::new(&mut builder);
    let expected_witness = expected.clone();
    builder.connect_biguint(&old.balance, &expected.balance);
    builder.connect_biguint(&new_locked_balance, &expected.locked_balance);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::<F>::new();

    let t = 7u64;
    pw.set_target(
        old_witness.locked_balance.limbs[0].0,
        F::from_canonical_u64(F::ORDER - t),
    )?;
    pw.set_target(
        old_witness.locked_balance.limbs[1].0,
        F::from_canonical_u64(t),
    )?;
    pw.set_target(old_witness.locked_balance.limbs[2].0, F::ZERO)?;
    for limb in &old_witness.balance.limbs {
        pw.set_target(limb.0, F::ZERO)?;
    }
    pw.set_target(old_witness.index_0, F::ZERO)?;

    let expected_value = BigUint::from(1_000u64 + t * ((1u64 << 32) - 1));
    pw.set_account_asset_target(
        &expected_witness,
        &AccountAsset {
            index_0: 0,
            balance: BigUint::ZERO,
            locked_balance: expected_value,
        },
    )?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
