// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use num::{BigInt, BigUint};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::account_delta::AccountDeltaTarget;
use crate::types::account_delta::position_delta::{
    PositionDelta, PositionDeltaTarget, PositionDeltaTargetWitness,
};
use crate::types::account_delta::public_pool_delta::{
    PublicPoolInfoDelta, PublicPoolInfoDeltaTarget, PublicPoolInfoDeltaWitness,
    PublicPoolShareDelta, PublicPoolShareDeltaTarget, PublicPoolShareDeltaWitness,
};
use crate::types::config::{BIG_U96_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::{
    ASSET_LIST_SIZE, ASSET_LIST_SIZE_BITS, NIL_ACCOUNT_INDEX, NIL_MASTER_ACCOUNT_INDEX,
    POSITION_LIST_SIZE, POSITION_LIST_SIZE_BITS, SHARES_DELTA_LIST_SIZE,
};

/// Similar to AccountDelta, but comes with all positions instead of one position and
/// a position tree root.
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct AccountDeltaFullLeaf {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "l1")]
    #[serde(deserialize_with = "deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint,
    #[serde(rename = "at", default)]
    pub account_type: u8,
    #[serde(rename = "aad")]
    #[serde(deserialize_with = "deserializers::all_aggregated_asset_deltas")]
    pub aggregated_asset_deltas: [BigInt; ASSET_LIST_SIZE],
    #[serde(rename = "pd")]
    #[serde(deserialize_with = "deserializers::positions_delta")]
    pub positions_delta: [PositionDelta; POSITION_LIST_SIZE],
    #[serde(rename = "ppsd")]
    #[serde(deserialize_with = "deserializers::public_pool_shares_delta")]
    pub public_pool_shares_delta: [PublicPoolShareDelta; SHARES_DELTA_LIST_SIZE],
    #[serde(rename = "ppid", default)]
    pub public_pool_info_delta: PublicPoolInfoDelta,
}

impl AccountDeltaFullLeaf {
    pub fn nil() -> Self {
        Self {
            account_index: NIL_ACCOUNT_INDEX,
            ..Self::default()
        }
    }
}

impl Default for AccountDeltaFullLeaf {
    fn default() -> Self {
        Self {
            account_index: NIL_MASTER_ACCOUNT_INDEX,
            l1_address: BigUint::ZERO,
            account_type: 0,
            aggregated_asset_deltas: array::from_fn(|_| BigInt::ZERO),
            positions_delta: array::from_fn(|_| PositionDelta::default()),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDelta::default()),
            public_pool_info_delta: PublicPoolInfoDelta::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccountDeltaFullLeafTarget {
    pub account_index: Target,
    pub l1_address: BigUintTarget,
    pub account_type: Target,
    pub aggregated_asset_deltas: [BigIntTarget; ASSET_LIST_SIZE],
    pub positions_delta: [PositionDeltaTarget; POSITION_LIST_SIZE],
    pub public_pool_shares_delta: [PublicPoolShareDeltaTarget; SHARES_DELTA_LIST_SIZE],
    pub public_pool_info_delta: PublicPoolInfoDeltaTarget,
}

impl Default for AccountDeltaFullLeafTarget {
    fn default() -> Self {
        AccountDeltaFullLeafTarget {
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),
            aggregated_asset_deltas: array::from_fn(|_| BigIntTarget::default()),
            positions_delta: array::from_fn(|_| PositionDeltaTarget::default()),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::default()),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::default(),
        }
    }
}

impl AccountDeltaFullLeafTarget {
    pub fn new(builder: &mut Builder) -> Self {
        AccountDeltaFullLeafTarget {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS),
            account_type: builder.add_virtual_target(),
            aggregated_asset_deltas: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),
            positions_delta: array::from_fn(|_| PositionDeltaTarget::new(builder)),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::new(builder)),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::new(builder),
        }
    }

    pub fn to_account_delta(&self, builder: &mut Builder) -> AccountDeltaTarget {
        AccountDeltaTarget {
            account_index: self.account_index,
            l1_address: self.l1_address.clone(),
            account_type: self.account_type,
            aggregated_asset_deltas: array::from_fn(|i| self.aggregated_asset_deltas[i].clone()),
            public_pool_shares_delta: self.public_pool_shares_delta,
            public_pool_info_delta: self.public_pool_info_delta.clone(),
            asset_delta_root: self.get_asset_delta_root(builder),
            position_delta_root: self.get_position_delta_root(builder),
            positions_delta: PositionDeltaTarget::default(),
            partial_hash: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> (HashOutTarget, BoolTarget) {
        self.to_account_delta(builder).hash_with_is_empty(builder)
    }

    pub fn get_asset_delta_root(&self, builder: &mut Builder) -> HashOutTarget {
        let mut level_hashes = self
            .aggregated_asset_deltas
            .iter()
            .map(|a| {
                let mut elements = vec![a.sign.target];
                elements.extend_from_slice(&a.abs.limbs.iter().map(|x| x.0).collect::<Vec<_>>());
                let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
                let empty_hash = builder.zero_hash_out();
                let is_empty = builder.is_zero_bigint(a);
                builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
            })
            .collect::<Vec<_>>();
        assert!((1 << ASSET_LIST_SIZE_BITS) == level_hashes.len());
        let mut iter_count = level_hashes.len() / 2;
        for _ in 0..ASSET_LIST_SIZE_BITS {
            for j in 0..iter_count {
                level_hashes[j] =
                    builder.hash_two_to_one(&level_hashes[2 * j], &level_hashes[2 * j + 1]);
            }
            iter_count /= 2;
        }
        level_hashes[0]
    }

    pub fn get_position_delta_root(&self, builder: &mut Builder) -> HashOutTarget {
        let mut level_hashes = self
            .positions_delta
            .iter()
            .map(|p| p.hash(builder))
            .collect::<Vec<_>>();
        level_hashes.push(builder.zero_hash_out()); // 256th
        assert!((1 << POSITION_LIST_SIZE_BITS) == level_hashes.len());
        let mut iter_count = level_hashes.len() / 2;
        for _ in 0..POSITION_LIST_SIZE_BITS {
            for j in 0..iter_count {
                level_hashes[j] =
                    builder.hash_two_to_one(&level_hashes[2 * j], &level_hashes[2 * j + 1]);
            }
            iter_count /= 2;
        }
        level_hashes[0]
    }

    pub fn print(&self, builder: &mut Builder, print_assets: bool, tag: &str) {
        builder.println(self.account_index, &format!("{} account_index", tag));
        builder.println_biguint(&self.l1_address, &format!("{}: l1_address", tag));
        builder.println(self.account_type, &format!("{} account_type", tag));
        self.public_pool_info_delta
            .print(builder, &format!("{} public_pool_info_delta", tag));

        if print_assets {
            for i in 0..ASSET_LIST_SIZE {
                builder.println_bigint(
                    &self.aggregated_asset_deltas[i],
                    &format!("{} aggregated_asset_deltas[{}]", tag, i),
                );
            }
        }
    }
}

pub trait AccountDeltaLeafTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_delta_leaf_target(
        &mut self,
        a: &AccountDeltaFullLeafTarget,
        b: &AccountDeltaFullLeaf,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountDeltaLeafTargetWitness<F> for T
{
    fn set_account_delta_leaf_target(
        &mut self,
        a: &AccountDeltaFullLeafTarget,
        b: &AccountDeltaFullLeaf,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_target(a.account_type, F::from_canonical_u8(b.account_type))?;
        for i in 0..b.aggregated_asset_deltas.len() {
            self.set_bigint_target(&a.aggregated_asset_deltas[i], &b.aggregated_asset_deltas[i])?;
        }
        for i in 0..b.positions_delta.len() {
            self.set_position_delta_target(&a.positions_delta[i], &b.positions_delta[i])?;
        }
        self.set_public_pool_info_delta(&a.public_pool_info_delta, &b.public_pool_info_delta)?;
        for i in 0..b.public_pool_shares_delta.len() {
            self.set_public_pool_share_delta(
                &a.public_pool_shares_delta[i],
                &b.public_pool_shares_delta[i],
            )?;
        }

        Ok(())
    }
}
