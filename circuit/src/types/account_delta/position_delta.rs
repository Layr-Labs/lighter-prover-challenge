// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::{BIGU16_U64_LIMBS, Builder};

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PositionDelta {
    #[serde(rename = "lfrps")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub funding_rate_prefix_sum_delta: BigInt, // value is in range [-2^62 + 1, 2^62 - 1], thus the diff is in range [-2^63 + 2, 2^63 - 2]

    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub position_delta: BigInt, // value is in range [-2^56 + 1, 2^56 - 1], thus the diff is in range [-2^57 + 2, 2^57 - 2]
}

#[derive(Debug, Clone, Default)]
pub struct PositionDeltaTarget {
    pub funding_rate_prefix_sum_delta: BigIntU16Target,
    pub position_delta: BigIntU16Target,
}

impl PositionDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PositionDeltaTarget {
            funding_rate_prefix_sum_delta: builder
                .add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            position_delta: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        PositionDeltaTarget {
            funding_rate_prefix_sum_delta: builder.zero_bigint_u16(),
            position_delta: builder.zero_bigint_u16(),
        }
    }

    pub fn select_position_delta(
        builder: &mut Builder,
        flag: BoolTarget,
        a: &Self,
        b: &Self,
    ) -> Self {
        Self {
            funding_rate_prefix_sum_delta: builder.select_bigint_u16(
                flag,
                &a.funding_rate_prefix_sum_delta,
                &b.funding_rate_prefix_sum_delta,
            ),
            position_delta: builder.select_bigint_u16(flag, &a.position_delta, &b.position_delta),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_bigint_u16(
            &self.funding_rate_prefix_sum_delta,
            &format!("{} funding_rate_prefix_sum_delta", tag),
        );
        builder.println_bigint_u16(&self.position_delta, &format!("{} position_delta", tag));
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let is_funding_rate_prefix_sum_delta_zero =
            builder.is_zero_bigint_u16(&self.funding_rate_prefix_sum_delta);
        let is_position_delta_zero = builder.is_zero_bigint_u16(&self.position_delta);
        builder.and(
            is_funding_rate_prefix_sum_delta_zero,
            is_position_delta_zero,
        )
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut elements = vec![self.funding_rate_prefix_sum_delta.sign.target];
        for limb in self.funding_rate_prefix_sum_delta.abs.limbs.iter() {
            elements.push(limb.0);
        }
        elements.push(self.position_delta.sign.target);
        for limb in self.position_delta.abs.limbs.iter() {
            elements.push(limb.0);
        }
        let nonzero_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let zero_hash = builder.zero_hash_out();

        let is_empty = self.is_empty(builder);
        builder.select_hash(is_empty, &zero_hash, &nonzero_hash)
    }
}

pub fn random_access_position_delta(
    builder: &mut Builder,
    access_index: Target,
    v: Vec<PositionDeltaTarget>,
) -> PositionDeltaTarget {
    assert!(v.len() % 64 == 0);
    PositionDeltaTarget {
        funding_rate_prefix_sum_delta: builder.random_access_bigint_u16(
            access_index,
            v.iter()
                .map(|x| x.funding_rate_prefix_sum_delta.clone())
                .collect(),
            BIGU16_U64_LIMBS,
        ),
        position_delta: builder.random_access_bigint_u16(
            access_index,
            v.iter().map(|x| x.position_delta.clone()).collect(),
            BIGU16_U64_LIMBS,
        ),
    }
}

pub trait PositionDeltaTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_position_delta_target(
        &mut self,
        a: &PositionDeltaTarget,
        b: &PositionDelta,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    PositionDeltaTargetWitness<F> for T
{
    fn set_position_delta_target(
        &mut self,
        a: &PositionDeltaTarget,
        b: &PositionDelta,
    ) -> Result<()> {
        self.set_bigint_u16_target(
            &a.funding_rate_prefix_sum_delta,
            &b.funding_rate_prefix_sum_delta,
        )?;
        self.set_bigint_u16_target(&a.position_delta, &b.position_delta)?;

        Ok(())
    }
}
