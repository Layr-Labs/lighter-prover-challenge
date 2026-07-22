// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::types::constants::POSITION_LIST_SIZE;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
pub struct PriceUpdates {
    #[serde(rename = "i")]
    #[serde(deserialize_with = "deserializers::price_updates")]
    #[serde(default = "deserializers::default_price_updates")]
    pub index_price: [u32; POSITION_LIST_SIZE],

    #[serde(rename = "m")]
    #[serde(deserialize_with = "deserializers::price_updates")]
    #[serde(default = "deserializers::default_price_updates")]
    pub mark_price: [u32; POSITION_LIST_SIZE],
}

impl Default for PriceUpdates {
    fn default() -> Self {
        Self {
            index_price: [0; POSITION_LIST_SIZE],
            mark_price: [0; POSITION_LIST_SIZE],
        }
    }
}

#[derive(Debug)]
pub struct PriceUpdatesTarget {
    // 32 bits each
    pub index_price: [Target; POSITION_LIST_SIZE],
    pub mark_price: [Target; POSITION_LIST_SIZE],
}

impl PriceUpdatesTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            index_price: builder
                .add_virtual_targets(POSITION_LIST_SIZE)
                .try_into()
                .unwrap(),
            mark_price: builder
                .add_virtual_targets(POSITION_LIST_SIZE)
                .try_into()
                .unwrap(),
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> QuinticExtensionTarget {
        let mut elements: Vec<Target> = Vec::with_capacity(2 * POSITION_LIST_SIZE);

        self.index_price
            .iter()
            .zip_eq(self.mark_price.iter())
            .for_each(|(&index_price, &mark_price)| {
                elements.push(index_price);
                elements.push(mark_price);
            });

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

pub trait PriceUpdatesWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_price_updates_target(&mut self, t: &PriceUpdatesTarget, n: &PriceUpdates) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PriceUpdatesWitness<F> for T {
    fn set_price_updates_target(&mut self, t: &PriceUpdatesTarget, n: &PriceUpdates) -> Result<()> {
        for i in 0..POSITION_LIST_SIZE {
            self.set_target(t.index_price[i], F::from_canonical_u32(n.index_price[i]))?;
            self.set_target(t.mark_price[i], F::from_canonical_u32(n.mark_price[i]))?;
        }

        Ok(())
    }
}
