// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gadgets/arithmetic_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use std::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use super::gadgets::arithmetic_u16::{CircuitBuilderU16, U16Target};
use crate::builder::Builder;
use crate::uint::u16::witness::GeneratedValuesU16;
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn split_u64_to_u16s_le(&mut self, x: Target, num_limbs: usize) -> Vec<U16Target> {
        assert!(num_limbs <= 4, "Can only split to at most 4 u16 limbs");

        if let Some(cached) = self.u16_split_cache.get(&x) {
            let mut result = cached.clone();
            if result.len() > num_limbs {
                panic!(
                    "Cached result has more limbs than requested. {} vs {}",
                    result.len(),
                    num_limbs
                );
            }

            result.resize(num_limbs, self.zero_u16());
            return result;
        }

        let limbs = (0..num_limbs)
            .map(|_| self.add_virtual_u16_target_safe())
            .collect::<Vec<_>>();

        self.add_simple_generator(SplitToU16Generator {
            x,
            limbs: limbs.clone(),
            _phantom: PhantomData,
        });

        if num_limbs != 4 {
            let mut acc = limbs.last().unwrap().0;
            let multiplier = self.constant_u64(1 << 16);
            for limb in limbs.iter().rev().skip(1) {
                acc = self.mul_add(acc, multiplier, limb.0);
            }
            self.connect(acc, x);
        } else {
            let multiplier_16 = self.constant_u64(1 << 16);
            let multiplier_32 = self.constant_u64(1 << 32);
            // Limbs are range-checked to be 16 bits, so low and high can fit 32 bits
            let low = U32Target(self.mul_add(limbs[1].0, multiplier_16, limbs[0].0));
            let high = U32Target(self.mul_add(limbs[3].0, multiplier_16, limbs[2].0));

            let result = self.mul_add(high.0, multiplier_32, low.0);
            self.connect(result, x);

            // If high is 2^32 - 1, then low must be 0. Because 2^32(2^32 - 1) + 1 = 2^64 - 2^32 + 1 which is prime itself
            // If high is <= 2^32 - 2, then 2^32(2^32 - 2) + 2^32 - 1 = 2^64 - 2^32 - 1 which is smaller than the prime.
            let max = self.constant_u64((1u64 << 32) - 1);
            let is_high_max = self.is_equal(high.0, max);
            self.conditional_assert_zero_u32(is_high_max, low);
        }

        self.u16_split_cache.insert(x, limbs.clone());

        limbs
    }
}

#[derive(Debug, Default)]
pub struct SplitToU16Generator<F: RichField + Extendable<D>, const D: usize> {
    x: Target,
    limbs: Vec<U16Target>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SplitToU16Generator<F, D>
{
    fn id(&self) -> String {
        "SplitToU16Generator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target_vec(&self.limbs.iter().map(|t| t.0).collect::<Vec<_>>())?;

        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let x = src.read_target()?;
        let limbs = src
            .read_target_vec()?
            .iter()
            .map(|t| U16Target(*t))
            .collect();

        Ok(Self {
            x,
            limbs,
            _phantom: PhantomData,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let x = witness.get_target(self.x);
        let mut acc = x.to_canonical_u64();

        for limb in &self.limbs {
            out_buffer.set_u16_target(*limb, acc as u16)?;
            acc >>= 16;
        }

        Ok(())
    }
}
