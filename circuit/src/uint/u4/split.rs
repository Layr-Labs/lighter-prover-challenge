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
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::builder::Builder;

impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn split_to_u4s_le(&mut self, x: Target, num_limbs: usize) -> Vec<Target> {
        let limbs = (0..num_limbs)
            .map(|_| {
                let target = self.add_virtual_target();
                self.register_range_check(target, 4);
                target
            })
            .collect::<Vec<_>>();

        self.add_simple_generator(SplitToU4Generator {
            x,
            limbs: limbs.clone(),
            _phantom: PhantomData,
        });

        let mut acc = *limbs.last().unwrap();
        let multiplier = self.constant_u64(1 << 4);
        for limb in limbs.iter().rev().skip(1) {
            acc = self.mul_add(acc, multiplier, *limb);
        }
        self.connect(acc, x);

        limbs
    }
}

#[derive(Debug, Default)]
pub struct SplitToU4Generator<F: RichField + Extendable<D>, const D: usize> {
    x: Target,
    limbs: Vec<Target>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SplitToU4Generator<F, D>
{
    fn id(&self) -> String {
        "SplitToU4Generator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target_vec(&self.limbs)?;

        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let x = src.read_target()?;
        let limbs = src.read_target_vec()?;

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
            out_buffer.set_target(*limb, F::from_canonical_u64(acc & 0xF))?;
            acc >>= 4;
        }

        Ok(())
    }
}
