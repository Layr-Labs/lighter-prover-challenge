// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::witness::{Witness, WitnessWrite};

use crate::uint::u16::gadgets::arithmetic_u16::U16Target;

pub trait WitnessU16<F: PrimeField64>: Witness<F> {
    fn set_u16_target(&mut self, target: U16Target, value: u16) -> Result<()>;
    fn get_u16_target(&self, target: U16Target) -> u16;
}

impl<T: Witness<F>, F: PrimeField64> WitnessU16<F> for T {
    fn set_u16_target(&mut self, target: U16Target, value: u16) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u16(value))
    }

    fn get_u16_target(&self, target: U16Target) -> u16 {
        let x_u64 = self.get_target(target.0).to_canonical_u64();
        let low = x_u64 as u16;
        let high = (x_u64 >> 32) as u16;
        if high != 0 {
            panic!(
                "High bits are not zero in u16Target: {:?}, value: {}",
                target, x_u64
            );
        }
        low
    }
}

pub trait GeneratedValuesU16<F: Field> {
    fn set_u16_target(&mut self, target: U16Target, value: u16) -> Result<()>;
}

impl<F: Field> GeneratedValuesU16<F> for GeneratedValues<F> {
    fn set_u16_target(&mut self, target: U16Target, value: u16) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u16(value))
    }
}
