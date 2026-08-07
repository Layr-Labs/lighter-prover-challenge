// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};

use crate::bigint::biguint::BigUintTarget;
use crate::builder::Builder;
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;

pub trait CircuitBuilderScalar<F: RichField + Extendable<D>, const D: usize> {
    fn register_scalar_public_input(&mut self, scalar: &BigUintTarget);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderScalar<F, D> for Builder<F, D> {
    fn register_scalar_public_input(&mut self, target: &BigUintTarget) {
        for limb in target.limbs.iter() {
            self.register_public_input(limb.0);
        }
    }
}

pub trait PartialWitnessScalar<F: RichField> {
    fn set_scalar_target(&mut self, target: &BigUintTarget, value: ECgFp5Scalar) -> Result<()>;
    fn get_scalar_target(&mut self, target: &BigUintTarget) -> ECgFp5Scalar;
}

impl<F: RichField> PartialWitnessScalar<F> for PartialWitness<F> {
    fn set_scalar_target(&mut self, target: &BigUintTarget, value: ECgFp5Scalar) -> Result<()> {
        let value = value.to_canonical_biguint();
        for (&limb, limb_value) in target.limbs.iter().zip(value.to_u32_digits()) {
            self.set_target(limb.0, F::from_canonical_u32(limb_value))?;
        }

        Ok(())
    }

    fn get_scalar_target(&mut self, target: &BigUintTarget) -> ECgFp5Scalar {
        let mut limbs = Vec::new();
        for limb in target.limbs.iter() {
            limbs.push(self.get_target(limb.0).to_canonical_u64() as u32);
        }

        let as_biguint = BigUint::from_slice(&limbs);
        ECgFp5Scalar::from_noncanonical_biguint(as_biguint)
    }
}
