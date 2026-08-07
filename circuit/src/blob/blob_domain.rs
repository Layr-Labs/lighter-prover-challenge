// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use lazy_static::lazy_static;
use num::{BigUint, Num};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::constants::BLOB_WIDTH;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::blob::bls12_381_scalar_field::{BLS12_381_SCALAR_LIMBS, BLS12381Scalar};
use crate::blob::roots_of_unity::ROOTS_OF_UNITY;
use crate::builder::Builder;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::types::config::D;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

pub(crate) fn get_brp_roots_of_unity_as_constant<F: RichField + Extendable<D>>(
    builder: &mut Builder<F, D>,
) -> [NonNativeTarget<BLS12381Scalar>; BLOB_WIDTH] {
    ROOTS_OF_UNITY
        .split(',')
        .map(big_from_str_base_10)
        .map(|root| {
            let mut root_big = builder.constant_biguint(&root);
            if root_big.limbs.len() < BLS12_381_SCALAR_LIMBS {
                root_big
                    .limbs
                    .resize(BLS12_381_SCALAR_LIMBS, builder.zero_u32());
            }
            builder.biguint_to_nonnative(&root_big)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn big_from_str_base_10(s: &str) -> BigUint {
    BigUint::from_str_radix(s, 10).unwrap()
}

lazy_static! {
    pub static ref CARDINALITY: usize = 4096;
    pub static ref CARDINALITY_INV: BigUint = big_from_str_base_10(
        "52423073447788513186850219087163459498374710080483563692275874603576291491841"
    );
    pub static ref GENERATOR: BigUint = big_from_str_base_10(
        "39033254847818212395286706435128746857159659164139250548781411570340225835782"
    );
    pub static ref GENERATOR_INV: BigUint = big_from_str_base_10(
        "25829815649260311651249373569448671287036547786131478959351418120540316250978"
    );
}
