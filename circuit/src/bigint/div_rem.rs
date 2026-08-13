// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/nonnative/gadgets/biguint.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::{Ok, Result};
use num::{BigUint, Integer, Zero};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::PartitionWitness;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use super::biguint::{
    BigUintTarget, CircuitBuilderBiguint, GeneratedValuesBigUint, WitnessBigUint,
};
use super::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::builder::Builder;
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::uint::u32::witness::{GeneratedValuesU32, WitnessU32};

pub trait CircuitBuilderBiguintDivRem<F: RichField + Extendable<D>, const D: usize> {
    /// Returns the quotient and remainder of a divided by b. If b is zero, returns (0, 0).
    fn div_rem_biguint(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, BigUintTarget);
    /// Like [`Self::div_rem_biguint`], but allocates the quotient with `quotient_num_limbs` limbs.
    /// The constraints are unsatisfiable if the quotient does not fit in that many limbs,
    /// so callers must guarantee the bound for all satisfiable witnesses.
    fn div_rem_biguint_trimmed(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> (BigUintTarget, BigUintTarget);
    /// Returns the quotient of a divided by b. If b is zero, returns 0.
    fn div_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    /// Returns the quotient of a divided by b with a quotient of `quotient_num_limbs` limbs.
    /// See [`Self::div_rem_biguint_trimmed`].
    fn div_biguint_trimmed(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> BigUintTarget;
    /// Returns the remainder of a divided by b. If b is zero, returns 0.
    fn rem_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    /// Returns the ceiling of the quotient of a divided by b. If b is zero, returns a.
    fn ceil_div_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    /// Returns the quotient of a divided by b, where a is a signed integer. If b is zero, returns 0.
    fn div_bigint_by_biguint(&mut self, a: &BigIntTarget, b: &BigUintTarget) -> BigIntTarget;
    /// Returns the quotient of a divided by b with a quotient of `quotient_num_limbs` limbs,
    /// where a is a signed integer. See [`Self::div_rem_biguint_trimmed`].
    fn div_bigint_by_biguint_trimmed(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> BigIntTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBiguintDivRem<F, D>
    for Builder<F, D>
{
    fn div_rem_biguint(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, BigUintTarget) {
        self.div_rem_biguint_trimmed(a, b, a.num_limbs())
    }

    fn div_rem_biguint_trimmed(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> (BigUintTarget, BigUintTarget) {
        assert!(quotient_num_limbs <= a.num_limbs());
        let key = (a.clone(), b.clone(), quotient_num_limbs);
        if let Some(result) = self.div_rem_biguint_cache.get(&key) {
            return result.clone();
        }

        let div = self.add_virtual_biguint_target_safe(quotient_num_limbs);
        let rem = self.add_virtual_biguint_target_safe(b.num_limbs());

        self.add_simple_generator(BigUintDivRemGenerator::<F, D> {
            a: a.clone(),
            b: b.clone(),
            div: div.clone(),
            rem: rem.clone(),
            _phantom: PhantomData,
        });

        let is_div_by_zero = self.is_zero_biguint(b);
        let is_not_div_by_zero = self.not(is_div_by_zero);
        self.conditional_assert_zero_biguint(is_div_by_zero, &div);
        self.conditional_assert_zero_biguint(is_div_by_zero, &rem);

        let div_b = self.mul_biguint(&div, b);
        let div_b_plus_rem = self.add_biguint(&div_b, &rem);
        self.conditional_assert_eq_biguint(is_not_div_by_zero, a, &div_b_plus_rem);

        self.conditional_assert_lt_biguint(is_not_div_by_zero, &rem, b);

        self.div_rem_biguint_cache
            .insert(key, (div.clone(), rem.clone()));

        (div, rem)
    }

    fn div_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let (div, _rem) = self.div_rem_biguint(a, b);
        div
    }

    fn div_biguint_trimmed(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> BigUintTarget {
        let (div, _rem) = self.div_rem_biguint_trimmed(a, b, quotient_num_limbs);
        div
    }

    fn div_bigint_by_biguint(&mut self, a: &BigIntTarget, b: &BigUintTarget) -> BigIntTarget {
        self.div_bigint_by_biguint_trimmed(a, b, a.abs.num_limbs())
    }

    fn div_bigint_by_biguint_trimmed(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        quotient_num_limbs: usize,
    ) -> BigIntTarget {
        let div_abs = self.div_biguint_trimmed(&a.abs, b, quotient_num_limbs);
        let div_bigint = self.biguint_to_bigint(&div_abs);
        BigIntTarget {
            abs: div_abs,
            sign: SignTarget::new_unsafe(self.mul(div_bigint.sign.target, a.sign.target)),
        }
    }

    fn ceil_div_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let (div, rem) = self.div_rem_biguint(a, b);
        let is_zero_rem = self.is_zero_biguint(&rem);
        let one = self.one_biguint();
        let div_plus_one = self.add_biguint(&div, &one);
        self.select_biguint(is_zero_rem, &div, &div_plus_one)
    }

    fn rem_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let (_div, rem) = self.div_rem_biguint(a, b);
        rem
    }
}

#[derive(Debug, Default)]
pub struct BigUintDivRemGenerator<F: RichField + Extendable<D>, const D: usize> {
    a: BigUintTarget,
    b: BigUintTarget,
    div: BigUintTarget,
    rem: BigUintTarget,
    _phantom: PhantomData<F>,
}

/// Divide a little-endian base-2^32 limb slice by one nonzero limb.
///
/// Quotient limbs are reported high-to-low because each quotient digit depends
/// on the remainder from the next higher input limb. The generator reverses
/// only the just-appended output segment afterwards, restoring the historical
/// low-to-high target order without allocating an intermediate digit vector.
#[inline]
fn divide_by_u32_limb(
    num_limbs: usize,
    divisor: u32,
    mut read_limb: impl FnMut(usize) -> u32,
    mut emit_quotient_limb: impl FnMut(usize, u32) -> Result<()>,
) -> Result<u32> {
    debug_assert_ne!(divisor, 0);
    let divisor = u64::from(divisor);
    let mut remainder = 0u64;
    for i in (0..num_limbs).rev() {
        let dividend = (remainder << 32) | u64::from(read_limb(i));
        emit_quotient_limb(i, (dividend / divisor) as u32)?;
        remainder = dividend % divisor;
    }
    Ok(remainder as u32)
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for BigUintDivRemGenerator<F, D>
{
    fn dependencies(&self) -> Vec<Target> {
        self.a
            .limbs
            .iter()
            .chain(&self.b.limbs)
            .map(|&l| l.0)
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // Most ranked fixed-integer divisions use a one-limb denominator. Do
        // their base-2^32 long division directly from witness limbs, avoiding
        // both heap-backed BigUint operands, the BigUint quotient/remainder,
        // and their digit extraction buffers. Arbitrary-width divisors retain
        // the exact generic path below.
        if self.b.num_limbs() == 1 {
            let divisor = witness.get_u32_target(self.b.limbs[0]);
            if divisor == 0 {
                for &target in &self.div.limbs {
                    out_buffer.set_u32_target(target, 0)?;
                }
                out_buffer.set_u32_target(self.rem.limbs[0], 0)?;
                return Ok(());
            }

            let output_start = out_buffer.target_values.len();
            let remainder = divide_by_u32_limb(
                self.a.num_limbs(),
                divisor,
                |i| witness.get_u32_target(self.a.limbs[i]),
                |i, quotient_limb| {
                    if i < self.div.num_limbs() {
                        out_buffer.set_u32_target(self.div.limbs[i], quotient_limb)
                    } else {
                        // The generic path's set_biguint_target asserts the
                        // same fit condition for a trimmed quotient target.
                        assert_eq!(quotient_limb, 0, "BigUint quotient does not fit");
                        Ok(())
                    }
                },
            )?;
            out_buffer.target_values[output_start..].reverse();
            out_buffer.set_u32_target(self.rem.limbs[0], remainder)?;
            return Ok(());
        }

        let a = witness.get_biguint_target(self.a.clone());
        let b = witness.get_biguint_target(self.b.clone());

        if b.is_zero() {
            out_buffer.set_biguint_target(&self.div, &BigUint::ZERO)?;
            out_buffer.set_biguint_target(&self.rem, &BigUint::ZERO)?;
            return Ok(());
        }

        let (div, rem) = a.div_rem(&b);
        out_buffer.set_biguint_target(&self.div, &div)?;
        out_buffer.set_biguint_target(&self.rem, &rem)
    }

    fn id(&self) -> String {
        "BigUintDivRemGenerator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target_vec(&self.a.limbs.iter().map(|&x| x.0).collect::<Vec<Target>>())?;
        dst.write_target_vec(&self.b.limbs.iter().map(|&x| x.0).collect::<Vec<Target>>())?;
        dst.write_target_vec(&self.div.limbs.iter().map(|&x| x.0).collect::<Vec<Target>>())?;
        dst.write_target_vec(&self.rem.limbs.iter().map(|&x| x.0).collect::<Vec<Target>>())?;

        IoResult::Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let a = src.read_target_vec()?;
        let b = src.read_target_vec()?;
        let div = src.read_target_vec()?;
        let rem = src.read_target_vec()?;

        IoResult::Ok(Self {
            a: BigUintTarget::from(a.iter().map(|&x| U32Target(x)).collect::<Vec<U32Target>>()),
            b: BigUintTarget::from(b.iter().map(|&x| U32Target(x)).collect::<Vec<U32Target>>()),
            div: BigUintTarget::from(
                div.iter()
                    .map(|&x| U32Target(x))
                    .collect::<Vec<U32Target>>(),
            ),
            rem: BigUintTarget::from(
                rem.iter()
                    .map(|&x| U32Target(x))
                    .collect::<Vec<U32Target>>(),
            ),
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod one_limb_division_tests {
    use num::{BigUint, Integer};

    use super::divide_by_u32_limb;

    #[test]
    fn one_limb_long_division_matches_biguint_raw_digits() {
        let cases: &[&[u32]] = &[
            &[0],
            &[1],
            &[u32::MAX],
            &[0, 1],
            &[u32::MAX, u32::MAX],
            &[0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210],
        ];
        for &limbs in cases {
            for divisor in [1, 2, 3, 17, 65_537, 0x8000_0001, u32::MAX] {
                let mut quotient = vec![0u32; limbs.len()];
                let remainder = divide_by_u32_limb(
                    limbs.len(),
                    divisor,
                    |i| limbs[i],
                    |i, value| {
                        quotient[i] = value;
                        Ok(())
                    },
                )
                .unwrap();

                let a = BigUint::new(limbs.to_vec());
                let (expected_quotient, expected_remainder) =
                    a.div_rem(&BigUint::from(divisor));
                let mut expected_digits = expected_quotient.to_u32_digits();
                expected_digits.resize(limbs.len(), 0);
                assert_eq!(quotient, expected_digits, "divisor={divisor}");
                assert_eq!(
                    remainder,
                    expected_remainder.to_u32_digits().first().copied().unwrap_or(0),
                    "divisor={divisor}"
                );
            }
        }
    }
}
