// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use num::bigint::BigUint;
use num::{Integer, One};
use plonky2::field::types::{Field, PrimeField, Sample};
use serde::{Deserialize, Serialize};

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::builder::Builder;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::types::config::{D, F};

/// The base field of the bls12381 elliptic curve.
///
/// Its order is
/// ```ignore
/// P = 0x73EDA753 299D7D48 3339D808 09A1D805 53BDA402 FFFE5BFE FFFFFFFF 00000001
/// ```
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct BLS12381Scalar(pub [u64; 4]);

pub const BLS12_381_SCALAR_LIMBS: usize = 8;

impl BLS12381Scalar {
    pub fn pow_to_const(
        builder: &mut Builder<F, D>,
        x: &NonNativeTarget<Self>,
        pow: usize,
    ) -> NonNativeTarget<Self> {
        if pow == 0 {
            let one_big = builder.one_biguint();
            return builder.biguint_to_nonnative(&one_big);
        }
        if pow == 1 {
            return x.clone();
        }

        let mut result = Self::pow_to_const(builder, x, pow / 2);
        result = builder.mul_nonnative(&result, &result);
        if pow % 2 == 1 {
            result = builder.mul_nonnative(&result, x);
        }
        result
    }
}

impl Field for BLS12381Scalar {
    const ZERO: Self = Self([0; 4]);
    const ONE: Self = Self([1, 0, 0, 0]);
    const TWO: Self = Self([2, 0, 0, 0]);
    const NEG_ONE: Self = Self([
        0xFFFFFFFF00000000,
        0x53BDA402FFFE5BFE,
        0x3339D80809A1D805,
        0x73EDA753299D7D48,
    ]);

    const TWO_ADICITY: usize = 32;
    const CHARACTERISTIC_TWO_ADICITY: usize = Self::TWO_ADICITY;

    // Sage: `g = GF(p).multiplicative_generator()`
    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self([7, 0, 0, 0]);

    // Sage: `g_2 = power_mod(g, (p - 1) // 2^6), p)`
    const POWER_OF_TWO_GENERATOR: Self = Self([
        0x0000_000e_ffff_fff1,
        0x17e3_63d3_0018_9c0f,
        0xff9c_5787_6f84_57b0,
        0x3513_3220_8fc5_a8c4,
    ]);

    const BITS: usize = 255;

    fn order() -> BigUint {
        BigUint::from_slice(&[
            0x00000001, 0xFFFFFFFF, 0xFFFE5BFE, 0x53BDA402, 0x09A1D805, 0x3339D808, 0x299D7D48,
            0x73EDA753,
        ])
    }
    fn characteristic() -> BigUint {
        Self::order()
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Fermat's Little Theorem
        Some(self.exp_biguint(&(Self::order() - BigUint::one() - BigUint::one())))
    }

    fn from_noncanonical_biguint(val: BigUint) -> Self {
        Self(
            val.to_u64_digits()
                .into_iter()
                .pad_using(4, |_| 0)
                .collect::<Vec<_>>()[..]
                .try_into()
                .expect("error converting to u64 array"),
        )
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        Self([n, 0, 0, 0])
    }

    #[inline]
    fn from_noncanonical_u128(n: u128) -> Self {
        Self([n as u64, (n >> 64) as u64, 0, 0])
    }

    #[inline]
    fn from_noncanonical_u96(n: (u64, u32)) -> Self {
        Self([n.0, n.1 as u64, 0, 0])
    }

    fn from_noncanonical_i64(n: i64) -> Self {
        let f = Self::from_canonical_u64(n.unsigned_abs());
        if n < 0 { -f } else { f }
    }

    fn from_noncanonical_u64(n: u64) -> Self {
        Self::from_canonical_u64(n)
    }
}

fn biguint_from_array(arr: [u64; 4]) -> BigUint {
    BigUint::from_slice(&[
        arr[0] as u32,
        (arr[0] >> 32) as u32,
        arr[1] as u32,
        (arr[1] >> 32) as u32,
        arr[2] as u32,
        (arr[2] >> 32) as u32,
        arr[3] as u32,
        (arr[3] >> 32) as u32,
    ])
}

impl Default for BLS12381Scalar {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for BLS12381Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.to_canonical_biguint() == other.to_canonical_biguint()
    }
}

impl Eq for BLS12381Scalar {}

impl Hash for BLS12381Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_canonical_biguint().hash(state)
    }
}

impl Display for BLS12381Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.to_canonical_biguint(), f)
    }
}

impl Debug for BLS12381Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.to_canonical_biguint(), f)
    }
}

impl Sample for BLS12381Scalar {
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        use num::bigint::RandBigInt;
        Self::from_noncanonical_biguint(rng.gen_biguint_below(&Self::order()))
    }
}

impl PrimeField for BLS12381Scalar {
    fn to_canonical_biguint(&self) -> BigUint {
        let mut result = biguint_from_array(self.0);
        if result >= Self::order() {
            result -= Self::order();
        }
        result
    }
}

impl Neg for BLS12381Scalar {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        if self.is_zero() {
            Self::ZERO
        } else {
            Self::from_noncanonical_biguint(Self::order() - self.to_canonical_biguint())
        }
    }
}

impl Add for BLS12381Scalar {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut result = self.to_canonical_biguint() + rhs.to_canonical_biguint();
        if result >= Self::order() {
            result -= Self::order();
        }
        Self::from_noncanonical_biguint(result)
    }
}

impl AddAssign for BLS12381Scalar {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for BLS12381Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Sub for BLS12381Scalar {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl SubAssign for BLS12381Scalar {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for BLS12381Scalar {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_noncanonical_biguint(
            (self.to_canonical_biguint() * rhs.to_canonical_biguint()).mod_floor(&Self::order()),
        )
    }
}

impl MulAssign for BLS12381Scalar {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for BLS12381Scalar {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl Div for BLS12381Scalar {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for BLS12381Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
