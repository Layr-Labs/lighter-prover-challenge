// Portions of this file are derived from plonky2
// Copyright (c) 2022-2025 The Plonky2 Authors
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

/// Quintuple over the extension field `E = <F as Extendable<D>>::Extension`,
/// representing elements in `E[u]/(u^5 - 3)` as `a0 + a1*u + ... + a4*u^4`.
#[derive(Copy, Clone)]
pub struct Quintuple<F: RichField + Extendable<D>, const D: usize>(
    pub [<F as Extendable<D>>::Extension; 5],
);

type E<F, const D: usize> = <F as Extendable<D>>::Extension;

impl<F: RichField + Extendable<D>, const D: usize> Quintuple<F, D> {
    pub const ZERO: Self = Self([E::<F, D>::ZERO; 5]);

    #[inline]
    pub fn one() -> Self {
        let mut a = [E::<F, D>::ZERO; 5];
        a[0] = E::<F, D>::ONE;
        Self(a)
    }

    #[inline]
    pub const fn from_coeffs(arr: [E<F, D>; 5]) -> Self {
        Self(arr)
    }

    #[inline]
    pub const fn to_coeffs(self) -> [E<F, D>; 5] {
        self.0
    }

    /// Convenience constructor.
    #[inline]
    pub const fn new(arr: [E<F, D>; 5]) -> Self {
        Self(arr)
    }

    /// Build from a slice (panics if `slice.len() != 5`).
    #[inline]
    pub fn from_slice(slice: &[E<F, D>]) -> Self {
        assert!(slice.len() == 5, "Quintuple::from_slice needs 5 coeffs");
        let arr: [E<F, D>; 5] = slice.try_into().ok().unwrap();
        Self(arr)
    }

    /// # Safety
    /// The caller must ensure that `slice.len() >= 5`.
    /// Otherwise, this function will cause undefined behavior.
    #[inline]
    pub unsafe fn from_slice_unchecked(slice: &[E<F, D>]) -> Self {
        debug_assert!(slice.len() >= 5);
        let arr = unsafe {
            [
                *slice.get_unchecked(0),
                *slice.get_unchecked(1),
                *slice.get_unchecked(2),
                *slice.get_unchecked(3),
                *slice.get_unchecked(4),
            ]
        };
        Self(arr)
    }

    #[inline]
    pub fn add_scalar(mut self, scalar: E<F, D>) -> Self {
        self.0[0] += scalar;
        self
    }

    #[inline]
    pub fn scalar_mul(&self, scalar: E<F, D>) -> Self {
        let mut res = self.0;
        for x in &mut res {
            *x *= scalar;
        }
        Self(res)
    }

    /// Schoolbook multiply with reduction by u^5 = 3 (W=3).
    #[inline]
    pub fn mul_quintic(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let w = E::<F, D>::from_canonical_u64(3);

        // convolution: d[0..=8] = sum_{i+j=s} a[i]*b[j]
        let mut d = [E::<F, D>::ZERO; 9];
        for s in 0..=8 {
            let mut acc = E::<F, D>::ZERO;
            for i in 0..=s {
                let j = s - i;
                if i < 5 && j < 5 {
                    acc += a[i] * b[j];
                }
            }
            d[s] = acc;
        }

        // reduction via u^5 = 3: c_k = d_k + 3 * d_{k+5}, for k = 0..4
        let mut c = [E::<F, D>::ZERO; 5];
        c.copy_from_slice(&d[0..5]);
        for s in 5..=8 {
            c[s - 5] += w * d[s];
        }

        Self(c)
    }
}

/* Conversions */

impl<F: RichField + Extendable<D>, const D: usize> From<[E<F, D>; 5]> for Quintuple<F, D> {
    #[inline]
    fn from(arr: [E<F, D>; 5]) -> Self {
        Self(arr)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TryFrom<&[E<F, D>]> for Quintuple<F, D> {
    type Error = &'static str;

    #[inline]
    fn try_from(slice: &[E<F, D>]) -> Result<Self, Self::Error> {
        if slice.len() != 5 {
            return Err("Quintuple: expected slice of length 5");
        }
        Ok(Self(slice.try_into().unwrap()))
    }
}

/* Formatting */

impl<F: RichField + Extendable<D>, const D: usize> Display for Quintuple<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({})", self.0[0])?;
        for i in 1..5 {
            write!(f, " + ({})*u^{i}", self.0[i])?;
        }
        Ok(())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Debug for Quintuple<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

/* Group ops */

impl<F: RichField + Extendable<D>, const D: usize> Neg for Quintuple<F, D> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut a = self.0;
        for x in &mut a {
            *x = -*x;
        }
        Self(a)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add for Quintuple<F, D> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut a = self.0;
        for i in 0..5 {
            a[i] += rhs.0[i];
        }
        Self(a)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> AddAssign for Quintuple<F, D> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub for Quintuple<F, D> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut a = self.0;
        for i in 0..5 {
            a[i] -= rhs.0[i];
        }
        Self(a)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SubAssign for Quintuple<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.0[i] -= rhs.0[i];
        }
    }
}

/* Ring ops */

impl<F: RichField + Extendable<D>, const D: usize> Mul for Quintuple<F, D> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_quintic(rhs)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> MulAssign for Quintuple<F, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_quintic(rhs);
    }
}

/* Iter helpers */

impl<F: RichField + Extendable<D>, const D: usize> Sum for Quintuple<F, D> {
    fn sum<I: Iterator<Item = Self>>(it: I) -> Self {
        it.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Product for Quintuple<F, D> {
    fn product<I: Iterator<Item = Self>>(it: I) -> Self {
        it.fold(Self::one(), |acc, x| acc * x)
    }
}
pub struct QuintupleTarget<const D: usize>(pub [ExtensionTarget<D>; 5]);

impl<const D: usize> QuintupleTarget<D> {
    #[inline]
    pub const fn new(arr: [ExtensionTarget<D>; 5]) -> Self {
        Self(arr)
    }

    /// Build from a slice (panics if `slice.len() != 5`).
    #[inline]
    pub fn from_slice(slice: &[ExtensionTarget<D>]) -> Self {
        assert!(
            slice.len() == 5,
            "QuintupleTarget::from_slice needs 5 coeffs"
        );
        let arr: [ExtensionTarget<D>; 5] = slice.try_into().ok().unwrap();
        Self(arr)
    }

    /// # Safety
    /// The caller must ensure that `slice.len() >= 5`.
    #[inline]
    pub unsafe fn from_slice_unchecked(slice: &[ExtensionTarget<D>]) -> Self {
        debug_assert!(slice.len() >= 5);
        Self([
            *unsafe { slice.get_unchecked(0) },
            *unsafe { slice.get_unchecked(1) },
            *unsafe { slice.get_unchecked(2) },
            *unsafe { slice.get_unchecked(3) },
            *unsafe { slice.get_unchecked(4) },
        ])
    }

    #[inline]
    pub const fn to_coeffs(self) -> [ExtensionTarget<D>; 5] {
        self.0
    }

    #[inline]
    pub fn as_coeffs(&self) -> &[ExtensionTarget<D>; 5] {
        &self.0
    }
}

#[inline]
pub fn add_quintuple<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    b: &QuintupleTarget<D>,
) -> QuintupleTarget<D> {
    let mut out = [builder.constant_extension(F::Extension::ZERO); 5];
    for i in 0..5 {
        out[i] = builder.add_extension(a.0[i], b.0[i]);
    }
    QuintupleTarget(out)
}

pub fn add_scalar<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    b: &ExtensionTarget<D>,
) -> QuintupleTarget<D> {
    let mut out = a.0;
    out[0] = builder.add_extension(out[0], *b);
    QuintupleTarget(out)
}

#[inline]
pub fn sub_quintuple<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    b: &QuintupleTarget<D>,
) -> QuintupleTarget<D> {
    let mut out = [builder.constant_extension(F::Extension::ZERO); 5];
    for i in 0..5 {
        out[i] = builder.sub_extension(a.0[i], b.0[i]);
    }
    QuintupleTarget(out)
}

/// Schoolbook multiply in E[u]/(u^5 - 3) with W=3.
#[inline]
pub fn mul_quintuple<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    b: &QuintupleTarget<D>,
) -> QuintupleTarget<D> {
    let zero = builder.constant_extension(F::Extension::ZERO);
    let w = builder.constant_extension(F::Extension::from_canonical_u64(3)); // W = 3

    let mut c = [zero; 5]; // accumulates c0..c4

    for i in 0..5 {
        for j in 0..5 {
            let prod = builder.mul_extension(a.0[i], b.0[j]);
            let s = i + j;
            if s < 5 {
                c[s] = builder.add_extension(c[s], prod);
            } else {
                // u^5 = 3 ⇒ u^s = 3 * u^{s-5}
                let prod_w = builder.mul_extension(prod, w);
                c[s - 5] = builder.add_extension(c[s - 5], prod_w);
            }
        }
    }

    QuintupleTarget(c)
}

/// Fused multiply-add: a*b + c in E[u]/(u^5 - 3).
#[inline]
pub fn mul_add_quintuple<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    b: &QuintupleTarget<D>,
    c: &QuintupleTarget<D>,
) -> QuintupleTarget<D> {
    let ab = mul_quintuple(builder, a, b);
    add_quintuple(builder, &ab, c)
}

#[inline]
pub fn mul_scalar_quintuple<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    a: &QuintupleTarget<D>,
    s: ExtensionTarget<D>,
) -> QuintupleTarget<D> {
    let mut out = [builder.constant_extension(F::Extension::ZERO); 5];
    for i in 0..5 {
        out[i] = builder.mul_extension(a.0[i], s);
    }
    QuintupleTarget(out)
}

#[derive(Copy, Clone, Debug)]
pub struct QuintupleBase<F: RichField + Extendable<D>, const D: usize>(pub [F; 5]);

impl<F: RichField + Extendable<D> + Copy, const D: usize> QuintupleBase<F, D> {
    pub const ZERO: Self = Self([F::ZERO; 5]);

    #[inline]
    pub fn one() -> Self {
        let mut a = [F::ZERO; 5];
        a[0] = F::ONE;
        Self(a)
    }

    #[inline]
    pub const fn new(arr: [F; 5]) -> Self {
        Self(arr)
    }

    #[inline]
    pub fn from_slice(slice: &[F]) -> Self {
        assert!(slice.len() == 5, "QuintupleBase::from_slice needs 5 coeffs");
        // Copy since F: Copy
        Self([slice[0], slice[1], slice[2], slice[3], slice[4]])
    }

    #[inline]
    pub const fn to_coeffs(self) -> [F; 5] {
        self.0
    }

    #[inline]
    pub fn as_array(&self) -> &[F; 5] {
        &self.0
    }

    #[inline]
    pub fn add_scalar(&self, k: F) -> Self {
        let [a0, a1, a2, a3, a4] = self.0;
        Self([a0 + k, a1, a2, a3, a4])
    }

    /// Scalar multiply
    #[inline]
    pub fn scalar_mul(&self, k: F) -> Self {
        let [a0, a1, a2, a3, a4] = self.0;
        Self([a0 * k, a1 * k, a2 * k, a3 * k, a4 * k])
    }

    /// Fused: `self[i] += a[i] * k`
    #[inline]
    pub fn mul_add_scalar(&mut self, a: &Self, k: F) {
        self.0[0] += a.0[0] * k;
        self.0[1] += a.0[1] * k;
        self.0[2] += a.0[2] * k;
        self.0[3] += a.0[3] * k;
        self.0[4] += a.0[4] * k;
    }

    /// Quintic ring multiply with reduction by `u^5 = 3` (same as your `Quintuple`).
    #[inline]
    pub fn mul_quintic(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let w = F::from_canonical_u64(3);

        // convolution d[0..=8]
        let mut d = [F::ZERO; 9];
        for s in 0..=8 {
            let mut acc = F::ZERO;
            for i in 0..=s {
                let j = s - i;
                if i < 5 && j < 5 {
                    acc += a[i] * b[j];
                }
            }
            d[s] = acc;
        }

        // reduction: c_k = d_k + 3 * d_{k+5}, k=0..4
        let mut c = [F::ZERO; 5];
        c.copy_from_slice(&d[0..5]);
        for s in 5..=8 {
            c[s - 5] += w * d[s];
        }
        Self(c)
    }
}

/* Group ops */

impl<F: RichField + Extendable<D> + Copy, const D: usize> Add for QuintupleBase<F, D> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut a = self.0;
        for i in 0..5 {
            a[i] += rhs.0[i];
        }
        Self(a)
    }
}

impl<F: RichField + Extendable<D> + Copy, const D: usize> AddAssign for QuintupleBase<F, D> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<F: RichField + Extendable<D> + Copy, const D: usize> Sub for QuintupleBase<F, D> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut a = self.0;
        for i in 0..5 {
            a[i] -= rhs.0[i];
        }
        Self(a)
    }
}

impl<F: RichField + Extendable<D> + Copy, const D: usize> SubAssign for QuintupleBase<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.0[i] -= rhs.0[i];
        }
    }
}

/* Scalar ops: q * k, k * q */

impl<F: RichField + Extendable<D> + Copy, const D: usize> Mul<F> for QuintupleBase<F, D> {
    type Output = Self;
    #[inline]
    fn mul(self, k: F) -> Self::Output {
        self.scalar_mul(k)
    }
}

impl<F: RichField + Extendable<D> + Copy, const D: usize> MulAssign<F> for QuintupleBase<F, D> {
    #[inline]
    fn mul_assign(&mut self, k: F) {
        self.0[0] *= k;
        self.0[1] *= k;
        self.0[2] *= k;
        self.0[3] *= k;
        self.0[4] *= k;
    }
}

/* Quintic ring ops: q * r with u^5 = 3 */

impl<F: RichField + Extendable<D> + Copy, const D: usize> Mul for QuintupleBase<F, D> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_quintic(rhs)
    }
}

impl<F: RichField + Extendable<D> + Copy, const D: usize> MulAssign for QuintupleBase<F, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_quintic(rhs);
    }
}
