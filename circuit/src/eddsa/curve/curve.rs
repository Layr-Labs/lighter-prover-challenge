// Portions of this file are derived from ecgfp5
// Copyright (c) 2022 Thomas Pornin
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::ops::Square;
use plonky2::field::types::{Field, Sample};
use rand::RngCore;

use super::base_field::InverseOrZero;
use crate::eddsa::curve::base_field::{Legendre, SquareRoot};
use crate::eddsa::curve::mul_table::*;
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::types::config::{F, const_f};

/// A curve point.
#[derive(Clone, Copy, Debug)]
pub struct ECgFp5Point {
    // Internally, we use the (x,u) fractional coordinates: for curve
    // point (x,y), we have (x,u) = (x,x/y) = (X/Z,U/T) (for the neutral
    // N, the u coordinate is 0).
    x: QuinticExtension<F>,
    z: QuinticExtension<F>,
    u: QuinticExtension<F>,
    t: QuinticExtension<F>,
}

/// A curve point in affine (x,u) coordinates. This is used internally
/// to make "windows" that speed up point multiplications.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AffinePoint {
    pub(crate) x: QuinticExtension<F>,
    pub(crate) u: QuinticExtension<F>,
}

/// A curve point in short Weirstrass form (x, y). This is used by the in-circuit representation
#[derive(Clone, Copy, Debug)]
pub struct WeierstrassPoint {
    pub(crate) x: QuinticExtension<F>,
    pub(crate) y: QuinticExtension<F>,
    pub(crate) is_inf: bool,
}

impl WeierstrassPoint {
    // curve equation `A` constants when in short Weierstrass form
    pub const A: QuinticExtension<F> = QuinticExtension([
        const_f(6148914689804861439),
        const_f(263),
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ]);

    pub const B: QuinticExtension<F> = QuinticExtension([
        const_f(15713893096167979237),
        const_f(6148914689804861265),
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ]);

    pub const NEUTRAL: Self = Self {
        x: QuinticExtension::<F>::ZERO,
        y: QuinticExtension::<F>::ZERO,
        is_inf: true,
    };

    pub const GENERATOR: Self = Self {
        x: QuinticExtension([
            const_f(11712523173042564207),
            const_f(14090224426659529053),
            const_f(13197813503519687414),
            const_f(16280770174934269299),
            const_f(15998333998318935536),
        ]),

        y: QuinticExtension([
            const_f(14639054205878357578),
            const_f(17426078571020221072),
            const_f(2548978194165003307),
            const_f(8663895577921260088),
            const_f(9793640284382595140),
        ]),
        is_inf: false,
    };

    pub fn encode(&self) -> QuinticExtension<F> {
        self.y / (ECgFp5Point::A / QuinticExtension::<F>::from_canonical_u16(3) - self.x)
    }

    pub fn decode(w: QuinticExtension<F>) -> Option<Self> {
        let e = w.square() - ECgFp5Point::A;
        let delta = e.square() - ECgFp5Point::B_MUL4;
        let r = delta.canonical_sqrt();
        let c = r.is_some();
        let r = r.unwrap_or(QuinticExtension::<F>::ZERO);

        let x1 = (e + r) / QuinticExtension::<F>::TWO;
        let x2 = (e - r) / QuinticExtension::<F>::TWO;

        let x = if x1.legendre() == F::ONE { x1 } else { x2 };

        let y = -w * x;
        let x = if c {
            x + ECgFp5Point::A / QuinticExtension::<F>::from_canonical_u16(3)
        } else {
            QuinticExtension::<F>::ZERO
        };
        let is_inf = !c;

        // If w == 0 then this is in fact a success.
        if c || w == QuinticExtension::<F>::ZERO {
            Some(WeierstrassPoint { x, y, is_inf })
        } else {
            None
        }
    }
}

impl PartialEq for WeierstrassPoint {
    fn eq(&self, other: &Self) -> bool {
        if self.is_inf && other.is_inf {
            true
        } else {
            self.x == other.x && self.y == other.y
        }
    }
}

impl Eq for WeierstrassPoint {}

impl Sample for ECgFp5Point {
    fn sample<R>(rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
    {
        let s = ECgFp5Scalar::sample(rng);
        ECgFp5Point::GENERATOR * s
    }
}

impl ECgFp5Point {
    // Curve equation 'a' constant.
    pub(crate) const A: QuinticExtension<F> =
        QuinticExtension([F::TWO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    pub const B1: u64 = 263;

    pub(crate) const B: QuinticExtension<F> =
        QuinticExtension([F::ZERO, const_f(Self::B1), F::ZERO, F::ZERO, F::ZERO]);

    // 2*b
    pub(crate) const B_MUL2: QuinticExtension<F> =
        QuinticExtension([F::ZERO, const_f(2 * Self::B1), F::ZERO, F::ZERO, F::ZERO]);
    // 4*b
    pub(crate) const B_MUL4: QuinticExtension<F> =
        QuinticExtension([F::ZERO, const_f(4 * Self::B1), F::ZERO, F::ZERO, F::ZERO]);
    // 16*b
    pub(crate) const B_MUL16: QuinticExtension<F> =
        QuinticExtension([F::ZERO, const_f(16 * Self::B1), F::ZERO, F::ZERO, F::ZERO]);

    /// The neutral point (neutral of the group law).
    pub const NEUTRAL: Self = Self {
        x: QuinticExtension::<F>::ZERO,
        z: QuinticExtension::<F>::ONE,
        u: QuinticExtension::<F>::ZERO,
        t: QuinticExtension::<F>::ONE,
    };

    /// The conventional generator (corresponding to encoding w = 4).
    pub const GENERATOR: Self = Self {
        x: QuinticExtension([
            const_f(12883135586176881569),
            const_f(4356519642755055268),
            const_f(5248930565894896907),
            const_f(2165973894480315022),
            const_f(2448410071095648785),
        ]),
        z: QuinticExtension::<F>::ONE,
        u: QuinticExtension([const_f(1), F::ZERO, F::ZERO, F::ZERO, F::ZERO]),
        t: QuinticExtension([const_f(4), F::ZERO, F::ZERO, F::ZERO, F::ZERO]),
    };

    /// Encode this point into a field element. Encoding is always
    /// canonical.
    pub fn encode(self) -> QuinticExtension<F> {
        // Encoded form is the value w = 1/u. GFpor the neutral (u == 0),
        // the encoded form is 0. Since our inversion over Gconst_f(p^5) already
        // yields 0 in that case, there is no need for any special code.
        self.t * self.u.inverse_or_zero()
    }

    /// Test whether a field element can be decoded into a point.
    /// returns `true` if decoding would work, `false` otherwise.
    pub fn validate(w: QuinticExtension<F>) -> bool {
        // Value w can be decoded if and only if it is zero, or
        // (w^2 - a)^2 - 4*b is a quadratic residue.
        let e = w.square() - Self::A;
        let delta = e.square() - Self::B_MUL4;
        w == QuinticExtension::<F>::ZERO || delta.legendre() == F::ONE
    }

    /// Attempt to decode a point from a field element
    pub fn decode(w: QuinticExtension<F>) -> Option<Self> {
        // Curve equation is y^2 = x*(x^2 + a*x + b); encoded value
        // is w = y/x. Dividing by x, we get the equation:
        //   x^2 - (w^2 - a)*x + b = 0
        // We solve for x and keep the solution which is not itself a
        // square (if there are solutions, exactly one of them will be
        // a square, and the other will not be a square).

        let e = w.square() - Self::A;
        let delta = e.square() - Self::B_MUL4;
        let r = delta.canonical_sqrt();
        let c = r.is_some();
        let r = r.unwrap_or(QuinticExtension::<F>::ZERO);

        let x1 = (e + r) / QuinticExtension::<F>::TWO;
        let x2 = (e - r) / QuinticExtension::<F>::TWO;
        let x = if x1.legendre() == F::ONE { x2 } else { x1 };

        // If c == true (delta is not a sqrt) then we want to get the neutral here; note that if
        // w == 0, then delta = a^2 - 4*b, which is not a square, and
        // thus we also get c == 0.
        let x = if c { x } else { QuinticExtension::<F>::ZERO };
        let z = QuinticExtension::<F>::ONE;
        let u = if c {
            QuinticExtension::<F>::ONE
        } else {
            QuinticExtension::<F>::ZERO
        };
        let t = if c { w } else { QuinticExtension::<F>::ONE };

        // If w == 0 then this is in fact a success.
        if c || w == QuinticExtension::<F>::ZERO {
            Some(Self { x, z, u, t })
        } else {
            None
        }
    }

    pub fn to_weierstrass(&self) -> WeierstrassPoint {
        let w = self.encode();
        WeierstrassPoint::decode(w).unwrap()
    }

    // General point addition. formulas are complete (no special case).
    fn set_add(&mut self, rhs: &Self) {
        // cost: 10M
        let (x1, z1, u1, _t1) = (self.x, self.z, self.u, self.t);
        let (x2, z2, u2, _t2) = (rhs.x, rhs.z, rhs.u, rhs.t);

        let t1 = x1 * x2;
        let t2 = z1 * z2;
        let t3 = u1 * u2;
        let t4 = _t1 * _t2;
        let t5 = (x1 + z1) * (x2 + z2) - t1 - t2;
        let t6 = (u1 + _t1) * (u2 + _t2) - t3 - t4;
        let t7 = t1 + t2 * Self::B;
        let t8 = t4 * t7;
        let t9 = t3 * (t5 * Self::B_MUL2 + t7.double());
        let t10 = (t4 + t3.double()) * (t5 + t7);

        self.x = (t10 - t8) * Self::B;
        self.z = t8 - t9;
        self.u = t6 * ((t2 * Self::B) - t1);
        self.t = t8 + t9;
    }

    // Add a point in affine coordinates to this one.
    fn set_add_affine(&mut self, rhs: &AffinePoint) {
        // cost: 8M
        let (x1, z1, u1, _t1) = (self.x, self.z, self.u, self.t);
        let (x2, u2) = (rhs.x, rhs.u);

        let t1 = x1 * x2;
        let t2 = z1;
        let t3 = u1 * u2;
        let t4 = _t1;
        let t5 = x1 + x2 * z1;
        let t6 = u1 + u2 * _t1;
        let t7 = t1 + t2 * Self::B;
        let t8 = t4 * t7;
        let t9 = t3 * (t5 * Self::B_MUL2 + t7.double());
        let t10 = (t4 + t3.double()) * (t5 + t7);

        self.x = (t10 - t8) * Self::B;
        self.u = t6 * (t2 * Self::B - t1);
        self.z = t8 - t9;
        self.t = t8 + t9;
    }

    // Subtract a point in affine coordinates from this one.
    fn set_sub_affine(&mut self, rhs: &AffinePoint) {
        self.set_add_affine(&AffinePoint {
            x: rhs.x,
            u: -rhs.u,
        })
    }

    fn set_neg(&mut self) {
        self.u = -self.u;
    }

    fn set_sub(&mut self, rhs: &Self) {
        self.set_add(&rhs.neg())
    }

    /// Specialized point doubling function (faster than using general
    /// addition on the point and itself).
    pub fn double(self) -> Self {
        let mut r = self;
        r.set_double();
        r
    }

    fn set_double(&mut self) {
        // cost: 4M+5S
        let (x, z, u, t) = (self.x, self.z, self.u, self.t);

        let t1 = z * t;
        let t2 = t1 * t;
        let x1 = t2.square();
        let z1 = t1 * u;
        let t3 = u.square();
        let w1 = t2 - (x + z).double() * t3;
        let t4 = z1.square();

        self.x = t4 * Self::B_MUL4;
        self.z = w1.square();
        self.u = (w1 + z1).square() - t4 - self.z;
        self.t = x1.double() - t4 * QuinticExtension::<F>::from_canonical_u64(4) - self.z;
    }

    /// Multiply this point by 2^n (i.e. n successive doublings). This is
    /// faster than calling the double() function n times.
    pub fn mdouble(self, n: u32) -> Self {
        let mut r = self;
        r.set_mdouble(n);
        r
    }

    fn set_mdouble(&mut self, n: u32) {
        // Handle corner cases (0 or 1 double).
        if n == 0 {
            return;
        }
        if n == 1 {
            self.set_double();
            return;
        }

        // cost: n*(2M+5S) + 2M+1S
        let (x0, z0, u0, t0) = (self.x, self.z, self.u, self.t);
        let mut t1 = z0 * t0;
        let mut t2 = t1 * t0;
        let x1 = t2.square();
        let z1 = t1 * u0;
        let mut t3 = u0.square();
        let mut w1 = t2 - (x0 + z0).double() * t3;
        let mut t4 = w1.square();
        let mut t5 = z1.square();
        let mut x = t5.square() * Self::B_MUL16;
        let mut w = x1.double() - t5 * QuinticExtension::<F>::from_canonical_u16(4) - t4;
        let mut z = (w1 + z1).square() - t4 - t5;

        for _ in 2..n {
            t1 = z.square();
            t2 = t1.square();
            t3 = w.square();
            t4 = t3.square();
            t5 = (w + z).square() - t1 - t3;
            z = t5 * ((x + t1).double() - t3);
            x = (t2 * t4) * Self::B_MUL16;
            w = -t4 - t2 * (Self::B_MUL4 - QuinticExtension::<F>::from_canonical_u16(4));
        }

        t1 = w.square();
        t2 = z.square();
        t3 = (w + z).square() - t1 - t2;
        w1 = t1 - (x + t2).double();
        self.x = t3.square() * Self::B;
        self.z = w1.square();
        self.u = t3 * w1;
        self.t = t1.double() * (t1 - t2.double()) - self.z;
    }

    /// Return `true` if this point is the neutral, `false` otherwise.
    pub fn is_neutral(self) -> bool {
        self.u == QuinticExtension::<F>::ZERO
    }

    /// Compare this point with another
    /// return `true` if they're equal`, `false` otherwise
    pub fn equals(self, rhs: Self) -> bool {
        self.u * rhs.t == rhs.u * self.t
    }

    // Convert points to affine coordinates.
    pub(crate) fn batch_to_affine(src: &[Self]) -> Vec<AffinePoint> {
        // We use a trick due to Montgomery: to compute the inverse of
        // x and of y, a single inversion suffices, with:
        //    1/x = y*(1/(x*y))
        //    1/y = x*(1/(x*y))
        // This extends to the case of inverting n values, with a total
        // cost of 1 inversion and 3*(n-1) multiplications.
        match src.len() {
            0 => Vec::new(),
            1 => {
                let p = src[0];
                let m1 = (p.z * p.t).inverse_or_zero();
                let res = AffinePoint {
                    x: p.x * p.t * m1,
                    u: p.u * p.z * m1,
                };

                vec![res]
            }
            n => {
                let mut res = vec![AffinePoint::NEUTRAL; n];
                // Compute product of all values to invert, and invert it.
                // We also use the x and u coordinates of the points in the
                // destination slice to keep track of the partial products.
                let mut m = src[0].z * src[0].t;
                for i in 1..n {
                    let x = m;
                    m *= src[i].z;
                    let u = m;
                    m *= src[i].t;

                    res[i] = AffinePoint { x, u };
                }

                m = m.inverse_or_zero();

                // Propagate back inverses.
                for i in (1..n).rev() {
                    res[i].u = src[i].u * res[i].u * m;
                    m *= src[i].t;
                    res[i].x = src[i].x * res[i].x * m;
                    m *= src[i].z;
                }
                res[0].u = src[0].u * src[0].z * m;
                m *= src[0].t;
                res[0].x = src[0].x * m;

                res
            }
        }
    }

    // Optimal window size should be 4 or 5 bits, depending on target
    // architecture. On an Intel i5-8259U ("Coffee Lake" core), a 5-bit
    // window seems very slightly better.
    const WINDOW: usize = 5;
    const WIN_SIZE: usize = 1 << ((Self::WINDOW - 1) as i32);

    fn make_window_affine(self) -> Vec<AffinePoint> {
        let mut tmp = [Self::NEUTRAL; Self::WIN_SIZE];
        tmp[0] = self;
        for i in 1..Self::WIN_SIZE {
            if (i & 1) == 0 {
                tmp[i] = self.add(tmp[i - 1]);
            } else {
                tmp[i] = tmp[i >> 1].double();
            }
        }

        Self::batch_to_affine(&tmp)
    }

    // Multiply this point by a scalar.
    fn set_mul(&mut self, s: &ECgFp5Scalar) {
        // Make a window with affine points.
        let win = self.make_window_affine();
        let mut digits = [0; (319 + Self::WINDOW) / Self::WINDOW];
        s.recode_signed(&mut digits, Self::WINDOW as i32);

        *self = AffinePoint::lookup_vartime(&win, *digits.last().unwrap()).to_point();
        for &digit in digits.iter().rev().skip(1) {
            self.set_mdouble(Self::WINDOW as u32);
            *self += AffinePoint::lookup(&win, digit);
        }
    }

    /// Multiply the conventional generator by a scalar.
    /// This function is faster than using the multiplication operator
    /// on the generator point.
    pub fn mulgen(s: ECgFp5Scalar) -> Self {
        let mut digits = [0i32; 64];
        s.recode_signed(&mut digits, 5);
        let mut p = AffinePoint::lookup(&MUL_TABLE_G0, digits[7]).to_point();

        p += AffinePoint::lookup(&MUL_TABLE_G40, digits[15]);
        p += AffinePoint::lookup(&MUL_TABLE_G80, digits[23]);
        p += AffinePoint::lookup(&MUL_TABLE_G120, digits[31]);
        p += AffinePoint::lookup(&MUL_TABLE_G160, digits[39]);
        p += AffinePoint::lookup(&MUL_TABLE_G200, digits[47]);
        p += AffinePoint::lookup(&MUL_TABLE_G240, digits[55]);
        p += AffinePoint::lookup(&MUL_TABLE_G280, digits[63]);
        for i in (0..7).rev() {
            p.set_mdouble(5);
            p += AffinePoint::lookup(&MUL_TABLE_G0, digits[i]);
            p += AffinePoint::lookup(&MUL_TABLE_G40, digits[i + 8]);
            p += AffinePoint::lookup(&MUL_TABLE_G80, digits[i + 16]);
            p += AffinePoint::lookup(&MUL_TABLE_G120, digits[i + 24]);
            p += AffinePoint::lookup(&MUL_TABLE_G160, digits[i + 32]);
            p += AffinePoint::lookup(&MUL_TABLE_G200, digits[i + 40]);
            p += AffinePoint::lookup(&MUL_TABLE_G240, digits[i + 48]);
            p += AffinePoint::lookup(&MUL_TABLE_G280, digits[i + 56]);
        }
        p
    }

    fn make_window_5(self) -> [Self; 16] {
        let mut win = [Self::NEUTRAL; 16];
        win[0] = self;
        for i in 1..win.len() {
            if (i & 1) == 0 {
                win[i] = self.add(win[i - 1]);
            } else {
                win[i] = win[i >> 1].double();
            }
        }
        win
    }

    fn lookup_vartime(win: &[Self], k: i32) -> Self {
        if k == 0 {
            Self::NEUTRAL
        } else if k > 0 {
            win[(k - 1) as usize]
        } else {
            -win[(-k - 1) as usize]
        }
    }

    /// Given scalars s and k, and point R, verify whether s*G + k*Q = R
    /// (with G being the curve conventional generator, and Q this instance).
    /// This is the main operation in Schnorr signature verification.
    /// WARNING: this function is not constant-time; use only on
    /// public data.
    pub fn verify_muladd_vartime(self, s: ECgFp5Scalar, k: ECgFp5Scalar, r: Self) -> bool {
        // We use a method by Antipa et al (SAC 2005), following the
        // description in: https://eprint.iacr.org/2020/454
        // We split k into two (signed) integers c0 and c1 such
        // that k = c0/c1 mod n; the integers c0 and c1 fit on 161 bits
        // each (including the signed bit). The verification is then:
        //    (s*c1)*G + c0*Q - c1*R = 0
        // We split s*c1 into two 160-bit halves, and use the precomputed
        // tables for G; thus, all scalars fit on 160 bits (+sign).
        //
        // Since formulas for multiple doublings favour long runs of
        // doublings, we do not use a wNAF representation; instead, we
        // make regular 5-bit (signed) windows.
        //
        // We use fractional coordinates for the Q and R windows; it is
        // not worth it converting them to affine.

        // Compute c0 and c1.
        let (c0, c1) = k.lagrange();

        // Compute t <- s*c1.
        let t = s * c1.to_scalar_vartime();

        // Recode multipliers.
        let mut tt = [0i32; 64];
        t.recode_signed(&mut tt, 5);
        let tt0 = &tt[..32];
        let tt1 = &tt[32..];
        let ss0 = c0.recode_signed_5();
        let ss1 = c1.recode_signed_5();

        // Make windows for this point (Q) and for -R.
        let win_q = self.make_window_5();
        let win_r = (-r).make_window_5();

        let mut p = Self::lookup_vartime(&win_q, ss0[32]);
        if ss1[32] != 0 {
            p += Self::lookup_vartime(&win_r, ss1[32]);
        }
        for i in (0..32).rev() {
            p.set_mdouble(5);
            if tt0[i] != 0 {
                p += AffinePoint::lookup_vartime(&MUL_TABLE_G0, tt0[i]);
            }
            if tt1[i] != 0 {
                p += AffinePoint::lookup_vartime(&MUL_TABLE_G160, tt1[i]);
            }
            if ss0[i] != 0 {
                p += Self::lookup_vartime(&win_q, ss0[i]);
            }
            if ss1[i] != 0 {
                p += Self::lookup_vartime(&win_r, ss1[i]);
            }
        }

        p == Self::NEUTRAL
    }
}

impl AffinePoint {
    pub(crate) const NEUTRAL: Self = Self {
        x: QuinticExtension::<F>::ZERO,
        u: QuinticExtension::<F>::ZERO,
    };

    fn to_point(self) -> ECgFp5Point {
        let Self { x, u } = self;
        ECgFp5Point {
            x,
            z: QuinticExtension::<F>::ONE,
            u,
            t: QuinticExtension::<F>::ONE,
        }
    }

    fn set_neg(&mut self) {
        self.u = -self.u;
    }

    // Lookup a point in a window. The win[] slice must contain values
    // i*P for i = 1 to n (win[0] contains P, win[1] contains 2*P, and
    // so on). Index value k is an integer in the -n to n range; returned
    // point is k*P.
    fn set_lookup(&mut self, win: &[Self], k: i32) {
        // sign = 0xFFFFFFFF if k < 0, 0x00000000 otherwise
        let sign = (k >> 31) as u32;
        // ka = abs(k)
        let ka = ((k as u32) ^ sign).wrapping_sub(sign);
        // km1 = ka - 1
        let km1 = ka.wrapping_sub(1);

        let mut x = QuinticExtension::<F>::ZERO;
        let mut u = QuinticExtension::<F>::ZERO;
        for i in 0..win.len() {
            let m = km1.wrapping_sub(i as u32);
            let c = (((m | m.wrapping_neg()) >> 31) as u64).wrapping_sub(1);
            x = if c == 0 { x } else { win[i].x };
            u = if c == 0 { u } else { win[i].u };
        }

        // If k < 0, then we must negate the point.
        let c = (sign as u64) | ((sign as u64) << 32);
        self.x = x;
        self.u = u;

        if c != 0 {
            self.u = -self.u;
        }
    }

    fn lookup(win: &[Self], k: i32) -> Self {
        let mut r = Self::NEUTRAL;
        r.set_lookup(win, k);
        r
    }

    // Same as lookup(), except this implementation is variable-time.
    fn lookup_vartime(win: &[Self], k: i32) -> Self {
        if k == 0 {
            Self::NEUTRAL
        } else if k > 0 {
            win[(k - 1) as usize]
        } else {
            -win[(-k - 1) as usize]
        }
    }
}

// We implement all the needed traits to allow use of the arithmetic
// operators on points. We support all combinations of operands
// either as Point structures, or pointers to Point structures. Some
// operations with AffinePoint structures are also implemented.

impl Add<ECgFp5Point> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&ECgFp5Point> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<ECgFp5Point> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&ECgFp5Point> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl Add<AffinePoint> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: AffinePoint) -> ECgFp5Point {
        let mut r = self;
        r.set_add_affine(&other);
        r
    }
}

impl Add<&AffinePoint> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &AffinePoint) -> ECgFp5Point {
        let mut r = self;
        r.set_add_affine(other);
        r
    }
}

impl Add<AffinePoint> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: AffinePoint) -> ECgFp5Point {
        let mut r = *self;
        r.set_add_affine(&other);
        r
    }
}

impl Add<&AffinePoint> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &AffinePoint) -> ECgFp5Point {
        let mut r = *self;
        r.set_add_affine(other);
        r
    }
}

impl Add<ECgFp5Point> for AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_add_affine(&self);
        r
    }
}

impl Add<&ECgFp5Point> for AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_add_affine(&self);
        r
    }
}

impl Add<ECgFp5Point> for &AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_add_affine(self);
        r
    }
}

impl Add<&ECgFp5Point> for &AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn add(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_add_affine(self);
        r
    }
}

impl AddAssign<ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn add_assign(&mut self, other: ECgFp5Point) {
        self.set_add(&other);
    }
}

impl AddAssign<&ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn add_assign(&mut self, other: &ECgFp5Point) {
        self.set_add(other);
    }
}

impl AddAssign<AffinePoint> for ECgFp5Point {
    #[inline(always)]
    fn add_assign(&mut self, other: AffinePoint) {
        self.set_add_affine(&other);
    }
}

impl AddAssign<&AffinePoint> for ECgFp5Point {
    #[inline(always)]
    fn add_assign(&mut self, other: &AffinePoint) {
        self.set_add_affine(other);
    }
}

impl Sub<ECgFp5Point> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&ECgFp5Point> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl Sub<ECgFp5Point> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&ECgFp5Point> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl Sub<AffinePoint> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: AffinePoint) -> ECgFp5Point {
        let mut r = self;
        r.set_sub_affine(&other);
        r
    }
}

impl Sub<&AffinePoint> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &AffinePoint) -> ECgFp5Point {
        let mut r = self;
        r.set_sub_affine(other);
        r
    }
}

impl Sub<AffinePoint> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: AffinePoint) -> ECgFp5Point {
        let mut r = *self;
        r.set_sub_affine(&other);
        r
    }
}

impl Sub<&AffinePoint> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &AffinePoint) -> ECgFp5Point {
        let mut r = *self;
        r.set_sub_affine(other);
        r
    }
}

impl Sub<ECgFp5Point> for AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_sub_affine(&self);
        r
    }
}

impl Sub<&ECgFp5Point> for AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_sub_affine(&self);
        r
    }
}

impl Sub<ECgFp5Point> for &AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_sub_affine(self);
        r
    }
}

impl Sub<&ECgFp5Point> for &AffinePoint {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn sub(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_sub_affine(self);
        r
    }
}

impl SubAssign<ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: ECgFp5Point) {
        self.set_sub(&other);
    }
}

impl SubAssign<&ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: &ECgFp5Point) {
        self.set_sub(other);
    }
}

impl SubAssign<AffinePoint> for ECgFp5Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: AffinePoint) {
        self.set_sub_affine(&other);
    }
}

impl SubAssign<&AffinePoint> for ECgFp5Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: &AffinePoint) {
        self.set_sub_affine(other);
    }
}

impl Neg for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn neg(self) -> ECgFp5Point {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn neg(self) -> ECgFp5Point {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Neg for AffinePoint {
    type Output = AffinePoint;

    #[inline(always)]
    fn neg(self) -> AffinePoint {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &AffinePoint {
    type Output = AffinePoint;

    #[inline(always)]
    fn neg(self) -> AffinePoint {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Mul<ECgFp5Scalar> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: ECgFp5Scalar) -> ECgFp5Point {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&ECgFp5Scalar> for ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: &ECgFp5Scalar) -> ECgFp5Point {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<ECgFp5Scalar> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: ECgFp5Scalar) -> ECgFp5Point {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&ECgFp5Scalar> for &ECgFp5Point {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: &ECgFp5Scalar) -> ECgFp5Point {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<ECgFp5Scalar> for ECgFp5Point {
    #[inline(always)]
    fn mul_assign(&mut self, other: ECgFp5Scalar) {
        self.set_mul(&other);
    }
}

impl MulAssign<&ECgFp5Scalar> for ECgFp5Point {
    #[inline(always)]
    fn mul_assign(&mut self, other: &ECgFp5Scalar) {
        self.set_mul(other);
    }
}

impl Mul<ECgFp5Point> for ECgFp5Scalar {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_mul(&self);
        r
    }
}

impl Mul<&ECgFp5Point> for ECgFp5Scalar {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_mul(&self);
        r
    }
}

impl Mul<ECgFp5Point> for &ECgFp5Scalar {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: ECgFp5Point) -> ECgFp5Point {
        let mut r = other;
        r.set_mul(self);
        r
    }
}

impl Mul<&ECgFp5Point> for &ECgFp5Scalar {
    type Output = ECgFp5Point;

    #[inline(always)]
    fn mul(self, other: &ECgFp5Point) -> ECgFp5Point {
        let mut r = *other;
        r.set_mul(self);
        r
    }
}

impl PartialEq<ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn eq(&self, other: &ECgFp5Point) -> bool {
        self.equals(*other)
    }
}

impl PartialEq<&ECgFp5Point> for ECgFp5Point {
    #[inline(always)]
    fn eq(&self, other: &&ECgFp5Point) -> bool {
        self.equals(**other)
    }
}

impl PartialEq<ECgFp5Point> for &ECgFp5Point {
    #[inline(always)]
    fn eq(&self, other: &ECgFp5Point) -> bool {
        self.equals(*other)
    }
}

impl Eq for ECgFp5Point {}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::quintic::QuinticExtension;
    use plonky2::field::types::{Field, Sample};
    use rand::thread_rng;

    use super::{AffinePoint, ECgFp5Point, WeierstrassPoint};
    use crate::eddsa::curve::base_field::InverseOrZero;
    use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
    use crate::types::config::{F, const_f};

    fn test_vectors() -> [QuinticExtension<F>; 8] {
        // P0 is neutral of G.
        // P1 is a random point in G (encoded as w1)
        // P2 = e*P1 in G (encoded as w2)
        // P3 = P1 + P2 (in G) (encoded as w3)
        // P4 = 2*P1 (in G) (encoded as w4)
        // P5 = 2*P2 (in G) (encoded as w5)
        // P6 = 2*P1 + P2 (in G) (encoded as w6)
        // P7 = P1 + 2*P2 (in G) (encoded as w7)

        let w0 = QuinticExtension::<F>::ZERO;
        let w1 = QuinticExtension([
            const_f(12539254003028696409),
            const_f(15524144070600887654),
            const_f(15092036948424041984),
            const_f(11398871370327264211),
            const_f(10958391180505708567),
        ]);
        let w2 = QuinticExtension([
            const_f(11001943240060308920),
            const_f(17075173755187928434),
            const_f(3940989555384655766),
            const_f(15017795574860011099),
            const_f(5548543797011402287),
        ]);
        let w3 = QuinticExtension([
            const_f(246872606398642312),
            const_f(4900963247917836450),
            const_f(7327006728177203977),
            const_f(13945036888436667069),
            const_f(3062018119121328861),
        ]);
        let w4 = QuinticExtension([
            const_f(8058035104653144162),
            const_f(16041715455419993830),
            const_f(7448530016070824199),
            const_f(11253639182222911208),
            const_f(6228757819849640866),
        ]);
        let w5 = QuinticExtension([
            const_f(10523134687509281194),
            const_f(11148711503117769087),
            const_f(9056499921957594891),
            const_f(13016664454465495026),
            const_f(16494247923890248266),
        ]);
        let w6 = QuinticExtension([
            const_f(12173306542237620),
            const_f(6587231965341539782),
            const_f(17027985748515888117),
            const_f(17194831817613584995),
            const_f(10056734072351459010),
        ]);
        let w7 = QuinticExtension([
            const_f(9420857400785992333),
            const_f(4695934009314206363),
            const_f(14471922162341187302),
            const_f(13395190104221781928),
            const_f(16359223219913018041),
        ]);

        [w0, w1, w2, w3, w4, w5, w6, w7]
    }

    #[test]
    fn test_basic_ops() {
        let [w0, w1, w2, w3, w4, w5, w6, w7] = test_vectors();

        // Values that should not decode successfully.
        let bww: [QuinticExtension<F>; 6] = [
            QuinticExtension([
                const_f(13557832913345268708),
                const_f(15669280705791538619),
                const_f(8534654657267986396),
                const_f(12533218303838131749),
                const_f(5058070698878426028),
            ]),
            QuinticExtension([
                const_f(135036726621282077),
                const_f(17283229938160287622),
                const_f(13113167081889323961),
                const_f(1653240450380825271),
                const_f(520025869628727862),
            ]),
            QuinticExtension([
                const_f(6727960962624180771),
                const_f(17240764188796091916),
                const_f(3954717247028503753),
                const_f(1002781561619501488),
                const_f(4295357288570643789),
            ]),
            QuinticExtension([
                const_f(4578929270179684956),
                const_f(3866930513245945042),
                const_f(7662265318638150701),
                const_f(9503686272550423634),
                const_f(12241691520798116285),
            ]),
            QuinticExtension([
                const_f(16890297404904119082),
                const_f(6169724643582733633),
                const_f(9725973298012340311),
                const_f(5977049210035183790),
                const_f(11379332130141664883),
            ]),
            QuinticExtension([
                const_f(13777379982711219130),
                const_f(14715168412651470168),
                const_f(17942199593791635585),
                const_f(6188824164976547520),
                const_f(15461469634034461986),
            ]),
        ];

        assert!(ECgFp5Point::validate(w0));
        assert!(ECgFp5Point::validate(w1));
        assert!(ECgFp5Point::validate(w2));
        assert!(ECgFp5Point::validate(w3));
        assert!(ECgFp5Point::validate(w4));
        assert!(ECgFp5Point::validate(w5));
        assert!(ECgFp5Point::validate(w6));
        assert!(ECgFp5Point::validate(w7));

        let p0 = ECgFp5Point::decode(w0).expect("w0 should successfully decode");
        let p1 = ECgFp5Point::decode(w1).expect("w1 should successfully decode");
        let p2 = ECgFp5Point::decode(w2).expect("w2 should successfully decode");
        let p3 = ECgFp5Point::decode(w3).expect("w3 should successfully decode");
        let p4 = ECgFp5Point::decode(w4).expect("w4 should successfully decode");
        let p5 = ECgFp5Point::decode(w5).expect("w5 should successfully decode");
        let p6 = ECgFp5Point::decode(w6).expect("w6 should successfully decode");
        let p7 = ECgFp5Point::decode(w7).expect("w7 should successfully decode");

        assert!(p0.is_neutral());
        assert!(!p1.is_neutral());
        assert!(!p2.is_neutral());
        assert!(!p3.is_neutral());
        assert!(!p4.is_neutral());
        assert!(!p5.is_neutral());
        assert!(!p6.is_neutral());
        assert!(!p7.is_neutral());

        assert_eq!(p0, p0);
        assert_eq!(p1, p1);
        assert_ne!(p0, p1);
        assert_ne!(p1, p0);
        assert_ne!(p1, p2);

        assert_eq!(p0.encode(), w0);
        assert_eq!(p1.encode(), w1);
        assert_eq!(p2.encode(), w2);
        assert_eq!(p3.encode(), w3);
        assert_eq!(p4.encode(), w4);
        assert_eq!(p5.encode(), w5);
        assert_eq!(p6.encode(), w6);
        assert_eq!(p7.encode(), w7);

        for &w in bww.iter() {
            assert!(!ECgFp5Point::validate(w));
            assert!(ECgFp5Point::decode(w).is_none());
        }

        assert_eq!((p1 + p2).encode(), w3);
        assert_eq!((p1 + p1).encode(), w4);
        assert_eq!(p2.double().encode(), w5);
        assert_eq!((p1.double() + p2).encode(), w6);
        assert_eq!((p1 + p2 + p2).encode(), w7);

        assert_eq!((p0.double()).encode(), QuinticExtension::<F>::ZERO);
        assert_eq!((p0 + p0).encode(), QuinticExtension::<F>::ZERO);
        assert_eq!((p0 + p1).encode(), w1);
        assert_eq!((p1 + p0).encode(), w1);

        for i in 0..10 {
            let q1 = p1.mdouble(i);
            let mut q2 = p1;
            for _ in 0..i {
                q2 = q2.double();
            }
            assert_eq!(q1, q2);
        }

        let p2_affine = AffinePoint {
            x: p2.x * p2.z.inverse_or_zero(),
            u: p2.u * p2.t.inverse_or_zero(),
        };
        assert_eq!(p1 + p2_affine, p1 + p2);
    }

    #[test]
    fn test_to_affine() {
        let w = QuinticExtension([
            const_f(12539254003028696409),
            const_f(15524144070600887654),
            const_f(15092036948424041984),
            const_f(11398871370327264211),
            const_f(10958391180505708567),
        ]);
        let p = ECgFp5Point::decode(w).expect("w should successfully decode");

        // Create an array of 8 points.
        let mut tab1 = [ECgFp5Point::NEUTRAL; 8];
        tab1[0] = p.double();
        for i in 1..tab1.len() {
            tab1[i] = tab1[0] + tab1[i - 1];
        }

        // Test conversion to affine coordinates.
        for n in 1..(tab1.len() + 1) {
            let tab2 = ECgFp5Point::batch_to_affine(&tab1);
            for i in 0..n {
                assert_eq!(tab1[i].z * tab2[i].x, tab1[i].x);
                assert_eq!(tab1[i].t * tab2[i].u, tab1[i].u);
            }
        }

        // Test lookup.
        let win = ECgFp5Point::batch_to_affine(&tab1);
        let p1_affine = AffinePoint::lookup(&win, 0);
        assert_eq!(p1_affine.x, QuinticExtension::<F>::ZERO);
        assert_eq!(p1_affine.u, QuinticExtension::<F>::ZERO);
        for i in 1..9 {
            let p2_affine = AffinePoint::lookup(&win, i as i32);
            assert_eq!(tab1[i - 1].z * p2_affine.x, tab1[i - 1].x);
            assert_eq!(tab1[i - 1].t * p2_affine.u, tab1[i - 1].u);

            let p3_affine = AffinePoint::lookup(&win, -(i as i32));
            assert_eq!(tab1[i - 1].z * p3_affine.x, tab1[i - 1].x);
            assert_eq!(tab1[i - 1].t * p3_affine.u, -tab1[i - 1].u);
        }
    }

    #[test]
    fn test_scalar_mul() {
        // w1 = encoding of a random point P1
        // ebuf = encoding of a random scalar e
        // w2 = encoding of P2 = e*P1
        let w1 = QuinticExtension([
            const_f(7534507442095725921),
            const_f(16658460051907528927),
            const_f(12417574136563175256),
            const_f(2750788641759288856),
            const_f(620002843272906439),
        ]);
        let ebuf: [u8; 40] = [
            0x1B, 0x18, 0x51, 0xC8, 0x1D, 0x22, 0xD4, 0x0D, 0x6D, 0x36, 0xEC, 0xCE, 0x54, 0x27,
            0x41, 0x66, 0x08, 0x14, 0x2F, 0x8F, 0xFF, 0x64, 0xB4, 0x76, 0x28, 0xCD, 0x3F, 0xF8,
            0xAA, 0x25, 0x16, 0xD4, 0xBA, 0xD0, 0xCC, 0x02, 0x1A, 0x44, 0x7C, 0x03,
        ];
        let w2 = QuinticExtension([
            const_f(9486104512504676657),
            const_f(14312981644741144668),
            const_f(5159846406177847664),
            const_f(15978863787033795628),
            const_f(3249948839313771192),
        ]);

        let p1 = ECgFp5Point::decode(w1).expect("w1 should successfully decode");
        let p2 = ECgFp5Point::decode(w2).expect("w2 should successfully decode");
        let (e, ce) = ECgFp5Scalar::try_from_noncanonical_bytes(&ebuf);

        assert!(ce == 0xFFFFFFFFFFFFFFFF);
        let q1 = p1 * e;
        assert!(q1 == p2);
        assert!(q1.encode() == w2);

        let q2 = e * p1;
        assert!(q2 == p2);
        assert!(q2.encode() == w2);
    }

    #[test]
    fn test_decode() {
        let [w0, w1, w2, w3, w4, w5, w6, w7] = test_vectors();

        let p0_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(6148914689804861440),
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ]),
            y: QuinticExtension::<F>::ZERO,
            is_inf: true,
        };
        let p0 = WeierstrassPoint::decode(w0).expect("w0 should successfully decode");
        assert_eq!(p0, p0_expected);

        let p1_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(7887569478949190020),
                const_f(11586418388990522938),
                const_f(13676447623055915878),
                const_f(5945168854809921881),
                const_f(16291886980725359814),
            ]),
            y: QuinticExtension([
                const_f(7556511254681645335),
                const_f(17611929280367064763),
                const_f(9410908488141053806),
                const_f(11351540010214108766),
                const_f(4846226015431423207),
            ]),
            is_inf: false,
        };
        let p1 = WeierstrassPoint::decode(w1).expect("w1 should successfully decode");
        assert_eq!(p1, p1_expected);

        let p2_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(11231216549003316587),
                const_f(17312878720767554617),
                const_f(5614299211412933260),
                const_f(2256199868722187419),
                const_f(14229722163821261464),
            ]),
            y: QuinticExtension([
                const_f(11740132275098847128),
                const_f(18250632754932612452),
                const_f(6988589976052950880),
                const_f(13612651576898186637),
                const_f(16040252831112129154),
            ]),
            is_inf: false,
        };
        let p2 = WeierstrassPoint::decode(w2).expect("w2 should successfully decode");
        assert_eq!(p2, p2_expected);

        let p3_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(567456832026211571),
                const_f(6401615614732569674),
                const_f(7303004494044972219),
                const_f(4332356015409706768),
                const_f(4663512734739523713),
            ]),
            y: QuinticExtension([
                const_f(13838792670272995877),
                const_f(11742686110311813089),
                const_f(17972799251722850796),
                const_f(8534723577625674697),
                const_f(3138422718990519265),
            ]),
            is_inf: false,
        };
        let p3 = WeierstrassPoint::decode(w3).expect("w3 should successfully decode");
        assert_eq!(p3, p3_expected);

        let p4_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(2626390539619063455),
                const_f(3069873143820007175),
                const_f(16481805966921623903),
                const_f(2169403494164322467),
                const_f(15849876939764656634),
            ]),
            y: QuinticExtension([
                const_f(8052493994140007067),
                const_f(12476750341447220703),
                const_f(7297584762312352412),
                const_f(4456043296886321460),
                const_f(17416054515469523789),
            ]),
            is_inf: false,
        };
        let p4 = WeierstrassPoint::decode(w4).expect("w4 should successfully decode");
        assert_eq!(p4, p4_expected);

        let p5_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(3378618241466923429),
                const_f(1600085176765664645),
                const_f(8450735902517439914),
                const_f(879305481131694650),
                const_f(9249368002914244868),
            ]),
            y: QuinticExtension([
                const_f(7063301786803892166),
                const_f(16450112846546843898),
                const_f(13291990378137922105),
                const_f(17122501309646837992),
                const_f(13551174888872382132),
            ]),
            is_inf: false,
        };
        let p5 = WeierstrassPoint::decode(w5).expect("w5 should successfully decode");
        assert_eq!(p5, p5_expected);

        let p6_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(12792842147978866906),
                const_f(10605017725125541653),
                const_f(7515179057747849898),
                const_f(4244613931017322576),
                const_f(5015379385130367832),
            ]),
            y: QuinticExtension([
                const_f(11618884250209642346),
                const_f(14788516166813429253),
                const_f(7317520700234795285),
                const_f(12825292405177435802),
                const_f(17658454967394645353),
            ]),
            is_inf: false,
        };
        let p6 = WeierstrassPoint::decode(w6).expect("w6 should successfully decode");
        assert_eq!(p6, p6_expected);

        let p7_expected = WeierstrassPoint {
            x: QuinticExtension([
                const_f(10440794216646581227),
                const_f(13992847258701590930),
                const_f(11213401763785319360),
                const_f(12830171931568113117),
                const_f(6220154342199499160),
            ]),
            y: QuinticExtension([
                const_f(7971683838841472962),
                const_f(1639066249976938469),
                const_f(15015315060237521031),
                const_f(10847769264696425470),
                const_f(9177491810370773777),
            ]),
            is_inf: false,
        };
        let p7 = WeierstrassPoint::decode(w7).expect("w7 should successfully decode");
        assert_eq!(p7, p7_expected);

        let w_gen = QuinticExtension::<F>::from_canonical_u16(4);
        let g = WeierstrassPoint::decode(w_gen).expect("w_gen should successfully decode");
        assert_eq!(g, WeierstrassPoint::GENERATOR);
    }

    #[test]
    fn test_decode_random() {
        let mut rng = thread_rng();
        for _ in 0..30 {
            let point = ECgFp5Point::sample(&mut rng);
            let encoded = point.encode();
            let decoded = ECgFp5Point::decode(encoded).expect("decoding should succeed");
            assert_eq!(point, decoded);

            let encoded = point.to_weierstrass().encode();
            let decoded = WeierstrassPoint::decode(encoded).expect("decoding should succeed");
            assert_eq!(point.to_weierstrass(), decoded);

            let decoded = ECgFp5Point::decode(encoded).expect("decoding should succeed");
            assert_eq!(point, decoded);
        }
    }
}
