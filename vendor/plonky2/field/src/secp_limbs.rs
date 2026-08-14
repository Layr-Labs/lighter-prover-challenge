//! 4-limb little-endian arithmetic for the secp256k1 base and scalar fields.
//! Marker: secp-limbs-1786693000.
//!
//! Replaces `BigUint` add/mul/neg on the hot witness path. Results are the
//! unique representative in `0..m`, so they are bit-identical to the previous
//! `to_canonical_biguint` + `mod_floor` path.

#![allow(dead_code)]

/// secp256k1 prime: `2^256 - 2^32 - 977`.
pub const SECP_P: [u64; 4] = [
    0xFFFF_FFFE_FFFF_FC2F,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// `2^256 ≡ SECP_R (mod p)` with `SECP_R = 2^32 + 977`.
const SECP_R: u128 = 0x1_0000_03D1;

/// secp256k1 group order.
pub const SECP_N: [u64; 4] = [
    0xBFD2_5E8C_D036_4141,
    0xBAAE_DCE6_AF48_A03B,
    0xFFFF_FFFF_FFFF_FFFE,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// `2^256 - n`, 129 bits: `2^256 ≡ N_PRIME (mod n)`.
const N_PRIME: [u64; 3] = [0x402D_A173_2FC9_BEBF, 0x4551_2319_50B7_5FC4, 0x1];

#[inline]
pub fn cmp4(a: [u64; 4], b: [u64; 4]) -> core::cmp::Ordering {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    core::cmp::Ordering::Equal
}

#[inline]
pub fn add4(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let (r0, c0) = a[0].overflowing_add(b[0]);
    let (r1, c1) = a[1].overflowing_add(c0 as u64);
    let (r1, c1b) = r1.overflowing_add(b[1]);
    let (r2, c2) = a[2].overflowing_add((c1 || c1b) as u64);
    let (r2, c2b) = r2.overflowing_add(b[2]);
    let (r3, c3) = a[3].overflowing_add((c2 || c2b) as u64);
    let (r3, c3b) = r3.overflowing_add(b[3]);
    ([r0, r1, r2, r3], c3 || c3b)
}

#[inline]
pub fn sub4(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let (r0, b0) = a[0].overflowing_sub(b[0]);
    let (r1, b1) = a[1].overflowing_sub(b0 as u64);
    let (r1, b1b) = r1.overflowing_sub(b[1]);
    let (r2, b2) = a[2].overflowing_sub((b1 || b1b) as u64);
    let (r2, b2b) = r2.overflowing_sub(b[2]);
    let (r3, b3) = a[3].overflowing_sub((b2 || b2b) as u64);
    let (r3, b3b) = r3.overflowing_sub(b[3]);
    ([r0, r1, r2, r3], b3 || b3b)
}

/// Any 4-limb value is `< 2^256`. Both moduli are `2^256 - ε` with `ε < 2^255`,
/// so one subtract yields the unique representative in `0..m`.
#[inline]
pub fn reduce4(a: [u64; 4], m: [u64; 4]) -> [u64; 4] {
    if cmp4(a, m) != core::cmp::Ordering::Less {
        sub4(a, m).0
    } else {
        a
    }
}

#[inline]
pub fn add_mod(a: [u64; 4], b: [u64; 4], m: [u64; 4]) -> [u64; 4] {
    let a = reduce4(a, m);
    let b = reduce4(b, m);
    let (s, carry) = add4(a, b);
    if carry || cmp4(s, m) != core::cmp::Ordering::Less {
        sub4(s, m).0
    } else {
        s
    }
}

#[inline]
pub fn neg_mod(a: [u64; 4], m: [u64; 4]) -> [u64; 4] {
    let a = reduce4(a, m);
    if a == [0; 4] {
        [0; 4]
    } else {
        sub4(m, a).0
    }
}

fn mul_wide(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let mut acc = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let t = acc[i + j] as u128 + (a[i] as u128) * (b[j] as u128) + carry;
            acc[i + j] = t as u64;
            carry = t >> 64;
        }
        let mut k = i + 4;
        while carry != 0 {
            let t = acc[k] as u128 + carry;
            acc[k] = t as u64;
            carry = t >> 64;
            k += 1;
        }
    }
    acc
}

/// Add `a * b` into the 8-limb accumulator (`b` is 3 limbs).
fn madd_4x3(t: &mut [u64; 8], a: [u64; 4], b: [u64; 3]) {
    for i in 0..4 {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..3 {
            let s = t[i + j] as u128 + (a[i] as u128) * (b[j] as u128) + carry;
            t[i + j] = s as u64;
            carry = s >> 64;
        }
        let mut k = i + 3;
        while carry != 0 {
            let s = t[k] as u128 + carry;
            t[k] = s as u64;
            carry = s >> 64;
            k += 1;
        }
    }
}

/// Fold a 512-bit product against `2^256 ≡ R (mod p)`.
fn reduce_secp_p(t: [u64; 8]) -> [u64; 4] {
    let r_limbs = [SECP_R as u64, (SECP_R >> 64) as u64, 0];
    let mut acc = t;
    for _ in 0..3 {
        if acc[4] == 0 && acc[5] == 0 && acc[6] == 0 && acc[7] == 0 {
            break;
        }
        let hi = [acc[4], acc[5], acc[6], acc[7]];
        acc[4] = 0;
        acc[5] = 0;
        acc[6] = 0;
        acc[7] = 0;
        madd_4x3(&mut acc, hi, r_limbs);
    }
    let r = [acc[0], acc[1], acc[2], acc[3]];
    let r = reduce4(r, SECP_P);
    reduce4(r, SECP_P)
}

/// Fold a 512-bit product against `2^256 ≡ N_PRIME (mod n)`.
fn reduce_secp_n(t: [u64; 8]) -> [u64; 4] {
    let mut acc = t;
    for _ in 0..4 {
        if acc[4] == 0 && acc[5] == 0 && acc[6] == 0 && acc[7] == 0 {
            break;
        }
        let hi = [acc[4], acc[5], acc[6], acc[7]];
        acc[4] = 0;
        acc[5] = 0;
        acc[6] = 0;
        acc[7] = 0;
        madd_4x3(&mut acc, hi, N_PRIME);
    }
    let r = [acc[0], acc[1], acc[2], acc[3]];
    let r = reduce4(r, SECP_N);
    reduce4(r, SECP_N)
}

#[inline]
pub fn mul_mod_p(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    reduce_secp_p(mul_wide(reduce4(a, SECP_P), reduce4(b, SECP_P)))
}

#[inline]
pub fn mul_mod_n(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    reduce_secp_n(mul_wide(reduce4(a, SECP_N), reduce4(b, SECP_N)))
}

#[cfg(test)]
mod tests {
    use num::bigint::BigUint;
    use num::Integer;

    use super::*;

    struct SplitMix(u64);
    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn limbs(&mut self) -> [u64; 4] {
            [self.next(), self.next(), self.next(), self.next()]
        }
    }

    fn to_big(a: [u64; 4]) -> BigUint {
        let mut acc = BigUint::from(0u8);
        for i in (0..4).rev() {
            acc = (acc << 64) + BigUint::from(a[i]);
        }
        acc
    }

    fn from_big(mut v: BigUint) -> [u64; 4] {
        let mut out = [0u64; 4];
        for i in 0..4 {
            let digits = v.to_u64_digits();
            out[i] = digits.first().copied().unwrap_or(0);
            v >>= 64;
        }
        out
    }

    fn p() -> BigUint {
        to_big(SECP_P)
    }
    fn n() -> BigUint {
        to_big(SECP_N)
    }

    #[test]
    fn add_mul_neg_match_biguint_p() {
        let mut rng = SplitMix(0xC0FF_EE00_D15E_A5E5);
        let modulus = p();
        for _ in 0..200 {
            let a = rng.limbs();
            let b = rng.limbs();
            let want_add = from_big((to_big(a) + to_big(b)) % &modulus);
            assert_eq!(add_mod(a, b, SECP_P), want_add, "add p");
            let want_neg = if to_big(a) % &modulus == BigUint::from(0u8) {
                [0; 4]
            } else {
                from_big((&modulus - (to_big(a) % &modulus)) % &modulus)
            };
            assert_eq!(neg_mod(a, SECP_P), want_neg, "neg p");
            let want_mul = from_big((to_big(a) * to_big(b)).mod_floor(&modulus));
            assert_eq!(mul_mod_p(a, b), want_mul, "mul p");
        }
        assert_eq!(add_mod(SECP_P, [1, 0, 0, 0], SECP_P), [1, 0, 0, 0]);
        assert_eq!(mul_mod_p([0; 4], rng.limbs()), [0; 4]);
        assert_eq!(mul_mod_p([1, 0, 0, 0], SECP_P), [0; 4]);
        let pm1 = sub4(SECP_P, [1, 0, 0, 0]).0;
        assert_eq!(mul_mod_p(pm1, pm1), [1, 0, 0, 0]);
    }

    #[test]
    fn add_mul_neg_match_biguint_n() {
        let mut rng = SplitMix(0xDEAD_BEEF_F00D_CAFE);
        let modulus = n();
        for _ in 0..200 {
            let a = rng.limbs();
            let b = rng.limbs();
            let want_add = from_big((to_big(a) + to_big(b)) % &modulus);
            assert_eq!(add_mod(a, b, SECP_N), want_add, "add n");
            let want_neg = if to_big(a) % &modulus == BigUint::from(0u8) {
                [0; 4]
            } else {
                from_big((&modulus - (to_big(a) % &modulus)) % &modulus)
            };
            assert_eq!(neg_mod(a, SECP_N), want_neg, "neg n");
            let want_mul = from_big((to_big(a) * to_big(b)).mod_floor(&modulus));
            assert_eq!(mul_mod_n(a, b), want_mul, "mul n");
        }
        assert_eq!(add_mod(SECP_N, [1, 0, 0, 0], SECP_N), [1, 0, 0, 0]);
        let nm1 = sub4(SECP_N, [1, 0, 0, 0]).0;
        assert_eq!(mul_mod_n(nm1, nm1), [1, 0, 0, 0]);
    }
}
