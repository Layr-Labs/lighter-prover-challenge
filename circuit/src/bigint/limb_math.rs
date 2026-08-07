//! Fixed-width u32-limb integer arithmetic for witness generators.
//!
//! The BigUint-family witness generators (nonnative field ops, div_rem) run
//! for every ECDSA signature and fee division on the ranked workload. Their
//! operand shapes are small and statically bounded (256-bit field elements,
//! 512-bit products), so the general heap-allocating `BigUint` machinery is
//! pure overhead. This module provides the three primitives those generators
//! need — comparison, schoolbook multiplication, and Knuth Algorithm D long
//! division — over caller-owned little-endian u32 digit slices, with no heap
//! allocation anywhere.
//!
//! All functions treat digit slices as little-endian (least significant limb
//! first) and tolerate leading (high-index) zero digits in inputs.

/// Maximum operand width, in u32 limbs, that the stack-array entry points
/// accept. 512-bit products of 256-bit nonnative elements need 16; the
/// div_rem generator falls back to `BigUint` beyond this cap.
pub const MAX_LIMBS: usize = 32;

/// Logical length of `x` ignoring leading zero digits.
#[inline]
pub fn significant_len(x: &[u32]) -> usize {
    let mut n = x.len();
    while n > 0 && x[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// Compares `a` and `b` as little-endian integers (leading zeros ignored).
pub fn cmp_limbs(a: &[u32], b: &[u32]) -> core::cmp::Ordering {
    let la = significant_len(a);
    let lb = significant_len(b);
    if la != lb {
        return la.cmp(&lb);
    }
    for i in (0..la).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    core::cmp::Ordering::Equal
}

/// Schoolbook multiplication: `out[..la+lb] = a * b`. `out` must be at least
/// `significant_len(a) + significant_len(b)` long; it is fully overwritten up
/// to that length (and zeroed there first).
pub fn mul_limbs(a: &[u32], b: &[u32], out: &mut [u32]) -> usize {
    let la = significant_len(a);
    let lb = significant_len(b);
    let lo = la + lb;
    debug_assert!(out.len() >= lo);
    for o in out[..lo].iter_mut() {
        *o = 0;
    }
    for (i, &ai) in a[..la].iter().enumerate() {
        if ai == 0 {
            continue;
        }
        let mut carry = 0u64;
        for (j, &bj) in b[..lb].iter().enumerate() {
            let t = ai as u64 * bj as u64 + out[i + j] as u64 + carry;
            out[i + j] = t as u32;
            carry = t >> 32;
        }
        out[i + lb] = carry as u32;
    }
    significant_len(&out[..lo])
}

/// Knuth Algorithm D (TAOCP 4.3.1) long division over base-2^32 digits:
/// computes `q = u / v` and `r = u % v`.
///
/// - `u` and `v` are little-endian; leading zeros are tolerated.
/// - `v` must be nonzero (the callers guard the zero-divisor case).
/// - `q_out` receives the quotient's low `q_out.len()` digits; any digits the
///   quotient does not need are zeroed. Panics (via the final assert) if the
///   quotient does not fit.
/// - `r_out` likewise receives the remainder (always fits in
///   `significant_len(v)` digits).
///
/// The implementation is the classical normalize / estimate-q̂ / multiply-
/// subtract / add-back loop, with the two-digit q̂ refinement that bounds the
/// add-back probability. Working storage is fixed stack arrays sized by
/// `MAX_LIMBS`; inputs longer than that are a caller bug (asserted).
pub fn div_rem_limbs(u: &[u32], v: &[u32], q_out: &mut [u32], r_out: &mut [u32]) {
    let m = significant_len(u);
    let n = significant_len(v);
    assert!(n > 0, "division by zero");
    assert!(m <= MAX_LIMBS && n <= MAX_LIMBS, "operand exceeds MAX_LIMBS");

    for d in q_out.iter_mut() {
        *d = 0;
    }
    for d in r_out.iter_mut() {
        *d = 0;
    }

    // u < v: quotient zero, remainder u.
    if cmp_limbs(&u[..m], &v[..n]) == core::cmp::Ordering::Less {
        assert!(r_out.len() >= m, "remainder does not fit");
        r_out[..m].copy_from_slice(&u[..m]);
        return;
    }

    // Single-digit divisor: one linear pass.
    if n == 1 {
        let d = v[0] as u64;
        let mut rem = 0u64;
        assert!(q_out.len() >= m, "quotient does not fit");
        for i in (0..m).rev() {
            let cur = (rem << 32) | u[i] as u64;
            q_out[i] = (cur / d) as u32;
            rem = cur % d;
        }
        assert!(!r_out.is_empty(), "remainder does not fit");
        r_out[0] = rem as u32;
        return;
    }

    // D1: normalize so the divisor's top digit has its high bit set.
    let shift = v[n - 1].leading_zeros();
    let mut un = [0u32; MAX_LIMBS + 1];
    let mut vn = [0u32; MAX_LIMBS];
    if shift == 0 {
        un[..m].copy_from_slice(&u[..m]);
        un[m] = 0;
        vn[..n].copy_from_slice(&v[..n]);
    } else {
        for i in (1..m).rev() {
            un[i] = (u[i] << shift) | (u[i - 1] >> (32 - shift));
        }
        un[0] = u[0] << shift;
        un[m] = u[m - 1] >> (32 - shift);
        for i in (1..n).rev() {
            vn[i] = (v[i] << shift) | (v[i - 1] >> (32 - shift));
        }
        vn[0] = v[0] << shift;
    }

    let quotient_digits = m - n + 1;
    assert!(q_out.len() >= quotient_digits, "quotient does not fit");

    // D2–D7: main loop over quotient digits, most significant first.
    let vtop = vn[n - 1] as u64;
    let vsecond = vn[n - 2] as u64;
    for j in (0..quotient_digits).rev() {
        // D3: estimate q̂ from the top two dividend digits.
        let top = ((un[j + n] as u64) << 32) | un[j + n - 1] as u64;
        let mut qhat = top / vtop;
        let mut rhat = top % vtop;
        while qhat >> 32 != 0 || qhat * vsecond > ((rhat << 32) | un[j + n - 2] as u64) {
            qhat -= 1;
            rhat += vtop;
            if rhat >> 32 != 0 {
                break;
            }
        }

        // D4: multiply and subtract `qhat * v` from the dividend window.
        let mut borrow = 0i64;
        let mut carry = 0u64;
        for i in 0..n {
            let p = qhat * vn[i] as u64 + carry;
            carry = p >> 32;
            let t = un[j + i] as i64 - (p as u32) as i64 - borrow;
            un[j + i] = t as u32;
            borrow = if t < 0 { 1 } else { 0 };
        }
        let t = un[j + n] as i64 - carry as i64 - borrow;
        un[j + n] = t as u32;

        // D5–D6: if we overshot (rare), add one divisor back.
        if t < 0 {
            qhat -= 1;
            let mut carry = 0u64;
            for i in 0..n {
                let s = un[j + i] as u64 + vn[i] as u64 + carry;
                un[j + i] = s as u32;
                carry = s >> 32;
            }
            un[j + n] = (un[j + n] as u64).wrapping_add(carry) as u32;
        }

        q_out[j] = qhat as u32;
    }

    // D8: denormalize the remainder.
    assert!(r_out.len() >= n, "remainder does not fit");
    if shift == 0 {
        r_out[..n].copy_from_slice(&un[..n]);
    } else {
        for i in 0..n - 1 {
            r_out[i] = (un[i] >> shift) | (un[i + 1] << (32 - shift));
        }
        r_out[n - 1] = un[n - 1] >> shift;
    }
}

#[cfg(test)]
mod tests {
    use num::BigUint;

    use super::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn to_biguint(d: &[u32]) -> BigUint {
        BigUint::new(d.to_vec())
    }

    fn check_div_rem(u: &[u32], v: &[u32]) {
        let mut q = [0u32; MAX_LIMBS];
        let mut r = [0u32; MAX_LIMBS];
        div_rem_limbs(u, v, &mut q, &mut r);
        let (eq, er) = num::Integer::div_rem(&to_biguint(u), &to_biguint(v));
        assert_eq!(to_biguint(&q), eq, "quotient mismatch u={u:?} v={v:?}");
        assert_eq!(to_biguint(&r), er, "remainder mismatch u={u:?} v={v:?}");
    }

    #[test]
    fn div_rem_limbs_matches_biguint() {
        let mut seed = 0xD1CE_D1CE_D1CE_D1CEu64;
        // Random shapes across the generator-relevant sizes, including the
        // 16/8 nonnative-reduction shape and ragged lengths.
        for (lu, lv) in [
            (1usize, 1usize),
            (2, 1),
            (2, 2),
            (4, 2),
            (4, 4),
            (8, 8),
            (9, 8),
            (16, 8),
            (16, 16),
            (17, 9),
            (32, 8),
            (32, 32),
        ] {
            for _ in 0..300 {
                let mut u = vec![0u32; lu];
                let mut v = vec![0u32; lv];
                for d in u.iter_mut() {
                    *d = splitmix64(&mut seed) as u32;
                }
                for d in v.iter_mut() {
                    *d = splitmix64(&mut seed) as u32;
                }
                // Occasionally force leading zeros and tiny divisors.
                if seed % 5 == 0 {
                    let k = (splitmix64(&mut seed) as usize) % lv;
                    for d in v[lv - 1 - k..].iter_mut() {
                        *d = 0;
                    }
                }
                if significant_len(&v) == 0 {
                    v[0] = 1;
                }
                check_div_rem(&u, &v);
            }
        }
    }

    #[test]
    fn div_rem_limbs_adversarial_addback_cases() {
        // Divisor tops chosen to exercise the q̂-overestimate and add-back
        // branches: high bit set, second digit small, dividend windows of
        // all-ones. These are the classical Algorithm D trap inputs.
        let cases: [(&[u32], &[u32]); 8] = [
            (&[0, 0, 0x8000_0000, 0x7FFF_FFFF], &[1, 0, 0x8000_0000]),
            (
                &[0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
                &[0xFFFF_FFFE, 0xFFFF_FFFF],
            ),
            (
                &[0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
                &[1, 0xFFFF_FFFF],
            ),
            (&[0, 0xFFFF_FFFE, 0x8000_0000], &[0xFFFF_FFFF, 0x8000_0000]),
            (&[0, 0, 0x8000_0000], &[1, 0x8000_0000]),
            (&[3, 0, 0x8000_0000], &[1, 0, 0x2000_0000]),
            (&[0, 0, 1, 0, 0, 0, 0, 1], &[0xFFFF_FFFF, 0, 0, 1]),
            (&[0xFFFF_FFFF; 16], &[0xFFFF_FFFF, 0, 0, 0, 0, 0, 0, 0x8000_0000]),
        ];
        for (u, v) in cases {
            check_div_rem(u, v);
        }
    }

    #[test]
    fn mul_limbs_matches_biguint() {
        let mut seed = 0x5EED_5EED_5EED_5EEDu64;
        for (la, lb) in [(1usize, 1usize), (2, 3), (4, 4), (8, 8), (8, 16), (16, 16)] {
            for _ in 0..200 {
                let mut a = vec![0u32; la];
                let mut b = vec![0u32; lb];
                for d in a.iter_mut() {
                    *d = splitmix64(&mut seed) as u32;
                }
                for d in b.iter_mut() {
                    *d = splitmix64(&mut seed) as u32;
                }
                if seed % 4 == 0 {
                    a[la - 1] = 0;
                }
                let mut out = [0u32; 2 * MAX_LIMBS];
                mul_limbs(&a, &b, &mut out);
                assert_eq!(
                    to_biguint(&out[..la + lb]),
                    to_biguint(&a) * to_biguint(&b),
                    "mul mismatch a={a:?} b={b:?}"
                );
            }
        }
    }

    #[test]
    fn cmp_and_len_basics() {
        assert_eq!(significant_len(&[0, 0, 0]), 0);
        assert_eq!(significant_len(&[1, 0, 0]), 1);
        assert_eq!(cmp_limbs(&[1, 2], &[1, 2, 0]), core::cmp::Ordering::Equal);
        assert_eq!(cmp_limbs(&[0, 1], &[0xFFFF_FFFF]), core::cmp::Ordering::Greater);
    }
}
