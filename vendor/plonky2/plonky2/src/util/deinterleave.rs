//! Value-exact stride-2 copies for Goldilocks-sized `Copy` elements.
//!
//! Production `GoldilocksField` is `#[repr(transparent)]` over `u64`. The
//! half-domain Range/U32 path and the even-row companion fill both walk
//! interleaved `[c0, c1, c0, c1, …]` or even-row `[x0, _, x1, _, …]` layouts
//! with a scalar gather. On Apple Silicon that gather is a `vld2q_u64`
//! deinterleave: same bytes, sequential loads. The scalar loop is the
//! reference and the fallback for odd tails / non-8-byte `T`.

use core::mem::{align_of, size_of};

#[inline]
const fn neon_u64_layout<T>() -> bool {
    size_of::<T>() == 8 && align_of::<T>() == 8
}

/// `dst[k] = src[2k]`. `src` must contain at least `2 * dst.len()` elements.
#[inline]
pub fn copy_even_indices<T: Copy>(src: &[T], dst: &mut [T]) {
    let n = dst.len();
    assert!(
        src.len() >= 2 * n,
        "copy_even_indices: src.len()={} < 2*dst.len()={}",
        src.len(),
        2 * n
    );

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if neon_u64_layout::<T>() {
        unsafe {
            copy_even_u64(
                core::slice::from_raw_parts(src.as_ptr().cast::<u64>(), src.len()),
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u64>(), n),
            );
        }
        return;
    }

    for k in 0..n {
        dst[k] = src[2 * k];
    }
}

/// Split interleaved `[e0, o0, e1, o1, …]` into contiguous `even` / `odd`.
/// `src.len() == 2 * even.len() == 2 * odd.len()`.
#[inline]
pub fn deinterleave_pairs<T: Copy>(src: &[T], even: &mut [T], odd: &mut [T]) {
    let n = even.len();
    assert_eq!(odd.len(), n, "deinterleave_pairs: even/odd length mismatch");
    assert_eq!(
        src.len(),
        2 * n,
        "deinterleave_pairs: src.len()={} != 2*n={}",
        src.len(),
        2 * n
    );

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if neon_u64_layout::<T>() {
        unsafe {
            deinterleave_u64(
                core::slice::from_raw_parts(src.as_ptr().cast::<u64>(), src.len()),
                core::slice::from_raw_parts_mut(even.as_mut_ptr().cast::<u64>(), n),
                core::slice::from_raw_parts_mut(odd.as_mut_ptr().cast::<u64>(), n),
            );
        }
        return;
    }

    for k in 0..n {
        even[k] = src[2 * k];
        odd[k] = src[2 * k + 1];
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[target_feature(enable = "neon")]
unsafe fn copy_even_u64(src: &[u64], dst: &mut [u64]) {
    use core::arch::aarch64::{vld2q_u64, vst1q_u64};
    let n = dst.len();
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let p = src_ptr.add(2 * i);
        let a = vld2q_u64(p);
        let b = vld2q_u64(p.add(4));
        let c = vld2q_u64(p.add(8));
        let d = vld2q_u64(p.add(12));
        let o = dst_ptr.add(i);
        vst1q_u64(o, a.0);
        vst1q_u64(o.add(2), b.0);
        vst1q_u64(o.add(4), c.0);
        vst1q_u64(o.add(6), d.0);
        i += 8;
    }
    while i + 2 <= n {
        let pair = vld2q_u64(src_ptr.add(2 * i));
        vst1q_u64(dst_ptr.add(i), pair.0);
        i += 2;
    }
    while i < n {
        *dst_ptr.add(i) = *src_ptr.add(2 * i);
        i += 1;
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[target_feature(enable = "neon")]
unsafe fn deinterleave_u64(src: &[u64], even: &mut [u64], odd: &mut [u64]) {
    use core::arch::aarch64::{vld2q_u64, vst1q_u64};
    let n = even.len();
    let src_ptr = src.as_ptr();
    let even_ptr = even.as_mut_ptr();
    let odd_ptr = odd.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let p = src_ptr.add(2 * i);
        let a = vld2q_u64(p);
        let b = vld2q_u64(p.add(4));
        let c = vld2q_u64(p.add(8));
        let d = vld2q_u64(p.add(12));
        let e = even_ptr.add(i);
        let o = odd_ptr.add(i);
        vst1q_u64(e, a.0);
        vst1q_u64(o, a.1);
        vst1q_u64(e.add(2), b.0);
        vst1q_u64(o.add(2), b.1);
        vst1q_u64(e.add(4), c.0);
        vst1q_u64(o.add(4), c.1);
        vst1q_u64(e.add(6), d.0);
        vst1q_u64(o.add(6), d.1);
        i += 8;
    }
    while i + 2 <= n {
        let pair = vld2q_u64(src_ptr.add(2 * i));
        vst1q_u64(even_ptr.add(i), pair.0);
        vst1q_u64(odd_ptr.add(i), pair.1);
        i += 2;
    }
    while i < n {
        *even_ptr.add(i) = *src_ptr.add(2 * i);
        *odd_ptr.add(i) = *src_ptr.add(2 * i + 1);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;

    #[test]
    fn copy_even_matches_scalar_u64() {
        for n in [0usize, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 256, 257] {
            let src: Vec<u64> = (0..2 * n as u64).map(|i| i.wrapping_mul(0x9e37) ^ 0x5555).collect();
            let mut got = vec![0u64; n];
            copy_even_indices(&src, &mut got);
            let expect: Vec<u64> = (0..n).map(|k| src[2 * k]).collect();
            assert_eq!(got, expect, "n={n}");
        }
    }

    #[test]
    fn deinterleave_pairs_matches_scalar_u64() {
        for n in [0usize, 1, 2, 3, 7, 8, 9, 16, 17, 63, 64, 255, 256, 257, 1024] {
            let src: Vec<u64> = (0..2 * n as u64)
                .map(|i| i.wrapping_mul(0x9e37) ^ 0xa5a5)
                .collect();
            let mut even = vec![0u64; n];
            let mut odd = vec![0u64; n];
            deinterleave_pairs(&src, &mut even, &mut odd);
            for k in 0..n {
                assert_eq!(even[k], src[2 * k], "even n={n} k={k}");
                assert_eq!(odd[k], src[2 * k + 1], "odd n={n} k={k}");
            }
        }
    }

    #[test]
    fn deinterleave_pairs_matches_scalar_goldilocks() {
        let n = 4096usize;
        let src: Vec<GoldilocksField> = (0..2 * n)
            .map(|i| GoldilocksField::from_canonical_u64((i as u64).wrapping_mul(13) + 7))
            .collect();
        let mut even = vec![GoldilocksField::ZERO; n];
        let mut odd = vec![GoldilocksField::ZERO; n];
        deinterleave_pairs(&src, &mut even, &mut odd);
        for k in 0..n {
            assert_eq!(even[k], src[2 * k]);
            assert_eq!(odd[k], src[2 * k + 1]);
        }
    }
}
