// Streaming-SVE (SME) Goldilocks FFT kernels for Apple M4-class cores.
//
// These run on the per-cluster SME block instead of the issuing core's own
// NEON/scalar pipes, so up to two designated "feeder" threads (one per
// P-cluster) add multiply throughput the machine otherwise leaves idle.
//
// Semantics are bit-identical to the NEON kernels in fft.rs:
//   * multiply: the reduce128 sequence of `mul_reduce_pair`
//     (neon_goldilocks_field.rs), with the (hi & EPSILON) * EPSILON product
//     computed by a 32x32->64 widening multiply (umullb) like the scalar
//     kernel's umull.
//   * add/sub: the double-correction sequence of gl_add_neon / gl_sub_neon.
// Raw u64 representatives are preserved everywhere: GoldilocksField values
// are compared and hashed in non-canonical form, so congruent-but-different
// results would change proof bytes.
//
// This file is the source of truth for gl_fft_ssve.s (checked in next to it),
// which is what the crate actually assembles via global_asm!. Regenerate with:
//   clang -O3 -march=armv8.7-a+sme2+sme-i16i64 -S gl_fft_ssve.c -o gl_fft_ssve.s
// (then keep the .arch directive at the top; see ssve_fft.rs).

#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#define GL_EPS 0xFFFFFFFFULL

static inline svuint64_t v_mul(svbool_t pg, svuint64_t a, svuint64_t b,
                               svuint64_t eps, svuint32_t eps32) __arm_streaming {
    svuint64_t lo = svmul_u64_x(pg, a, b);
    svuint64_t hi = svmulh_u64_x(pg, a, b);
    svuint64_t hi_hi = svlsr_n_u64_x(pg, hi, 32);
    svuint64_t t0 = svsub_u64_x(pg, lo, hi_hi);
    svbool_t borrow = svcmplt_u64(pg, lo, hi_hi);
    t0 = svsub_u64_m(borrow, t0, eps);
    svuint64_t t1 = svmullb_u64(svreinterpret_u32_u64(hi), eps32);
    svuint64_t res = svadd_u64_x(pg, t0, t1);
    svbool_t carry = svcmplt_u64(pg, res, t1);
    res = svadd_u64_m(carry, res, eps);
    return res;
}

static inline svuint64_t v_add2(svbool_t pg, svuint64_t x, svuint64_t y,
                                svuint64_t eps) __arm_streaming {
    svuint64_t sum = svadd_u64_x(pg, x, y);
    svbool_t over = svcmplt_u64(pg, sum, x);
    svuint64_t sum2 = svadd_u64_m(over, sum, eps);
    svbool_t over2 = svcmplt_u64(pg, sum2, sum);
    return svadd_u64_m(over2, sum2, eps);
}

static inline svuint64_t v_sub2(svbool_t pg, svuint64_t x, svuint64_t y,
                                svuint64_t eps) __arm_streaming {
    svuint64_t diff = svsub_u64_x(pg, x, y);
    svbool_t under = svcmplt_u64(pg, x, y);
    svuint64_t adj = svsel_u64(under, eps, svdup_u64(0));
    svuint64_t diff2 = svsub_u64_x(pg, diff, adj);
    svbool_t under2 = svcmplt_u64(pg, diff, adj);
    return svsub_u64_m(under2, diff2, eps);
}

// Fused two-layer pass, mirroring fft_classic_simd_two_layers_neon.
// Blocks of 4q = A|B|C|D quarters; w1_row[q]; w2_row[2q] = w2lo|w2hi.
// Requires q >= 16 (lg_half_m >= 4) and len a multiple of 4q.
__arm_locally_streaming void gl_fft_fused2_ssve(uint64_t *values, size_t len,
                                                size_t lg_half_m,
                                                const uint64_t *w1_row,
                                                const uint64_t *w2_row) {
    svbool_t pg = svptrue_b64();
    svuint64_t eps = svdup_u64(GL_EPS);
    svuint32_t eps32 = svdup_u32(0xFFFFFFFFu);
    size_t q = (size_t)1 << lg_half_m;
    size_t vl = svcntd();
    const uint64_t *w2lo = w2_row, *w2hi = w2_row + q;
    for (size_t base = 0; base + 4 * q <= len; base += 4 * q) {
        uint64_t *A = values + base;
        uint64_t *B = A + q;
        uint64_t *C = B + q;
        uint64_t *D = C + q;
        for (size_t j = 0; j + 2 * vl <= q; j += 2 * vl) {
            svuint64_t w1a = svld1_u64(pg, w1_row + j);
            svuint64_t w1b = svld1_u64(pg, w1_row + j + vl);
            svuint64_t Ba = svld1_u64(pg, B + j), Bb = svld1_u64(pg, B + j + vl);
            svuint64_t Da = svld1_u64(pg, D + j), Db = svld1_u64(pg, D + j + vl);
            svuint64_t Ca = svld1_u64(pg, C + j), Cb = svld1_u64(pg, C + j + vl);
            svuint64_t Aa = svld1_u64(pg, A + j), Ab = svld1_u64(pg, A + j + vl);
            svuint64_t t1a = v_mul(pg, w1a, Ba, eps, eps32);
            svuint64_t t1b = v_mul(pg, w1b, Bb, eps, eps32);
            svuint64_t t2a = v_mul(pg, w1a, Da, eps, eps32);
            svuint64_t t2b = v_mul(pg, w1b, Db, eps, eps32);
            svuint64_t cd0a = v_add2(pg, Ca, t2a, eps), cd0b = v_add2(pg, Cb, t2b, eps);
            svuint64_t cd1a = v_sub2(pg, Ca, t2a, eps), cd1b = v_sub2(pg, Cb, t2b, eps);
            svuint64_t w2la = svld1_u64(pg, w2lo + j), w2lb = svld1_u64(pg, w2lo + j + vl);
            svuint64_t w2ha = svld1_u64(pg, w2hi + j), w2hb = svld1_u64(pg, w2hi + j + vl);
            svuint64_t t3a = v_mul(pg, w2la, cd0a, eps, eps32);
            svuint64_t t3b = v_mul(pg, w2lb, cd0b, eps, eps32);
            svuint64_t t4a = v_mul(pg, w2ha, cd1a, eps, eps32);
            svuint64_t t4b = v_mul(pg, w2hb, cd1b, eps, eps32);
            svuint64_t ab0a = v_add2(pg, Aa, t1a, eps), ab0b = v_add2(pg, Ab, t1b, eps);
            svuint64_t ab1a = v_sub2(pg, Aa, t1a, eps), ab1b = v_sub2(pg, Ab, t1b, eps);
            svst1_u64(pg, A + j, v_add2(pg, ab0a, t3a, eps));
            svst1_u64(pg, A + j + vl, v_add2(pg, ab0b, t3b, eps));
            svst1_u64(pg, C + j, v_sub2(pg, ab0a, t3a, eps));
            svst1_u64(pg, C + j + vl, v_sub2(pg, ab0b, t3b, eps));
            svst1_u64(pg, B + j, v_add2(pg, ab1a, t4a, eps));
            svst1_u64(pg, B + j + vl, v_add2(pg, ab1b, t4b, eps));
            svst1_u64(pg, D + j, v_sub2(pg, ab1a, t4a, eps));
            svst1_u64(pg, D + j + vl, v_sub2(pg, ab1b, t4b, eps));
        }
    }
}

// Single radix-2 layer, mirroring fft_classic_simd_single_layer_neon.
// Sub-blocks of m = 2*half; pair (k+j, k+half+j) with omega_row[j].
// Requires half >= 16 (lg_half_m >= 4).
__arm_locally_streaming void gl_fft_single_ssve(uint64_t *values, size_t len,
                                                size_t lg_half_m,
                                                const uint64_t *omega_row) {
    svbool_t pg = svptrue_b64();
    svuint64_t eps = svdup_u64(GL_EPS);
    svuint32_t eps32 = svdup_u32(0xFFFFFFFFu);
    size_t half = (size_t)1 << lg_half_m;
    size_t m = half << 1;
    size_t vl = svcntd();
    for (size_t k = 0; k + m <= len; k += m) {
        uint64_t *U = values + k;
        uint64_t *V = U + half;
        for (size_t j = 0; j + 2 * vl <= half; j += 2 * vl) {
            svuint64_t wa = svld1_u64(pg, omega_row + j);
            svuint64_t wb = svld1_u64(pg, omega_row + j + vl);
            svuint64_t va = svld1_u64(pg, V + j), vb = svld1_u64(pg, V + j + vl);
            svuint64_t ua = svld1_u64(pg, U + j), ub = svld1_u64(pg, U + j + vl);
            svuint64_t ta = v_mul(pg, wa, va, eps, eps32);
            svuint64_t tb = v_mul(pg, wb, vb, eps, eps32);
            svst1_u64(pg, U + j, v_add2(pg, ua, ta, eps));
            svst1_u64(pg, U + j + vl, v_add2(pg, ub, tb, eps));
            svst1_u64(pg, V + j, v_sub2(pg, ua, ta, eps));
            svst1_u64(pg, V + j + vl, v_sub2(pg, ub, tb, eps));
        }
    }
}
