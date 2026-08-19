#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

// Compile-time Poseidon2 round constants (same values as config.rs).
// File-scope constant arrays keep the round loops compact so register
// pressure stays near the tip kernels, while still removing device-buffer
// loads of the parameters table for every RC add.
constant ulong POSEIDON2_EXTERNAL_RC[8][12] = {
    { 0xd70193d17ab3b7d6UL, 0xa2c3662a78a9162bUL, 0x7a9fda827556ad44UL, 0xe8d5501818c99643UL, 0x4c7a8fced4d5fd38UL, 0x55ab38985c0c513dUL, 0x28a17bd016210b0bUL, 0x8f8277679ec32fa8UL, 0x768b3c3d68a460e9UL, 0x872a022eb559d941UL, 0xd1316dd4b3b97973UL, 0xa7b608e578321000UL },
    { 0x3fa02c87b0bee026UL, 0x7a38f0022e13c31eUL, 0x00c054f3c5e8d20dUL, 0x439f50f4bca7242fUL, 0x4d0938aa57cd517fUL, 0xb2e03ac5fb6b9a7dUL, 0xe29d1f4237bedca8UL, 0x05b7c844bc99b848UL, 0x91cc0b73f34e17edUL, 0x876e4427694bd755UL, 0x67002ae0725c612dUL, 0x05351f20e0b6315fUL },
    { 0x2e3b9ef5457eb60bUL, 0xd9ac17618c3783ddUL, 0x0807528ad8874bcfUL, 0xc78d546a455d2a0eUL, 0xf8b930c81e2481f0UL, 0x712707d8dff3b041UL, 0xdcb8c0aa0b9d34c3UL, 0x9baddbdf2ee3a468UL, 0x2dd16d50c5176c78UL, 0x89eac5cfbc075cd3UL, 0x2a741dea181587f3UL, 0x1a4d6aa85a113d84UL },
    { 0x4d736286a2387e34UL, 0x8bad5dfc4fcb3ee3UL, 0x84fbd03adb77c56aUL, 0x8d5cdd1a23ec53a2UL, 0x036f08f08fff28ecUL, 0xb717a3f4dbdfb443UL, 0x58a074b5509d645cUL, 0xf92bf834e4b87718UL, 0x1541c3a0baa5ac4bUL, 0x22149e6783e67692UL, 0x9be8b5d9e112476fUL, 0x41e0969f62babb76UL },
    { 0xbc585ad3b9443dbbUL, 0xf28dd3206975cbb1UL, 0xdd8815e53ca045e0UL, 0xde82c416b9e701baUL, 0xc5cb875233afa025UL, 0x7212697cd897ffa9UL, 0x67844790aa63cfd7UL, 0xdc0b9cfa97fe65c3UL, 0xe8fe091869a82070UL, 0x62902bb2e413c6d1UL, 0x29f9f5001fb84f57UL, 0xbe1014796ef5f8beUL },
    { 0x71feb53e9bdba19cUL, 0x251054f592ebb71cUL, 0xe1a57643a4bb284bUL, 0xa4ba6f87a45b739bUL, 0x2c1fcade0b958c49UL, 0xbbb424cda9a3e360UL, 0x2ca647354c5f3f54UL, 0xc9277b64d152e084UL, 0xdbc9ac97445eff17UL, 0x6f6cdf3198969f70UL, 0x1de29d14fa76d8f1UL, 0x73337458a8cc1d19UL },
    { 0xb87e775e2fb3ab23UL, 0xf166a1c7a565c80bUL, 0xb24be06f426c747fUL, 0xc281e8c49482ce00UL, 0x51974c3b3b726c2dUL, 0x87444cf8caf7d619UL, 0x7c362f827a580cedUL, 0x9567af14667647a0UL, 0xcbf0473cbec54e37UL, 0xe3209dedeff4f620UL, 0xd43ad94e45a4c4eeUL, 0x976981ee73f41768UL },
    { 0xef707a224e207258UL, 0x2fc779e10e6362eeUL, 0x29b5ee60ad8c891fUL, 0x96b37b39d8bfd667UL, 0x877df68a8b22e733UL, 0x5c41746f562c8d9fUL, 0x0c9d76751052b71aUL, 0xfb3465341bf1c087UL, 0xa0d14dc614d15eb1UL, 0xdc27d17136906fa6UL, 0x482e163b05ec397fUL, 0x0273a462992366efUL },
};
constant ulong POSEIDON2_INTERNAL_RC[22] = {
    0xa571418d95897b60UL, 0x8f32676574fcf6d3UL, 0x731102d4e3fb1bbeUL, 0x0330f08328a82d2bUL, 0x7f0449b6557f785dUL, 0x62f06210658dcbcbUL, 0xd5a98af9f89c458bUL, 0x77ec69083a346385UL, 0xef7ca48bbc27f890UL, 0x53e9652f61eac532UL, 0xa71c634abff4f0ccUL, 0xb16f5f0d7e28ea29UL, 0xc9dde31d0a003ab2UL, 0x2ddadf9775902533UL, 0xe4fa73fb16408b47UL, 0x90242ebc00d2ee59UL, 0xbb02dffd9f381982UL, 0xdea328364c50907cUL, 0x1395d3b924857cf8UL, 0x7d3ead0d5aec04e6UL, 0xc2f12be3fed74668UL, 0x0ba3c338f8c3d285UL,
};


inline void add_epsilon_u32(thread uint& lo, thread uint& hi, uint active) {
    uint old0 = lo;
    lo -= active;
    uint old1 = hi;
    hi += active & (uint)(old0 != 0);
    uint overflow = active & (uint)(hi < old1);
    old0 = lo;
    lo -= overflow;
    hi += overflow & (uint)(old0 != 0);
}

inline void sub_epsilon_u32(thread uint& lo, thread uint& hi, uint active) {
    uint old0 = lo;
    lo += active;
    uint old1 = hi;
    hi -= active & (uint)(old0 != 0xffffffffU);
    uint underflow = active & (uint)(hi > old1);
    old0 = lo;
    lo += underflow;
    hi -= underflow & (uint)(old0 != 0xffffffffU);
}

// Multiplying by four is two doublings; a doubling is one field add.
inline ulong gl_add(ulong a, ulong b);
inline ulong gl_quadruple(ulong a) {
    ulong twice = gl_add(a, a);
    return gl_add(twice, twice);
}

inline ulong gl_add(ulong a, ulong b) {
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong sum = a + b;
    ulong carry = sum < a;
    sum += carry * GOLDILOCKS_EPSILON;
    ulong carry2 = (carry != 0UL) && (sum < GOLDILOCKS_EPSILON);
    return sum + carry2 * GOLDILOCKS_EPSILON;
#else
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    uint r0 = a0 + b0;
    uint carry0 = (uint)(r0 < a0);
    uint r1 = a1 + b1;
    uint carry1 = (uint)(r1 < a1);
    uint next = r1 + carry0;
    carry1 += (uint)(next < r1);
    r1 = next;
    add_epsilon_u32(r0, r1, carry1);
    return ((ulong)r1 << 32) | (ulong)r0;
#endif
}

// Adds a canonical field element `b < p` to an arbitrary u64 representative.
// If the native sum carries, its wrapped value is at most p - 2, because
// `a <= 2^64 - 1` and `b <= p - 1`. Adding EPSILON therefore cannot carry a
// second time. Round constants satisfy this bound, so their additions can
// omit the general path's unreachable second correction fold.
inline ulong gl_add_canonical_rhs(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint r0 = av.x + bv.x;
    uint carry0 = (uint)(r0 < av.x);
    uint r1 = av.y + bv.y;
    uint carry1 = (uint)(r1 < av.y);
    uint next = r1 + carry0;
    carry1 += (uint)(next < r1);
    r1 = next;

    uint old0 = r0;
    r0 -= carry1;
    r1 += carry1 & (uint)(old0 != 0);
    return ((ulong)r1 << 32) | (ulong)r0;
}

inline ulong gl_sub(ulong a, ulong b) {
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong diff = a - b;
    ulong under = diff > a;
    diff -= under * GOLDILOCKS_EPSILON;
    ulong under2 = (under != 0UL) && (diff > (~0UL - GOLDILOCKS_EPSILON));
    return diff - under2 * GOLDILOCKS_EPSILON;
#else
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    uint r0 = a0 - b0;
    uint borrow0 = (uint)(r0 > a0);
    uint r1 = a1 - b1;
    uint under = (uint)(r1 > a1);
    uint next = r1 - borrow0;
    under += (uint)(next > r1);
    r1 = next;
    sub_epsilon_u32(r0, r1, under);
    return ((ulong)r1 << 32) | (ulong)r0;
#endif
}

// Subtracts a canonical field element `b < p` from an arbitrary u64
// representative. On native underflow the wrapped difference is at least
// `2^64 - (p - 1) = 2^32`, so subtracting EPSILON cannot underflow again.
// Literal bit/radix constants satisfy this bound and need only one fold.
inline ulong gl_sub_canonical_rhs(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint r0 = av.x - bv.x;
    uint borrow0 = (uint)(r0 > av.x);
    uint r1 = av.y - bv.y;
    uint under = (uint)(r1 > av.y);
    uint next = r1 - borrow0;
    under += (uint)(next > r1);
    r1 = next;

    uint old0 = r0;
    r0 += under;
    r1 -= under & (uint)(old0 != 0xffffffffU);
    return ((ulong)r1 << 32) | (ulong)r0;
}

// Final step of the 128-bit Goldilocks reduction shared by gl_mul and
// gl_mul_add. On entry (r0, r1) are the low and high 32-bit limbs of the
// residue and `top` is its 2^64 weight, one of -1, 0, +1, so the value is
// r + top * 2^64, and 2^64 == EPSILON (mod p).
//
// Neither fold can leave the 64-bit range, so unlike gl_add this needs no
// second correction round. Writing the reduced product as
//     V = low + h0 * EPSILON - h1
// with low < 2^64 and h0, h1 < 2^32 bounds it by
//     -(2^32 - 1) <= V <= (2^64 - 1) + (2^32 - 1)^2 = 2^65 - 2^33.
// When top is +1 the residue is r = V - 2^64 <= 2^64 - 2^33, so r + EPSILON
// stays below 2^64 and the add cannot overflow. When top is -1 the residue is
// r = V + 2^64 >= 2^64 - 2^32 + 1 > EPSILON, so the subtract cannot borrow.
// One unconditional 64-bit add of top * EPSILON therefore replaces both
// add_epsilon_u32 and sub_epsilon_u32, including their unreachable second
// correction rounds. gl_add and gl_sub keep theirs: they know nothing about
// their operands and their second rounds are genuinely reachable.
inline ulong reduce_top(uint r0, uint r1, int top) {
    ulong r = ((ulong)r1 << 32) | (ulong)r0;
    return r + (ulong)(((long)top << 32) - (long)top);
}

// Full 128-bit product of two 64-bit operands, delivered as four 32-bit limbs.
//
// `a * b` and `metal::mulhi(a, b)` are lowered as two independent 64-bit
// expansions that each rebuild the 32x32 partial products they need; the
// backend does not share them. Computing the four products once and assembling
// only the limbs the reduction actually consumes removes that duplication and
// the 64-bit pack/unpack around it.
//
// With B = 2^32 and a = a1*B + a0, b = b1*B + b0:
//   t = p01 + (p00 >> 32) <= (B-1)^2 + (B-1) = B^2 - B, so t cannot wrap.
//   m = t + p10 <= 2B^2 - 3B + 1 wraps at most once; `carry` is that 65th bit.
//   low  = (m << 32) | (uint)p00 -> l0 = (uint)p00, l1 = (uint)m.
//   high = p11 + (m >> 32) + carry * B, which is < B^2, so h1 cannot wrap.
//
// The operands are split with a `uint2` bitcast rather than shift-and-truncate.
// Both name the same two halves on a little-endian target, but `(uint)(a >> 32)`
// asks the backend for a 64-bit shift it then has to narrow, while the bitcast
// is a pure reinterpretation of the register pair the value already occupies.
// This is the multiply every hash lane runs four times per S-box, and dropping
// those two shifts is worth ~9% of the leaf-hash kernel.
inline void mul_128(
    ulong a,
    ulong b,
    thread uint& l0,
    thread uint& l1,
    thread uint& h0,
    thread uint& h1) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint a0 = av.x;
    uint a1 = av.y;
    uint b0 = bv.x;
    uint b1 = bv.y;
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong t = p01 + (p00 >> 32);
    ulong m = t + p10;
    uint carry = (uint)(m < t);

    l0 = (uint)p00;
    l1 = (uint)m;

    uint q0 = (uint)p11;
    uint q1 = (uint)(p11 >> 32);
    uint mh = (uint)(m >> 32);
    h0 = q0 + mh;
    h1 = q1 + (uint)(h0 < q0) + carry;
}

// Goldilocks multiplication with 32-bit reduction after the native product.
// If low = l0 + l1*B and high = h0 + h1*B for B = 2^32, then
//   low + high*B^2 = (l0 - h0 - h1) + (l1 + h0)*B  (mod p).
// Normalizing those two signed base-B coefficients only needs uint
// add/subtract/carry operations; the only ulong multiplies left are the
// product's low and high halves.
inline ulong gl_mul(ulong a, ulong b) {
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong high_high = high >> 32;
    ulong high_low = high & GOLDILOCKS_EPSILON;
    ulong reduced = low - high_high;
    if (reduced > low) {
        reduced -= GOLDILOCKS_EPSILON;
    }
    ulong addend = high_low * GOLDILOCKS_EPSILON;
    ulong result = reduced + addend;
    return result + (result < reduced) * GOLDILOCKS_EPSILON;
#else
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);

    uint r0 = l0 - h0;
    uint borrow = (uint)(r0 > l0);
    uint next = r0 - h1;
    borrow += (uint)(next > r0);
    r0 = next;

    uint r1 = l1 + h0;
    uint carry = (uint)(r1 < l1);
    next = r1 - borrow;
    uint under = (uint)(next > r1);
    r1 = next;

    return reduce_top(r0, r1, (int)carry - (int)under);
#endif
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

// A lazy value is (lo, hi) representing hi * 2^32 + lo, with both halves held
// in full 64-bit registers. Splitting the operand at the 32-bit boundary makes
// addition carry-free: every accumulator below sums field elements with
// coefficients totalling at most 28, so each half stays under
// 28 * (2^32 - 1) < 2^37 and neither can overflow. lazy_add is then two
// independent 64-bit adds; the single-word (v, c) form needed a third
// operation per add to recover the carry a 64-bit add cannot report to MSL.
// One materialize per element per layer still replaces the per-add reduction
// chains, and every materialized output is an ordinary u64 representative, so
// downstream arithmetic and gl_canonicalize see the same canonical field
// values as the strict per-op path.
struct lazy_t {
    ulong lo;
    ulong hi;
};

inline lazy_t lazy_of(ulong v) {
    return { v & GOLDILOCKS_EPSILON, v >> 32 };
}

inline lazy_t lazy_add(lazy_t a, lazy_t b) {
    return { a.lo + b.lo, a.hi + b.hi };
}

// Collapses (lo, hi) to an ordinary 64-bit representative. Splitting both
// halves at the 32-bit boundary as
//   lo = lh * 2^32 + ll,  hi = hh * 2^32 + hl   (lh, hh < 2^5)
// rewrites the value as
//   hi * 2^32 + lo == hh * 2^64 + (hl + lh) * 2^32 + ll,
// so with u = (hl + lh) mod 2^32 and e = hh + carry(hl + lh) <= 32 and
// 2^64 == EPSILON == 2^32 - 1 (mod p) the residue is exactly
//   (u + e) * 2^32 + (ll - e).
// Both coefficients are single 32-bit adds whose carry out and borrow out are
// the 2^64 weight of the result, which is the shape reduce_top already folds:
// when that weight is +1 the high limb is at most 32, because u + e >= 2^64
// forces u >= 2^32 - 32, so the EPSILON add cannot wrap; when it is -1 the high
// limb is 0xffffffff, so the subtract cannot borrow. Everything above the four
// limb extractions is 32-bit work, and the wide compares plus the trailing
// correction round of the 64-bit form are gone.
inline ulong lazy_materialize(lazy_t a) {
    uint ll = (uint)a.lo;
    uint lh = (uint)(a.lo >> 32);
    uint hl = (uint)a.hi;
    uint hh = (uint)(a.hi >> 32);
    uint u = hl + lh;
    uint e = hh + (uint)(u < hl);
    uint r0 = ll - e;
    uint borrow = (uint)(r0 > ll);
    uint r1 = u + e;
    uint carry = (uint)(r1 < u);
    uint next = r1 - borrow;
    uint under = (uint)(next > r1);
    r1 = next;
    return reduce_top(r0, r1, (int)carry - (int)under);
}

// r = a * b + addend (mod p): the 128-bit product absorbs the addend before
// one shared reduction, deleting the separate post-multiply gl_add. mulhi is
// at most 2^64 - 2, so the absorbed carry cannot overflow the high word. The
// reduction below is byte-identical to gl_mul's.
inline ulong gl_mul_add(ulong a, ulong b, ulong addend) {
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);

    // Absorb the addend into the low limbs and propagate its single carry into
    // h0. h1 cannot wrap: the 128-bit product of two 64-bit operands is at most
    // 2^128 - 2^65 + 1, so h1 == 0xffffffff forces h0 <= 0xfffffffe.
    uint d0 = (uint)addend;
    uint d1 = (uint)(addend >> 32);
    uint s0 = l0 + d0;
    uint c0 = (uint)(s0 < l0);
    uint s1 = l1 + d1;
    uint c1 = (uint)(s1 < l1);
    uint s1b = s1 + c0;
    c1 += (uint)(s1b < s1);
    l0 = s0;
    l1 = s1b;
    uint hh0 = h0 + c1;
    h1 += (uint)(hh0 < h0);
    h0 = hh0;

    uint r0 = l0 - h0;
    uint borrow = (uint)(r0 > l0);
    uint next = r0 - h1;
    borrow += (uint)(next > r0);
    r0 = next;

    uint r1 = l1 + h0;
    uint carry = (uint)(r1 < l1);
    next = r1 - borrow;
    uint under = (uint)(next > r1);
    r1 = next;

    return reduce_top(r0, r1, (int)carry - (int)under);
}

// x^7 by the addition chain 1 -> 2 -> 3 -> 6 -> 7. Every length-four chain for
// x^7 costs the same four multiplies, but this one keeps only `value` and one
// running power live at a time, and three of the four multiplies take `value`
// or a square as an operand, so the backend reuses one operand's limb split
// instead of re-splitting a fresh pair. The 2/4/3/7 chain is one step shallower
// and measured consistently slower for both reasons.
inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value3 = gl_mul(value, value2);
    ulong value6 = gl_mul(value3, value3);
    return gl_mul(value, value6);
}

inline void mat4(thread lazy_t* values) {
    lazy_t x0 = values[0];
    lazy_t x1 = values[1];
    lazy_t x2 = values[2];
    lazy_t x3 = values[3];
    lazy_t t01 = lazy_add(x0, x1);
    lazy_t t23 = lazy_add(x2, x3);
    lazy_t total = lazy_add(t01, t23);

    values[0] = lazy_add(lazy_add(total, t01), x1);
    values[1] = lazy_add(lazy_add(lazy_add(total, x1), x2), x2);
    values[2] = lazy_add(lazy_add(total, t23), x3);
    values[3] = lazy_add(lazy_add(lazy_add(total, x3), x0), x0);
}

inline void external_linear_layer(thread ulong state[12]) {
    lazy_t lazy[12];
    for (uint i = 0; i < 12; ++i) {
        lazy[i] = lazy_of(state[i]);
    }
    mat4(lazy);
    mat4(lazy + 4);
    mat4(lazy + 8);

    lazy_t sums[4];
    for (uint i = 0; i < 4; ++i) {
        sums[i] = lazy_add(lazy_add(lazy[i], lazy[i + 4]), lazy[i + 8]);
    }
    for (uint i = 0; i < 12; ++i) {
        state[i] = lazy_materialize(lazy_add(lazy[i], sums[i & 3]));
    }
}

// Same carry-free split as lazy_t: twelve operands keep both halves below
// 12 * (2^32 - 1) < 2^36.
inline ulong sum_state(thread const ulong state[12]) {
    lazy_t sum = lazy_of(state[0]);
    for (uint i = 1; i < 12; ++i) {
        sum = lazy_add(sum, lazy_of(state[i]));
    }
    return lazy_materialize(sum);
}

inline void internal_linear_layer(thread ulong state[12]) {
    ulong sum = sum_state(state);
    // Poseidon2's internal diagonal is fixed by the hash configuration. Spell
    // it as immediates so the Metal compiler can specialize the constant
    // operand of all 264 internal-round multiplications per permutation.
    state[0] = gl_mul_add(state[0], 0xc3b6c08e23ba9300UL, sum);
    state[1] = gl_mul_add(state[1], 0xd84b5de94a324fb6UL, sum);
    state[2] = gl_mul_add(state[2], 0x0d0c371c5b35b84fUL, sum);
    state[3] = gl_mul_add(state[3], 0x7964f570e7188037UL, sum);
    state[4] = gl_mul_add(state[4], 0x5daf18bbd996604bUL, sum);
    state[5] = gl_mul_add(state[5], 0x6743bc47b9595257UL, sum);
    state[6] = gl_mul_add(state[6], 0x5528b9362c59bb70UL, sum);
    state[7] = gl_mul_add(state[7], 0xac45e25b7127b68bUL, sum);
    state[8] = gl_mul_add(state[8], 0xa2077d7dfbb606b5UL, sum);
    state[9] = gl_mul_add(state[9], 0xf3faac6faee378aeUL, sum);
    state[10] = gl_mul_add(state[10], 0x0c6388b51545e883UL, sum);
    state[11] = gl_mul_add(state[11], 0xd27dbb6944917b60UL, sum);
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
// Parameter buffer remains in the ABI for host binding; sponge arithmetic uses
// compile-time constant tables so RC adds do not depend on device buffer loads.
// Round loops stay compact (unlike full unroll) to preserve occupancy.
inline void poseidon2(thread ulong state[12], constant ulong* /*parameters*/) {
    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]));
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        state[0] = pow7(gl_add_canonical_rhs(state[0], POSEIDON2_INTERNAL_RC[round]));
        internal_linear_layer(state);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]));
        }
        external_linear_layer(state);
    }
}
