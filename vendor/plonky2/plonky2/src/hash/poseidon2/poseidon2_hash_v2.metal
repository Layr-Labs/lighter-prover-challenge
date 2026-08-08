#include <metal_stdlib>
using namespace metal;

// Sponge/Merkle-path Poseidon2 kernels with every round-constant addition
// folded into the linear layer that produces the state it applies to, so the
// 118 standalone full-reduction gl_add chains per permutation disappear, and
// with the linear layers' materializing reductions specialized to their
// proven-small addends (a <2^38 carry-count multiple of EPSILON needs one
// conditional EPSILON fold, not the general 17-op chain).
//
// This is a separate source file, compiled at Metal context construction on
// the prewarm background thread, because the committed poseidon2.metallib
// pins poseidon2.metal byte-for-byte (`metallib_matches_shader_source`) and
// this machine has no offline Metal toolchain to regenerate the AIR: editing
// poseidon2.metal would leave a stale artifact that the runtime silently
// prefers. poseidon2.metal stays byte-identical and keeps serving every
// non-hash kernel through the prebuilt metallib; the four kernels here are
// fetched from this source-compiled library instead.
//
// Arithmetic contract, identical to poseidon2.metal's: every ulong holds an
// arbitrary u64 representative of its field value (2^64 = EPSILON mod p),
// every operation is exact mod p over such representatives for ARBITRARY u64
// inputs, and gl_canonicalize at the output boundary maps any u64 to the
// canonical value (one conditional subtract suffices because 2p > 2^64).
// The differential tests assert bit-equality of kernel outputs against both
// the poseidon2.metal kernels and the strict-arithmetic reference build.

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

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
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    uint l0 = (uint)low;
    uint l1 = (uint)(low >> 32);
    uint h0 = (uint)high;
    uint h1 = (uint)(high >> 32);

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

    int top = (int)carry - (int)under;
    add_epsilon_u32(r0, r1, (uint)(top > 0));
    sub_epsilon_u32(r0, r1, (uint)(top < 0));
    return ((ulong)r1 << 32) | (ulong)r0;
#endif
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

// A lazy value is (v, c) representing v + c * 2^64; since 2^64 = EPSILON
// (mod p), its field value is v + c * EPSILON. Raw u64 adds accumulate the
// wrap count instead of folding EPSILON back per addition. Carry counts in
// the layers below stay <= 28 (see external_linear_layer_rc), so the
// materializing addend c * EPSILON is < 2^37 and gl_add_small's single
// conditional fold is exact. Every materialized output is an ordinary u64
// representative, so downstream arithmetic and gl_canonicalize see the same
// canonical field values as the strict per-op path.
struct lazy_t {
    ulong v;
    uint c;
};

inline lazy_t lazy_of(ulong v) {
    return { v, 0u };
}

inline lazy_t lazy_add(lazy_t a, lazy_t b) {
    ulong next = a.v + b.v;
    return { next, a.c + b.c + (uint)(next < a.v) };
}

// Folds a small addend into an arbitrary u64 representative: exact mod p
// whenever small < 2^64 - 2^32. The raw add wraps at most once; a wrap means
// the sum lost 2^64 = EPSILON (mod p), and the wrapped sum is < small, so
// adding EPSILON back cannot wrap again (small + 2^32 < 2^64). Replaces the
// general gl_add reduction chain (~17 uint ops) with an add, a compare and a
// conditional add for the materialize sites whose addend is a carry-count
// multiple of EPSILON (< 2^38 everywhere below).
inline ulong gl_add_small(ulong v, ulong small) {
    ulong sum = v + small;
    return sum + (sum < v ? GOLDILOCKS_EPSILON : 0UL);
}

// r = a * b + addend (mod p): the 128-bit product absorbs the addend before
// one shared reduction, deleting the separate post-multiply gl_add. mulhi is
// at most 2^64 - 2, so the absorbed carry cannot overflow the high word. The
// reduction below is byte-identical to gl_mul's.
inline ulong gl_mul_add(ulong a, ulong b, ulong addend) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong low2 = low + addend;
    high += (ulong)(low2 < low);
    uint l0 = (uint)low2;
    uint l1 = (uint)(low2 >> 32);
    uint h0 = (uint)high;
    uint h1 = (uint)(high >> 32);

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

    int top = (int)carry - (int)under;
    add_epsilon_u32(r0, r1, (uint)(top > 0));
    sub_epsilon_u32(r0, r1, (uint)(top < 0));
    return ((ulong)r1 << 32) | (ulong)r0;
}

inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value4 = gl_mul(value2, value2);
    ulong value3 = gl_mul(value, value2);
    return gl_mul(value3, value4);
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

// r = a * b + addend0 + addend1 (mod p), REQUIRING b < p. The 128-bit
// product absorbs both addends before one shared reduction. With
// b <= p - 1 = 2^64 - 2^32, mulhi(a, b) <= 2^64 - 2^32 - 1, so absorbing two
// carries into the high word cannot overflow it. The reduction below is
// byte-identical to gl_mul's. Used only with the internal-diagonal constants
// as b, which are canonical.
inline ulong gl_mul_add2(ulong a, ulong b, ulong addend0, ulong addend1) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong low2 = low + addend0;
    high += (ulong)(low2 < low);
    ulong low3 = low2 + addend1;
    high += (ulong)(low3 < low2);
    uint l0 = (uint)low3;
    uint l1 = (uint)(low3 >> 32);
    uint h0 = (uint)high;
    uint h1 = (uint)(high >> 32);

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

    int top = (int)carry - (int)under;
    add_epsilon_u32(r0, r1, (uint)(top > 0));
    sub_epsilon_u32(r0, r1, (uint)(top < 0));
    return ((ulong)r1 << 32) | (ulong)r0;
}

// External linear layer with the NEXT round's constants folded into the lazy
// accumulation before each lane's materialize: rc_lanes is 12 when the layer
// feeds a full external round, 1 when it feeds the first internal round
// (only lane 0 takes a constant there), 0 for the output layer. Every call
// site passes a literal rc_lanes, so the per-lane bound check folds away
// after inlining.
//
// Carry bound for the materialize: mat4 outputs carry c <= 6 (each output is
// a tree of <= 6 raw adds over c = 0 inputs), sums[] c <= 6+6+1+6+1 = 20,
// the lane total c <= 6+20+1 = 27, plus 1 for the folded constant = 28, so
// the addend is < 28 * 2^32 < 2^37 and gl_add_small is exact.
inline void external_linear_layer_rc(
    thread ulong state[12],
    constant ulong* rc,
    uint rc_lanes) {
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
        lazy_t t = lazy_add(lazy[i], sums[i & 3]);
        if (i < rc_lanes) {
            ulong next = t.v + rc[i];
            t.c += (uint)(next < t.v);
            t.v = next;
        }
        state[i] = gl_add_small(t.v, (ulong)t.c * GOLDILOCKS_EPSILON);
    }
}

inline ulong sum_state(thread const ulong state[12]) {
    ulong sum = 0;
    uint carries = 0;
    for (uint i = 0; i < 12; ++i) {
        ulong next = sum + state[i];
        carries += next < sum;
        sum = next;
    }
    // carries <= 11, so the addend is < 11 * 2^32 < 2^36: gl_add_small is
    // exact.
    return gl_add_small(sum, (ulong)carries * GOLDILOCKS_EPSILON);
}

// Internal linear layer with the NEXT round's constant(s) absorbed into the
// per-lane fused multiply-add as a second addend: rc_lanes = 1 folds the
// next internal round's lane-0 constant, rc_lanes = 12 folds external round
// 4's full constant vector after the last internal round. Legal because the
// diagonal entries are canonical (< p), which is what caps gl_mul_add2's
// high word (see its comment).
inline void internal_linear_layer_rc(
    thread ulong state[12],
    constant ulong* diagonal,
    constant ulong* rc,
    uint rc_lanes) {
    ulong sum = sum_state(state);
    for (uint i = 0; i < 12; ++i) {
        state[i] = i < rc_lanes
            ? gl_mul_add2(state[i], diagonal[i], sum, rc[i])
            : gl_mul_add(state[i], diagonal[i], sum);
    }
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
//
// Identical constants in the identical order as poseidon2.metal's
// poseidon2(), but each constant is added by the linear layer that PRODUCES
// the state it applies to (folded into the lazy chain or the fused
// multiply-add) instead of by a standalone gl_add in the consuming round:
//
//   initial layer            folds external constants of round 0
//   round r < 3 layer        folds external constants of round r+1
//   round 3 layer            folds internal constant 0 into lane 0
//   internal r < 21 layer    folds internal constant r+1 into lane 0
//   internal round 21 layer  folds external constants of round 4
//   round 4 <= r < 7 layer   folds external constants of round r+1
//   round 7 (output) layer   no constants remain
inline void poseidon2(thread ulong state[12], constant ulong* parameters) {
    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;

    external_linear_layer_rc(state, external_constants, 12u);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(state[i]);
        }
        if (round < 3) {
            external_linear_layer_rc(
                state, external_constants + (round + 1) * 12, 12u);
        } else {
            external_linear_layer_rc(state, internal_constants, 1u);
        }
    }

    for (uint round = 0; round < 22; ++round) {
        state[0] = pow7(state[0]);
        if (round < 21) {
            internal_linear_layer_rc(
                state, diagonal, internal_constants + round + 1, 1u);
        } else {
            internal_linear_layer_rc(
                state, diagonal, external_constants + 4 * 12, 12u);
        }
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(state[i]);
        }
        if (round < 7) {
            external_linear_layer_rc(
                state, external_constants + (round + 1) * 12, 12u);
        } else {
            external_linear_layer_rc(state, external_constants, 0u);
        }
    }
}

kernel void poseidon2_hash_leaves_rcfold(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    const device ulong* input = leaves + (ulong)gid * leaf_width;
    device ulong* output = hashes + (ulong)gid * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(input[i]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            state[i] = gl_canonicalize(input[offset + i]);
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

kernel void poseidon2_hash_leaves_colmajor_rcfold(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    // Input is column-major: column j occupies leaves[j * leaf_count..(j + 1) *
    // leaf_count] in natural row order, so adjacent threads read adjacent
    // addresses. The digest of natural row gid belongs to tree leaf
    // reverse_bits(gid), so outputs scatter by bit reversal.
    uint out_row = log_leaf_count == 0
        ? gid
        : (reverse_bits(gid) >> (32 - log_leaf_count));
    device ulong* output = hashes + (ulong)out_row * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(leaves[(ulong)i * leaf_count + gid]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            state[i] = gl_canonicalize(leaves[(ulong)(offset + i) * leaf_count + gid]);
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

kernel void poseidon2_hash_parents_rcfold(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& parent_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= parent_count) {
        return;
    }

    const device ulong* input = children + (ulong)gid * 8;
    ulong state[12] = { 0 };
    for (uint i = 0; i < 8; ++i) {
        state[i] = input[i];
    }
    poseidon2(state, parameters);

    device ulong* output = parents + (ulong)gid * 4;
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

// One sponge absorption pass over a group of at most eight natural-order
// columns, with the running 12-lane state parked in `state` between passes
// (column-major: lane i of row gid at state[i * leaf_count + gid]). The
// final pass writes the four-lane digests to `hashes` at the bit-reversed
// row, exactly like poseidon2_hash_leaves_colmajor. Splitting the sponge by
// column group lets the CPU compute group g+1's LDE columns while the GPU
// absorbs group g; the arithmetic per pass is identical to the fused
// kernel's corresponding loop iteration.
kernel void poseidon2_absorb_pass_rcfold(
    const device ulong* leaves [[buffer(0)]],
    device ulong* state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
    constant uint& chunk_size [[buffer(7)]],
    constant uint& first_pass [[buffer(8)]],
    constant uint& final_pass [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }
    ulong st[12] = { 0 };
    if (first_pass == 0u) {
        for (uint i = 0; i < 12; ++i) {
            st[i] = state[(ulong)i * leaf_count + gid];
        }
    }
    for (uint i = 0; i < chunk_size; ++i) {
        st[i] = gl_canonicalize(leaves[(ulong)(col_start + i) * leaf_count + gid]);
    }
    poseidon2(st, parameters);
    if (final_pass != 0u) {
        uint out_row = log_leaf_count == 0
            ? gid
            : (reverse_bits(gid) >> (32 - log_leaf_count));
        device ulong* output = hashes + (ulong)out_row * 4;
        for (uint i = 0; i < 4; ++i) {
            output[i] = gl_canonicalize(st[i]);
        }
    } else {
        for (uint i = 0; i < 12; ++i) {
            state[(ulong)i * leaf_count + gid] = st[i];
        }
    }
}
