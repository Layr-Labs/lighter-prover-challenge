#include <metal_stdlib>
using namespace metal;

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
inline void mul_128(
    ulong a,
    ulong b,
    thread uint& l0,
    thread uint& l1,
    thread uint& h0,
    thread uint& h1) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
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

// Collapses (lo, hi) to an ordinary 64-bit representative. With
//   hi = hh * 2^32 + hl  and  2^64 == EPSILON (mod p),
//   hi * 2^32 + lo == hl * 2^32 + lo + hh * EPSILON.
// `a.hi << 32` already discards hh, so s is the first two terms and c1 its
// 65th bit. hh + c1 < 2^6, so the EPSILON fold is at most 2^38: if s wrapped,
// s < 2^37 and the fold cannot wrap again; if it did not, one more fold of
// EPSILON onto a value below 2^38 cannot either. Both correction rounds after
// the first are therefore unreachable.
inline ulong lazy_materialize(lazy_t a) {
    ulong s = (a.hi << 32) + a.lo;
    ulong extra = (a.hi >> 32) + (ulong)(s < a.lo);
    ulong t = s + ((extra << 32) - extra);
    return t + (t < s ? GOLDILOCKS_EPSILON : 0UL);
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

inline void internal_linear_layer(thread ulong state[12], constant ulong* diagonal) {
    ulong sum = sum_state(state);
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_mul_add(state[i], diagonal[i], sum);
    }
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
inline void poseidon2(thread ulong state[12], constant ulong* parameters) {
    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;

    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add(state[i], external_constants[round * 12 + i]));
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        state[0] = pow7(gl_add(state[0], internal_constants[round]));
        internal_linear_layer(state, diagonal);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add(state[i], external_constants[round * 12 + i]));
        }
        external_linear_layer(state);
    }
}

inline ulong poseidon2_gate_wire(
    const device ulong* wires,
    uint column,
    uint rows,
    uint row) {
    return wires[(ulong)column * rows + row];
}

inline void poseidon2_gate_emit(
    ulong constraint,
    constant ulong* alpha_powers,
    thread ulong accumulators[2],
    thread uint& constraint_index) {
    accumulators[0] =
        gl_mul_add(constraint, alpha_powers[constraint_index], accumulators[0]);
    accumulators[1] =
        gl_mul_add(constraint, alpha_powers[123 + constraint_index], accumulators[1]);
    ++constraint_index;
}

// Evaluates the filtered Poseidon2Gate contribution to both quotient
// challenges at one natural-order quotient-domain point. The alpha powers
// already include the prefix occupied by the non-gate vanishing terms, so
// these two values can be added to the CPU evaluator after it has reduced the
// permutation argument.
kernel void poseidon2_gate_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant ulong* alpha_powers [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& selector_column [[buffer(8)]],
    constant uint& gate_index [[buffer(9)]],
    constant uint& group_start [[buffer(10)]],
    constant uint& group_end [[buffer(11)]],
    constant uint& include_unused_selector [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
    ulong filter = 1;
    for (uint i = group_start; i < group_end; ++i) {
        if (i != gate_index) {
            filter = gl_mul(filter, gl_sub((ulong)i, selector));
        }
    }
    if (include_unused_selector != 0u) {
        filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
    }

    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;
    ulong accumulators[2] = { 0, 0 };
    uint constraint_index = 0;

    ulong swap = poseidon2_gate_wire(wires, 24, lde_rows, source_row);
    poseidon2_gate_emit(
        gl_mul(swap, gl_sub(swap, 1)),
        alpha_powers,
        accumulators,
        constraint_index);

    ulong state[12];
    for (uint i = 0; i < 4; ++i) {
        ulong lhs = poseidon2_gate_wire(wires, i, lde_rows, source_row);
        ulong rhs = poseidon2_gate_wire(wires, i + 4, lde_rows, source_row);
        ulong delta = poseidon2_gate_wire(wires, 25 + i, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(gl_mul(swap, gl_sub(rhs, lhs)), delta),
            alpha_powers,
            accumulators,
            constraint_index);
        state[i] = gl_add(lhs, delta);
        state[i + 4] = gl_sub(rhs, delta);
    }
    for (uint i = 8; i < 12; ++i) {
        state[i] = poseidon2_gate_wire(wires, i, lde_rows, source_row);
    }

    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = gl_add(state[i], external_constants[round * 12 + i]);
        }
        if (round != 0) {
            uint saved_start = 29 + (round - 1) * 12;
            for (uint i = 0; i < 12; ++i) {
                ulong saved = poseidon2_gate_wire(
                    wires,
                    saved_start + i,
                    lde_rows,
                    source_row);
                poseidon2_gate_emit(
                    gl_sub(state[i], saved),
                    alpha_powers,
                    accumulators,
                    constraint_index);
                state[i] = saved;
            }
        }
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(state[i]);
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        ulong saved = poseidon2_gate_wire(wires, 65 + round, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(gl_add(state[0], internal_constants[round]), saved),
            alpha_powers,
            accumulators,
            constraint_index);
        state[0] = pow7(saved);
        internal_linear_layer(state, diagonal);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = gl_add(state[i], external_constants[round * 12 + i]);
        }
        uint saved_start = 87 + (round - 4) * 12;
        for (uint i = 0; i < 12; ++i) {
            ulong saved = poseidon2_gate_wire(
                wires,
                saved_start + i,
                lde_rows,
                source_row);
            poseidon2_gate_emit(
                gl_sub(state[i], saved),
                alpha_powers,
                accumulators,
                constraint_index);
            state[i] = pow7(saved);
        }
        external_linear_layer(state);
    }

    for (uint i = 0; i < 12; ++i) {
        ulong expected = poseidon2_gate_wire(wires, 12 + i, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(state[i], expected),
            alpha_powers,
            accumulators,
            constraint_index);
    }

    output[(ulong)gid * 2] = gl_canonicalize(gl_mul(filter, accumulators[0]));
    output[(ulong)gid * 2 + 1] = gl_canonicalize(gl_mul(filter, accumulators[1]));
}

inline void range_check_gate_emit(
    ulong constraint,
    constant ulong* alpha_powers,
    uint alpha_stride,
    thread ulong accumulators[2],
    uint constraint_index) {
    accumulators[0] =
        gl_mul_add(constraint, alpha_powers[constraint_index], accumulators[0]);
    accumulators[1] = gl_mul_add(
        constraint, alpha_powers[alpha_stride + constraint_index], accumulators[1]);
}

// Each RangeCheck metadata record is ten uints:
//   selector column, gate index, group start/end, include UNUSED selector,
//   operation count, base-4 limbs per operation, final-limb range (2 or 4),
//   then two unused words that keep both record kinds the same stride.
// It is followed by promoted-family records with the same five selector words,
// then kind (arithmetic=0, subtraction=1, add-many=2, byte-decomposition=3,
// quintic-multiplication=4, quintic-squaring=5, random-access=6,
// exponentiation=7, equality=8, reducing=9), operation or copy count, and
// three explicit kind words. Random access uses the final words for index
// bits, extra constants, and the raw constant-column base; equality carries
// its constants column and reducing its extension-coefficient flag in the
// addend-count word.
// The result-limb count is what makes the subtraction and add-many branches
// width-generic: a `2 * result_limbs`-bit word recomposes from that many
// base-4 limbs and its overflow weight is `1 << (2 * result_limbs)`, which
// covers the 16-, 32- and 48-bit production gates with one code path.
// Every gate starts again at constraint row zero. This matches the CPU's
// shared row accumulator: reducing each filtered gate locally with the same
// alpha powers and then adding the results is linear in those row values.

// Select within one contiguous eight-item block using the low three index bits.
// Keeping only this block and the at-most-eight block results private avoids a
// 64-word private array for the audited six-bit gate.
inline ulong random_access_select_8(
    const device ulong* wires,
    uint lde_rows,
    uint source_row,
    ulong list_base,
    ulong bit_base,
    uint block) {
    ulong items[8];
    for (uint i = 0; i < 8u; ++i) {
        ulong column = list_base + (ulong)block * 8u + i;
        items[i] = wires[column * lde_rows + source_row];
    }
    uint level_size = 8u;
    for (uint level = 0; level < 3u; ++level) {
        ulong b = wires[(bit_base + level) * lde_rows + source_row];
        for (uint k = 0; k < level_size / 2u; ++k) {
            ulong x = items[2u * k];
            ulong y = items[2u * k + 1u];
            items[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
        }
        level_size /= 2u;
    }
    return items[0];
}

// Multiply two degree-1 coefficient vectors with three field multiplications.
// The middle coefficient uses the standard Karatsuba cross term.
inline void mul2_coeffs(
    ulong a0,
    ulong a1,
    ulong b0,
    ulong b1,
    thread ulong& c0,
    thread ulong& c1,
    thread ulong& c2) {
    c0 = gl_mul(a0, b0);
    c2 = gl_mul(a1, b1);
    c1 = gl_sub(
        gl_sub(gl_mul(gl_add(a0, a1), gl_add(b0, b1)), c0),
        c2);
}

// Multiply two degree-2 coefficient vectors when the highest coefficient
// a2*b2 is already available. The remaining four coefficients need five
// field multiplications.
inline void mul3_coeffs_known_c4(
    ulong a0,
    ulong a1,
    ulong a2,
    ulong b0,
    ulong b1,
    ulong b2,
    ulong c4,
    thread ulong& c0,
    thread ulong& c1,
    thread ulong& c2,
    thread ulong& c3) {
    c0 = gl_mul(a0, b0);
    c2 = gl_mul(a1, b1);
    c1 = gl_sub(
        gl_sub(gl_mul(gl_add(a0, a1), gl_add(b0, b1)), c0),
        c2);
    c3 = gl_sub(
        gl_sub(gl_mul(gl_add(a1, a2), gl_add(b1, b2)), c2),
        c4);
    c2 = gl_add(
        c2,
        gl_sub(
            gl_sub(gl_mul(gl_add(a0, a2), gl_add(b0, b2)), c0),
            c4));
}

// General degree-2 product. Keeping the five output coefficients separate
// lets the quintic extension product reuse its existing a[] storage without
// increasing live arrays.
inline void mul3_coeffs(
    ulong a0,
    ulong a1,
    ulong a2,
    ulong b0,
    ulong b1,
    ulong b2,
    thread ulong& c0,
    thread ulong& c1,
    thread ulong& c2,
    thread ulong& c3,
    thread ulong& c4) {
    c4 = gl_mul(a2, b2);
    mul3_coeffs_known_c4(
        a0, a1, a2, b0, b1, b2, c4, c0, c1, c2, c3);
}

kernel void range_check_gate_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* metadata [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& alpha_stride [[buffer(8)]],
    constant uint& range_count [[buffer(9)]],
    constant uint& u32_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong total[2] = { 0, 0 };
    for (uint range_index = 0; range_index < range_count; ++range_index) {
        constant uint* spec = metadata + range_index * 10u;
        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint num_ops = spec[5];
        uint num_aux = spec[6];
        uint final_limb_range = spec[7];

        ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        ulong gate_accumulators[2] = { 0, 0 };
        uint constraint_index = 0;
        for (uint op = 0; op < num_ops; ++op) {
            ulong input = wires[(ulong)op * lde_rows + source_row];
            ulong aux_base = (ulong)num_ops + (ulong)num_aux * op;
            ulong computed = wires[(aux_base + num_aux - 1u) * lde_rows + source_row];
            for (uint remaining = num_aux - 1u; remaining > 0u; --remaining) {
                uint j = remaining - 1u;
                ulong limb = wires[(aux_base + j) * lde_rows + source_row];
                computed = gl_add(gl_mul(computed, 4), limb);
            }
            range_check_gate_emit(
                gl_sub(computed, input),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);

            for (uint j = 0; j < num_aux; ++j) {
                ulong x = wires[(aux_base + j) * lde_rows + source_row];
                ulong constraint;
                if (j + 1u == num_aux && final_limb_range == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    // x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3),
                    // exactly the production CPU specialization.
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        }

        total[0] = gl_mul_add(filter, gate_accumulators[0], total[0]);
        total[1] = gl_mul_add(filter, gate_accumulators[1], total[1]);
    }

    constant uint* u32_metadata = metadata + range_count * 10u;
    for (uint u32_index = 0; u32_index < u32_count; ++u32_index) {
        constant uint* spec = u32_metadata + u32_index * 10u;
        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint kind = spec[5];
        uint num_ops = spec[6];
        uint num_addends = spec[7];
        uint result_limbs = spec[8];
        uint num_carry_limbs = spec[9];
        // Overflow weight of the recomposed word: 2^16, 2^32 or 2^48.
        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        ulong gate_accumulators[2] = { 0, 0 };
        uint constraint_index = 0;
        if (kind == 0u) {
            // U32ArithmeticGate: six routed words followed by 32 base-4
            // output limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 6u;
                ulong multiplicand_0 = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong multiplicand_1 = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong addend = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong output_low = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong output_high = wires[(routed_base + 4u) * lde_rows + source_row];
                ulong inverse = wires[(routed_base + 5u) * lde_rows + source_row];

                ulong high_diff = gl_sub(0xffffffffUL, output_high);
                ulong high_not_max = gl_sub(gl_mul(inverse, high_diff), 1);
                range_check_gate_emit(
                    gl_mul(high_not_max, output_low),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong computed = gl_add(gl_mul(multiplicand_0, multiplicand_1), addend);
                ulong combined = gl_add(gl_mul(output_high, 4294967296UL), output_low);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 6u + (ulong)op * 32u;
                ulong combined_low = 0;
                ulong combined_high = 0;
                for (uint remaining = 32u; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    if (j < 16u) {
                        combined_low = gl_add(gl_mul(combined_low, 4), x);
                    } else {
                        combined_high = gl_add(gl_mul(combined_high, 4), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_low, output_low),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_high, output_high),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 1u) {
            // U16/U32/U48 SubtractionGate: five routed words followed by
            // `result_limbs` base-4 result limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 5u;
                ulong input_x = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong input_y = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong input_borrow = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong output_result = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong output_borrow = wires[(routed_base + 4u) * lde_rows + source_row];
                ulong result_initial = gl_sub(gl_sub(input_x, input_y), input_borrow);
                ulong borrowed = gl_add(
                    result_initial,
                    gl_mul(word_base, output_borrow));
                range_check_gate_emit(
                    gl_sub(output_result, borrowed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 5u + (ulong)op * result_limbs;
                ulong recomposed = 0;
                for (uint remaining = result_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    recomposed = gl_add(gl_mul(recomposed, 4), x);
                }
                range_check_gate_emit(
                    gl_sub(recomposed, output_result),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_mul(output_borrow, gl_sub(1, output_borrow)),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 2u) {
            // U16/U32 AddManyGate: num_addends inputs, carry/result/output-carry,
            // then `result_limbs` result and `num_carry_limbs` carry base-4
            // limbs per operation.
            uint routed_per_op = num_addends + 3u;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong computed = wires[(routed_base + num_addends) * lde_rows + source_row];
                for (uint j = 0; j < num_addends; ++j) {
                    computed = gl_add(
                        computed,
                        wires[(routed_base + j) * lde_rows + source_row]);
                }
                ulong output_result =
                    wires[(routed_base + num_addends + 1u) * lde_rows + source_row];
                ulong output_carry =
                    wires[(routed_base + num_addends + 2u) * lde_rows + source_row];
                ulong combined = gl_add(gl_mul(output_carry, word_base), output_result);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                uint total_limbs = result_limbs + num_carry_limbs;
                ulong limb_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * total_limbs;
                ulong combined_result = 0;
                ulong combined_carry = 0;
                for (uint remaining = total_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    if (j < result_limbs) {
                        combined_result = gl_add(gl_mul(combined_result, 4), x);
                    } else {
                        combined_carry = gl_add(gl_mul(combined_carry, 4), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_result, output_result),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_carry, output_carry),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 3u) {
            // ByteDecompositionGate: per operation, one routed sum wire and
            // `num_addends` routed byte wires (the metadata word carries the
            // byte count for this kind), then four base-4 aux limbs per
            // byte. Constraint order matches the CPU gate exactly: aux range
            // products in ascending wire order, one base-4 recomposition per
            // byte, then the base-256 byte-to-sum recomposition.
            uint num_limbs = num_addends;
            uint routed_per_op = 1u + num_limbs;
            uint aux_per_op = 4u * num_limbs;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong aux_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * aux_per_op;
                for (uint j = 0; j < aux_per_op; ++j) {
                    ulong x = wires[(aux_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }
                for (uint byte_index = 0; byte_index < num_limbs; ++byte_index) {
                    ulong chunk = aux_base + (ulong)byte_index * 4u;
                    ulong recomposed = wires[(chunk + 3u) * lde_rows + source_row];
                    for (uint remaining = 3u; remaining > 0u; --remaining) {
                        uint k = remaining - 1u;
                        recomposed = gl_add(
                            gl_mul(recomposed, 4),
                            wires[(chunk + k) * lde_rows + source_row]);
                    }
                    ulong byte_value =
                        wires[(routed_base + 1u + byte_index) * lde_rows + source_row];
                    range_check_gate_emit(
                        gl_sub(recomposed, byte_value),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }
                ulong recomposed_sum =
                    wires[(routed_base + num_limbs) * lde_rows + source_row];
                for (uint remaining = num_limbs - 1u; remaining > 0u; --remaining) {
                    uint k = remaining - 1u;
                    recomposed_sum = gl_add(
                        gl_mul(recomposed_sum, 256),
                        wires[(routed_base + 1u + k) * lde_rows + source_row]);
                }
                ulong expected_sum = wires[routed_base * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(recomposed_sum, expected_sum),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 4u) {
            // QuinticMultiplicationGate: fifteen routed words per operation
            // (five limbs each for a, b and the claimed product c). The five
            // constraints are the product limbs reduced by u^5 = 3, minus
            // the claimed output limbs, in ascending limb order exactly like
            // the CPU accumulator. Split at u^3: L=A0*B0, H=A1*B1 and
            // C=(A0+A1)(B0+B1)-L-H. Its highest coefficient is the already
            // computed L[4]=a2*b2, so it is reused. The two 3x3 products and
            // one 2x2 product use 6/5/3 field multiplications, reducing the
            // full product from 25 to 14.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 15u;
                ulong a[5];
                ulong b[5];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * lde_rows + source_row];
                    b[j] = wires[(routed_base + 5u + j) * lde_rows + source_row];
                }

                // Store L in d[0..5) and H in d[5..8). After both products
                // are complete, the original low limbs are no longer needed;
                // reuse a[] for the middle product and then the cross term C.
                ulong d[8];
                mul3_coeffs(
                    a[0], a[1], a[2], b[0], b[1], b[2],
                    d[0], d[1], d[2], d[3], d[4]);
                mul2_coeffs(a[3], a[4], b[3], b[4], d[5], d[6], d[7]);

                a[0] = gl_add(a[0], a[3]);
                a[1] = gl_add(a[1], a[4]);
                b[0] = gl_add(b[0], b[3]);
                b[1] = gl_add(b[1], b[4]);
                mul3_coeffs_known_c4(
                    a[0], a[1], a[2], b[0], b[1], b[2], d[4],
                    a[0], a[1], a[2], a[3]);
                for (uint k = 0; k < 4u; ++k) {
                    a[k] = gl_sub(a[k], d[k]);
                    if (k < 3u) {
                        a[k] = gl_sub(a[k], d[5u + k]);
                    }
                }

                // b[] is dead after the middle product; reuse it for the five
                // reduced output coefficients instead of adding a live array.
                b[0] = gl_add(d[0], gl_mul(3, a[2]));
                b[1] = gl_add(d[1], gl_mul(3, gl_add(a[3], d[5])));
                b[2] = gl_add(d[2], gl_mul(3, d[6]));
                b[3] = gl_add(gl_add(d[3], a[0]), gl_mul(3, d[7]));
                b[4] = gl_add(d[4], a[1]);
                for (uint k = 0; k < 5u; ++k) {
                    ulong c = wires[(routed_base + 10u + k) * lde_rows + source_row];
                    range_check_gate_emit(
                        gl_sub(b[k], c),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }
            }
        } else if (kind == 5u) {
            // QuinticSquaringGate: ten routed words per operation (input
            // limbs a then output limbs c) plus ten temporary wires. Each
            // constraint checks one accumulation step of the squaring
            // against its temporary or output, in the exact CPU emission
            // order.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 10u;
                ulong temp_base = (ulong)num_ops * 10u + (ulong)op * 10u;
                ulong a[5];
                ulong c[5];
                ulong extra[10];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * lde_rows + source_row];
                    c[j] = wires[(routed_base + 5u + j) * lde_rows + source_row];
                }
                for (uint j = 0; j < 10u; ++j) {
                    extra[j] = wires[(temp_base + j) * lde_rows + source_row];
                }

                // c[0]
                range_check_gate_emit(
                    gl_sub(gl_mul(a[0], a[0]), extra[0]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[1]), a[4]), extra[0]), extra[1]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[3]), extra[1]), c[0]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                // c[1]
                range_check_gate_emit(
                    gl_sub(gl_mul(gl_mul(3, a[3]), a[3]), extra[2]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[1]), extra[2]), extra[3]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[4]), extra[3]), c[1]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                // c[2]
                range_check_gate_emit(
                    gl_sub(gl_mul(a[1], a[1]), extra[4]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[2]), extra[4]), extra[5]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[3]), a[4]), extra[5]), c[2]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                // c[3]
                range_check_gate_emit(
                    gl_sub(gl_mul(gl_mul(3, a[4]), a[4]), extra[6]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[3]), extra[6]), extra[7]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[2]), extra[7]), c[3]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                // c[4]
                range_check_gate_emit(
                    gl_sub(gl_mul(a[2], a[2]), extra[8]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[4]), extra[8]), extra[9]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[3]), extra[9]), c[4]),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 6u) {
            uint bits = num_addends;
            uint num_extra_constants = result_limbs;
            uint constant_base = num_carry_limbs;
            uint num_copies = num_ops;
            ulong vec_size = 1UL << bits;
            ulong routed_per_copy = vec_size + 2u;
            ulong extra_wire_base = routed_per_copy * num_copies;
            ulong bit_base = extra_wire_base + num_extra_constants;

            for (uint copy = 0; copy < num_copies; ++copy) {
                ulong copy_base = routed_per_copy * copy;

                // RandomAccessGate emits boolean constraints for b_0 upward.
                for (uint i = 0; i < bits; ++i) {
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * lde_rows + source_row];
                    range_check_gate_emit(
                        gl_mul(b, gl_sub(b, 1)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }

                // Reconstruct the little-endian index in the CPU's exact
                // reverse-bit `acc.double() + b` order.
                ulong reconstructed_index = 0;
                for (uint remaining = bits; remaining > 0u; --remaining) {
                    uint i = remaining - 1u;
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * lde_rows + source_row];
                    reconstructed_index = gl_add(
                        gl_add(reconstructed_index, reconstructed_index), b);
                }
                ulong access_index = wires[copy_base * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(reconstructed_index, access_index),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                // Fold each eight-item block in ascending pair order, then fold
                // block results with the remaining bits in the same order.
                ulong block_results[8];
                uint block_count = (uint)(vec_size / 8u);
                ulong list_base = copy_base + 2u;
                ulong copy_bit_base = bit_base + (ulong)copy * bits;
                for (uint block = 0; block < block_count; ++block) {
                    block_results[block] = random_access_select_8(
                        wires, lde_rows, source_row, list_base, copy_bit_base, block);
                }
                uint level_size = block_count;
                for (uint i = 3u; i < bits; ++i) {
                    ulong b = wires[(copy_bit_base + i) * lde_rows + source_row];
                    for (uint k = 0; k < level_size / 2u; ++k) {
                        ulong x = block_results[2u * k];
                        ulong y = block_results[2u * k + 1u];
                        block_results[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
                    }
                    level_size /= 2u;
                }
                ulong claimed_element = wires[(copy_base + 1u) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(block_results[0], claimed_element),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }

            // Raw local constants follow all gate and lookup selectors.
            for (uint i = 0; i < num_extra_constants; ++i) {
                ulong local_constant = constants[
                    ((ulong)constant_base + i) * lde_rows + source_row];
                ulong extra_wire = wires[
                    (extra_wire_base + i) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(local_constant, extra_wire),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 7u) {
            // ExponentiationGate: wire 0 is the base, wires 1..=n the power
            // bits in little-endian order, wire 1+n the output and wires
            // 2+n..2+2n the running intermediate values; `num_ops` carries n.
            // The accumulation walks the bits big-endian and seeds the chain
            // with ONE rather than a square, exactly as the CPU evaluator.
            uint num_power_bits = num_ops;
            ulong exponent_base = wires[(ulong)0 * lde_rows + source_row];
            for (uint i = 0; i < num_power_bits; ++i) {
                ulong previous;
                if (i == 0u) {
                    previous = 1;
                } else {
                    ulong last = wires[((ulong)2u + num_power_bits + i - 1u) * lde_rows
                                       + source_row];
                    previous = gl_mul(last, last);
                }
                ulong current_bit =
                    wires[((ulong)1u + (num_power_bits - i - 1u)) * lde_rows + source_row];
                ulong multiplier =
                    gl_add(gl_mul(current_bit, exponent_base), gl_sub(1, current_bit));
                ulong intermediate =
                    wires[((ulong)2u + num_power_bits + i) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(gl_mul(previous, multiplier), intermediate),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
            ulong output_value = wires[((ulong)1u + num_power_bits) * lde_rows + source_row];
            ulong final_intermediate =
                wires[((ulong)1u + 2u * num_power_bits) * lde_rows + source_row];
            range_check_gate_emit(
                gl_sub(output_value, final_intermediate),
                alpha_powers, alpha_stride, gate_accumulators,
                constraint_index++);
        } else if (kind == 8u) {
            // EqualityGate: three routed words per operation (x, y, equal)
            // followed by three unrouted temporaries (diff, invdiff, prod).
            // The addend slot carries the constants column holding the gate's
            // first constant, its "one" value.
            uint constant_column = num_addends;
            ulong const_0 = constants[(ulong)constant_column * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 3u;
                ulong x = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong y = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong equal = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong temporary_base = (ulong)num_ops * 3u + (ulong)op * 3u;
                ulong difference = wires[(temporary_base + 0u) * lde_rows + source_row];
                ulong inverse = wires[(temporary_base + 1u) * lde_rows + source_row];
                ulong product = wires[(temporary_base + 2u) * lde_rows + source_row];

                range_check_gate_emit(
                    gl_sub(gl_sub(x, y), difference),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(difference, inverse), product),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(product, difference), difference),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(const_0, product), equal),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 9u) {
            // ReducingGate / ReducingExtensionGate at D == 2. Wires 0..2 are
            // the output, 2..4 alpha, 4..6 the incoming accumulator, then one
            // (base) or two (extension) wires per coefficient, then one
            // two-wire accumulator per step except the last, which aliases the
            // output. Each step emits the two components of
            // `acc * alpha + coeff - next_acc` in that order.
            //
            // The quadratic extension is F[x]/(x^2 - 7) for Goldilocks:
            //   (a0 + a1 x)(b0 + b1 x) = (a0 b0 + 7 a1 b1) + (a0 b1 + a1 b0) x.
            uint extension_coeffs = num_addends;
            uint coeff_wires = extension_coeffs != 0u ? 2u : 1u;
            uint coeff_start = 6u;
            uint acc_start = coeff_start + num_ops * coeff_wires;
            ulong alpha_0 = wires[(ulong)2u * lde_rows + source_row];
            ulong alpha_1 = wires[(ulong)3u * lde_rows + source_row];
            ulong acc_0 = wires[(ulong)4u * lde_rows + source_row];
            ulong acc_1 = wires[(ulong)5u * lde_rows + source_row];
            for (uint i = 0; i < num_ops; ++i) {
                uint next_start = (i + 1u == num_ops) ? 0u : acc_start + 2u * i;
                ulong next_0 = wires[(ulong)next_start * lde_rows + source_row];
                ulong next_1 = wires[((ulong)next_start + 1u) * lde_rows + source_row];

                uint coeff_wire = coeff_start + i * coeff_wires;
                ulong coeff_0 = wires[(ulong)coeff_wire * lde_rows + source_row];
                ulong coeff_1 = extension_coeffs != 0u
                    ? wires[((ulong)coeff_wire + 1u) * lde_rows + source_row]
                    : 0;

                ulong product_0 = gl_add(
                    gl_mul(acc_0, alpha_0),
                    gl_mul(7, gl_mul(acc_1, alpha_1)));
                ulong product_1 = gl_add(
                    gl_mul(acc_0, alpha_1),
                    gl_mul(acc_1, alpha_0));
                range_check_gate_emit(
                    gl_sub(gl_add(product_0, coeff_0), next_0),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(product_1, coeff_1), next_1),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                acc_0 = next_0;
                acc_1 = next_1;
            }
        } else {
            // The Rust encoder rejects unknown discriminants; if a malformed
            // record reaches the shader, make its selected row unsatisfiable.
            range_check_gate_emit(
                1, alpha_powers, alpha_stride, gate_accumulators,
                constraint_index++);
        }

        total[0] = gl_mul_add(filter, gate_accumulators[0], total[0]);
        total[1] = gl_mul_add(filter, gate_accumulators[1], total[1]);
    }

    output[(ulong)gid * 2] = gl_canonicalize(total[0]);
    output[(ulong)gid * 2 + 1] = gl_canonicalize(total[1]);
}

kernel void poseidon2_hash_leaves(
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

// Writes the state of the zero-padded, coset-shifted coefficient array as it
// exists after `reverse_index_bits_in_place` plus the first `rate_bits`
// replication rounds of `fft_classic`: position i of column `col` holds
// shift^k * coeffs[col][k] with k = reverse_bits(i >> rate_bits, log_degree).
kernel void ntt_prepare(
    const device ulong* coeffs [[buffer(0)]],
    const device ulong* shift_pows [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant uint& degree [[buffer(3)]],
    constant uint& lde_size [[buffer(4)]],
    constant uint& log_degree [[buffer(5)]],
    constant uint& rate_bits [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= lde_size) {
        return;
    }
    uint col = gid.y;
    uint k = log_degree == 0
        ? 0
        : (reverse_bits(i >> rate_bits) >> (32 - log_degree));
    ulong value = gl_mul(shift_pows[k], coeffs[(ulong)col * degree + k]);
    out[(ulong)col * lde_size + i] = value;
}

// One radix-2 decimation-in-time butterfly stage over every column, matching
// fft_classic: (u, v) := (u + w*v, u - w*v) with w = roots[j]. The final stage
// canonicalizes so downstream consumers see canonical representations.
kernel void ntt_stage(
    device ulong* values [[buffer(0)]],
    const device ulong* roots [[buffer(1)]],
    constant uint& lde_size [[buffer(2)]],
    constant uint& log_half_m [[buffer(3)]],
    constant uint& canonicalize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint half_butterflies = lde_size >> 1;
    if (t >= half_butterflies) {
        return;
    }
    ulong colbase = (ulong)gid.y * lde_size;
    uint half_m = 1u << log_half_m;
    uint j = t & (half_m - 1u);
    uint base = ((t >> log_half_m) << (log_half_m + 1u));
    uint u_index = base + j;
    uint v_index = u_index + half_m;

    ulong u = values[colbase + u_index];
    ulong v = values[colbase + v_index];
    ulong w = roots[j];
    ulong wv = gl_mul(w, v);
    ulong out_u = gl_add(u, wv);
    ulong out_v = gl_sub(u, wv);
    if (canonicalize != 0u) {
        out_u = gl_canonicalize(out_u);
        out_v = gl_canonicalize(out_v);
    }
    values[colbase + u_index] = out_u;
    values[colbase + v_index] = out_v;
}

// Converts a forward-FFT output into IFFT coefficients, matching plonky2's
// `ifft`: coeffs[i] = fft_out[(n - i) mod n] * n^{-1}, canonicalized so the
// CPU-side readback and the downstream LDE prepare see canonical values.
kernel void ifft_finalize(
    const device ulong* fft_out [[buffer(0)]],
    device ulong* coeffs [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& n_inv [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= n) {
        return;
    }
    ulong colbase = (ulong)gid.y * n;
    uint src = (n - i) & (n - 1u);
    coeffs[colbase + i] = gl_canonicalize(gl_mul(fft_out[colbase + src], n_inv));
}

kernel void poseidon2_hash_leaves_colmajor(
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

kernel void poseidon2_hash_parents(
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
kernel void poseidon2_absorb_pass(
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
