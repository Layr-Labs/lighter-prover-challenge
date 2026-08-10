#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;

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
}

inline ulong gl_sub(ulong a, ulong b) {
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
}

inline ulong reduce_top(uint r0, uint r1, int top) {
    ulong r = ((ulong)r1 << 32) | (ulong)r0;
    return r + (ulong)(((long)top << 32) - (long)top);
}

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

inline ulong gl_mul(ulong a, ulong b) {
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
}

inline ulong gl_mul_add(ulong a, ulong b, ulong addend) {
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);

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

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

// Standalone replacement for only the permutation quotient kernel in the
// checked-in library. The ABI is deliberately identical to poseidon2.metal.
kernel void permutation_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants_sigmas [[buffer(1)]],
    const device ulong* zs_partial_products [[buffer(2)]],
    const device ulong* shifted_points [[buffer(3)]],
    device ulong* output [[buffer(4)]],
    constant ulong* alpha_powers [[buffer(5)]],
    constant ulong* challenges [[buffer(6)]],
    constant uint& lde_rows [[buffer(7)]],
    constant uint& quotient_rows [[buffer(8)]],
    constant uint& step [[buffer(9)]],
    constant uint& next_step [[buffer(10)]],
    constant uint& sigma_start [[buffer(11)]],
    constant uint& num_routed_wires [[buffer(12)]],
    constant uint& num_partial_products [[buffer(13)]],
    constant uint& chunk_size [[buffer(14)]],
    constant uint& alpha_stride [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    uint next_gid = (gid + next_step) & (quotient_rows - 1u);
    uint next_source_row = next_gid * step;
    uint num_chunks = num_partial_products + 1u;
    ulong x = shifted_points[gid];
    ulong totals[2] = { 0, 0 };

    ulong beta0 = challenges[0];
    ulong beta1 = challenges[1];
    ulong gamma0 = challenges[2];
    ulong gamma1 = challenges[3];
    for (uint chunk = 0; chunk < num_chunks; ++chunk) {
        uint j_start = chunk * chunk_size;
        uint j_end = min(j_start + chunk_size, num_routed_wires);
        ulong wire = wires[(ulong)j_start * lde_rows + source_row];
        ulong sigma = constants_sigmas[
            (ulong)(sigma_start + j_start) * lde_rows + source_row];
        ulong beta_k0 = challenges[4u + j_start];
        ulong beta_k1 = challenges[4u + num_routed_wires + j_start];
        ulong wg = gl_add(wire, gamma0);
        ulong numerator0 = gl_mul_add(beta_k0, x, wg);
        ulong denominator0 = gl_mul_add(beta0, sigma, wg);
        wg = gl_add(wire, gamma1);
        ulong numerator1 = gl_mul_add(beta_k1, x, wg);
        ulong denominator1 = gl_mul_add(beta1, sigma, wg);
        for (uint j = j_start + 1u; j < j_end; ++j) {
            wire = wires[(ulong)j * lde_rows + source_row];
            sigma = constants_sigmas[
                (ulong)(sigma_start + j) * lde_rows + source_row];
            beta_k0 = challenges[4u + j];
            beta_k1 = challenges[4u + num_routed_wires + j];
            wg = gl_add(wire, gamma0);
            numerator0 = gl_mul(numerator0, gl_mul_add(beta_k0, x, wg));
            denominator0 = gl_mul(denominator0, gl_mul_add(beta0, sigma, wg));
            wg = gl_add(wire, gamma1);
            numerator1 = gl_mul(numerator1, gl_mul_add(beta_k1, x, wg));
            denominator1 = gl_mul(denominator1, gl_mul_add(beta1, sigma, wg));
        }

        uint previous_column0 = chunk == 0u ? 0u : 1u + chunk;
        ulong previous0 = zs_partial_products[
            (ulong)previous_column0 * lde_rows + source_row];
        ulong next0;
        if (chunk < num_partial_products) {
            next0 = zs_partial_products[(ulong)(2u + chunk) * lde_rows + source_row];
        } else {
            next0 = zs_partial_products[next_source_row];
        }
        ulong term0 = gl_sub(
            gl_mul(previous0, numerator0), gl_mul(next0, denominator0));
        uint alpha_index0 = 2u + chunk;
        totals[0] = gl_mul_add(term0, alpha_powers[alpha_index0], totals[0]);
        totals[1] = gl_mul_add(
            term0, alpha_powers[alpha_stride + alpha_index0], totals[1]);

        uint previous_column1 = chunk == 0u
            ? 1u
            : 1u + num_partial_products + chunk;
        ulong previous1 = zs_partial_products[
            (ulong)previous_column1 * lde_rows + source_row];
        ulong next1;
        if (chunk < num_partial_products) {
            uint next_column1 = 2u + num_partial_products + chunk;
            next1 = zs_partial_products[(ulong)next_column1 * lde_rows + source_row];
        } else {
            next1 = zs_partial_products[(ulong)lde_rows + next_source_row];
        }
        ulong term1 = gl_sub(
            gl_mul(previous1, numerator1), gl_mul(next1, denominator1));
        uint alpha_index1 = 2u + num_chunks + chunk;
        totals[0] = gl_mul_add(term1, alpha_powers[alpha_index1], totals[0]);
        totals[1] = gl_mul_add(
            term1, alpha_powers[alpha_stride + alpha_index1], totals[1]);
    }

    output[(ulong)gid * 2u] = gl_canonicalize(totals[0]);
    output[(ulong)gid * 2u + 1u] = gl_canonicalize(totals[1]);
}
