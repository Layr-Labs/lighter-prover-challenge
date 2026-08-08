#include <metal_stdlib>
using namespace metal;

// This deliberately stays separate from poseidon2.metal. The promoted
// precompiled library remains valid for every existing kernel, while this
// small source-only library extends its range-check quotient output with the
// two downstream U32 bit-permutation gates.
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

inline ulong gl_mul(ulong a, ulong b) {
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
}

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

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

inline void emit_constraint(
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

// Runs after range_check_gate_quotient in the same command buffer. Its metadata
// partition begins after the legacy records, so this kernel starts from the
// legacy kernel's canonical output and adds only kinds 10 through 13.
kernel void interleave_gate_quotient_patch(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* metadata [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& alpha_stride [[buffer(8)]],
    constant uint& u32_start [[buffer(9)]],
    constant uint& u32_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong total[2] = { output[(ulong)gid * 2u], output[(ulong)gid * 2u + 1u] };
    constant uint* u32_metadata = metadata + u32_start * 10u;
    for (uint u32_index = 0; u32_index < u32_count; ++u32_index) {
        constant uint* spec = u32_metadata + u32_index * 10u;
        uint kind = spec[5];
        if (kind < 10u || kind > 13u) {
            continue;
        }

        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint num_ops = spec[6];
        uint num_addends = spec[7];

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
        if (kind == 10u) {
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 2u;
                ulong bit_base = (ulong)num_ops * 2u + (ulong)op * 32u;
                ulong reconstructed = 0;
                ulong reconstructed_interleaved = 0;
                for (uint j = 0; j < 32u; ++j) {
                    ulong bit = wires[(bit_base + j) * lde_rows + source_row];
                    reconstructed = gl_add(gl_mul(reconstructed, 2), bit);
                    reconstructed_interleaved =
                        gl_add(gl_mul(reconstructed_interleaved, 4), bit);
                }
                emit_constraint(
                    gl_sub(reconstructed, wires[routed_base * lde_rows + source_row]),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                emit_constraint(
                    gl_sub(
                        reconstructed_interleaved,
                        wires[(routed_base + 1u) * lde_rows + source_row]),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                for (uint j = 0; j < 32u; ++j) {
                    ulong bit = wires[(bit_base + j) * lde_rows + source_row];
                    emit_constraint(
                        gl_mul(bit, gl_sub(bit, 1)),
                        alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                }
            }
        } else if (kind == 11u) {
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 4u;
                ulong bit_base = (ulong)num_ops * 4u + (ulong)op * 64u;
                ulong high = 0;
                ulong low = 0;
                ulong evens = 0;
                ulong odds = 0;
                for (uint j = 0; j < 64u; ++j) {
                    ulong bit = wires[(bit_base + j) * lde_rows + source_row];
                    if (j < 32u) {
                        high = gl_add(gl_mul(high, 2), bit);
                    } else {
                        low = gl_add(gl_mul(low, 2), bit);
                    }
                    if ((j & 1u) == 0u) {
                        evens = gl_add(gl_mul(evens, 2), bit);
                    } else {
                        odds = gl_add(gl_mul(odds, 2), bit);
                    }
                }

                ulong inverse = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong high_not_max =
                    gl_sub(gl_mul(inverse, gl_sub(0xffffffffUL, high)), 1);
                emit_constraint(
                    gl_mul(high_not_max, low),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                emit_constraint(
                    gl_sub(
                        gl_add(gl_mul(high, 4294967296UL), low),
                        wires[routed_base * lde_rows + source_row]),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                emit_constraint(
                    gl_sub(evens, wires[(routed_base + 1u) * lde_rows + source_row]),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                emit_constraint(
                    gl_sub(odds, wires[(routed_base + 2u) * lde_rows + source_row]),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                for (uint j = 0; j < 64u; ++j) {
                    ulong bit = wires[(bit_base + j) * lde_rows + source_row];
                    emit_constraint(
                        gl_mul(bit, gl_sub(bit, 1)),
                        alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                }
            }
        } else if (kind == 12u) {
            // SelectionGate: four routed values per operation (selector, x,
            // y, result), followed by one temporary per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 4u;
                ulong temporary = (ulong)num_ops * 4u + op;
                ulong selector_bit = wires[routed_base * lde_rows + source_row];
                ulong x = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong y = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong result = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong temp = wires[temporary * lde_rows + source_row];
                emit_constraint(
                    gl_sub(gl_sub(gl_mul(selector_bit, y), y), temp),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
                emit_constraint(
                    gl_sub(gl_sub(gl_mul(selector_bit, x), temp), result),
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
            }
        } else {
            // BaseSumGate<2/4>: num_ops carries the little-endian limb count
            // and num_addends carries the compile-time base.
            ulong computed = 0;
            for (uint remaining = num_ops; remaining > 0u; --remaining) {
                ulong limb = wires[(ulong)remaining * lde_rows + source_row];
                computed = gl_add(gl_mul(computed, num_addends), limb);
            }
            emit_constraint(
                gl_sub(computed, wires[source_row]),
                alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
            for (uint limb_index = 0; limb_index < num_ops; ++limb_index) {
                ulong limb = wires[((ulong)limb_index + 1u) * lde_rows + source_row];
                ulong constraint;
                if (num_addends == 2u) {
                    constraint = gl_mul(limb, gl_sub(limb, 1));
                } else {
                    ulong y = gl_mul(limb, gl_sub(limb, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                emit_constraint(
                    constraint,
                    alpha_powers, alpha_stride, gate_accumulators, constraint_index++);
            }
        }

        total[0] = gl_mul_add(filter, gate_accumulators[0], total[0]);
        total[1] = gl_mul_add(filter, gate_accumulators[1], total[1]);
    }

    output[(ulong)gid * 2u] = gl_canonicalize(total[0]);
    output[(ulong)gid * 2u + 1u] = gl_canonicalize(total[1]);
}
