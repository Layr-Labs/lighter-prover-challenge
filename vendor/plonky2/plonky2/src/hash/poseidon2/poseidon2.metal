#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

inline ulong gl_add(ulong a, ulong b) {
    ulong sum = a + b;
    ulong carry = sum < a;
    sum += carry * GOLDILOCKS_EPSILON;
    ulong carry2 = (carry != 0UL) && (sum < GOLDILOCKS_EPSILON);
    return sum + carry2 * GOLDILOCKS_EPSILON;
}

inline ulong gl_mul(ulong a, ulong b) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong high_high = high >> 32;
    ulong high_low = high & GOLDILOCKS_EPSILON;

    // Since 2^64 = 2^32 - 1 and 2^32(2^32 - 1) = -1 modulo p,
    // low + high * 2^64 reduces to low - high_high + high_low * (2^32 - 1).
    ulong reduced = low - high_high;
    if (reduced > low) {
        reduced -= GOLDILOCKS_EPSILON;
    }
    ulong addend = high_low * GOLDILOCKS_EPSILON;
    ulong result = reduced + addend;
    return result + (result < reduced) * GOLDILOCKS_EPSILON;
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value4 = gl_mul(value2, value2);
    ulong value3 = gl_mul(value, value2);
    return gl_mul(value3, value4);
}

inline void mat4(thread ulong* values) {
    ulong x0 = values[0];
    ulong x1 = values[1];
    ulong x2 = values[2];
    ulong x3 = values[3];
    ulong t01 = gl_add(x0, x1);
    ulong t23 = gl_add(x2, x3);
    ulong total = gl_add(t01, t23);

    values[0] = gl_add(gl_add(total, t01), x1);
    values[1] = gl_add(gl_add(gl_add(total, x1), x2), x2);
    values[2] = gl_add(gl_add(total, t23), x3);
    values[3] = gl_add(gl_add(gl_add(total, x3), x0), x0);
}

inline void external_linear_layer(thread ulong state[12]) {
    mat4(state);
    mat4(state + 4);
    mat4(state + 8);

    ulong sums[4];
    for (uint i = 0; i < 4; ++i) {
        sums[i] = gl_add(gl_add(state[i], state[i + 4]), state[i + 8]);
    }
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_add(state[i], sums[i & 3]);
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
    return gl_add(sum, (ulong)carries * GOLDILOCKS_EPSILON);
}

inline void internal_linear_layer(thread ulong state[12], constant ulong* diagonal) {
    ulong sum = sum_state(state);
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_add(sum, gl_mul(state[i], diagonal[i]));
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

// Column-major leaf hashing: element i of natural-order point `gid` is at
// polys[i * leaf_count + gid], so adjacent threads read adjacent addresses.
// Hashes are written in natural point order; a separate bit-reversal pass
// places them in Merkle leaf order.
kernel void poseidon2_hash_leaves_cols(
    const device ulong* polys [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    device ulong* output = hashes + (ulong)gid * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(polys[(ulong)i * leaf_count + gid]);
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
            state[i] = gl_canonicalize(polys[(ulong)(offset + i) * leaf_count + gid]);
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

// Permute natural-order leaf hashes into Merkle leaf order:
// dst[j] = src[reverse_bits(j)].
kernel void bit_reverse_rows(
    const device ulong* src [[buffer(0)]],
    device ulong* dst [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& log_row_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) {
        return;
    }
    uint source_row = log_row_count == 0
        ? gid
        : reverse_bits(gid) >> (32u - log_row_count);
    const device ulong* source = src + (ulong)source_row * 4;
    device ulong* destination = dst + (ulong)gid * 4;
    for (uint i = 0; i < 4; ++i) {
        destination[i] = source[i];
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
