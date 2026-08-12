#include "poseidon2.metal"

// Absorbs one or two complete sponge-rate groups while the twelve-lane state
// remains in registers. The promoted streamed builder used two separate
// `poseidon2_absorb_pass` dispatches for a prepared sixteen-column span, which
// wrote and reread all twelve state lanes between the two permutations.
kernel void poseidon2_absorb16(
    const device ulong* leaves [[buffer(0)]],
    device ulong* state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
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
    // This specialization is dispatched only for a complete pair of rate
    // groups. Fixed trip counts let the Metal compiler fully unroll both
    // overwrites instead of carrying the generic pass-size loop per leaf.
    for (uint i = 0; i < 8; ++i) {
        st[i] = gl_canonicalize(
            leaves[(ulong)(col_start + i) * leaf_count + gid]);
    }
    poseidon2(st, parameters);
    for (uint i = 0; i < 8; ++i) {
        st[i] = gl_canonicalize(
            leaves[(ulong)(col_start + 8 + i) * leaf_count + gid]);
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
