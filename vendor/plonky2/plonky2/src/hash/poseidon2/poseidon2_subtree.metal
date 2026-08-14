#include "poseidon2.metal"

// Reduces 256 child digests per 128-thread group through as many as eight
// ordinary binary Merkle levels. The broad first level retains one hash per
// thread; subsequent levels halve the active threads and exchange canonical
// digests through ping-pong threadgroup arrays. Every intermediate level is
// still written to the complete global tree for authentication paths.
kernel void poseidon2_hash_subtree128(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& child_count [[buffer(3)]],
    constant uint& levels [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong4 even_level[128];
    threadgroup ulong4 odd_level[128];
    ulong state[12];

    const uint first_child = group * 256 + tid * 2;
    for (uint i = 0; i < 8; ++i) {
        state[i] = children[(ulong)first_child * 4 + i];
    }
    for (uint i = 8; i < 12; ++i) {
        state[i] = 0;
    }
    poseidon2(state, parameters);
    ulong4 digest = ulong4(
        gl_canonicalize(state[0]),
        gl_canonicalize(state[1]),
        gl_canonicalize(state[2]),
        gl_canonicalize(state[3]));
    even_level[tid] = digest;
    const uint first_parent = group * 128 + tid;
    for (uint i = 0; i < 4; ++i) {
        parents[(ulong)first_parent * 4 + i] = digest[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint level = 1; level < levels; ++level) {
        const uint active = 128 >> level;
        if (tid < active) {
            ulong4 left;
            ulong4 right;
            if ((level & 1) != 0) {
                left = even_level[tid * 2];
                right = even_level[tid * 2 + 1];
            } else {
                left = odd_level[tid * 2];
                right = odd_level[tid * 2 + 1];
            }
            for (uint i = 0; i < 4; ++i) {
                state[i] = left[i];
                state[4 + i] = right[i];
            }
            for (uint i = 8; i < 12; ++i) {
                state[i] = 0;
            }
            poseidon2(state, parameters);
            digest = ulong4(
                gl_canonicalize(state[0]),
                gl_canonicalize(state[1]),
                gl_canonicalize(state[2]),
                gl_canonicalize(state[3]));
            if ((level & 1) != 0) {
                odd_level[tid] = digest;
            } else {
                even_level[tid] = digest;
            }

            const uint level_offset = child_count - (child_count >> level);
            const uint output_digest = level_offset + group * active + tid;
            for (uint i = 0; i < 4; ++i) {
                parents[(ulong)output_digest * 4 + i] = digest[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
