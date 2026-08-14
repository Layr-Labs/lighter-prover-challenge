#include "poseidon2.metal"

// Computes two consecutive binary Merkle levels in one invocation. Every
// thread owns four child digests, writes the two ordinary first-level parents
// needed by authentication paths, and keeps those parents in registers for
// the grandparent compression. Thus the mathematical tree and complete node
// layout are unchanged while the second level avoids an intermediate device
// read and a separate grid dispatch.
kernel void poseidon2_hash_parent2(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    device ulong* grandparents [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& grandparent_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= grandparent_count) {
        return;
    }

    const device ulong* input = children + (ulong)gid * 16;
    device ulong* middle = parents + (ulong)gid * 8;
    ulong left[4];
    ulong state[12] = { 0 };

    for (uint i = 0; i < 8; ++i) {
        state[i] = input[i];
    }
    poseidon2(state, parameters);
    for (uint i = 0; i < 4; ++i) {
        left[i] = gl_canonicalize(state[i]);
        middle[i] = left[i];
    }

    for (uint i = 0; i < 12; ++i) {
        state[i] = 0;
    }
    for (uint i = 0; i < 8; ++i) {
        state[i] = input[8 + i];
    }
    poseidon2(state, parameters);
    for (uint i = 0; i < 4; ++i) {
        ulong right = gl_canonicalize(state[i]);
        middle[4 + i] = right;
        state[4 + i] = right;
        state[i] = left[i];
    }
    for (uint i = 8; i < 12; ++i) {
        state[i] = 0;
    }

    poseidon2(state, parameters);
    device ulong* output = grandparents + (ulong)gid * 4;
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}
