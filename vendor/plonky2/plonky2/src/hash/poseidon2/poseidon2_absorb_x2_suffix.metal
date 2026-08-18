// ===========================================================================
// Two-groups-per-pass streamed sponge absorb (side library).
//
// This file is NOT compiled into poseidon2.metallib. The build tool
// concatenates poseidon2.metal + this suffix into one translation unit and
// compiles it to poseidon2-absorb-x2.metallib, so `poseidon2`,
// `gl_canonicalize`, `reverse_bits` and every other helper below are
// byte-for-byte the frontier definitions. A Rust test pins that relationship
// (`absorb_x2_metallib_matches_sources` in metal.rs). The frontier metallib
// and pipeline archive stay byte-identical (they are device/compiler-keyed).
//
// Why: measured GPU execution attribution puts `poseidon2_absorb_pass` at
// ~42% of all GPU execution, at ~46M permutations/s — roughly half the
// per-permutation rate of the fused single-launch tree builds — because
// every interior pass pays a full 12-lane state round trip through device
// memory and launches one permutation per thread. Absorbing two full
// eight-column groups per launch keeps the state in registers between the
// two permutations, halving the state traffic and the launch count for the
// paired portion of the build.
//
// Bit-exactness: the sequence of field operations is identical to two
// consecutive `poseidon2_absorb_pass` launches over the same groups — the
// interior store-then-reload of the state that this kernel skips is a value
// identity. Both chunks are the full eight columns by construction (the
// host pairs only full groups), so the absorb loops are the fixed-width
// specialization of the single-pass kernel's runtime-width loop.
// ===========================================================================

kernel void poseidon2_absorb_pass_x2(
    const device ulong* leaves [[buffer(0)]],
    device ulong* state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
    constant uint& first_pass [[buffer(7)]],
    constant uint& final_pass [[buffer(8)]],
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
    for (uint i = 0; i < 8u; ++i) {
        st[i] = gl_canonicalize(leaves[(ulong)(col_start + i) * leaf_count + gid]);
    }
    poseidon2(st, parameters);
    for (uint i = 0; i < 8u; ++i) {
        st[i] = gl_canonicalize(
            leaves[(ulong)(col_start + 8u + i) * leaf_count + gid]);
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

// Four full eight-column groups per launch: the same identity argument as
// the x2 kernel (each skipped interior state store/reload is a value
// identity, every chunk is the full eight columns), with three interior
// round trips and three launches removed per quad instead of one.
kernel void poseidon2_absorb_pass_x4(
    const device ulong* leaves [[buffer(0)]],
    device ulong* state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
    constant uint& first_pass [[buffer(7)]],
    constant uint& final_pass [[buffer(8)]],
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
    for (uint pass = 0; pass < 4u; ++pass) {
        uint pass_start = col_start + pass * 8u;
        for (uint i = 0; i < 8u; ++i) {
            st[i] = gl_canonicalize(
                leaves[(ulong)(pass_start + i) * leaf_count + gid]);
        }
        poseidon2(st, parameters);
    }
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
