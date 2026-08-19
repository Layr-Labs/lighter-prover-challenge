// Compiled together with poseidon2.metal so this kernel can reuse the exact
// Goldilocks and Poseidon2 primitives from the promoted shader.
kernel void poseidon2_absorb_final_x2_parent(
    const device ulong* leaves [[buffer(0)]],
    const device ulong* parked_state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
    constant uint& chunk_size [[buffer(7)]],
    constant uint& first_pass [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint parent_count = leaf_count >> 1;
    if (gid >= parent_count || log_leaf_count == 0u) {
        return;
    }

    // Adjacent children in the committed bit-reversed leaf level originate
    // from natural rows separated by half the domain.
    uint natural_0 = log_leaf_count == 1u
        ? 0u
        : (reverse_bits(gid) >> (32u - (log_leaf_count - 1u)));
    uint natural_1 = natural_0 + parent_count;

    ulong state_0[12] = { 0 };
    ulong state_1[12] = { 0 };
    if (first_pass == 0u) {
        for (uint i = 0; i < 12; ++i) {
            state_0[i] = parked_state[(ulong)i * leaf_count + natural_0];
            state_1[i] = parked_state[(ulong)i * leaf_count + natural_1];
        }
    }
    for (uint i = 0; i < chunk_size; ++i) {
        ulong column = (ulong)(col_start + i) * leaf_count;
        state_0[i] = gl_canonicalize(leaves[column + natural_0]);
        state_1[i] = gl_canonicalize(leaves[column + natural_1]);
    }
    poseidon2(state_0, parameters);
    poseidon2(state_1, parameters);

    ulong digest_0[4];
    ulong digest_1[4];
    device ulong* leaf_0 = hashes + (ulong)(2u * gid) * 4u;
    device ulong* leaf_1 = leaf_0 + 4u;
    for (uint i = 0; i < 4; ++i) {
        digest_0[i] = gl_canonicalize(state_0[i]);
        digest_1[i] = gl_canonicalize(state_1[i]);
        leaf_0[i] = digest_0[i];
        leaf_1[i] = digest_1[i];
    }

    ulong parent_state[12] = { 0 };
    for (uint i = 0; i < 4; ++i) {
        parent_state[i] = digest_0[i];
        parent_state[i + 4] = digest_1[i];
    }
    poseidon2(parent_state, parameters);
    device ulong* parent = hashes + ((ulong)leaf_count + gid) * 4u;
    for (uint i = 0; i < 4; ++i) {
        parent[i] = gl_canonicalize(parent_state[i]);
    }
}

// Folds the live coefficient prefix of one Goldilocks quadratic-extension
// polynomial by arity 16. Input is extension-major `[c0.0, c0.1, c1.0, ...]`;
// output is two base-field columns so `ntt_prepare` can consume it directly.
kernel void fri_fold_ext2_arity16(
    const device ulong* coeffs [[buffer(0)]],
    device ulong* folded [[buffer(1)]],
    constant ulong* beta [[buffer(2)]],
    constant uint& live_chunks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= live_chunks) {
        return;
    }

    ulong acc_0 = coeffs[((ulong)gid * 16u + 15u) * 2u];
    ulong acc_1 = coeffs[((ulong)gid * 16u + 15u) * 2u + 1u];
    for (uint i = 15u; i-- > 0u;) {
        ulong product_0 = gl_add(
            gl_mul(acc_0, beta[0]),
            gl_mul(7UL, gl_mul(acc_1, beta[1])));
        ulong product_1 = gl_add(
            gl_mul(acc_0, beta[1]),
            gl_mul(acc_1, beta[0]));
        ulong offset = ((ulong)gid * 16u + i) * 2u;
        acc_0 = gl_add(coeffs[offset], product_0);
        acc_1 = gl_add(coeffs[offset + 1u], product_1);
    }
    folded[gid] = acc_0;
    folded[(ulong)live_chunks + gid] = acc_1;
}

// Reinterprets two natural-order extension-evaluation columns as the 32
// natural-order columns of arity-16 FRI leaves. Merkle's column-major shader
// performs the remaining row bit reversal. No arithmetic and no CPU round trip.
kernel void fri_reorder_ext2_arity16(
    const device ulong* evaluations [[buffer(0)]],
    device ulong* leaves [[buffer(1)]],
    constant uint& evaluation_count [[buffer(2)]],
    constant uint& leaf_count [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint column = gid.y;
    if (row >= leaf_count || column >= 32u) {
        return;
    }
    uint value_in_leaf = column >> 1u;
    uint limb = column & 1u;
    uint segment = reverse_bits(value_in_leaf) >> 28u;
    leaves[(ulong)column * leaf_count + row] =
        evaluations[(ulong)limb * evaluation_count +
                    (ulong)segment * leaf_count + row];
}
