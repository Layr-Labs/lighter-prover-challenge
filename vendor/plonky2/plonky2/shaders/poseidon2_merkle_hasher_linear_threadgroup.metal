// Linear indexing Merkle tree hashing with Poseidon2
// Uses poseidon2_permute() instead of poseidon_permute_const().
// Identical tree structure and memory layout as the Poseidon variant.

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon2_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;

// Uniforms (same struct as Poseidon variant)
struct LinearThreadgroupUniforms {
    uint level;
    uint subtree_digests_len;
    uint subtree_leaves_len;
    uint leaf_size;
    uint leaf_count;
    uint subtree_count;
    uint grid_width;
};

inline uint compute_linear_leaf_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint in_subtree_idx
) {
    return subtree_idx * subtree_digests_len
         + (subtree_digests_len - subtree_leaves_len)
         + in_subtree_idx;
}

inline uint compute_linear_internal_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint level,
    uint index_in_level
) {
    uint level_start = subtree_digests_len - 2 * subtree_leaves_len + (subtree_leaves_len >> level);
    return subtree_idx * subtree_digests_len + level_start + index_in_level;
}

// Leaf hashing kernel using Poseidon2
kernel void poseidon2_hash_leaves_linear_threadgroup(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    uint subtree_idx = thread_id / uniforms.subtree_leaves_len;
    uint in_subtree_idx = thread_id % uniforms.subtree_leaves_len;

    uint digest_idx = compute_linear_leaf_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint input_offset = thread_id * uniforms.leaf_size;
    uint output_offset = digest_idx * 4;

    // hash_or_noop logic
    if (uniforms.leaf_size <= 4) {
        for (uint i = 0; i < 4; i++) {
            if (i < uniforms.leaf_size) {
                Fp val = leaf_inputs[input_offset + i];
                output[output_offset + i] = static_cast<ulong>(val);
            } else {
                output[output_offset + i] = 0;
            }
        }
    } else {
        Fp p2_state[12];

        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint offset = input_offset;
        uint num_full_rounds = uniforms.leaf_size / 8;

        for (uint i = 0; i < num_full_rounds; i++) {
            p2_state[0] = leaf_inputs[offset];
            p2_state[1] = leaf_inputs[offset + 1];
            p2_state[2] = leaf_inputs[offset + 2];
            p2_state[3] = leaf_inputs[offset + 3];

            p2_state[4] = leaf_inputs[offset + 4];
            p2_state[5] = leaf_inputs[offset + 5];
            p2_state[6] = leaf_inputs[offset + 6];
            p2_state[7] = leaf_inputs[offset + 7];

            poseidon2_permute(p2_state);
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            poseidon2_permute(p2_state);
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Internal tree level hashing kernel using Poseidon2
kernel void poseidon2_hash_tree_level_linear_threadgroup(
    device ulong * output[[buffer(0)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(1)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];

    threadgroup ulong * shared_children = tg_memory;

    uint nodes_per_subtree = uniforms.subtree_leaves_len >> uniforms.level;
    uint total_nodes = nodes_per_subtree * uniforms.subtree_count;

    if (thread_id >= total_nodes) {
        return;
    }

    uint subtree_idx = thread_id / nodes_per_subtree;
    uint index_in_level = thread_id % nodes_per_subtree;

    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;
    uint right_child_in_level = index_in_level * 2 + 1;

    uint left_child_idx, right_child_idx;

    if (child_level == 0) {
        left_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            left_child_in_level
        );
        right_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            right_child_in_level
        );
    } else {
        left_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            left_child_in_level
        );
        right_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            right_child_in_level
        );
    }

    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;
    uint shared_base = local_id * 8;

    shared_children[shared_base + 0] = output[left_offset + 0];
    shared_children[shared_base + 1] = output[left_offset + 1];
    shared_children[shared_base + 2] = output[left_offset + 2];
    shared_children[shared_base + 3] = output[left_offset + 3];
    shared_children[shared_base + 4] = output[right_offset + 0];
    shared_children[shared_base + 5] = output[right_offset + 1];
    shared_children[shared_base + 6] = output[right_offset + 2];
    shared_children[shared_base + 7] = output[right_offset + 3];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    Fp p2_state[12];
    p2_state[0] = Fp(shared_children[shared_base + 0]);
    p2_state[1] = Fp(shared_children[shared_base + 1]);
    p2_state[2] = Fp(shared_children[shared_base + 2]);
    p2_state[3] = Fp(shared_children[shared_base + 3]);
    p2_state[4] = Fp(shared_children[shared_base + 4]);
    p2_state[5] = Fp(shared_children[shared_base + 5]);
    p2_state[6] = Fp(shared_children[shared_base + 6]);
    p2_state[7] = Fp(shared_children[shared_base + 7]);
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    poseidon2_permute(p2_state);

    uint parent_idx = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        uniforms.level,
        index_in_level
    );

    uint parent_offset = parent_idx * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}

// Cap hashing kernel using Poseidon2
kernel void poseidon2_hash_caps_linear_threadgroup(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.subtree_count) {
        return;
    }

    uint subtree_base = gid * uniforms.subtree_digests_len;

    uint left_idx = subtree_base;
    uint right_idx = subtree_base + 1;

    Fp p2_state[12];

    p2_state[0] = Fp(digests[left_idx * 4]);
    p2_state[1] = Fp(digests[left_idx * 4 + 1]);
    p2_state[2] = Fp(digests[left_idx * 4 + 2]);
    p2_state[3] = Fp(digests[left_idx * 4 + 3]);

    p2_state[4] = Fp(digests[right_idx * 4]);
    p2_state[5] = Fp(digests[right_idx * 4 + 1]);
    p2_state[6] = Fp(digests[right_idx * 4 + 2]);
    p2_state[7] = Fp(digests[right_idx * 4 + 3]);

    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    poseidon2_permute(p2_state);

    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
