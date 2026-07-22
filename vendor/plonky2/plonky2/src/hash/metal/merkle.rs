//! GPU Merkle tree construction using Metal Poseidon2 shaders.
//!
//! The GPU computes in BFS layout. This module converts the output
//! to the in-order layout used by elliottech's plonky2.

use metal::*;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::HashOut;
use crate::hash::metal::runtime::MetalRuntime;
use crate::hash::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::hash::metal::utils::{from_buf_raw, get_size_for_count, LinearUniforms};
use crate::util::log2_strict;

/// Convert BFS-layout digests from the GPU to in-order layout used by plonky2.
///
/// GPU BFS layout for a subtree with n leaves (subtree_digests_len = 2*(n-1)):
///   Level d-1 (root's children, 2 nodes): positions [0, 1]
///   Level d-2 (4 nodes): positions [2, 5]
///   ...
///   Level 0 (n leaves): positions [n-2, 2n-3]
///
/// In-order layout (recursive interleaved):
///   [left_subtree || left_digest || right_digest || right_subtree]
fn convert_bfs_to_inorder(
    bfs_digests: &[HashOut<GoldilocksField>],
    subtree_leaves_len: usize,
    num_subtrees: usize,
) -> Vec<HashOut<GoldilocksField>> {
    let subtree_digests_len = 2 * (subtree_leaves_len - 1);
    let total_digests = subtree_digests_len * num_subtrees;
    let mut inorder = vec![HashOut::<GoldilocksField>::default(); total_digests];
    let d = log2_strict(subtree_leaves_len);

    for subtree_idx in 0..num_subtrees {
        let bfs_base = subtree_idx * subtree_digests_len;
        let inorder_base = subtree_idx * subtree_digests_len;

        for level in 0..d {
            let nodes_at_level = subtree_leaves_len >> level;
            let bfs_start = if level == 0 {
                subtree_leaves_len - 2
            } else {
                (subtree_leaves_len >> level) - 2
            };

            for j in 0..nodes_at_level {
                let bfs_pos = bfs_base + bfs_start + j;
                let inorder_pos = inorder_base + compute_inorder_pos(d, level, j);
                inorder[inorder_pos] = bfs_digests[bfs_pos];
            }
        }
    }

    inorder
}

/// Compute the in-order position of a node at (level, index) in a subtree of depth d.
///
/// For a subtree with n = 2^d leaves:
///   - Total digests: S(d) = 2*(n-1) = 2^{d+1} - 2
///   - Left subtree occupies [0, S(d-1)-1]
///   - Left digest at position S(d)/2 - 1 = 2^d - 2
///   - Right digest at position S(d)/2 = 2^d - 1
///   - Right subtree occupies [S(d)/2 + 1, S(d)-1]
fn compute_inorder_pos(d: usize, level: usize, index: usize) -> usize {
    if d == 1 {
        // 2 leaves → positions [0, 1]
        debug_assert!(level == 0);
        return index;
    }

    if level == d - 1 {
        // Root's children (2 nodes)
        return (1 << d) - 2 + index;
    }

    let half_nodes = 1 << (d - 1 - level);
    let left_subtree_size = 2 * ((1 << (d - 1)) - 1); // S(d-1)

    if index < half_nodes {
        // In left subtree
        compute_inorder_pos(d - 1, level, index)
    } else {
        // In right subtree: offset = S(d-1) + 2 (left subtree + 2 parent slots)
        left_subtree_size + 2 + compute_inorder_pos(d - 1, level, index - half_nodes)
    }
}

impl MetalRuntime {
    /// Hash merkle tree using Poseidon2 with linear layout + threadgroup optimization.
    /// Returns digests in BFS layout (to be converted by caller or internally).
    pub fn hash_merkle_tree_poseidon2_linear_threadgroup_buf(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>) {
        let leaf_count = 1usize << tree_height;

        assert!(
            cap_height < tree_height,
            "cap height must be less than tree height"
        );

        let num_caps = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        // GPU computes in BFS layout with subtree_digests_len = 2*(n-1)
        let subtree_digests_len = 2 * (subtree_leaves_len - 1);
        let total_digests = subtree_digests_len * num_caps;
        let num_layers = tree_height - cap_height;

        let buffer_size = (total_digests * 4) * std::mem::size_of::<u64>();
        let (digests_buffer, digests_from_pool) =
            self.alloc_aligned_tracked_with_pool_hint(buffer_size);

        if !digests_from_pool {
            unsafe {
                let ptr = digests_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(buffer_size));
            }
        }

        let caps_size = (num_caps * 4) * std::mem::size_of::<u64>();
        let (caps_buffer, caps_from_pool) = self.alloc_aligned_tracked_with_pool_hint(caps_size);
        if !caps_from_pool {
            unsafe {
                let ptr = caps_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(caps_size));
            }
        }

        let pipeline_hash_leaves =
            self.get_poseidon2_hash_leaves_linear_threadgroup_pipeline_state();
        let pipeline_hash_tree_level =
            self.get_poseidon2_hash_tree_level_linear_threadgroup_pipeline_state();
        let pipeline_hash_caps = self.get_poseidon2_hash_caps_linear_threadgroup_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();

        let mut uniforms = LinearUniforms {
            level: 0,
            subtree_digests_len: subtree_digests_len as u32,
            subtree_leaves_len: subtree_leaves_len as u32,
            leaf_size: leaf_length as u32,
            leaf_count: leaf_count as u32,
            subtree_count: num_caps as u32,
            grid_width: 0,
        };

        // Hash all leaves
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads = pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
            let num_threads =
                get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (leaf_count as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.set_threadgroup_memory_length(0, 0);
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Hash internal tree levels
        for level in 1..num_layers {
            uniforms.level = level as u32;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);
            encoder.set_buffer(0, Some(&*digests_buffer), 0);

            let nodes_at_this_level = subtree_leaves_len >> level;
            let total_nodes = nodes_at_this_level * num_caps;
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads = pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads =
                get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (total_nodes as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            encoder.set_bytes(
                1,
                std::mem::size_of::<LinearUniforms>() as u64,
                &uniforms as *const LinearUniforms as *const core::ffi::c_void,
            );

            let child_cache_size = (num_threads as usize) * 8 * std::mem::size_of::<u64>();
            encoder.set_threadgroup_memory_length(0, child_cache_size as u64);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Compute cap hashes
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_caps);
            encoder.set_buffer(0, Some(&*caps_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            let simd_width = pipeline_hash_caps.thread_execution_width() as usize;
            let max_threads = pipeline_hash_caps.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(num_caps, simd_width, max_threads) as u64;
            let lg = (num_caps as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = MTLSize {
                width: lg.max(1),
                height: 1,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: num_threads.min(num_caps as u64),
                height: 1,
                depth: 1,
            };

            encoder.set_threadgroup_memory_length(0, 0);
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // GPU produced BFS layout — convert to in-order for plonky2
        let digests_ptr = (*digests_buffer).contents() as *const HashOut<GoldilocksField>;
        let bfs_digests =
            unsafe { from_buf_raw::<HashOut<GoldilocksField>>(digests_ptr, total_digests) };
        let digests = convert_bfs_to_inorder(&bfs_digests, subtree_leaves_len, num_caps);

        let caps_ptr = (*caps_buffer).contents() as *mut HashOut<GoldilocksField>;
        let caps = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, num_caps) };

        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

        (digests, caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inorder_pos_depth2() {
        // 4 leaves, 6 digests
        // In-order: [h0, h1, A=hash(h0,h1), B=hash(h2,h3), h2, h3]
        // BFS: level 1 (2 nodes) at [0,1], level 0 (4 leaves) at [2..5]
        assert_eq!(compute_inorder_pos(2, 1, 0), 2); // A
        assert_eq!(compute_inorder_pos(2, 1, 1), 3); // B
        assert_eq!(compute_inorder_pos(2, 0, 0), 0); // h0
        assert_eq!(compute_inorder_pos(2, 0, 1), 1); // h1
        assert_eq!(compute_inorder_pos(2, 0, 2), 4); // h2
        assert_eq!(compute_inorder_pos(2, 0, 3), 5); // h3
    }

    #[test]
    fn test_inorder_pos_depth3() {
        // 8 leaves, 14 digests
        // In-order: [h0, h1, A, B, h2, h3, C, D, h4, h5, E, F, h6, h7]
        assert_eq!(compute_inorder_pos(3, 2, 0), 6); // C
        assert_eq!(compute_inorder_pos(3, 2, 1), 7); // D
        assert_eq!(compute_inorder_pos(3, 1, 0), 2); // A
        assert_eq!(compute_inorder_pos(3, 1, 1), 3); // B
        assert_eq!(compute_inorder_pos(3, 1, 2), 10); // E
        assert_eq!(compute_inorder_pos(3, 1, 3), 11); // F
        assert_eq!(compute_inorder_pos(3, 0, 0), 0); // h0
        assert_eq!(compute_inorder_pos(3, 0, 4), 8); // h4
        assert_eq!(compute_inorder_pos(3, 0, 7), 13); // h7
    }
}
