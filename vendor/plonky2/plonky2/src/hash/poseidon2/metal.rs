use core::ffi::c_void;
use core::mem::{size_of, size_of_val};
use core::slice;
use std::sync::{LazyLock, Mutex};

use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSUInteger,
};
use objc::rc::autoreleasepool;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::poseidon2::config::{
    EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, MATRIX_DIAG_12_U64,
};

const SHADER_SOURCE: &str = include_str!("poseidon2.metal");
const MIN_GPU_PERMUTATIONS: usize = 1 << 14;

/// Target size of one staged slice of leaf input. Leaves are copied into the
/// GPU-visible buffers one slice at a time and each slice is committed as soon
/// as it is filled, so the copy of slice `k + 1` overlaps the GPU hashing of
/// slice `k` instead of running before any GPU work is queued.
const STAGE_TARGET_BYTES: usize = 8 << 20;
/// Leaves per stage are rounded down to a multiple of this so that the output
/// offset of every stage (`leaf_index * 4 * 8` bytes) stays 256-byte aligned.
const STAGE_LEAF_ALIGN: usize = 8;

/// Buffers are rounded up to a multiple of this before allocation so that
/// slightly different request sizes can share a pooled allocation.
const POOL_GRANULARITY: usize = 1 << 20;
/// Maximum number of pooled buffers kept alive between calls.
const POOL_SLOTS: usize = 48;
/// Maximum number of bytes kept alive by the pool.
const POOL_BYTES: usize = 1 << 30;

/// Minimum number of field elements before the leaf copy is spread over the
/// rayon pool.
const PARALLEL_FILL_ELEMENTS: usize = 1 << 13;
/// Minimum subtree size before the digest layout pass forks. A subtree of
/// `1 << 10` leaves lays out ~64 KiB of digests, which comfortably covers the
/// `join` overhead.
const PARALLEL_LAYOUT_LEAVES: usize = 1 << 10;

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    leaf_pipeline: ComputePipelineState,
    parent_pipeline: ComputePipelineState,
    parameters: Buffer,
    /// Recycled `StorageModeShared` buffers. Allocating (and first-touching)
    /// hundreds of megabytes of shared storage per commitment is expensive, and
    /// the sizes repeat across commitments, so buffers are handed back here
    /// once the command buffers that referenced them have completed.
    pool: Mutex<Vec<Buffer>>,
}

// `MetalContext` is shared immutably: `Device`, `CommandQueue`,
// `ComputePipelineState` and `Buffer` are declared `Send + Sync` by
// `metal`'s `foreign_obj_type!`, and command-buffer creation off a
// `MTLCommandQueue` is documented as thread safe. The only mutable state is
// the buffer pool, which has its own short-lived lock that is never held
// across GPU work. Being a `static` below, the compiler checks `Sync` for us.
static CONTEXT: LazyLock<Result<MetalContext, String>> = LazyLock::new(MetalContext::new);

pub(crate) fn build_merkle_tree<F: RichField>(
    leaves: &[Vec<F>],
    cap_height: usize,
) -> Option<(Vec<HashOut<F>>, Vec<HashOut<F>>)> {
    let leaf_count = leaves.len();
    let leaf_width = leaves.first()?.len();
    if F::ORDER != 0xffff_ffff_0000_0001
        || leaves.iter().any(|leaf| leaf.len() != leaf_width)
        || leaf_count > u32::MAX as usize
        || leaf_width > u32::MAX as usize
    {
        return None;
    }

    let leaf_permutations = if leaf_width <= 4 {
        0
    } else {
        leaf_width.div_ceil(8) * leaf_count
    };
    let parent_permutations = leaf_count - (1usize << cap_height);
    if leaf_permutations + parent_permutations < MIN_GPU_PERMUTATIONS {
        return None;
    }

    let context = match &*CONTEXT {
        Ok(context) => context,
        Err(error) => {
            log::warn!("Metal Poseidon2 unavailable; using CPU Merkle hashing: {error}");
            return None;
        }
    };

    match autoreleasepool(|| context.build(leaves, cap_height)) {
        Ok(tree) => Some(tree),
        Err(error) => {
            log::warn!("Metal Poseidon2 failed; using CPU Merkle hashing: {error}");
            None
        }
    }
}

impl MetalContext {
    fn new() -> Result<Self, String> {
        autoreleasepool(|| {
            let device = Device::system_default().ok_or("no Metal device")?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &options)
                .map_err(|error| format!("shader compilation failed: {error}"))?;
            let leaf_function = library
                .get_function("poseidon2_hash_leaves", None)
                .map_err(|error| format!("leaf kernel unavailable: {error}"))?;
            let parent_function = library
                .get_function("poseidon2_hash_parents", None)
                .map_err(|error| format!("parent kernel unavailable: {error}"))?;
            let leaf_pipeline = device
                .new_compute_pipeline_state_with_function(&leaf_function)
                .map_err(|error| format!("leaf pipeline creation failed: {error}"))?;
            let parent_pipeline = device
                .new_compute_pipeline_state_with_function(&parent_function)
                .map_err(|error| format!("parent pipeline creation failed: {error}"))?;

            let mut parameter_values = Vec::with_capacity(130);
            parameter_values.extend(EXTERNAL_CONSTANTS.into_iter().flatten());
            parameter_values.extend(INTERNAL_CONSTANTS);
            parameter_values.extend(MATRIX_DIAG_12_U64);
            debug_assert_eq!(parameter_values.len(), 130);
            let parameters = device.new_buffer_with_data(
                parameter_values.as_ptr().cast::<c_void>(),
                size_of_val(parameter_values.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            Ok(Self {
                queue: device.new_command_queue(),
                device,
                leaf_pipeline,
                parent_pipeline,
                parameters,
                pool: Mutex::new(Vec::new()),
            })
        })
    }

    /// Returns a shared-storage buffer of at least `bytes`, reusing a pooled
    /// allocation when one is available.
    fn take_buffer(&self, bytes: usize) -> Buffer {
        let bytes = bytes.max(size_of::<u64>());
        if let Ok(mut pool) = self.pool.lock() {
            let mut best: Option<(usize, u64)> = None;
            for (index, buffer) in pool.iter().enumerate() {
                let length = buffer.length();
                if length < bytes as u64 {
                    continue;
                }
                match best {
                    Some((_, best_length)) if best_length <= length => {}
                    _ => best = Some((index, length)),
                }
            }
            if let Some((index, _)) = best {
                return pool.swap_remove(index);
            }
        }
        let length = bytes.next_multiple_of(POOL_GRANULARITY);
        self.device
            .new_buffer(length as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Hands a buffer back to the pool. The caller must have observed the
    /// command buffers referencing it complete.
    fn return_buffer(&self, buffer: Buffer) {
        let Ok(mut pool) = self.pool.lock() else {
            return;
        };
        pool.push(buffer);
        loop {
            let total: u64 = pool.iter().map(|buffer| buffer.length()).sum();
            if pool.len() <= POOL_SLOTS && total <= POOL_BYTES as u64 {
                break;
            }
            let smallest = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, buffer)| buffer.length())
                .map(|(index, _)| index);
            match smallest {
                Some(index) => {
                    pool.swap_remove(index);
                }
                None => break,
            }
        }
    }

    fn build<F: RichField>(
        &self,
        leaves: &[Vec<F>],
        cap_height: usize,
    ) -> Result<(Vec<HashOut<F>>, Vec<HashOut<F>>), String> {
        let leaf_count = leaves.len();
        let leaf_width = leaves[0].len();
        let cap_count = 1usize << cap_height;
        let total_node_count = 2 * leaf_count - cap_count;

        let output_len = total_node_count
            .checked_mul(4)
            .ok_or("Metal Merkle output length overflow")?;
        let output_bytes = output_len
            .checked_mul(size_of::<u64>())
            .ok_or("Metal Merkle output size overflow")?;
        let leaf_bytes = leaf_width
            .checked_mul(size_of::<u64>())
            .ok_or("Metal leaf input size overflow")?;
        // Guards the per-stage `stage_len * leaf_bytes` products below.
        if leaf_count.checked_mul(leaf_bytes).is_none() {
            return Err("Metal leaf input size overflow".to_owned());
        }

        let output_buffer = self.take_buffer(output_bytes);
        let mut staged: Vec<Buffer> = Vec::new();
        let mut command_buffers: Vec<&CommandBufferRef> = Vec::new();

        // Leaf hashing, staged so that the host copy of one slice overlaps the
        // GPU hashing of the previous one. Command buffers submitted to a
        // single `MTLCommandQueue` execute in commit order, so the parent
        // passes below still observe every leaf digest.
        let stage_leaves = if leaf_bytes == 0 {
            leaf_count.max(1)
        } else {
            (STAGE_TARGET_BYTES / leaf_bytes / STAGE_LEAF_ALIGN * STAGE_LEAF_ALIGN)
                .max(STAGE_LEAF_ALIGN)
        };

        let leaf_width_u32 = leaf_width as u32;
        let mut start = 0usize;
        while start < leaf_count {
            let end = (start + stage_leaves).min(leaf_count);
            let stage_len = end - start;
            let input_buffer = self.take_buffer(stage_len * leaf_bytes);
            if leaf_bytes != 0 {
                let input = unsafe {
                    slice::from_raw_parts_mut(
                        input_buffer.contents().cast::<u64>(),
                        stage_len * leaf_width,
                    )
                };
                fill_input(input, &leaves[start..end], leaf_width);
            }

            let stage_len_u32 = stage_len as u32;
            let command_buffer = self.queue.new_command_buffer();
            let leaf_encoder = command_buffer.new_compute_command_encoder();
            leaf_encoder.set_compute_pipeline_state(&self.leaf_pipeline);
            leaf_encoder.set_buffer(0, Some(&input_buffer), 0);
            leaf_encoder.set_buffer(
                1,
                Some(&output_buffer),
                (start * 4 * size_of::<u64>()) as NSUInteger,
            );
            leaf_encoder.set_buffer(2, Some(&self.parameters), 0);
            leaf_encoder.set_bytes(
                3,
                size_of::<u32>() as NSUInteger,
                (&leaf_width_u32 as *const u32).cast::<c_void>(),
            );
            leaf_encoder.set_bytes(
                4,
                size_of::<u32>() as NSUInteger,
                (&stage_len_u32 as *const u32).cast::<c_void>(),
            );
            dispatch(&leaf_encoder, &self.leaf_pipeline, stage_len);
            leaf_encoder.end_encoding();
            command_buffer.commit();

            staged.push(input_buffer);
            command_buffers.push(command_buffer);
            start = end;
        }

        let mut level_offsets = Vec::with_capacity(leaf_count.ilog2() as usize + 1);
        let mut level_offset = 0usize;
        let mut child_count = leaf_count;
        level_offsets.push(level_offset);
        let command_buffer = self.queue.new_command_buffer();
        while child_count > cap_count {
            let parent_count = child_count / 2;
            let child_offset = level_offset;
            level_offset += child_count * 4;
            level_offsets.push(level_offset);

            let parent_count_u32 = parent_count as u32;
            let parent_encoder = command_buffer.new_compute_command_encoder();
            parent_encoder.set_compute_pipeline_state(&self.parent_pipeline);
            parent_encoder.set_buffer(
                0,
                Some(&output_buffer),
                (child_offset * size_of::<u64>()) as NSUInteger,
            );
            parent_encoder.set_buffer(
                1,
                Some(&output_buffer),
                (level_offset * size_of::<u64>()) as NSUInteger,
            );
            parent_encoder.set_buffer(2, Some(&self.parameters), 0);
            parent_encoder.set_bytes(
                3,
                size_of::<u32>() as NSUInteger,
                (&parent_count_u32 as *const u32).cast::<c_void>(),
            );
            dispatch(&parent_encoder, &self.parent_pipeline, parent_count);
            parent_encoder.end_encoding();

            child_count = parent_count;
        }

        command_buffer.commit();
        command_buffers.push(command_buffer);
        command_buffer.wait_until_completed();
        for command_buffer in &command_buffers {
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                // The buffers are deliberately dropped rather than pooled: on
                // an aborted submission it is not worth reasoning about which
                // of them the driver may still touch.
                return Err(format!("command buffer ended with status {status:?}"));
            }
        }

        for buffer in staged {
            self.return_buffer(buffer);
        }

        let tree = {
            let nodes = unsafe {
                slice::from_raw_parts(output_buffer.contents().cast::<u64>(), output_len)
            };
            tree_from_levels(nodes, &level_offsets, leaf_count, cap_height)
        };
        self.return_buffer(output_buffer);
        Ok(tree)
    }
}

fn fill_input<F: RichField>(input: &mut [u64], leaves: &[Vec<F>], leaf_width: usize) {
    if input.len() >= PARALLEL_FILL_ELEMENTS {
        input
            .par_chunks_mut(leaf_width)
            .zip(leaves.par_iter())
            .for_each(|(destination, leaf)| {
                for (destination, value) in destination.iter_mut().zip(leaf.iter()) {
                    *destination = value.to_noncanonical_u64();
                }
            });
    } else {
        for (destination, value) in input.iter_mut().zip(leaves.iter().flatten()) {
            *destination = value.to_noncanonical_u64();
        }
    }
}

fn dispatch(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    thread_count: usize,
) {
    let execution_width = pipeline.thread_execution_width();
    let group_width = pipeline
        .max_total_threads_per_threadgroup()
        .min(64)
        .max(execution_width);
    encoder.dispatch_threads(
        MTLSize {
            width: thread_count as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    );
}

fn tree_from_levels<F: RichField>(
    nodes: &[u64],
    level_offsets: &[usize],
    leaf_count: usize,
    cap_height: usize,
) -> (Vec<HashOut<F>>, Vec<HashOut<F>>) {
    let cap_count = 1usize << cap_height;
    let subtree_leaf_count = leaf_count / cap_count;
    let subtree_digest_count = 2 * (subtree_leaf_count - 1);
    let mut digests = vec![HashOut::ZERO; 2 * (leaf_count - cap_count)];
    let mut cap = vec![HashOut::ZERO; cap_count];

    if subtree_digest_count == 0 {
        for (cap_index, root) in cap.iter_mut().enumerate() {
            *root = read_node(nodes, level_offsets[0], cap_index);
        }
    } else {
        digests
            .par_chunks_mut(subtree_digest_count)
            .zip(cap.par_iter_mut())
            .enumerate()
            .for_each(|(cap_index, (digests, root))| {
                *root = fill_subtree_layout(
                    digests,
                    nodes,
                    level_offsets,
                    cap_index * subtree_leaf_count,
                    subtree_leaf_count,
                );
            });
    }
    (digests, cap)
}

fn fill_subtree_layout<F: RichField>(
    digests: &mut [HashOut<F>],
    nodes: &[u64],
    level_offsets: &[usize],
    start_leaf: usize,
    leaf_count: usize,
) -> HashOut<F> {
    if leaf_count == 1 {
        return read_node(nodes, level_offsets[0], start_leaf);
    }

    let (left_half, right_half) = digests.split_at_mut(digests.len() / 2);
    let (left_root, left_digests) = left_half.split_last_mut().unwrap();
    let (right_root, right_digests) = right_half.split_first_mut().unwrap();
    let half = leaf_count / 2;
    let (left_value, right_value) = if leaf_count >= PARALLEL_LAYOUT_LEAVES {
        plonky2_maybe_rayon::join(
            || fill_subtree_layout(left_digests, nodes, level_offsets, start_leaf, half),
            || fill_subtree_layout(right_digests, nodes, level_offsets, start_leaf + half, half),
        )
    } else {
        (
            fill_subtree_layout(left_digests, nodes, level_offsets, start_leaf, half),
            fill_subtree_layout(right_digests, nodes, level_offsets, start_leaf + half, half),
        )
    };
    *left_root = left_value;
    *right_root = right_value;

    let level = leaf_count.ilog2() as usize;
    read_node(nodes, level_offsets[level], start_leaf / leaf_count)
}

fn read_node<F: RichField>(nodes: &[u64], level_offset: usize, index: usize) -> HashOut<F> {
    let offset = level_offset + index * 4;
    HashOut {
        elements: core::array::from_fn(|i| F::from_canonical_u64(nodes[offset + i])),
    }
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, PrimeField64};
    use crate::hash::merkle_tree::{capacity_up_to_mut, fill_digests_buf};
    use crate::hash::poseidon2::hash::Poseidon2Hash;

    fn context() -> &'static MetalContext {
        CONTEXT
            .as_ref()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn sample_leaves(leaf_count: usize, width: usize, seed: u64) -> Vec<Vec<GoldilocksField>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..leaf_count)
            .map(|leaf| {
                (0..width)
                    .map(|column| {
                        let raw = match (leaf * width + column) & 7 {
                            0 => 0,
                            1 => 1,
                            2 => GoldilocksField::ORDER - 1,
                            3 => GoldilocksField::ORDER,
                            4 => GoldilocksField::ORDER + 1,
                            5 => u64::MAX,
                            _ => rng.next_u64(),
                        };
                        GoldilocksField(raw)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn metal_merkle_matches_cpu_across_sponge_boundaries() {
        for width in [0, 1, 4, 5, 8, 9, 16, 17, 31, 64, 137] {
            let leaves = sample_leaves(64, width, 0x4d45_5441_4c32);
            for cap_height in [0, 3, 6] {
                let gpu = autoreleasepool(|| context().build(&leaves, cap_height)).unwrap();
                let cpu = cpu_tree(&leaves, cap_height);
                assert_tree_eq(&gpu, &cpu, width, cap_height);
            }
        }
    }

    /// Exercises the multi-stage leaf path, where the leaf input spans several
    /// staged command buffers and every stage after the first writes at a
    /// non-zero offset into the shared output buffer.
    #[test]
    fn metal_merkle_matches_cpu_across_input_stages() {
        // 32 x 65536 = 16 MiB, two stages, very wide leaves.
        for (leaf_count, width, cap_heights) in [
            (32usize, 65_536usize, &[0usize, 2][..]),
            // 8192 x 200 = 12.5 MiB, two stages with a realistic aspect ratio
            // and thirteen parent levels.
            (8_192, 200, &[0, 4][..]),
        ] {
            let leaves = sample_leaves(leaf_count, width, 0x5354_4147_45);
            for &cap_height in cap_heights {
                let gpu = autoreleasepool(|| context().build(&leaves, cap_height)).unwrap();
                let cpu = cpu_tree(&leaves, cap_height);
                assert_tree_eq(&gpu, &cpu, width, cap_height);
            }
        }
    }

    /// The context is shared without a lock, so concurrent commitments must
    /// each get their own command buffers and their own pooled staging
    /// buffers.
    #[test]
    fn metal_merkle_is_concurrency_safe() {
        let shapes: Vec<(usize, usize, usize)> = vec![
            (1024, 33, 0),
            (512, 137, 3),
            (2048, 8, 4),
            (256, 4097, 2),
            (128, 1, 1),
            (4096, 12, 6),
        ];
        let expected: Vec<_> = shapes
            .iter()
            .map(|&(leaf_count, width, cap_height)| {
                let leaves = sample_leaves(leaf_count, width, (leaf_count * width) as u64);
                let cpu = cpu_tree(&leaves, cap_height);
                (leaves, cap_height, width, cpu)
            })
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let expected = &expected;
                scope.spawn(move || {
                    for _ in 0..3 {
                        for (leaves, cap_height, width, cpu) in expected {
                            let gpu =
                                autoreleasepool(|| context().build(leaves, *cap_height)).unwrap();
                            assert_tree_eq(&gpu, cpu, *width, *cap_height);
                        }
                    }
                });
            }
        });
    }

    fn cpu_tree(
        leaves: &[Vec<GoldilocksField>],
        cap_height: usize,
    ) -> (
        Vec<HashOut<GoldilocksField>>,
        Vec<HashOut<GoldilocksField>>,
    ) {
        let cap_len = 1 << cap_height;
        let digest_len = 2 * (leaves.len() - cap_len);
        let mut digests = Vec::with_capacity(digest_len);
        let mut cap = Vec::with_capacity(cap_len);
        let digest_buffer: &mut [MaybeUninit<HashOut<GoldilocksField>>] =
            capacity_up_to_mut(&mut digests, digest_len);
        let cap_buffer: &mut [MaybeUninit<HashOut<GoldilocksField>>] =
            capacity_up_to_mut(&mut cap, cap_len);
        fill_digests_buf::<GoldilocksField, Poseidon2Hash>(
            digest_buffer,
            cap_buffer,
            leaves,
            cap_height,
        );
        unsafe {
            digests.set_len(digest_len);
            cap.set_len(cap_len);
        }
        (digests, cap)
    }

    fn assert_tree_eq(
        actual: &(
            Vec<HashOut<GoldilocksField>>,
            Vec<HashOut<GoldilocksField>>,
        ),
        expected: &(
            Vec<HashOut<GoldilocksField>>,
            Vec<HashOut<GoldilocksField>>,
        ),
        width: usize,
        cap_height: usize,
    ) {
        assert_eq!(actual.0.len(), expected.0.len());
        assert_eq!(actual.1.len(), expected.1.len());
        for (index, (actual, expected)) in actual
            .0
            .iter()
            .chain(&actual.1)
            .zip(expected.0.iter().chain(&expected.1))
            .enumerate()
        {
            let actual = actual.elements.map(|value| value.to_canonical_u64());
            let expected = expected.elements.map(|value| value.to_canonical_u64());
            assert_eq!(
                actual, expected,
                "width {width}, cap height {cap_height}, node {index}"
            );
        }
    }
}
