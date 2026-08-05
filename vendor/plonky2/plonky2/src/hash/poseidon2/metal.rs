use core::ffi::c_void;
use core::mem::{size_of, size_of_val};
use core::slice;
use std::sync::{LazyLock, Mutex};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize, NSUInteger,
};
use objc::rc::autoreleasepool;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::poseidon2::config::{
    EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, MATRIX_DIAG_12_U64,
};

const SHADER_SOURCE: &str = include_str!("poseidon2.metal");
const MIN_GPU_PERMUTATIONS: usize = 1 << 14;

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    leaf_pipeline: ComputePipelineState,
    parent_pipeline: ComputePipelineState,
    parameters: Buffer,
    input_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
}

static CONTEXT: LazyLock<Result<Mutex<MetalContext>, String>> =
    LazyLock::new(MetalContext::new);

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
    let mut context = match context.lock() {
        Ok(context) => context,
        Err(_) => {
            log::warn!("Metal Poseidon2 context lock poisoned; using CPU Merkle hashing");
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
    fn new() -> Result<Mutex<Self>, String> {
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

            Ok(Mutex::new(Self {
                queue: device.new_command_queue(),
                device,
                leaf_pipeline,
                parent_pipeline,
                parameters,
                input_buffer: None,
                output_buffer: None,
            }))
        })
    }

    fn build<F: RichField>(
        &mut self,
        leaves: &[Vec<F>],
        cap_height: usize,
    ) -> Result<(Vec<HashOut<F>>, Vec<HashOut<F>>), String> {
        let leaf_count = leaves.len();
        let leaf_width = leaves[0].len();
        let cap_count = 1usize << cap_height;
        let total_node_count = 2 * leaf_count - cap_count;

        let input_len = leaf_count
            .checked_mul(leaf_width)
            .ok_or("Metal leaf input length overflow")?;
        let input_bytes = input_len
            .checked_mul(size_of::<u64>())
            .ok_or("Metal leaf input size overflow")?;
        let input_buffer_bytes = input_bytes.max(size_of::<u64>()) as u64;
        if self
            .input_buffer
            .as_ref()
            .map_or(true, |buffer| buffer.length() < input_buffer_bytes)
        {
            self.input_buffer = Some(self.device.new_buffer(
                input_buffer_bytes,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        let input_buffer = self.input_buffer.as_ref().unwrap();
        let input = unsafe {
            slice::from_raw_parts_mut(input_buffer.contents().cast::<u64>(), input_len)
        };
        if leaf_width != 0 {
            input
                .par_chunks_mut(leaf_width)
                .zip(leaves.par_iter())
                .for_each(|(destination, leaf)| {
                    destination
                        .iter_mut()
                        .zip(leaf)
                        .for_each(|(destination, value)| {
                            *destination = value.to_noncanonical_u64();
                        });
                });
        }

        let output_len = total_node_count
            .checked_mul(4)
            .ok_or("Metal Merkle output length overflow")?;
        let output_bytes = output_len
            .checked_mul(size_of::<u64>())
            .ok_or("Metal Merkle output size overflow")?;
        let output_buffer_bytes = output_bytes as u64;
        if self
            .output_buffer
            .as_ref()
            .map_or(true, |buffer| buffer.length() < output_buffer_bytes)
        {
            self.output_buffer = Some(self.device.new_buffer(
                output_buffer_bytes,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        let output_buffer = self.output_buffer.as_ref().unwrap();

        let leaf_count_u32 = leaf_count as u32;
        let leaf_width_u32 = leaf_width as u32;
        let command_buffer = self.queue.new_command_buffer();
        let leaf_encoder = command_buffer.new_compute_command_encoder();
        leaf_encoder.set_compute_pipeline_state(&self.leaf_pipeline);
        leaf_encoder.set_buffer(0, Some(&input_buffer), 0);
        leaf_encoder.set_buffer(1, Some(&output_buffer), 0);
        leaf_encoder.set_buffer(2, Some(&self.parameters), 0);
        leaf_encoder.set_bytes(
            3,
            size_of::<u32>() as NSUInteger,
            (&leaf_width_u32 as *const u32).cast::<c_void>(),
        );
        leaf_encoder.set_bytes(
            4,
            size_of::<u32>() as NSUInteger,
            (&leaf_count_u32 as *const u32).cast::<c_void>(),
        );
        dispatch(&leaf_encoder, &self.leaf_pipeline, leaf_count);
        leaf_encoder.end_encoding();

        let mut level_offsets = Vec::with_capacity(leaf_count.ilog2() as usize + 1);
        let mut level_offset = 0usize;
        let mut child_count = leaf_count;
        level_offsets.push(level_offset);
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
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "command buffer ended with status {:?}",
                command_buffer.status()
            ));
        }

        let nodes = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<u64>(), output_len)
        };
        Ok(tree_from_levels(
            nodes,
            &level_offsets,
            leaf_count,
            cap_height,
        ))
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
        cap.par_iter_mut()
            .enumerate()
            .for_each(|(cap_index, root)| {
                *root = read_node(nodes, level_offsets[0], cap_index);
            });
    } else {
        digests
            .par_chunks_mut(subtree_digest_count)
            .zip(cap.par_iter_mut())
            .enumerate()
            .for_each(|(cap_index, (subtree_digests, root))| {
                *root = fill_subtree_layout(
                    subtree_digests,
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
    *left_root = fill_subtree_layout(
        left_digests,
        nodes,
        level_offsets,
        start_leaf,
        half,
    );
    *right_root = fill_subtree_layout(
        right_digests,
        nodes,
        level_offsets,
        start_leaf + half,
        half,
    );

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

    #[test]
    fn metal_merkle_matches_cpu_across_sponge_boundaries() {
        let mut rng = StdRng::seed_from_u64(0x4d45_5441_4c32);
        for width in [0, 1, 4, 5, 8, 9, 16, 17, 31, 64, 137] {
            let leaves = (0..64)
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
                .collect::<Vec<_>>();

            for cap_height in [0, 3, 6] {
                let mut context = CONTEXT
                    .as_ref()
                    .as_ref()
                    .unwrap_or_else(|error| panic!("{error}"))
                    .lock()
                    .unwrap();
                let gpu = autoreleasepool(|| context.build(&leaves, cap_height)).unwrap();
                let cpu = cpu_tree(&leaves, cap_height);
                assert_tree_eq(&gpu, &cpu, width, cap_height);
            }
        }
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
