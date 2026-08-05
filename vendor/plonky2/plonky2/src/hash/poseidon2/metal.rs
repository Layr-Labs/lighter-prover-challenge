use core::ffi::c_void;
use core::mem::{size_of, size_of_val};
use core::slice;
use std::sync::{Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant};

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

/// Commit-stagger depth = number of pooled `MetalContext`s (see `MetalPool`):
/// how many threads may be inside the commit-heavy section (host pack, GPU
/// dispatch, readback) at once. Checked-out contexts are the permits.
///
/// MEASURED 2026-08-05 on an M4 Max at leaf concurrency 4 (52 leaf proofs,
/// smoke fixture), plain pool without the GPU-inflight fence: 1 context beats
/// 2 and 4 (57.6s vs 59.4s / 59.3s wall). The single context's lock waits
/// (60.8s aggregate) overlap other leaves' CPU compute, and the lock acts as
/// a free pipeline stagger: with 4 contexts every leaf commits at once (GPU
/// latency 13.3s -> 41.3s multiplexed, readback 14 -> 25s rayon-contended),
/// then every leaf computes at once. *With* the `GPU_INFLIGHT` fence below,
/// deeper is monotonically better (57.8 / 57.3 / 54.9 / 54.3s wall at depth
/// 1/2/3/4, K=4): kernels keep solo latency (13.3s aggregate at every depth)
/// while pack/readback of different builds overlap the serialized GPU stream,
/// so depth 4 = "enough contexts that checkout never convoys" and the fence
/// provides the only serialization that matters. On the ranked M4 Pro the
/// bench layer runs leaf concurrency 2 (cores/4), so 4 contexts are already
/// beyond saturation there; unused contexts allocate no staging memory.
///
/// The env variables are local profiling overrides only (the ranked sandbox
/// clears the environment, so ranked always runs the compiled default).
const DEFAULT_COMMIT_STAGGER: usize = 4;

fn pool_size() -> usize {
    for key in ["LIGHTER_COMMIT_STAGGER", "LIGHTER_METAL_CONTEXTS"] {
        if let Some(value) = std::env::var_os(key) {
            if let Some(n) = value.to_str().and_then(|v| v.parse::<usize>().ok()) {
                return n.clamp(1, 8);
            }
        }
    }
    DEFAULT_COMMIT_STAGGER
}

/// At most one command buffer in flight on the GPU: builders that are past
/// packing queue here instead of multiplexing the GPU (multiplexed kernels
/// finish ~3x slower in aggregate, see notes/08). Held only across
/// commit -> completed and released before readback, so the next builder's
/// kernels run while this one lays out its digests.
static GPU_INFLIGHT: Mutex<()> = Mutex::new(());

/// Immutable Metal state shared by all pooled contexts. `MTLDevice`,
/// `MTLComputePipelineState` and `MTLBuffer` are thread-safe; `parameters` is
/// written once at creation and only read (by the GPU) afterwards.
struct MetalShared {
    device: Device,
    leaf_pipeline: ComputePipelineState,
    parent_pipeline: ComputePipelineState,
    parameters: Buffer,
}

/// Mutable per-slot state: a command queue plus grow-only cached staging
/// buffers. A context is exclusively checked out of the pool for the duration
/// of one build, so the buffers are never shared between concurrent builds.
struct MetalContext {
    queue: CommandQueue,
    input_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
}

/// A fixed pool of `MetalContext`s over one shared device. Concurrent Merkle
/// builds (up to 4 leaf provers plus the chain/pre/block workers) each check
/// out their own context, so the host-side packing/readback and the GPU
/// execution of different builds overlap instead of serializing on one global
/// lock; the GPU schedules command buffers from the queues concurrently.
struct MetalPool {
    shared: MetalShared,
    idle: Mutex<Vec<MetalContext>>,
    available: Condvar,
}

/// RAII checkout of a pool slot: returns the context to the pool on drop.
struct PooledContext<'a> {
    pool: &'a MetalPool,
    context: Option<MetalContext>,
}

impl Drop for PooledContext<'_> {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            let mut idle = self
                .pool
                .idle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            idle.push(context);
            self.pool.available.notify_one();
        }
    }
}

static POOL: LazyLock<Result<MetalPool, String>> = LazyLock::new(MetalPool::new);

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

    let pool = match &*POOL {
        Ok(pool) => pool,
        Err(error) => {
            log::warn!("Metal Poseidon2 unavailable; using CPU Merkle hashing: {error}");
            return None;
        }
    };

    let wait_start = Instant::now();
    let mut checked_out = pool.checkout();
    let wait = wait_start.elapsed();
    let context = checked_out
        .context
        .as_mut()
        .expect("pooled Metal context is present until drop");

    match autoreleasepool(|| context.build(&pool.shared, leaves, cap_height, wait)) {
        Ok(tree) => Some(tree),
        Err(error) => {
            log::warn!("Metal Poseidon2 failed; using CPU Merkle hashing: {error}");
            None
        }
    }
}

impl MetalPool {
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

            let size = pool_size();
            let contexts = (0..size)
                .map(|_| MetalContext {
                    queue: device.new_command_queue(),
                    input_buffer: None,
                    output_buffer: None,
                })
                .collect::<Vec<_>>();
            log::debug!("metal pool: {size} contexts");

            Ok(Self {
                shared: MetalShared {
                    device,
                    leaf_pipeline,
                    parent_pipeline,
                    parameters,
                },
                idle: Mutex::new(contexts),
                available: Condvar::new(),
            })
        })
    }

    fn checkout(&self) -> PooledContext<'_> {
        let mut idle = self.idle.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            if let Some(context) = idle.pop() {
                return PooledContext {
                    pool: self,
                    context: Some(context),
                };
            }
            idle = self
                .available
                .wait(idle)
                .unwrap_or_else(|error| error.into_inner());
        }
    }
}

impl MetalContext {
    fn build<F: RichField>(
        &mut self,
        shared: &MetalShared,
        leaves: &[Vec<F>],
        cap_height: usize,
        wait: Duration,
    ) -> Result<(Vec<HashOut<F>>, Vec<HashOut<F>>), String> {
        let leaf_count = leaves.len();
        let leaf_width = leaves[0].len();
        let cap_count = 1usize << cap_height;
        let total_node_count = 2 * leaf_count - cap_count;

        let pack_start = Instant::now();
        let input_len = leaf_count
            .checked_mul(leaf_width)
            .ok_or("Metal leaf input length overflow")?;
        let input_bytes = input_len
            .checked_mul(size_of::<u64>())
            .ok_or("Metal leaf input size overflow")?;
        if self
            .input_buffer
            .as_ref()
            .map_or(true, |buffer| buffer.length() < input_bytes.max(size_of::<u64>()) as u64)
        {
            self.input_buffer = Some(shared.device.new_buffer(
                input_bytes.max(size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        let input_buffer = self.input_buffer.as_ref().unwrap();
        let input = unsafe {
            slice::from_raw_parts_mut(input_buffer.contents().cast::<u64>(), input_len)
        };
        if leaf_width != 0 {
            input
                .par_chunks_exact_mut(leaf_width)
                .zip(leaves.par_iter())
                .for_each(|(destination, source)| {
                    for (destination, value) in destination.iter_mut().zip(source) {
                        *destination = value.to_noncanonical_u64();
                    }
                });
        }

        let output_len = total_node_count
            .checked_mul(4)
            .ok_or("Metal Merkle output length overflow")?;
        let output_bytes = output_len
            .checked_mul(size_of::<u64>())
            .ok_or("Metal Merkle output size overflow")?;
        if self
            .output_buffer
            .as_ref()
            .map_or(true, |buffer| buffer.length() < output_bytes as u64)
        {
            self.output_buffer = Some(
                shared
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
            );
        }
        let output_buffer = self.output_buffer.as_ref().unwrap();
        let pack = pack_start.elapsed();

        let leaf_count_u32 = leaf_count as u32;
        let leaf_width_u32 = leaf_width as u32;
        let command_buffer = self.queue.new_command_buffer();
        let leaf_encoder = command_buffer.new_compute_command_encoder();
        leaf_encoder.set_compute_pipeline_state(&shared.leaf_pipeline);
        leaf_encoder.set_buffer(0, Some(&input_buffer), 0);
        leaf_encoder.set_buffer(1, Some(&output_buffer), 0);
        leaf_encoder.set_buffer(2, Some(&shared.parameters), 0);
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
        dispatch(&leaf_encoder, &shared.leaf_pipeline, leaf_count);
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
            parent_encoder.set_compute_pipeline_state(&shared.parent_pipeline);
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
            parent_encoder.set_buffer(2, Some(&shared.parameters), 0);
            parent_encoder.set_bytes(
                3,
                size_of::<u32>() as NSUInteger,
                (&parent_count_u32 as *const u32).cast::<c_void>(),
            );
            dispatch(&parent_encoder, &shared.parent_pipeline, parent_count);
            parent_encoder.end_encoding();

            child_count = parent_count;
        }

        let gpu_wait_start = Instant::now();
        let gpu_permit = GPU_INFLIGHT
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let gpu_wait = gpu_wait_start.elapsed();

        let gpu_start = Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "command buffer ended with status {:?}",
                command_buffer.status()
            ));
        }
        let gpu = gpu_start.elapsed();
        // Release before readback: the next builder's kernels overlap our
        // digest layout.
        drop(gpu_permit);

        let readback_start = Instant::now();
        let nodes = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<u64>(), output_len)
        };
        let tree = tree_from_levels(nodes, &level_offsets, leaf_count, cap_height);
        let readback = readback_start.elapsed();
        log::debug!(
            "metal merkle: leaves={leaf_count} width={leaf_width} wait_ms={:.1} pack_ms={:.1} gpu_wait_ms={:.1} gpu_ms={:.1} readback_ms={:.1}",
            wait.as_secs_f64() * 1e3,
            pack.as_secs_f64() * 1e3,
            gpu_wait.as_secs_f64() * 1e3,
            gpu.as_secs_f64() * 1e3,
            readback.as_secs_f64() * 1e3,
        );
        Ok(tree)
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
            .par_chunks_exact_mut(subtree_digest_count)
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
                let pool = POOL
                    .as_ref()
                    .unwrap_or_else(|error| panic!("{error}"));
                let mut checked_out = pool.checkout();
                let context = checked_out
                    .context
                    .as_mut()
                    .expect("pooled Metal context is present until drop");
                let gpu = autoreleasepool(|| {
                    context.build(&pool.shared, &leaves, cap_height, Duration::ZERO)
                })
                .unwrap();
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
