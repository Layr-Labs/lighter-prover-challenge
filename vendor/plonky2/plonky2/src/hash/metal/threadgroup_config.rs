//! Threadgroup size configuration for Merkle tree GPU operations.
//!
//! Provides adaptive threadgroup sizing based on tree size and device limits.

/// Environment variable for Merkle threadgroup size override
const MERKLE_THREADGROUP_SIZE_ENV: &str = "METAL_MERKLE_THREADGROUP_SIZE";

/// Default threadgroup sizes for Merkle operations (tuned via examples/threadgroup_tuning.rs)
/// Benchmark results on Apple M4 (2026-01-30):
///   - 8k leaves: 64 optimal (0.0122s)
///   - 16k leaves: 128 optimal (0.0198s)
///   - 64k leaves: 128 optimal (0.0702s)
///   - 128k leaves: 128 optimal (0.1362s)
///   - 16M+ leaves (2^24+): 64 reduces memory contention in bandwidth-bound regime
const MERKLE_THREADGROUP_SIZE_SMALL: usize = 64; // For small trees (<16k leaves)
const MERKLE_THREADGROUP_SIZE_MEDIUM: usize = 128; // For medium trees (16k-64k leaves)
const MERKLE_THREADGROUP_SIZE_LARGE: usize = 128; // For large trees (64k-16M leaves)
const MERKLE_THREADGROUP_SIZE_VERY_LARGE: usize = 64; // For very large trees (>= 16M leaves, 2^24+)

/// Get threadgroup size from environment or use default based on leaf count
///
/// Parameters:
/// - `leaf_count`: Number of leaves in the tree (for adaptive sizing)
/// - `simd_width`: The pipeline's thread_execution_width (SIMD width)
/// - `max_threads`: The pipeline's max_total_threads_per_threadgroup (device limit)
///
/// Returns a threadgroup size that is:
/// - A multiple of simd_width (for efficient SIMD execution)
/// - At least simd_width (to avoid zero-thread dispatches)
/// - At most max_threads (device limit)
pub fn get_merkle_threadgroup_size(
    leaf_count: usize,
    simd_width: usize,
    max_threads: usize,
) -> usize {
    // Check for environment variable override
    if let Ok(val) = std::env::var(MERKLE_THREADGROUP_SIZE_ENV) {
        if let Ok(size) = val.parse::<usize>() {
            if size.is_power_of_two() && size >= 32 && size <= 1024 {
                // Clamp override to [simd_width, max_threads] then align to SIMD width
                let clamped = size.clamp(simd_width, max_threads);
                let aligned = (clamped / simd_width) * simd_width;
                // Ensure at least simd_width (handles case where clamped < simd_width somehow)
                return aligned.max(simd_width);
            }
        }
    }

    // Use adaptive sizing based on tree size
    // For very large trees (2^24+), use smaller threadgroups to reduce memory contention
    let base_size = if leaf_count >= (1 << 24) {
        MERKLE_THREADGROUP_SIZE_VERY_LARGE
    } else if leaf_count >= (1 << 16) {
        MERKLE_THREADGROUP_SIZE_LARGE
    } else if leaf_count >= (1 << 13) {
        MERKLE_THREADGROUP_SIZE_MEDIUM
    } else {
        MERKLE_THREADGROUP_SIZE_SMALL
    };

    // Clamp to [simd_width, max_threads] and align to SIMD width multiple
    let clamped = base_size.clamp(simd_width, max_threads);
    let aligned = (clamped / simd_width) * simd_width;
    // Ensure at least SIMD width
    aligned.max(simd_width)
}

/// Get current Merkle threadgroup configuration for diagnostics
/// Returns (small_default, medium_default, large_default, very_large_default, env_override)
#[allow(dead_code)]
pub fn get_merkle_threadgroup_config() -> (usize, usize, usize, usize, Option<usize>) {
    let override_val = std::env::var(MERKLE_THREADGROUP_SIZE_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&size: &usize| size.is_power_of_two() && size >= 32 && size <= 1024);

    (
        MERKLE_THREADGROUP_SIZE_SMALL,
        MERKLE_THREADGROUP_SIZE_MEDIUM,
        MERKLE_THREADGROUP_SIZE_LARGE,
        MERKLE_THREADGROUP_SIZE_VERY_LARGE,
        override_val,
    )
}

/// Compute effective threadgroup size for a given leaf count using a specific pipeline
/// This is useful for diagnostics to see what size will actually be used
#[allow(dead_code)]
pub fn compute_effective_threadgroup_size(
    leaf_count: usize,
    simd_width: usize,
    max_threads: usize,
) -> usize {
    get_merkle_threadgroup_size(leaf_count, simd_width, max_threads)
}
