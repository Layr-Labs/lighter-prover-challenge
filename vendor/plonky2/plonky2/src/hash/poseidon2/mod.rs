pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    is_exclusive_gpu_phase, prewarm as prewarm_gpu, prewarm_large_column_store,
    prewarm_streamed_merkle_buffers, set_exclusive_gpu_phase, spine_backlog_add,
};

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

/// False on platforms without the Metal-backed exclusive proving phases.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn is_exclusive_gpu_phase() -> bool {
    false
}

/// No-op fallback so a process entry point can request GPU pre-warming
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu() {}

/// No-op fallback for the chain-backlog GPU priority hint on platforms
/// without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn spine_backlog_add(_delta: isize) {}

/// No-op fallback for the large-store prewarm on platforms without the
/// Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_large_column_store(_bytes: u64) {}

/// No-op fallback for streamed Merkle buffer prewarming on platforms without
/// the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_streamed_merkle_buffers(_leaf_count: usize, _cap_height: usize) {}

#[cfg(test)]
pub mod p3;
