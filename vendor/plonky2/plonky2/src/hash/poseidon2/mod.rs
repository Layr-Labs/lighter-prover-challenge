pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    enter_final_column_store_phase, is_exclusive_gpu_phase, leave_final_column_store_phase,
    prewarm as prewarm_gpu, prewarm_large_column_store, set_exclusive_gpu_phase, spine_backlog_add,
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

/// No-op fallback for final-phase Metal buffer-pool trimming.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn enter_final_column_store_phase(_required_bytes: &[u64]) {}

/// No-op fallback for final-phase Metal buffer-pool trimming.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn leave_final_column_store_phase() {}

#[cfg(test)]
pub mod p3;
