pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{enter_concurrent_pipeline, exit_concurrent_pipeline, set_exclusive_gpu_phase};

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn enter_concurrent_pipeline() {}

#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn exit_concurrent_pipeline() {}

#[cfg(test)]
pub mod p3;
