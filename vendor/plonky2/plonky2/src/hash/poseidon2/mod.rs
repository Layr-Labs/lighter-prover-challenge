pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    configure_shared_column_cache, is_exclusive_gpu_phase, prewarm as prewarm_gpu,
    set_exclusive_gpu_phase, with_shared_column_cache_key,
};

/// No-op fallbacks keep circuit embedding portable while the ranked macOS
/// worker alone supplies the file-backed Metal implementation.
#[cfg(all(
    feature = "std",
    not(all(target_arch = "aarch64", target_os = "macos"))
))]
pub fn configure_shared_column_cache(_directory: &std::path::Path, _participants: usize) {}

#[cfg(all(
    feature = "std",
    not(all(target_arch = "aarch64", target_os = "macos"))
))]
pub fn with_shared_column_cache_key<T>(_key: &str, build: impl FnOnce() -> T) -> T {
    build()
}

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

#[cfg(test)]
pub mod p3;
