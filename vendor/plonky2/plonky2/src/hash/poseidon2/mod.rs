pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    generate_binary_archive, prewarm as prewarm_gpu, set_exclusive_gpu_phase, set_gpu_archive_dir,
};

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

/// No-op fallback so a process entry point can request GPU pre-warming
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu() {}

/// No-op fallback so a process entry point can nominate a directory for the GPU
/// binary archive unconditionally on platforms without the Metal backend.
#[cfg(all(feature = "std", not(all(target_arch = "aarch64", target_os = "macos"))))]
pub fn set_gpu_archive_dir(_dir: std::path::PathBuf) {}

/// Fallback so the archive generator builds on platforms without the Metal
/// backend, where there is no archive to generate.
#[cfg(all(feature = "std", not(all(target_arch = "aarch64", target_os = "macos"))))]
#[doc(hidden)]
pub fn generate_binary_archive(_path: &std::path::Path) -> Result<(), String> {
    Err("the Metal backend is not available on this target".to_owned())
}

#[cfg(test)]
pub mod p3;
