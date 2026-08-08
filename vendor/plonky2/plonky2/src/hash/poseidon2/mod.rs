pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    prewarm as prewarm_gpu, prewarm_with_archive as prewarm_gpu_with_archive,
    serialize_pipeline_archive, set_exclusive_gpu_phase,
};

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

/// No-op fallback so a process entry point can request GPU pre-warming
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu() {}

/// No-op fallback for installing a host-specific Metal archive on platforms
/// without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu_with_archive(_bytes: &'static [u8]) {}

/// Unsupported build hosts emit an empty artifact and preserve their current
/// runtime backend.
#[cfg(all(
    feature = "std",
    not(all(target_arch = "aarch64", target_os = "macos"))
))]
pub fn serialize_pipeline_archive(_path: &std::path::Path) -> Result<(), String> {
    Err("Metal binary archives require macOS on Apple Silicon".to_owned())
}

#[cfg(test)]
pub mod p3;
