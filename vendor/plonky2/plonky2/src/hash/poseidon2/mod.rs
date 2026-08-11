pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    is_exclusive_gpu_phase, prewarm as prewarm_gpu,
    prewarm_with_archive as prewarm_gpu_with_archive,
    prewarm_with_required_archive as prewarm_gpu_with_required_archive, set_exclusive_gpu_phase,
    verify_required_archive_hits as verify_required_gpu_archive_hits,
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

/// No-op archive installation on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu_with_archive(_bytes: &'static [u8]) {}

/// No-op required-archive installation on platforms without the Metal
/// backend. The requirement applies only when Metal can be selected.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn prewarm_gpu_with_required_archive(
    _bytes: &'static [u8],
    _archive_output_path: &std::path::Path,
) {
}

/// No-op completion gate on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn verify_required_gpu_archive_hits() {}

#[cfg(test)]
pub mod p3;
