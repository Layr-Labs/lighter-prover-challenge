pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    generate_pipeline_archive, is_exclusive_gpu_phase, pipeline_archive_matches_shader,
    compiles_from_source,
    prewarm as prewarm_gpu,
    verify_pipeline_archive,
    set_exclusive_gpu_phase,
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

#[cfg(test)]
pub mod p3;
