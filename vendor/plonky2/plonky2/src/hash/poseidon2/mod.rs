pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::set_exclusive_gpu_phase;

/// No-op fallback so callers can toggle the exclusive-phase GPU routing hint
/// unconditionally on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn set_exclusive_gpu_phase(_enabled: bool) {}

/// Forces the shared Metal context (device, queue, and the shader library
/// compiled from source at runtime) to initialize now. Callers use this to
/// overlap the one-time shader compilation with other startup work instead of
/// paying it on the first GPU tree build inside the proving pipeline.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub fn warm_metal_context() {
    metal::warm_context();
}

/// No-op fallback on platforms without the Metal backend.
#[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
pub fn warm_metal_context() {}

#[cfg(test)]
pub mod p3;
