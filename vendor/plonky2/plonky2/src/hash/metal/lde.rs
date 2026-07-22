//! GPU low-degree-extension (LDE) extension point.
//!
//! Enabling `metal-lde` does not change proving behavior yet. LDE continues to
//! use the CPU implementation in `plonky2_field::polynomial`. A future Metal
//! FFT implementation should dispatch from that path only when this check is
//! true and must produce field elements identical to the CPU implementation.

/// Reports whether a Metal device is available for a future GPU LDE path.
///
/// This is an availability hook only; no LDE work is dispatched to Metal yet.
pub(crate) fn available() -> bool {
    metal::Device::system_default().is_some()
}
