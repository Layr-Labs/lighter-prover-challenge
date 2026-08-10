pub mod config;
pub mod hash;

#[cfg(feature = "std")]
use std::cell::Cell;

#[cfg(feature = "std")]
thread_local! {
    /// Marks the fixed light recursive-chain proof on its owning thread so its
    /// d14 commitment trees can use congestion-aware backend routing.
    static LIGHT_CHAIN_GPU_ROUTING: Cell<bool> = const { Cell::new(false) };
}

#[cfg(all(
    feature = "std",
    any(
        test,
        all(target_arch = "aarch64", target_os = "macos")
    )
))]
pub(crate) fn is_light_chain_gpu_routing() -> bool {
    LIGHT_CHAIN_GPU_ROUTING.get()
}

#[cfg(feature = "std")]
struct LightChainGpuRoutingReset(bool);

#[cfg(feature = "std")]
impl Drop for LightChainGpuRoutingReset {
    fn drop(&mut self) {
        LIGHT_CHAIN_GPU_ROUTING.set(self.0);
    }
}

/// Runs the fixed light recursive-chain proof with congestion-aware routing.
/// The mark is scheduling-only; either backend hashes the same Merkle tree.
#[cfg(feature = "std")]
pub fn with_light_chain_gpu_routing<R>(f: impl FnOnce() -> R) -> R {
    let previous = LIGHT_CHAIN_GPU_ROUTING.replace(true);
    let _reset = LightChainGpuRoutingReset(previous);
    f()
}

/// Runs `f` without changing scheduling when the standard library is disabled.
#[cfg(not(feature = "std"))]
pub fn with_light_chain_gpu_routing<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(any(
    test,
    all(feature = "std", target_arch = "aarch64", target_os = "macos")
))]
fn serial_critical_tree_uses_gpu(
    exclusive: bool,
    gpu_idle: bool,
    light_chain: bool,
    leaf_width: usize,
) -> bool {
    exclusive || gpu_idle || (leaf_width > 64 && !light_chain)
}

#[cfg(any(
    test,
    all(feature = "std", target_arch = "aarch64", target_os = "macos")
))]
fn serial_critical_storage_uses_gpu(
    exclusive: bool,
    gpu_idle: bool,
    leaf_width: usize,
) -> bool {
    exclusive || gpu_idle || leaf_width > 64
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod metal;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub use metal::{
    is_exclusive_gpu_phase, prewarm as prewarm_gpu, set_exclusive_gpu_phase,
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

#[cfg(test)]
mod routing_tests {
    use super::{
        serial_critical_storage_uses_gpu, serial_critical_tree_uses_gpu,
        with_light_chain_gpu_routing,
    };

    #[test]
    fn light_chain_wide_tree_escapes_only_a_busy_nonexclusive_gpu() {
        assert!(!serial_critical_tree_uses_gpu(false, false, true, 136));
        assert!(serial_critical_tree_uses_gpu(false, true, true, 136));
        assert!(serial_critical_tree_uses_gpu(false, false, false, 136));
        assert!(!serial_critical_tree_uses_gpu(false, false, true, 64));
        assert!(!serial_critical_tree_uses_gpu(false, false, false, 64));
        assert!(serial_critical_tree_uses_gpu(true, false, true, 136));

        // The CPU tree still retains GPU-visible wire columns so later
        // quotient kernels remain available.
        assert!(serial_critical_storage_uses_gpu(false, false, 136));
        assert!(!serial_critical_storage_uses_gpu(false, false, 64));
    }

    #[cfg(feature = "std")]
    #[test]
    fn light_chain_routing_scope_nests_and_unwinds() {
        assert!(!super::is_light_chain_gpu_routing());
        assert_eq!(
            with_light_chain_gpu_routing(|| {
                assert!(super::is_light_chain_gpu_routing());
                with_light_chain_gpu_routing(|| {
                    assert!(super::is_light_chain_gpu_routing());
                });
                7
            }),
            7
        );
        assert!(!super::is_light_chain_gpu_routing());

        let _ = std::panic::catch_unwind(|| {
            with_light_chain_gpu_routing(|| panic!("routing-scope-test"));
        });
        assert!(!super::is_light_chain_gpu_routing());
    }
}
