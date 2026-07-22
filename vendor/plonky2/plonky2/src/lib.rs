#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub extern crate alloc;

/// Re-export of `plonky2_field`.
#[doc(inline)]
pub use plonky2_field as field;

pub mod batch_fri;
pub mod fri;
pub mod gadgets;
pub mod gates;
pub mod hash;
pub mod iop;
pub mod plonk;
pub mod recursion;
pub mod util;

/// Reset Metal allocation counters before an instrumented proving run.
#[cfg(feature = "metal")]
pub fn reset_metal_gpu_allocation_stats() {
    hash::metal::tracking::reset_allocation_stats();
}

/// Return the number of Metal buffer allocations in the current process.
#[cfg(feature = "metal")]
pub fn metal_gpu_allocation_count() -> usize {
    hash::metal::tracking::get_allocation_count()
}

#[cfg(test)]
mod lookup_test;
