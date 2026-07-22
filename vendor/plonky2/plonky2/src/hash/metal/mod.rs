//! Metal GPU acceleration for Poseidon2 Merkle tree construction and quotient polynomial evaluation.

pub(crate) mod buffer_pool;
pub(crate) mod gpu_thread;
#[cfg(feature = "metal-lde")]
pub(crate) mod lde;
pub(crate) mod merkle;
pub(crate) mod quotient;
pub(crate) mod runtime;
pub(crate) mod threadgroup_config;
pub(crate) mod tracking;
pub(crate) mod utils;

pub(crate) use runtime::RUNTIME;
