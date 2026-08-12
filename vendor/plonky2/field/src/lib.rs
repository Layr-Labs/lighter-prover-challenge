#![allow(incomplete_features)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![feature(specialization)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub(crate) mod arch;

/// Two Goldilocks elements multiplied through a single interleaved reduction
/// block. Re-exported on its own so callers outside this crate can reach the
/// paired assembly — the Poseidon2 partial-round S-box has products that are
/// independent of each other and wants them scheduled together — without the
/// `arch` tree as a whole becoming public API.
#[cfg(target_arch = "aarch64")]
pub use arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;

pub mod batch_util;
pub mod cosets;
pub mod extension;
pub mod fft;
pub mod goldilocks_extensions;
pub mod goldilocks_field;
pub mod interpolation;
pub mod ops;
pub mod packable;
pub mod packed;
pub mod polynomial;
pub mod secp256k1_base;
pub mod secp256k1_scalar;
pub mod types;
pub mod zero_poly_coset;

#[cfg(test)]
mod field_testing;

#[cfg(test)]
mod prime_field_testing;
