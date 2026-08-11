//! Streaming-SVE (SME) feeder kernels for the Goldilocks FFT.
//!
//! Apple M4-class chips expose one SME block per core cluster — an execution
//! unit with true 64-bit vector `MUL`/`UMULH` that nothing else in this
//! prover touches. A thread that enters streaming mode runs its butterflies
//! on the cluster's SME block instead of its own scalar/NEON pipes, so under
//! full rayon load a couple of "feeder" threads add multiply throughput the
//! machine otherwise leaves idle.
//!
//! Two rules keep this additive rather than contended:
//! * at most `LIGHTER_SSVE_TOKENS` (default 2 — one per P-cluster on M4 Pro)
//!   threads stream at any moment, enforced by a global permit counter;
//! * only large passes take the streaming path (the permit is held for one
//!   whole kernel call, microseconds to sub-millisecond).
//!
//! The kernels (`ssve/gl_fft_ssve.s`, generated from `ssve/gl_fft_ssve.c`)
//! are bit-identical to `fft_classic_simd_two_layers_neon` /
//! `fft_classic_simd_single_layer_neon`: same reduce128 multiply as
//! `mul_reduce_pair`, same double-correction add/sub as
//! `gl_add_neon`/`gl_sub_neon`, so raw non-canonical `u64` representatives —
//! which are compared and hashed directly — are preserved exactly.
//!
//! Fail-safe: the streaming path is entered only when
//! `hw.optional.arm.FEAT_SME2` reports 1 (and `LIGHTER_SSVE` is not `0`);
//! otherwise every call falls through to the NEON kernels unchanged.

use core::sync::atomic::{AtomicI32, AtomicU8, Ordering};

use crate::goldilocks_field::GoldilocksField;

#[cfg(target_os = "macos")]
core::arch::global_asm!(include_str!("ssve/gl_fft_ssve.s"), options(raw));

#[cfg(target_os = "macos")]
extern "C" {
    fn gl_fft_fused2_ssve(
        values: *mut u64,
        len: usize,
        lg_half_m: usize,
        w1_row: *const u64,
        w2_row: *const u64,
    );
    fn gl_fft_single_ssve(values: *mut u64, len: usize, lg_half_m: usize, omega_row: *const u64);
    fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut core::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// 0 = undetermined, 1 = enabled, 2 = disabled.
static STATE: AtomicU8 = AtomicU8::new(0);
/// Streaming permits; configured on first detection.
static PERMITS: AtomicI32 = AtomicI32::new(2);

#[cfg(target_os = "macos")]
fn detect() -> bool {
    if let Some(v) = std::env::var_os("LIGHTER_SSVE") {
        if v == "0" || v == "off" {
            return false;
        }
    }
    if let Some(v) = std::env::var_os("LIGHTER_SSVE_TOKENS") {
        if let Some(n) = v.to_str().and_then(|s| s.parse::<i32>().ok()) {
            if (1..=8).contains(&n) {
                PERMITS.store(n, Ordering::Relaxed);
            }
        }
    }
    let mut val: u32 = 0;
    let mut len: usize = core::mem::size_of::<u32>();
    let ok = unsafe {
        sysctlbyname(
            c"hw.optional.arm.FEAT_SME2".as_ptr().cast(),
            (&mut val as *mut u32).cast(),
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    ok == 0 && val != 0
}

#[cfg(not(target_os = "macos"))]
fn detect() -> bool {
    false
}

#[inline]
fn available() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = detect();
            STATE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Held while a thread runs a streaming kernel; capped at the permit count.
pub struct FeederToken(());

impl Drop for FeederToken {
    fn drop(&mut self) {
        PERMITS.fetch_add(1, Ordering::Release);
    }
}

/// Try to become a feeder for one kernel call. `None` when SME is absent,
/// disabled, or all permits are taken — callers then use the NEON path.
#[inline]
pub fn try_feeder() -> Option<FeederToken> {
    if !available() {
        return None;
    }
    #[allow(deprecated)] // fetch_update: stable across wider toolchain range than try_update
    PERMITS
        .fetch_update(Ordering::Acquire, Ordering::Relaxed, |p| {
            (p > 0).then(|| p - 1)
        })
        .ok()
        .map(|_| FeederToken(()))
}

/// Fused two-layer pass; caller guarantees the same shape the NEON kernel
/// requires (blocks of `4q`, `w1_row[q]`, `w2_row[2q]`) plus `lg_half_m >= 4`.
#[cfg(target_os = "macos")]
pub fn fused2(
    values: &mut [GoldilocksField],
    lg_half_m: usize,
    w1_row: &[GoldilocksField],
    w2_row: &[GoldilocksField],
) {
    let q = 1usize << lg_half_m;
    debug_assert!(lg_half_m >= 4);
    debug_assert!(w1_row.len() >= q && w2_row.len() >= 2 * q);
    debug_assert_eq!(values.len() % (4 * q), 0);
    unsafe {
        gl_fft_fused2_ssve(
            values.as_mut_ptr().cast(),
            values.len(),
            lg_half_m,
            w1_row.as_ptr().cast(),
            w2_row.as_ptr().cast(),
        );
    }
}

/// Single radix-2 layer; caller guarantees the NEON kernel's shape and
/// `lg_half_m >= 4`.
#[cfg(target_os = "macos")]
pub fn single_layer(values: &mut [GoldilocksField], lg_half_m: usize, omega_row: &[GoldilocksField]) {
    debug_assert!(lg_half_m >= 4);
    debug_assert!(omega_row.len() >= (1 << lg_half_m));
    unsafe {
        gl_fft_single_ssve(
            values.as_mut_ptr().cast(),
            values.len(),
            lg_half_m,
            omega_row.as_ptr().cast(),
        );
    }
}

#[cfg(not(target_os = "macos"))]
pub fn fused2(
    _values: &mut [GoldilocksField],
    _lg_half_m: usize,
    _w1_row: &[GoldilocksField],
    _w2_row: &[GoldilocksField],
) {
    unreachable!("try_feeder never succeeds off macOS")
}

#[cfg(not(target_os = "macos"))]
pub fn single_layer(
    _values: &mut [GoldilocksField],
    _lg_half_m: usize,
    _omega_row: &[GoldilocksField],
) {
    unreachable!("try_feeder never succeeds off macOS")
}
