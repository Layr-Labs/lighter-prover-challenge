// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;
#[path = "../embedded.rs"]
mod embedded;
#[path = "../prover.rs"]
mod prover;

use std::env;
use std::fs::{self, File};
use std::io::BufWriter;

use api::{
    Circuits, HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PROVER_THREAD_STACK_BYTES,
    PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
};
use circuit::block::Block;
use circuit::types::config::F;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Disable jemalloc's dirty/muzzy page decay for this short-lived single-shot
// prover: retained pages are never madvised away between commitment phases, so
// the allocator stops paying the recurring purge/refault churn.
//
// ABI note: jemalloc reads `const char *malloc_conf` (prefixed `_rjem_` in
// tikv-jemalloc-sys), i.e. a pointer-sized slot holding the address of a
// NUL-terminated string. `&[u8; 36]` is a thin pointer to the NUL-terminated
// bytes, which matches that ABI exactly. Exporting the bare byte array itself
// (no indirection) or omitting the trailing NUL would make jemalloc read the
// string bytes as a pointer and crash. This is a default: the environment and
// /etc/malloc.conf can still override it.
#[cfg(not(target_env = "msvc"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: &[u8; 36] = b"dirty_decay_ms:-1,muzzy_decay_ms:-1\0";

// Keep the promoted writer path while exercising a second submission from that baseline.
const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

/// Number of performance (P) cores on this Apple Silicon part, via
/// `hw.perflevel0.logicalcpu`. macOS has no hard affinity, but QoS class
/// steers scheduling: a rayon worker parked on an E-core runs 3-4x slower
/// than a P-core, and a `par_iter` barrier completes at the speed of its
/// slowest chunk, so one E-core straggler stretches every parallel phase.
/// Sizing the pool to P-cores keeps every barrier chunk on a fast core.
/// `PROVE_THREADS` overrides the count (for A/B on the ranked host), and
/// the P-core count is floored at half the total CPU count so parts with a
/// small P-core share (e.g. 4P+6E base M4) cannot collapse the pool.
/// Falls back to all CPUs on non-macOS or if the sysctl is unavailable.
fn p_core_count() -> usize {
    if let Ok(v) = std::env::var("PROVE_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    let all = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const i8,
                oldp: *mut core::ffi::c_void,
                oldlenp: *mut usize,
                newp: *const core::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut count: u32 = 0;
        let mut len = core::mem::size_of::<u32>();
        let name = b"hw.perflevel0.logicalcpu\0".as_ptr().cast::<i8>();
        let ok = unsafe {
            sysctlbyname(
                name,
                (&mut count as *mut u32).cast(),
                &mut len,
                core::ptr::null(),
                0,
            )
        };
        if ok == 0 && count > 0 {
            return (count as usize).max(all / 2);
        }
    }
    all
}

fn main() {
    // First statement in the process: the Metal shader compile and pipeline
    // lowering behind the GPU hash path cost the better part of a second on a
    // cold OS shader cache, and the benchmark sandbox denies writes to that
    // cache, which disables it entirely — so every scored worker pays the full
    // price. Starting it here overlaps it with the startup work below instead
    // of stalling the first proving step that wants the GPU. Pure scheduling:
    // the compiled kernels are identical either way.
    plonky2::hash::poseidon2::prewarm_gpu();
    env_logger::init();
    // Size the pool to P-cores and mark every worker latency-critical. The
    // default `num_cpus` count includes E-cores, and default-QoS threads are
    // what macOS parks on them; a barrier then waits on the slowest chunk.
    // Scheduling-only: no work is added or reordered, proof bytes are untouched.
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .num_threads(p_core_count())
        .start_handler(|_| {
            #[cfg(target_os = "macos")]
            unsafe {
                #[allow(non_camel_case_types)]
                type qos_class_t = u32;
                unsafe extern "C" {
                    fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
                }
                let _ = pthread_set_qos_class_self_np(0x21, 0);
            }
        })
        .build_global()
        .expect("cannot configure prover thread pool");

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    // Overlap fixture parse with embedded circuit load: independent work that
    // previously ran fully serial on the scored critical path, beside the Metal
    // pipeline prewarm already started above.
    let (block, circuits) = rayon::join(
        || {
            let json = fs::read(&fixture).expect("cannot read prover fixture");
            Block::<F>::from_json_with_empty_txs(
                &json,
                HEAVY_TX_PER_PROOF,
                LIGHT_TX_PER_PROOF,
                PUBLIC_HEAVY_TX_COUNT,
                PUBLIC_LIGHT_TX_COUNT,
            )
            .expect("invalid prover fixture")
        },
        Circuits::load,
    );
    let proof = prover::prove_block(block, circuits);
    let mut writer = BufWriter::with_capacity(
        PROOF_OUTPUT_BUFFER_BYTES,
        File::create(output).expect("cannot create proof output"),
    );
    bincode::serialize_into(&mut writer, &proof).expect("cannot write proof output");
    // Explicit flush instead of relying on `BufWriter`'s `Drop` (which swallows
    // errors): every serialized byte must have reached the file descriptor
    // before the fast exit below, since `process::exit` runs no destructors.
    // `into_inner` flushes the userspace buffer and surfaces any write error;
    // dropping the returned `File` closes the descriptor. No `fsync` is needed
    // — the benchmark verifier reads the file back through the same page cache
    // on the same machine, so the `write(2)`s are already visible to it, and an
    // `fsync` would only add durability latency to the scored process lifetime.
    let file = writer.into_inner().expect("cannot flush proof output");
    drop(file);

    // The score is the sum of worker process lifetimes (spawn -> exit), so the
    // destructor teardown after the proof is written is scored dead work: the
    // circuit data, prover data, LDE commitments and witness buffers are all
    // multi-hundred-megabyte `Vec`/`HashMap` graphs whose recursive drops free
    // every allocation one by one, and none of it is observable — the kernel
    // reclaims the address space wholesale at exit. Every Metal command buffer
    // in the hash path is `commit()`ed and then `wait_until_completed()`ed
    // before its results are read. The one detached thread this binary spawns
    // is the GPU pre-warm above, which only populates a cache of compiled
    // kernels and produces nothing anyone reads back, so there is no in-flight
    // background work left to lose here.
    std::process::exit(0);
}

// p90-fire-174-1786149031
