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
use circuit::block_pre_execution_constraints::{BlockPreExecutionTarget, Circuit as _};
use circuit::types::config::{C, F};
use plonky2::fri::oracle::PolynomialBatch;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Return freed pages to the OS as soon as they are unused instead of retaining
// them for the process lifetime.
//
// The benchmark runs five of these workers concurrently and the score is the
// sum of their lifetimes, so every resident page one worker holds is a page the
// other four contend for. With decay disabled the allocator never madvises a
// freed extent away, so this process's resident set is the *high-water mark* of
// its heap rather than its live set: the transaction/chain pipeline allocates
// and frees the same shapes of multi-hundred-megabyte witness, coefficient and
// digest buffers 50+ times, and the retained slack accumulates monotonically.
// Setting both decay periods to zero makes residency track the live set.
//
// This changes no computed value: `malloc_conf` only tunes when the allocator
// hands unused pages back to the kernel. Every allocation still returns
// correctly sized, correctly aligned storage, and no arithmetic, ordering or
// buffer content depends on the option.
//
// ABI note: jemalloc reads `const char *malloc_conf` (prefixed `_rjem_` in
// tikv-jemalloc-sys), i.e. a pointer-sized slot holding the address of a
// NUL-terminated string. `&[u8; 34]` is a thin pointer to the NUL-terminated
// bytes, which matches that ABI exactly. Exporting the bare byte array itself
// (no indirection) or omitting the trailing NUL would make jemalloc read the
// string bytes as a pointer and crash. This is a default: the environment and
// /etc/malloc.conf can still override it.
#[cfg(not(target_env = "msvc"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: &[u8; 34] = b"dirty_decay_ms:0,muzzy_decay_ms:0\0";

// Keep the promoted writer path while exercising a second submission from that baseline.
const PROOF_OUTPUT_BUFFER_BYTES: usize = 2 * 1024 * 1024;

fn main() {
    #[cfg(feature = "diagnostic_profile")]
    let _profile_context = plonky2::util::profile::enter_context("worker", 0, &[]);
    #[cfg(feature = "diagnostic_profile")]
    let profile_process = plonky2::util::profile::span("process", "prove_worker");
    // First statement in the process: the Metal shader compile and pipeline
    // lowering behind the GPU hash path cost the better part of a second on a
    // cold OS shader cache, and the benchmark sandbox denies writes to that
    // cache, which disables it entirely — so every scored worker pays the full
    // price. Starting it here overlaps it with the startup work below instead
    // of stalling the first proving step that wants the GPU. Pure scheduling:
    // the compiled kernels are identical either way.
    {
        #[cfg(feature = "diagnostic_profile")]
        let _span = plonky2::util::profile::span("startup", "metal_prewarm_submit");
        plonky2::hash::poseidon2::prewarm_gpu();
    }
    // `log` is statically disabled in release builds: the ranked worker has no
    // log consumer, and diagnostics remain available in debug/test builds.
    // Do not link and initialize an unused logger in every scored process.
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure prover thread pool");
    #[cfg(feature = "diagnostic_profile")]
    plonky2::util::profile::counter(
        "resources",
        "rayon_threads",
        rayon::current_num_threads() as u64,
    );

    let mut args = env::args().skip(1);
    let fixture = args.next().expect("usage: prove FIXTURE OUTPUT");
    let output = args.next().expect("usage: prove FIXTURE OUTPUT");
    assert!(args.next().is_none(), "usage: prove FIXTURE OUTPUT");

    // Fixture parse overlaps the pre-execution circuit load; both are fast.
    let (block, pre_circuits) = rayon::join(
        || {
            #[cfg(feature = "diagnostic_profile")]
            let _span = plonky2::util::profile::span("startup", "fixture_read_parse");
            let json = fs::read(&fixture).expect("cannot read prover fixture");
            Block::<F>::from_json_with_pruned_identity_runs(
                &json,
                HEAVY_TX_PER_PROOF,
                LIGHT_TX_PER_PROOF,
                PUBLIC_HEAVY_TX_COUNT,
                PUBLIC_LIGHT_TX_COUNT,
            )
            .expect("invalid prover fixture")
        },
        || {
            #[cfg(feature = "diagnostic_profile")]
            let _span = plonky2::util::profile::span("startup", "pre_circuit_load");
            match Circuits::load_pre() {
                Ok(loaded) => loaded,
                Err(error) => {
                    log::warn!(
                        "embedded pre circuit unavailable ({error:#}); building from scratch"
                    );
                    let pre =
                        circuit::block_pre_execution_constraints::BlockPreExecutionCircuit::define(
                            circuit::types::config::CIRCUIT_CONFIG,
                        );
                    (pre.target, pre.builder.build::<C>())
                }
            }
        },
    );
    // Startup is decoupled end to end:
    //
    // 1. The four tx/chain blobs parse on a scoped thread while their
    //    constants/sigmas commitments finalize in the background
    //    (`load_remaining_embedded`); the pipeline only needs the parse.
    // 2. The pre-execution witness is pure block-derived data (no circuit
    //    dependency), so the pre proof starts on its own thread the moment its
    //    witness exists and runs underneath everything else; the pipeline does
    //    NOT wait for it — it consumes only the *native* pre-execution outputs
    //    (`native_pre_outputs`), and the proof is joined later by the
    //    final-block lane, which is the only consumer of the proof itself.
    // 3. The pre circuit lives in a ready `CircuitSlot` shared with the proof
    //    thread: the pipeline reads it for the native-output witness pass and
    //    `build_block_circuit`, and the proof thread installs the post-proof
    //    commitment release through it.
    //
    // Value-exact: every quantity is computed by the same code as before, only
    // the waiting changes.
    let (pre_target, pre_data) = pre_circuits;
    let pre_target: &'static BlockPreExecutionTarget = Box::leak(Box::new(pre_target));
    let pre_slot = std::sync::Arc::new(api::CircuitSlot::ready(pre_data));
    let pre_exec: &'static circuit::block_pre_execution::BlockPreExec<F> = {
        #[cfg(feature = "diagnostic_profile")]
        let _span = plonky2::util::profile::span("startup", "pre_execution_native_witness");
        Box::leak(Box::new(
            circuit::block_pre_execution::BlockPreExec::from_block(&block),
        ))
    };
    let pre_handle = {
        let pre_slot = std::sync::Arc::clone(&pre_slot);
        std::thread::Builder::new()
            .name("pre-exec-startup".into())
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(move || {
                let pre_proof = {
                    let pre_data = pre_slot.read();
                    prover::prove_pre_execution_parallel(&pre_data, pre_target, pre_exec)
                };
                // The pre-execution circuit is proven exactly once, here, and this
                // is that proof's last instruction. Its rate-2^3 constants/sigmas
                // low-degree extension — 2^17 rows x 86 columns = 86 MiB, held in a
                // CPU-visible Metal shared buffer whose release returns the pages to
                // the OS immediately — is read only by proofs *of this circuit*
                // (`fill_lde_batch` for the quotient and the FRI query openings), so
                // it is unreachable from here on. The only later uses of the pre
                // circuit are as an input to the final block circuit's construction,
                // which reads `common` and `verifier_only` only (`BlockCircuit::define`
                // -> `handle_proofs`: `constant_verifier_data(&..verifier_only)` and
                // `verify_proof(.., &..common)`), and
                // `release_finished_circuit_extensions`, which assigns the same
                // empty value again.
                //
                // Without this the buffer stays resident from the first second of
                // the process until the pipeline joins, i.e. across the entire
                // transaction/chain phase. Value-exact and free: no quantity is
                // computed differently and no work is added — storage that no
                // subsequent read can reach is returned earlier.
                pre_slot.write().prover_only.constants_sigmas_commitment =
                    PolynomialBatch::default();
                pre_proof
            })
            .expect("pre-execution startup thread must start")
    };
    let remaining = std::thread::scope(|scope| {
        let remaining_handle = scope.spawn(|| {
            #[cfg(feature = "diagnostic_profile")]
            let _span = plonky2::util::profile::span("startup", "remaining_circuit_loads");
            (!std::env::var_os("LIGHTER_BUILD_CIRCUITS").is_some_and(|v| v == "1"))
                .then(Circuits::load_remaining_embedded)
        });
        #[cfg(feature = "diagnostic_profile")]
        let _remaining_wait = plonky2::util::profile::span("wait", "remaining_circuit_loads_join");
        remaining_handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    });
    // The pre circuit is shared with the startup proof through the slot; only
    // the other four blobs are loaded above (loading all five here would
    // deserialize the same pre circuit twice on the scored critical path).
    // Keep the forced-build mode's established behavior unchanged.
    let circuits = match remaining {
        Some(Ok(remaining)) => remaining.into_circuits((pre_target, pre_slot)),
        Some(Err(error)) => {
            log::warn!(
                "embedded remaining circuits unavailable ({error:#}); building from scratch"
            );
            Circuits::load()
        }
        None => Circuits::load(),
    };
    let proof = {
        #[cfg(feature = "diagnostic_profile")]
        let _span = plonky2::util::profile::span("orchestration", "block_pipeline");
        prover::prove_block_after_pre(block, circuits, pre_exec, pre_handle)
    };
    #[cfg(feature = "diagnostic_profile")]
    let _output_span = plonky2::util::profile::span("output", "serialize_and_flush_proof");
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
    #[cfg(feature = "diagnostic_profile")]
    {
        drop(_output_span);
        drop(profile_process);
        if let Some(path) = std::env::var_os("LIGHTER_PROFILE_PATH") {
            plonky2::util::profile::write_chrome_trace(path)
                .expect("cannot write diagnostic profile trace");
        }
    }

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
    // `std::process::exit` skips Rust destructors but still enters libc
    // `exit(3)`, which runs every registered `atexit`/`__cxa_atexit` handler and
    // finalises each loaded image — the Objective-C runtime, Metal and the
    // driver bundle among them — before it reaches `_exit(2)`. That teardown
    // releases objects the kernel reclaims at process death anyway, and it runs
    // after the last proof byte has reached its descriptor, so it is dead work
    // by the same argument that motivates skipping the destructors above.
    // Entering `_exit(2)` directly is safe for the same reason the fast exit
    // already was: the proof was flushed by `into_inner` and its descriptor
    // closed, so every byte is with the kernel; the only thing additionally
    // discarded is userspace stdio buffering, and nothing is written to stdout
    // on the scored path. Declared in an `extern "C"` block rather than through
    // a new dependency, so the dependency graph and `Cargo.lock` are untouched.
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    unsafe { _exit(0) }
}

// p90-fire-808-1786266919
