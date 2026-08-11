// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Builds the five startup circuits at compile time and serializes them into
//! OUT_DIR blobs that `src/embedded.rs` embeds into the prove binary.
//!
//! Compilation runs in the benchmark's untimed CI job, so the multi-second
//! circuit construction here is free; the scored worker process then loads
//! the blobs in a fraction of the build time (`Circuits::from_embedded`).
//!
//! The circuits and their parameters must match `src/api.rs`
//! (`Circuits::new`/`PathCircuits::new`) exactly; the ignored test
//! `embedded_matches_rebuilt` in `src/embedded.rs` is the equality oracle for
//! that. Cargo re-runs this script whenever the `circuit` or `plonky2` crates
//! change (they are build-dependencies), so the blobs cannot go stale.
//!
//! Set `LIGHTER_SKIP_EMBED=1` to write empty blobs instead (the runtime then
//! falls back to building circuits from scratch); use this to A/B the
//! mechanism or to cut compile time while iterating on unrelated code.
//!
//! On macOS the build also runs a disposable Objective-C helper that asks the
//! build host's Metal device to serialize the exact checked-in Poseidon2
//! pipelines. Set `LIGHTER_SKIP_METAL_ARCHIVE=1` to embed an empty archive and
//! force the generic runtime path.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::embed::serialize_embedded;
use circuit::types::config::{C, CIRCUIT_CONFIG};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

// Mirrors of the `src/api.rs` constants (a build script cannot import from
// the crate it builds). Divergence is caught by `embedded_matches_rebuilt`:
// the freshly built and embedded circuits would differ in `circuit_digest`.
const CHAIN_ID: u32 = 304;
const HEAVY_TX_PER_PROOF: usize = 4;
const LIGHT_TX_PER_PROOF: usize = 10;
const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

const BLOB_NAMES: [&str; 5] = [
    "pre.embed",
    "heavy_tx.embed",
    "heavy_chain.embed",
    "light_tx.embed",
    "light_chain.embed",
];
const METAL_PIPELINE_ARCHIVE: &str = "poseidon2-pipelines.binary.metallib";
const MAX_METAL_PIPELINE_ARCHIVE_BYTES: u64 = 256 << 20;

/// Standalone ARC helper used only by the unscored macOS build job.
///
/// Keeping every Objective-C object under ARC avoids the ownership bug in
/// `metal-rs::URL::new_with_string` that made earlier in-process generators
/// crash while draining an autorelease pool. The helper does not need the
/// optional Metal command-line toolchain: it loads the committed AIR library
/// through Metal.framework, records either the hot eleven-descriptor roster or
/// the exact generic ten-descriptor roster, serializes the device binary slice,
/// and proves every entry is a hard archive hit in a fresh process.
const METAL_ARCHIVE_HELPER_SOURCE: &str = r#"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <stdio.h>
#import <unistd.h>

static void mark(const char *stage, NSString *name) {
    if (name != nil) {
        fprintf(stderr, "metal-archive-helper: stage=%s name=%s\n",
                stage, name.UTF8String);
    } else {
        fprintf(stderr, "metal-archive-helper: stage=%s\n", stage);
    }
    fflush(stderr);
}

static int fail(NSString *operation, NSError *error) {
    if (error != nil) {
        fprintf(stderr, "metal-archive-helper: %s: %s\n",
                operation.UTF8String, error.localizedDescription.UTF8String);
    } else {
        fprintf(stderr, "metal-archive-helper: %s\n", operation.UTF8String);
    }
    fflush(stderr);
    return 1;
}

__attribute__((noreturn)) static void succeed_without_teardown(void) {
    // The parent has all the output it needs. This is a disposable child, so
    // exit without draining ARC/autorelease objects backed by Metal compiler
    // state; a fresh process independently reopens and hard-verifies the file.
    fflush(NULL);
    _exit(0);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr,
                    "usage: metal-archive-helper generate11|verify11|generate10|verify10 "
                    "INPUT.metallib "
                    "OUTPUT.binary.metallib\n");
            return 2;
        }

        NSString *mode = [NSString stringWithUTF8String:argv[1]];
        NSString *inputPath = [NSString stringWithUTF8String:argv[2]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[3]];
        if (mode == nil || inputPath == nil || outputPath == nil) {
            return fail(@"paths are not valid UTF-8", nil);
        }
        BOOL generating = [mode isEqualToString:@"generate11"] ||
                          [mode isEqualToString:@"generate10"];
        BOOL verifying = [mode isEqualToString:@"verify11"] ||
                         [mode isEqualToString:@"verify10"];
        BOOL includesHot = [mode hasSuffix:@"11"];
        if (!generating && !verifying) {
            return fail(@"mode must select generate/verify and an 11/10 roster", nil);
        }

        NSError *error = nil;
        NSData *bytes = [NSData dataWithContentsOfFile:inputPath
                                               options:NSDataReadingMappedIfSafe
                                                 error:&error];
        if (bytes == nil || bytes.length == 0) {
            return fail(@"reading input metallib", error);
        }

        mark("device-begin", nil);
        __attribute__((objc_precise_lifetime)) id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return fail(@"no system Metal device", nil);
        }
        mark("device-done", nil);

        // DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the mapped NSData bytes, so
        // the MTLLibrary never borrows storage with a shorter lifetime.
        dispatch_data_t libraryData = dispatch_data_create(
            bytes.bytes, bytes.length, dispatch_get_main_queue(),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (libraryData == nil) {
            return fail(@"creating dispatch data", nil);
        }
        error = nil;
        mark("library-begin", nil);
        __attribute__((objc_precise_lifetime)) id<MTLLibrary> library =
            [device newLibraryWithData:libraryData error:&error];
        if (library == nil) {
            return fail(@"loading input metallib", error);
        }
        mark("library-done", nil);

        __attribute__((objc_precise_lifetime)) NSMutableArray<NSString *> *names =
            [NSMutableArray arrayWithArray:@[
            @"poseidon2_hash_leaves",
            @"poseidon2_hash_leaves_colmajor",
            @"poseidon2_hash_parents",
            @"poseidon2_absorb_pass",
            @"ntt_prepare",
            @"ntt_stage",
            @"ifft_finalize",
            @"poseidon2_gate_quotient",
            @"range_check_gate_quotient",
            @"permutation_quotient",
        ]];
        if (includesHot) {
            [names insertObject:@"poseidon2_hash_leaves_colmajor_hot" atIndex:2];
        }
        NSUInteger expectedCount = includesHot ? 11 : 10;
        if (names.count != expectedCount) {
            return fail(@"archive roster has an unexpected function count", nil);
        }

        __attribute__((objc_precise_lifetime))
        NSMutableArray<MTLComputePipelineDescriptor *> *pipelines =
            [NSMutableArray arrayWithCapacity:names.count];
        for (NSString *name in names) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (function == nil) {
                return fail([@"resolving function " stringByAppendingString:name], nil);
            }
            MTLComputePipelineDescriptor *pipeline =
                [[MTLComputePipelineDescriptor alloc] init];
            pipeline.computeFunction = function;
            [pipelines addObject:pipeline];
        }

        NSURL *outputURL = [NSURL fileURLWithPath:outputPath isDirectory:NO];
        if (generating) {
            mark("archive-create-begin", nil);
            __attribute__((objc_precise_lifetime)) MTLBinaryArchiveDescriptor *archiveDescriptor =
                [[MTLBinaryArchiveDescriptor alloc] init];
            error = nil;
            __attribute__((objc_precise_lifetime)) id<MTLBinaryArchive> archive =
                [device newBinaryArchiveWithDescriptor:archiveDescriptor error:&error];
            if (archive == nil) {
                return fail(@"creating empty binary archive", error);
            }
            mark("archive-create-done", nil);
            for (NSUInteger index = 0; index < pipelines.count; index++) {
                mark("add-begin", names[index]);
                error = nil;
                if (![archive addComputePipelineFunctionsWithDescriptor:pipelines[index]
                                                                   error:&error]) {
                    return fail([@"adding pipeline " stringByAppendingString:names[index]],
                                error);
                }
                mark("add-done", names[index]);
            }
            [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];
            mark("serialize-begin", nil);
            error = nil;
            if (![archive serializeToURL:outputURL error:&error]) {
                return fail(@"serializing binary archive", error);
            }
            mark("serialize-done", nil);

            // Do not leave this lexical scope: doing so releases `archive`
            // before `_exit`, which can re-enter the Metal compiler/driver
            // teardown path that this disposable helper exists to isolate.
            error = nil;
            NSDictionary<NSFileAttributeKey, id> *attributes =
                [[NSFileManager defaultManager] attributesOfItemAtPath:outputPath error:&error];
            NSNumber *fileSize = attributes[NSFileSize];
            if (attributes == nil || fileSize == nil || fileSize.unsignedLongLongValue == 0) {
                return fail(@"serialized archive is empty", error);
            }
            fprintf(stdout,
                    "metal-archive-helper: generated %lu/%lu descriptors on %s; %llu bytes\n",
                    (unsigned long)pipelines.count, (unsigned long)expectedCount,
                    device.name.UTF8String, fileSize.unsignedLongLongValue);
            succeed_without_teardown();
        }

        error = nil;
        NSDictionary<NSFileAttributeKey, id> *attributes =
            [[NSFileManager defaultManager] attributesOfItemAtPath:outputPath error:&error];
        NSNumber *fileSize = attributes[NSFileSize];
        if (attributes == nil || fileSize == nil || fileSize.unsignedLongLongValue == 0) {
            return fail(@"serialized archive is empty", error);
        }

        mark("reopen-begin", nil);
        __attribute__((objc_precise_lifetime)) MTLBinaryArchiveDescriptor *loadDescriptor =
            [[MTLBinaryArchiveDescriptor alloc] init];
        loadDescriptor.url = outputURL;
        error = nil;
        __attribute__((objc_precise_lifetime)) id<MTLBinaryArchive> loaded =
            [device newBinaryArchiveWithDescriptor:loadDescriptor error:&error];
        if (loaded == nil) {
            return fail(@"reopening serialized archive", error);
        }
        mark("reopen-done", nil);

        __attribute__((objc_precise_lifetime))
        NSMutableArray<id<MTLComputePipelineState>> *states =
            [NSMutableArray arrayWithCapacity:pipelines.count];
        for (NSUInteger index = 0; index < pipelines.count; index++) {
            MTLComputePipelineDescriptor *pipeline = pipelines[index];
            pipeline.binaryArchives = @[loaded];
            mark("verify-begin", names[index]);
            error = nil;
            id<MTLComputePipelineState> state =
                [device newComputePipelineStateWithDescriptor:pipeline
                                                      options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                   reflection:NULL
                                                        error:&error];
            if (state == nil || error != nil) {
                return fail([@"hard archive miss for " stringByAppendingString:names[index]], error);
            }
            [states addObject:state];
            mark("verify-done", names[index]);
        }

        fprintf(stdout,
                "metal-archive-helper: verified %lu/%lu hard hits in a fresh process on %s; "
                "%llu bytes\n",
                (unsigned long)pipelines.count, (unsigned long)names.count,
                device.name.UTF8String, fileSize.unsignedLongLongValue);
        succeed_without_teardown();
    }
}
"#;

fn write_blob(out_dir: &Path, name: &str, bytes: &[u8]) {
    let path = out_dir.join(name);
    std::fs::write(&path, bytes).unwrap_or_else(|error| {
        panic!("cannot write embedded circuit blob {}: {error}", path.display())
    });
    println!(
        "cargo:warning=embedded circuit blob {name}: {:.2} MiB",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
}

fn build_path_blobs(tx_per_proof: usize, tx_mode: u8) -> (Vec<u8>, Vec<u8>) {
    // Same construction as `PathCircuits::new`.
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
    let tx_target: BlockTxTarget = tx.target;
    let tx_data = tx.builder.build::<C>();

    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
    let chain_target = chain.target;
    let chain_data = chain.builder.build::<C>();

    let tx_blob = serialize_embedded(&tx_target, &tx_data)
        .expect("serializing block transaction circuit for embedding");
    let chain_blob = serialize_embedded(&chain_target, &chain_data)
        .expect("serializing block transaction chain circuit for embedding");
    (tx_blob, chain_blob)
}

fn write_empty_metal_archive(path: &Path) {
    std::fs::write(path, []).unwrap_or_else(|error| {
        panic!(
            "cannot write Metal archive fallback {}: {error}",
            path.display()
        )
    });
}

fn successful_stdout(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_metal_archive_helper(
    executable: &Path,
    mode: &str,
    metallib: &Path,
    archive: &Path,
    out_dir: &Path,
) -> Result<String, String> {
    let output = Command::new(executable)
        .arg(mode)
        .arg(metallib)
        .arg(archive)
        .env("TMPDIR", out_dir)
        .env("HOME", out_dir)
        .env("CFFIXED_USER_HOME", out_dir)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("starting Metal archive helper {mode} failed: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(format!(
            "Metal archive helper {mode} exited with {}: {stderr}",
            output.status
        ));
    }
    Ok(stdout)
}

fn build_verified_metal_archive_variant(
    helper_executable: &Path,
    metallib: &Path,
    child_path: &Path,
    out_dir: &Path,
    roster: usize,
) -> Result<u64, String> {
    assert!(matches!(roster, 10 | 11));
    let _ = std::fs::remove_file(child_path);
    let generate_mode = format!("generate{roster}");
    let verify_mode = format!("verify{roster}");
    let generated_marker = format!("generated {roster}/{roster} descriptors");
    let verified_marker = format!("verified {roster}/{roster} hard hits in a fresh process");

    let generation = run_metal_archive_helper(
        helper_executable,
        &generate_mode,
        metallib,
        child_path,
        out_dir,
    );
    match generation {
        Ok(stdout) if stdout.contains(&generated_marker) => println!("cargo:warning={stdout}"),
        Ok(stdout) => println!(
            "cargo:warning=Metal archive {roster}-roster generator returned unexpected output ({stdout}); attempting independent verification"
        ),
        Err(error) => println!(
            "cargo:warning=Metal archive {roster}-roster generator terminated ({error}); attempting independent verification of any serialized output"
        ),
    }

    // A Metal driver may finish serialization and then fault while ARC tears
    // down compiler objects. Generator status is therefore diagnostic only.
    // A bounded nonempty file is accepted exclusively after a fresh process
    // reopens it and proves the complete selected roster is a hard hit.
    let bytes = std::fs::metadata(child_path)
        .map_err(|error| format!("Metal archive helper produced no {roster}-roster archive: {error}"))
        .and_then(|metadata| {
            (metadata.len() > 0 && metadata.len() <= MAX_METAL_PIPELINE_ARCHIVE_BYTES)
                .then_some(metadata.len())
                .ok_or_else(|| {
                    format!(
                        "Metal archive helper {roster}-roster output size {} is outside 1..={MAX_METAL_PIPELINE_ARCHIVE_BYTES}",
                        metadata.len()
                    )
                })
        })?;

    // Reopen in a second process so the hard-hit proof cannot inherit any
    // process-local compiler/library state from archive generation.
    let stdout = run_metal_archive_helper(
        helper_executable,
        &verify_mode,
        metallib,
        child_path,
        out_dir,
    )?;
    if !stdout.contains(&verified_marker) {
        return Err(format!(
            "Metal archive {roster}-roster verifier returned unexpected output: {stdout}"
        ));
    }
    println!("cargo:warning={stdout}");
    Ok(bytes)
}

/// Builds a device archive from the exact checked-in Metal IR library.
///
/// Apple's device-built archive format contains one GPU-architecture slice and
/// is deployable to other compatible devices. Both ranked jobs require M4
/// Apple Silicon, but runtime still treats portability as untrusted: the whole
/// selected roster must hard-hit before any archived state is published. The
/// hot eleven-entry roster is preferred; a verified exact-generic ten-entry
/// archive is the fallback. Only failure of both embeds an empty stub.
fn build_metal_pipeline_archive(out_dir: &Path, poseidon_dir: &Path) {
    let path = out_dir.join(METAL_PIPELINE_ARCHIVE);
    let child_path = out_dir.join("poseidon2-pipelines.child.binary.metallib");
    let helper_source = out_dir.join("poseidon2-metal-archive-helper.m");
    let helper_executable = out_dir.join("poseidon2-metal-archive-helper");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&child_path);
    let _ = std::fs::remove_file(&helper_executable);

    if std::env::var_os("LIGHTER_SKIP_METAL_ARCHIVE").is_some_and(|value| value == "1") {
        write_empty_metal_archive(&path);
        println!(
            "cargo:warning=LIGHTER_SKIP_METAL_ARCHIVE=1: embedded Metal archive is an empty fallback stub"
        );
        return;
    }

    // Build scripts are host executables. A cross-check on Linux still needs
    // the include_bytes! target to exist, but must not touch Metal.framework.
    if !cfg!(target_os = "macos") {
        write_empty_metal_archive(&path);
        println!(
            "cargo:warning=Metal framework unavailable on this build host; embedded empty fallback"
        );
        return;
    }

    if let Err(error) = std::fs::write(&helper_source, METAL_ARCHIVE_HELPER_SOURCE) {
        write_empty_metal_archive(&path);
        println!(
            "cargo:warning=writing Metal archive helper failed ({error}); embedded empty fallback"
        );
        return;
    }

    let compiled = Command::new("/usr/bin/clang")
        .args([
            "-fobjc-arc",
            "-fblocks",
            "-Wall",
            "-Wextra",
            "-framework",
            "Foundation",
            "-framework",
            // Apple requires command-line Metal clients that do not otherwise
            // use graphics to link CoreGraphics before asking for the system
            // default device. The Rust `metal` crate gets this transitively
            // through its default `core-graphics-types/link` feature; this
            // standalone helper must request it explicitly.
            "CoreGraphics",
            "-framework",
            "Metal",
            "-o",
        ])
        .arg(&helper_executable)
        .arg(&helper_source)
        .env("TMPDIR", out_dir)
        .env("HOME", out_dir)
        .env("CFFIXED_USER_HOME", out_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("starting Objective-C compiler failed: {error}"))
        .and_then(|status| {
            status
                .success()
                .then_some(())
                .ok_or_else(|| format!("Objective-C compiler exited with {status}"))
        });

    if let Err(error) = compiled {
        write_empty_metal_archive(&path);
        println!(
            "cargo:warning=Metal archive helper unavailable ({error}); embedded empty fallback"
        );
        return;
    }

    // First preserve the full-upside path: the hot library and all eleven
    // descriptors. If the giant hot kernel faults the offline compiler (or
    // any hard hit fails), retry the byte-exact 59c generic library's ten
    // established descriptors. Runtime distinguishes the archive by strict
    // probes and lowers only the hot leaf pipeline asynchronously in 10 mode.
    let hot_metallib = poseidon_dir.join("poseidon2.metallib");
    let generic_metallib = poseidon_dir.join("poseidon2.generic.metallib");
    let translated = build_verified_metal_archive_variant(
        &helper_executable,
        &hot_metallib,
        &child_path,
        out_dir,
        11,
    )
    .map(|bytes| (bytes, 11usize))
    .or_else(|hot_error| {
        println!(
            "cargo:warning=hot 11-roster Metal archive unavailable ({hot_error}); retrying exact generic 10-roster archive"
        );
        build_verified_metal_archive_variant(
            &helper_executable,
            &generic_metallib,
            &child_path,
            out_dir,
            10,
        )
        .map(|bytes| (bytes, 10usize))
        .map_err(|generic_error| {
            format!(
                "hot 11-roster attempt failed ({hot_error}); generic 10-roster attempt failed ({generic_error})"
            )
        })
    });

    match translated {
        Ok((bytes, roster)) => {
            let sha256 = successful_stdout(
                Command::new("/usr/bin/shasum")
                    .args(["-a", "256"])
                    .arg(&child_path),
            );
            std::fs::rename(&child_path, &path).unwrap_or_else(|error| {
                panic!("cannot publish Metal archive {}: {error}", path.display())
            });
            println!(
                "cargo:warning=embedded verified {roster}-roster device Metal archive: {:.2} MiB",
                bytes as f64 / (1024.0 * 1024.0)
            );
            if let Some(sha256) = sha256 {
                println!("cargo:warning=Metal archive SHA-256: {sha256}");
            }
        }
        Err(error) => {
            let _ = std::fs::remove_file(&child_path);
            write_empty_metal_archive(&path);
            println!(
                "cargo:warning=Metal archive helper unavailable ({error}); embedded empty fallback"
            );
        }
    }
}

fn main() {
    // A dependency change (circuit/, vendor/plonky2/) rebuilds this script and
    // re-runs it regardless of these directives; bench's own sources do not
    // affect the blobs, so they are deliberately not tracked.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_EMBED");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_METAL_ARCHIVE");

    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"));
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );
    let poseidon_dir = manifest_dir.join("../vendor/plonky2/plonky2/src/hash/poseidon2");
    println!(
        "cargo:rerun-if-changed={}",
        poseidon_dir.join("poseidon2.metallib").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        poseidon_dir.join("poseidon2.generic.metallib").display()
    );
    build_metal_pipeline_archive(&out_dir, &poseidon_dir);

    if std::env::var_os("LIGHTER_SKIP_EMBED").is_some_and(|v| v == "1") {
        for name in BLOB_NAMES {
            write_blob(&out_dir, name, &[]);
        }
        println!("cargo:warning=LIGHTER_SKIP_EMBED=1: embedded circuit blobs are empty stubs");
        return;
    }

    // Circuit construction needs deep stacks (recursive gadget definition) on
    // both the spawning thread and the rayon workers, exactly like the prove
    // binary configures them at startup.
    rayon::ThreadPoolBuilder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .build_global()
        .expect("cannot configure build-script thread pool");

    std::thread::Builder::new()
        .stack_size(PROVER_THREAD_STACK_BYTES)
        .spawn(move || {
            // Same layout as `Circuits::new`: pre-execution circuit in
            // parallel with the heavy and light transaction paths.
            let (pre_blob, (heavy_blobs, light_blobs)) = rayon::join(
                || {
                    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                    let pre_target = pre.target;
                    let pre_data = pre.builder.build::<C>();
                    serialize_embedded(&pre_target, &pre_data)
                        .expect("serializing block pre-execution circuit for embedding")
                },
                || {
                    rayon::join(
                        || build_path_blobs(HEAVY_TX_PER_PROOF, TX_HEAVY),
                        || build_path_blobs(LIGHT_TX_PER_PROOF, TX_LIGHT),
                    )
                },
            );

            write_blob(&out_dir, "pre.embed", &pre_blob);
            write_blob(&out_dir, "heavy_tx.embed", &heavy_blobs.0);
            write_blob(&out_dir, "heavy_chain.embed", &heavy_blobs.1);
            write_blob(&out_dir, "light_tx.embed", &light_blobs.0);
            write_blob(&out_dir, "light_chain.embed", &light_blobs.1);
        })
        .expect("circuit build thread must start")
        .join()
        .expect("circuit build thread must finish");
}
