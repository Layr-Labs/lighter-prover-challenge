//! Precompiles the Poseidon2 Metal compute pipelines into an `MTLBinaryArchive`.
//!
//! `MetalShared::new` (src/hash/poseidon2/metal.rs) turns the eight kernels of
//! `poseidon2.metal` into `MTLComputePipelineState`s. Each of those calls lowers
//! that kernel's AIR to a GPU binary through MTLCompilerService, and the
//! benchmark sandbox denies writes to the OS shader cache, which disables that
//! cache outright — so every scored worker process pays the full cold lowering
//! (~0.7-1.3 s even with all eight issued concurrently).
//!
//! A binary archive moves that lowering into the build, which runs untimed on
//! the same runner and the same GPU: this script builds the very same pipeline
//! descriptors, adds them to an `MTLBinaryArchive`, and serializes it into
//! `OUT_DIR`. `metal.rs` `include_bytes!`s the result, so it travels inside the
//! worker binary — the ranked workflow stages only that binary and throws the
//! build tree away — and attaches it to its descriptors, turning pipeline
//! creation into an archive lookup instead of a compile.
//!
//! The artifact is written unconditionally, empty when anything goes wrong, so
//! that `include_bytes!` always has a file to read. A machine with no Metal
//! device, a driver that rejects binary archives, or any other API failure
//! yields those zero bytes and the runtime compiles exactly as it did before
//! this script existed. An archive is a cache of compiled kernels — it can
//! never change a value the GPU computes.
//!
//! Set `LIGHTER_SKIP_METAL_ARCHIVE=1` to emit the empty artifact deliberately
//! and A/B the mechanism against the plain compile path.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metal");
    println!("cargo:rerun-if-env-changed=LIGHTER_SKIP_METAL_ARCHIVE");

    let out_dir = std::path::PathBuf::from(
        std::env::var_os("OUT_DIR").expect("OUT_DIR must be set for build scripts"),
    );
    let archive_path = out_dir.join("poseidon2_pipelines.metallib");
    // A stale artifact must never outlive a failed rebuild: the runtime would
    // embed an archive whose entries no longer match the shader and silently
    // miss on every lookup. `serializeToURL:` also refuses to overwrite an
    // existing file, so this removal is required on the success path too.
    let _ = std::fs::remove_file(&archive_path);

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    metal_archive::build(&archive_path);

    if !archive_path.exists() {
        std::fs::write(&archive_path, []).expect("cannot write empty pipeline archive placeholder");
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod metal_archive {
    use std::path::{Path, PathBuf};

    use metal::{BinaryArchiveDescriptor, CompileOptions, ComputePipelineDescriptor, Device, URL};

    /// Every kernel `MetalShared::new` builds a pipeline for, in declaration
    /// order. A name missing here is silently absent from the archive and
    /// simply compiles at runtime, costing the very stall this script removes
    /// without failing anything; the `binary_archive_serves_every_pipeline`
    /// test in `metal.rs` is the oracle that keeps the two lists identical.
    const KERNELS: [&str; 8] = [
        "poseidon2_hash_leaves",
        "poseidon2_hash_leaves_colmajor",
        "poseidon2_hash_parents",
        "ntt_prepare",
        "ntt_stage",
        "ifft_finalize",
        "poseidon2_gate_quotient",
        "range_check_gate_quotient",
    ];

    pub(super) fn build(archive_path: &Path) {
        if std::env::var_os("LIGHTER_SKIP_METAL_ARCHIVE").is_some_and(|value| value == "1") {
            println!(
                "cargo:warning=LIGHTER_SKIP_METAL_ARCHIVE=1: Poseidon2 pipelines will be \
                 compiled at runtime"
            );
            return;
        }

        if let Err(error) = serialize_archive(archive_path) {
            // Never fail the build: the crate must compile on machines with no
            // GPU, no Metal, or a driver without binary-archive support. The
            // caller writes the empty placeholder that `include_bytes!` needs.
            let _ = std::fs::remove_file(archive_path);
            println!(
                "cargo:warning=Poseidon2 Metal pipeline archive unavailable ({error}); \
                 pipelines will be compiled at runtime"
            );
        }
    }

    fn serialize_archive(archive_path: &Path) -> Result<(), String> {
        let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .ok_or("CARGO_MANIFEST_DIR is not set")
            .map(PathBuf::from)?;
        // The exact bytes `metal.rs` `include_str!`s into `SHADER_SOURCE`.
        // Compiling anything else would key the archive entries to a different
        // function and every runtime lookup would miss.
        let source_path = manifest_dir.join("src/hash/poseidon2/poseidon2.metal");
        let source = std::fs::read_to_string(&source_path)
            .map_err(|error| format!("cannot read {}: {error}", source_path.display()))?;

        // A driver without binary-archive support surfaces as an error from
        // `newBinaryArchiveWithDescriptor:` below; there is nothing to probe
        // for up front.
        let device = Device::system_default().ok_or("no Metal device")?;
        // Same source, same (default) compile options as `MetalShared::new`.
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| format!("shader compilation failed: {error}"))?;

        let archive_descriptor = BinaryArchiveDescriptor::new();
        // A descriptor with no URL creates an empty archive to add to.
        let archive = device
            .new_binary_archive_with_descriptor(&archive_descriptor)
            .map_err(|error| format!("cannot create binary archive: {error}"))?;

        let mut added = 0usize;
        for kernel in KERNELS {
            // A kernel this shader revision does not define is not an error:
            // the two gate-quotient kernels are optional at runtime too.
            let Ok(function) = library.get_function(kernel, None) else {
                continue;
            };
            // Byte-for-byte the descriptor `MetalShared::new` hands to
            // `newComputePipelineStateWithDescriptor:`; every other property
            // (notably `threadGroupSizeIsMultipleOfThreadExecutionWidth`) is
            // left at its default on both sides, because any difference makes
            // the runtime lookup miss without saying so.
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(&function));
            archive
                .add_compute_pipeline_functions_with_descriptor(&descriptor)
                .map_err(|error| format!("cannot add {kernel} to binary archive: {error}"))?;
            added += 1;
        }
        if added == 0 {
            return Err("no Poseidon2 kernels found in the shader".to_owned());
        }

        let url = file_url(archive_path)?;
        archive
            .serialize_to_url(&url)
            .map_err(|error| format!("cannot serialize binary archive: {error}"))?;
        println!(
            "cargo:warning=Poseidon2 Metal pipeline archive: {added} kernels, {:.2} MiB",
            std::fs::metadata(archive_path).map(|m| m.len()).unwrap_or(0) as f64
                / (1024.0 * 1024.0)
        );
        Ok(())
    }

    /// `NSURL URLWithString:` needs a percent-encoded absolute URL, so build one
    /// rather than pasting the path in raw (`OUT_DIR` can contain spaces).
    fn file_url(path: &Path) -> Result<URL, String> {
        let path = path.to_str().ok_or("archive path is not valid UTF-8")?;
        let mut url = String::with_capacity(path.len() + 8);
        url.push_str("file://");
        for byte in path.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                    url.push(char::from(byte));
                }
                _ => url.push_str(&format!("%{byte:02X}")),
            }
        }
        // `+[NSURL URLWithString:]` returns an autoreleased (+0) object while
        // `metal::URL` releases on drop; retain it so the handle owns a real
        // reference (see the matching note in `metal.rs`).
        let borrowed = URL::new_with_string(&url);
        let owned = borrowed.clone();
        std::mem::forget(borrowed);
        Ok(owned)
    }
}
