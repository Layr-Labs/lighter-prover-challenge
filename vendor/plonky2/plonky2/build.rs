//! Builds a device-binary Metal pipeline archive on the ranked M4 builder.
//!
//! The committed metallib contains portable Metal IR. That skips the MSL
//! front end, but every sandboxed worker still lowers the IR to GPU binaries
//! because its normal Metal cache is read-only. The official build job and
//! benchmark job both run on M4-labelled hosts, so doing that lowering here
//! lets the worker load a GPU-family-specific binary archive instead.

use std::path::Path;
use std::{env, fs};

const ARCHIVE_FILE: &str = "poseidon2.binary.metallib";

fn main() {
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metallib");

    let output = Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(ARCHIVE_FILE);
    // `include_bytes!` needs the file even on non-Metal builders and whenever
    // archive generation is unavailable. An empty artifact means "use the
    // existing runtime lowering path".
    fs::write(&output, []).expect("cannot initialize Metal archive output");

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if let Err(error) = macos::build_archive(&output) {
        println!("cargo:warning=Metal binary archive unavailable: {error}");
        fs::write(&output, []).expect("cannot restore empty Metal archive output");
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod macos {
    use std::fs;
    use std::path::Path;

    use metal::{
        BinaryArchiveDescriptor, ComputePipelineDescriptor, Device, DeviceRef, LibraryRef, URL,
    };

    const KERNELS: [&str; 10] = [
        "poseidon2_hash_leaves",
        "poseidon2_hash_leaves_colmajor",
        "poseidon2_hash_parents",
        "poseidon2_absorb_pass",
        "ntt_prepare",
        "ntt_stage",
        "ifft_finalize",
        "poseidon2_gate_quotient",
        "range_check_gate_quotient",
        "permutation_quotient",
    ];

    pub(super) fn build_archive(output: &Path) -> Result<(), String> {
        let device = Device::system_default().ok_or("no Metal device on build host")?;
        let library_path = Path::new(&std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
            .join("src/hash/poseidon2/poseidon2.metallib");
        let library_bytes = fs::read(&library_path)
            .map_err(|error| format!("cannot read {}: {error}", library_path.display()))?;
        let library = device
            .new_library_with_data(&library_bytes)
            .map_err(|error| format!("cannot load committed metallib: {error}"))?;

        let archive_descriptor = BinaryArchiveDescriptor::new();
        let archive = device
            .new_binary_archive_with_descriptor(&archive_descriptor)
            .map_err(|error| format!("cannot create binary archive: {error}"))?;

        for name in KERNELS {
            add_pipeline(&device, &library, &archive, name)?;
        }

        // `serializeToURL` creates the file itself. Remove the zero-length
        // placeholder only after all descriptors have compiled successfully.
        fs::remove_file(output)
            .map_err(|error| format!("cannot replace archive placeholder: {error}"))?;
        let url = URL::new_with_string(&format!("file://{}", output.display()));
        let serialized = archive
            .serialize_to_url(&url)
            .map_err(|error| format!("cannot serialize binary archive: {error}"))?;
        if !serialized {
            return Err("Metal declined to serialize the binary archive".to_owned());
        }
        let length = fs::metadata(output)
            .map_err(|error| format!("cannot inspect serialized archive: {error}"))?
            .len();
        if length == 0 {
            return Err("serialized binary archive is empty".to_owned());
        }
        println!("cargo:warning=embedded {length}-byte M4 Metal binary archive");
        Ok(())
    }

    fn add_pipeline(
        device: &DeviceRef,
        library: &LibraryRef,
        archive: &metal::BinaryArchiveRef,
        name: &str,
    ) -> Result<(), String> {
        let function = library
            .get_function(name, None)
            .map_err(|error| format!("kernel {name} unavailable: {error}"))?;
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_binary_archives(&[archive]);
        let added = archive
            .add_compute_pipeline_functions_with_descriptor(&descriptor)
            .map_err(|error| format!("cannot archive {name}: {error}"))?;
        if !added {
            return Err(format!("Metal declined to archive {name}"));
        }
        // Materialize the state before serialization, mirroring Apple's
        // device-built archive workflow.
        device
            .new_compute_pipeline_state(&descriptor)
            .map_err(|error| format!("cannot compile archived {name}: {error}"))?;
        Ok(())
    }
}
