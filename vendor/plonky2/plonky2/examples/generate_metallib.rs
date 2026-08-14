//! Regenerates poseidon2.metallib as a fat archive (air64 + applegpu) using
//! the same MTLBinaryArchive sequence as PR #4183. Run with:
//!
//! cargo run --release --manifest-path vendor/plonky2/plonky2/Cargo.toml
//!   --example generate_metallib --features parallel,std
//!
//! The shader source path and output path are hardcoded relative to the
//! workspace root; run this from the repository root. The 10 kernels are the
//! same list METALLIB_REQUIRED_KERNELS in src/hash/poseidon2/metal.rs uses
//! for the load probe.
//!
//! On a GPU mismatch at load time (e.g. an M5-generated applegpu slice on an
//! M4 Pro), Metal ignores the slice and lowers the air64 archive instead -
//! the same path the pre-#4183 tip used. This generator does NOT produce a
//! wrong proof; the worst case is the ~0.20 s fatlib advantage is lost.
//!
//! This example is a build-time tool only; it is never compiled into the
//! prover binary and is not part of the submission surface.

use metal::*;
use std::path::Path;

const SHADER_PATH: &str = "vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal";
const OUTPUT_PATH: &str = "vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metallib";

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

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {:?}", device.name());

    let source = std::fs::read_to_string(SHADER_PATH).expect("cannot read shader source");
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .expect("MSL compile failed");

    let descriptor = BinaryArchiveDescriptor::new();
    let archive = device
        .new_binary_archive_with_descriptor(&descriptor)
        .expect("cannot create binary archive");

    for name in KERNELS {
        let function = library
            .get_function(name, None)
            .unwrap_or_else(|_| panic!("kernel {name} not found in compiled library"));
        let pipeline_desc = ComputePipelineDescriptor::new();
        pipeline_desc.set_compute_function(Some(&function));
        let ok = archive
            .add_compute_pipeline_functions_with_descriptor(&pipeline_desc)
            .unwrap_or_else(|e| panic!("add_compute_pipeline_functions failed for {name}: {e}"));
        assert!(ok, "archive rejected kernel {name}");
        println!("archived: {name}");
    }

    let url = URL::new_with_string(&format!(
        "file://{}",
        std::env::current_dir()
            .expect("no cwd")
            .join(OUTPUT_PATH)
            .display()
    ));
    let ok = archive
        .serialize_to_url(&url)
        .expect("serialize_to_url failed");
    assert!(ok, "archive serialization returned false");
    println!("wrote: {OUTPUT_PATH}");

    // Sanity: reload the archive and probe the kernels, mirroring
    // metallib_loads_and_exposes_every_kernel.
    let bytes = std::fs::read(OUTPUT_PATH).expect("cannot read written metallib");
    let reloaded = device
        .new_library_with_data(&bytes)
        .expect("written metallib does not load");
    for name in KERNELS {
        assert!(
            reloaded.get_function(name, None).is_ok(),
            "written metallib missing kernel {name}"
        );
    }
    println!("reload probe passed: {} kernels", KERNELS.len());
    let size = std::fs::metadata(OUTPUT_PATH).expect("no output").len();
    println!("final size: {size} bytes");
}
