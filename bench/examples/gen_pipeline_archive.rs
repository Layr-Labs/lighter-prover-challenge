//! Regenerates the committed `MTLBinaryArchive` of the Poseidon2 compute
//! pipelines.
//!
//!     cargo run --release --example gen_pipeline_archive
//!
//! An `example` rather than a `bin` on purpose: `cargo build --release` does
//! not build examples, so this costs the scored build nothing, and it is not a
//! build script — `build.rs` must never call Metal.
//!
//! All the work is in `plonky2::hash::poseidon2::generate_pipeline_archive`, so
//! the generator reads the same shader source and the same kernel list the
//! runtime does and cannot drift from them.
fn main() {
    let default = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2_pipelines.metalar"
    );
    let out = std::env::args().nth(1).unwrap_or_else(|| default.to_string());
    let path = std::path::Path::new(&out);

    match plonky2::hash::poseidon2::generate_pipeline_archive(path) {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("pipeline archive generation FAILED: {error}");
            std::process::exit(1);
        }
    }
}
