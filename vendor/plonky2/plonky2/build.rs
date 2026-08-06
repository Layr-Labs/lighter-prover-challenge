//! Best-effort compilation of the Metal Poseidon2 shader to a `.metallib` at
//! build time so the worker never pays a runtime MSL compile (the first run
//! of a new shader source otherwise takes a multi-second cold-cache hit in
//! MTLCompilerService). If the Metal toolchain is unavailable the build still
//! succeeds and the runtime falls back to compiling the shader from source.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(poseidon2_metallib)");
    let shader = PathBuf::from("src/hash/poseidon2/poseidon2.metal");
    println!("cargo:rerun-if-changed={}", shader.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let metallib = out_dir.join("poseidon2.metallib");

    let compiled = cfg!(target_os = "macos")
        && Command::new("/usr/bin/xcrun")
            .args(["-sdk", "macosx", "metal"])
            .arg(&shader)
            .arg("-o")
            .arg(&metallib)
            .status()
            .is_ok_and(|status| status.success())
        && metallib.metadata().is_ok_and(|meta| meta.len() > 0);

    if compiled {
        println!("cargo:rustc-cfg=poseidon2_metallib");
    } else {
        // Keep include_bytes! in the (cfg'd-out) consumer harmless on every
        // build configuration by always producing the file.
        let _ = fs::write(&metallib, []);
        println!(
            "cargo:warning=Metal toolchain unavailable; poseidon2 shader will compile from source at runtime"
        );
    }
}
