use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only relevant on macOS (Metal is Apple-only).
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_src = manifest_dir.join("src/hash/poseidon2/poseidon2.metal");
    let metallib_dst = manifest_dir.join("src/hash/poseidon2/poseidon2.metallib");

    // Tell Cargo to re-run when the shader source changes.
    println!("cargo:rerun-if-changed={}", shader_src.display());

    // Check whether the Metal toolchain is available.
    let metal_ok = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !metal_ok {
        // No Metal toolchain (e.g. Command Line Tools only). Keep the
        // checked-in metallib; runtime will use whatever is embedded.
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let air = out_dir.join("poseidon2.air");

    // Compile MSL source -> AIR.
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&shader_src)
        .arg("-o")
        .arg(&air)
        .status();

    if !status.map(|s| s.success()).unwrap_or(false) {
        println!("cargo:warning=poseidon2.metal compilation failed, using checked-in metallib");
        return;
    }

    // Link AIR -> metallib, overwriting the checked-in artifact so that
    // include_bytes! picks up the freshly compiled version.
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&metallib_dst)
        .status();

    if status.map(|s| s.success()).unwrap_or(false) {
        println!("cargo:warning=poseidon2.metallib recompiled from source");
    } else {
        println!("cargo:warning=metallib linking failed, using checked-in metallib");
    }
}
