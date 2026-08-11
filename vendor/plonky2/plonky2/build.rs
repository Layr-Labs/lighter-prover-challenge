use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_src = manifest_dir.join("src/hash/poseidon2/poseidon2.metal");
    let metallib_out = out_dir.join("poseidon2.metallib");

    println!("cargo:rerun-if-changed={}", shader_src.display());

    let is_macos = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");

    // On macOS, try to compile the shader source to a real metallib.
    if is_macos {
        let metal_ok = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if metal_ok {
            let air = out_dir.join("poseidon2.air");
            let c1 = Command::new("xcrun")
                .args(["-sdk", "macosx", "metal", "-c"])
                .arg(&shader_src)
                .arg("-o").arg(&air)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if c1 {
                let c2 = Command::new("xcrun")
                    .args(["-sdk", "macosx", "metallib"])
                    .arg(&air)
                    .arg("-o").arg(&metallib_out)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if c2 && metallib_out.is_file() {
                    println!("cargo:rustc-env=LIGHTER_METALLIB={}", metallib_out.display());
                    return;
                }
            }
        }
    }

    // No metal compiler available: write a dummy metallib so include_bytes!
    // compiles. At runtime, new_library_with_data will fail on this invalid
    // data and the code falls back to new_library_with_source(SHADER_SOURCE),
    // which compiles the CURRENT .metal source (including any new kernel
    // branches) using the GPU driver's built-in compiler.
    let _ = std::fs::write(&metallib_out, b"EMPTY");
    println!("cargo:rustc-env=LIGHTER_METALLIB={}", metallib_out.display());
}
