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

    if is_macos {
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

    // Fallback: copy checked-in metallib
    let checked_in = manifest_dir.join("src/hash/poseidon2/poseidon2.metallib");
    if checked_in.is_file() {
        let _ = std::fs::copy(&checked_in, &metallib_out);
    } else {
        let _ = std::fs::write(&metallib_out, b"EMPTY");
    }
    println!("cargo:rustc-env=LIGHTER_METALLIB={}", metallib_out.display());
}
