use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn run_xcrun(arguments: &[&str]) {
    let status = Command::new("xcrun")
        .args(arguments)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch xcrun {arguments:?}: {error}"));
    assert!(status.success(), "xcrun {arguments:?} failed with {status}");
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("src/hash/poseidon2");
    let source = shader_dir.join("poseidon2.metal");
    let prebuilt = shader_dir.join("poseidon2.metallib");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let air = out_dir.join("poseidon2.air");
    let metallib = out_dir.join("poseidon2.metallib");

    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", prebuilt.display());

    let target_is_macos = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let host_is_macos = env::var("HOST")
        .map(|host| host.ends_with("-apple-darwin"))
        .unwrap_or(false);
    if target_is_macos && host_is_macos {
        let source = source.to_str().expect("Metal source path is not UTF-8");
        let air = air.to_str().expect("AIR output path is not UTF-8");
        let metallib_path = metallib
            .to_str()
            .expect("metallib output path is not UTF-8");
        run_xcrun(&["-sdk", "macosx", "metal", "-c", source, "-o", air]);
        run_xcrun(&[
            "-sdk",
            "macosx",
            "metallib",
            air,
            "-o",
            metallib_path,
        ]);
    } else {
        fs::copy(&prebuilt, &metallib).unwrap_or_else(|error| {
            panic!(
                "failed to stage {} as {}: {error}",
                prebuilt.display(),
                metallib.display()
            )
        });
    }

    println!(
        "cargo:rustc-env=PLONKY2_POSEIDON2_METALLIB={}",
        metallib.display()
    );
}
