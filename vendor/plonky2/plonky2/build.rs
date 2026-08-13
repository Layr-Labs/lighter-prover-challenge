use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("src/hash/poseidon2/poseidon2.metal");
    let committed = manifest.join("src/hash/poseidon2/poseidon2.metallib");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("poseidon2.metallib");

    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", committed.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("src/hash/poseidon2/gen_metallib.m").display()
    );

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "macos"
        && arch == "aarch64"
        && try_generate(&manifest, &src, &out)
    {
        println!("cargo:rustc-env=LIGHTER_DEVICE_METALLIB=1");
        return;
    }
    println!("cargo:warning=using committed poseidon2.metallib (device archive not generated)");
    std::fs::copy(&committed, &out).unwrap_or_else(|error| {
        panic!("copy committed metallib to OUT_DIR: {error}");
    });
}

fn try_generate(manifest: &Path, src: &Path, out: &Path) -> bool {
    let gen_src = manifest.join("src/hash/poseidon2/gen_metallib.m");
    let gen_bin = PathBuf::from(env::var("OUT_DIR").unwrap()).join("gen_metallib");
    let clang = Path::new("/usr/bin/clang");
    if !clang.exists() {
        return false;
    }
    let status = Command::new(clang)
        .args([
            "-fobjc-arc",
            "-O2",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-o",
        ])
        .arg(&gen_bin)
        .arg(&gen_src)
        .status();
    let Ok(status) = status else {
        return false;
    };
    if !status.success() {
        return false;
    }
    let status = Command::new(&gen_bin).arg(src).arg(out).status();
    matches!(status, Ok(status) if status.success() && out.is_file())
}
