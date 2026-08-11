//! Builds the optional portable metallib containing the fused NTT kernel.

use std::path::Path;
use std::process::{Command, Output};
use std::{env, fs};

const FUSED_METALLIB: &str = "poseidon2_fused.metallib";

fn main() {
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metal");
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2_fused.metal");

    let output = Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(FUSED_METALLIB);
    // `include_bytes!` needs the artifact on all build hosts. Empty means the
    // worker loads the committed frontier metallib and uses radix-2 stages.
    fs::write(&output, []).expect("cannot initialize fused Metal library output");

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        if let Err(error) = build_fused_metallib(&output) {
            println!("cargo:warning=fused Metal library unavailable: {error}");
            fs::write(&output, []).expect("cannot restore empty fused Metal library output");
        }
    }
}

fn build_fused_metallib(output: &Path) -> Result<(), String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?;
    let source_dir = Path::new(&manifest_dir).join("src/hash/poseidon2");
    let source = source_dir.join("poseidon2_fused.metal");
    let air = output.with_extension("air");

    clear_if_present(&air)?;
    let first = run_metal(&source, &source_dir, &air)?;
    let compiled = if first.status.success() {
        first
    } else if optional_toolchain_missing(&first) {
        install_metal_toolchain()?;
        let _ = Command::new("/usr/bin/xcrun").arg("--kill-cache").status();
        clear_if_present(&air)?;
        run_metal(&source, &source_dir, &air)?
    } else {
        return Err(command_failure("metal", &first));
    };
    if !compiled.status.success() {
        return Err(command_failure("metal retry", &compiled));
    }

    // `metallib` creates its output and may reject the empty placeholder.
    fs::remove_file(output)
        .map_err(|error| format!("cannot replace fused library placeholder: {error}"))?;
    let linked = Command::new("/usr/bin/xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(output)
        .output()
        .map_err(|error| format!("cannot launch xcrun metallib: {error}"))?;
    if !linked.status.success() {
        return Err(command_failure("metallib", &linked));
    }
    let length = fs::metadata(output)
        .map_err(|error| format!("cannot inspect fused metallib: {error}"))?
        .len();
    if length == 0 {
        return Err("fused metallib is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte portable fused Metal library");
    Ok(())
}

fn run_metal(source: &Path, include: &Path, air: &Path) -> Result<Output, String> {
    Command::new("/usr/bin/xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(source)
        .arg("-I")
        .arg(include)
        .arg("-o")
        .arg(air)
        .output()
        .map_err(|error| format!("cannot launch xcrun metal: {error}"))
}

fn install_metal_toolchain() -> Result<(), String> {
    println!("cargo:warning=optional Metal toolchain absent; installing it once");
    let output = Command::new("/usr/bin/xcodebuild")
        .args(["-downloadComponent", "MetalToolchain"])
        .output()
        .map_err(|error| format!("cannot launch xcodebuild: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_failure("xcodebuild -downloadComponent", &output))
    }
}

fn clear_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot clear {}: {error}", path.display())),
    }
}

fn optional_toolchain_missing(output: &Output) -> bool {
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .to_ascii_lowercase();
    diagnostic.contains("unable to find utility")
        || diagnostic.contains("metal toolchain")
        || diagnostic.contains("metaltoolchain")
}

fn command_failure(command: &str, output: &Output) -> String {
    format!(
        "{command} exited with {}; stdout: {}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim()
    )
}
