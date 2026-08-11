//! Builds an offline Metal pipeline archive on the ranked M4 builder.
//!
//! The committed metallib contains portable Metal IR. `metal-tt` translates
//! the listed pipeline descriptors without opening a Metal device in Cargo's
//! build process. If the optional toolchain or translation is unavailable,
//! the empty embedded artifact selects the established runtime path.

use std::path::Path;
use std::process::{Command, Output};
use std::{env, fs};

const ARCHIVE_FILE: &str = "poseidon2.binary.metallib";
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
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metallib");

    let output = Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(ARCHIVE_FILE);
    // `include_bytes!` needs this file on every platform. Zero bytes means
    // that runtime pipeline construction must use the proven frontier path.
    fs::write(&output, []).expect("cannot initialize Metal archive output");

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        if let Err(error) = build_archive(&output) {
            println!("cargo:warning=offline Metal binary archive unavailable: {error}");
            fs::write(&output, []).expect("cannot restore empty Metal archive output");
        }
    }
}

fn build_archive(output: &Path) -> Result<(), String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?;
    let library = Path::new(&manifest_dir).join("src/hash/poseidon2/poseidon2.metallib");
    if !library.is_file() {
        return Err(format!(
            "committed metallib is missing: {}",
            library.display()
        ));
    }

    let config = output.with_file_name("poseidon2.mtlp-json");
    fs::write(&config, archive_configuration(&library))
        .map_err(|error| format!("cannot write {}: {error}", config.display()))?;
    // `metal-tt` creates its output and may reject an existing placeholder.
    fs::remove_file(output)
        .map_err(|error| format!("cannot replace archive placeholder: {error}"))?;

    let first = run_metal_tt(&library, &config, output)?;
    let result = if first.status.success() {
        first
    } else if optional_toolchain_missing(&first) {
        install_metal_toolchain()?;
        let _ = Command::new("/usr/bin/xcrun").arg("--kill-cache").status();
        // Do not let a partial first output poison the translator retry.
        match fs::remove_file(output) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot clear partial archive: {error}")),
        }
        run_metal_tt(&library, &config, output)?
    } else {
        return Err(command_failure("metal-tt", &first));
    };

    if !result.status.success() {
        return Err(command_failure("metal-tt retry", &result));
    }
    let length = fs::metadata(output)
        .map_err(|error| format!("cannot inspect translated archive: {error}"))?
        .len();
    if length == 0 {
        return Err("translated archive is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte offline Metal binary archive");
    Ok(())
}

fn run_metal_tt(library: &Path, config: &Path, output: &Path) -> Result<Output, String> {
    Command::new("/usr/bin/xcrun")
        .args(["-sdk", "macosx", "metal-tt"])
        .arg(library)
        .arg(config)
        .arg("-o")
        .arg(output)
        .output()
        .map_err(|error| format!("cannot launch xcrun metal-tt: {error}"))
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

fn archive_configuration(library: &Path) -> String {
    let pipelines = KERNELS
        .iter()
        .map(|name| format!(r#"{{"compute_function":"alias:Poseidon2#{name}"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"libraries":{{"paths":[{{"label":"Poseidon2","path":"{}"}}]}},"pipelines":{{"compute_pipelines":[{pipelines}]}}}}"#,
        json_escape(&library.to_string_lossy())
    )
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character < ' ' => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
}
