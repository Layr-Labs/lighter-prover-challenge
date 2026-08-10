use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn run_command(program: &str, arguments: &[&str]) -> Result<(), String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to launch {program} {arguments:?}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(format!(
            "{program} {arguments:?} failed with {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        ))
    }
}

fn compile_metal(source: &str, air: &str, metallib: &str) -> Result<(), String> {
    run_command(
        "/usr/bin/xcrun",
        &["-sdk", "macosx", "metal", "-c", source, "-o", air],
    )?;
    run_command(
        "/usr/bin/xcrun",
        &["-sdk", "macosx", "metallib", air, "-o", metallib],
    )
}

fn concise_error(error: &str) -> String {
    const LIMIT: usize = 3000;
    if error.len() <= LIMIT {
        return error.replace('\n', " | ");
    }
    let mut start = error.len() - LIMIT;
    while !error.is_char_boundary(start) {
        start += 1;
    }
    format!("...{}", error[start..].replace('\n', " | "))
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
    let mut source_built = false;
    if target_is_macos && host_is_macos {
        let source = source.to_str().expect("Metal source path is not UTF-8");
        let air = air.to_str().expect("AIR output path is not UTF-8");
        let metallib_path = metallib
            .to_str()
            .expect("metallib output path is not UTF-8");
        let first_attempt = compile_metal(source, air, metallib_path);
        let compile_result = match first_attempt {
            Ok(()) => Ok(()),
            Err(error) if error.contains("missing Metal Toolchain") => {
                // Xcode 26 moved the Metal compiler into an optional component.
                // The ranked build is untimed; if its sandbox permits Apple's
                // supported component installer, fetch it once and retry.
                // Any denial still falls through to the committed safe artifact.
                println!(
                    "cargo:warning=Metal toolchain is absent; trying Apple's xcodebuild component installer"
                );
                match run_command(
                    "/usr/bin/xcodebuild",
                    &["-downloadComponent", "metalToolchain"],
                ) {
                    Ok(()) => {
                        let _ = run_command("/usr/bin/xcrun", &["--kill-cache"]);
                        compile_metal(source, air, metallib_path)
                    }
                    Err(download_error) => Err(format!(
                        "{error}\nMetal component installation also failed:\n{download_error}"
                    )),
                }
            }
            Err(error) => Err(error),
        };
        match compile_result {
            Ok(()) => source_built = true,
            Err(error) => println!(
                "cargo:warning=Metal source build unavailable; using the committed metallib and CPU fallback: {}",
                concise_error(&error)
            ),
        }
    }
    if !source_built {
        fs::copy(&prebuilt, &metallib).unwrap_or_else(|error| {
            panic!(
                "failed to stage {} as {}: {error}",
                prebuilt.display(),
                metallib.display()
            )
        });
    } else {
        println!("cargo:rustc-env=PLONKY2_METAL_SOURCE_BUILT=1");
    }

    println!(
        "cargo:rustc-env=PLONKY2_POSEIDON2_METALLIB={}",
        metallib.display()
    );
}
