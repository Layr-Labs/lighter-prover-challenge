//! Builds the optional portable metallib containing the two-level Merkle parent kernel.

use std::path::Path;
use std::process::{Command, Output};
use std::{env, fs};

const AUX_METALLIB: &str = "poseidon2_parent2.metallib";
const ARCHIVER_SOURCE: &str = r#"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

static int fail(const char *stage, NSError *error) {
    const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
    fprintf(stderr, "%s: %s\n", stage, detail);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: parent2-archiver source output\n");
            return 2;
        }
        NSString *source_path = [NSString stringWithUTF8String:argv[1]];
        NSString *output_path = [NSString stringWithUTF8String:argv[2]];
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:source_path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (source == nil) return fail("read source", error);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (library == nil) return fail("compile source", error);
        id<MTLFunction> function = [library newFunctionWithName:@"poseidon2_hash_parent2"];
        if (function == nil) {
            fprintf(stderr, "parent2 kernel missing after source compilation\n");
            return 1;
        }

        MTLBinaryArchiveDescriptor *archive_desc = [MTLBinaryArchiveDescriptor new];
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:archive_desc
                                                                       error:&error];
        if (archive == nil) return fail("create binary archive", error);
        MTLComputePipelineDescriptor *pipeline_desc = [MTLComputePipelineDescriptor new];
        pipeline_desc.computeFunction = function;
        if (![archive addComputePipelineFunctionsWithDescriptor:pipeline_desc error:&error]) {
            return fail("add parent2 pipeline", error);
        }
        NSURL *output_url = [NSURL fileURLWithPath:output_path];
        if (![archive serializeToURL:output_url error:&error]) {
            return fail("serialize binary archive", error);
        }

        // Prove that the serialized bytes are directly loadable as a Metal
        // library and retain the kernel entry-point type before Cargo embeds
        // them in the ranked worker.
        NSData *data = [NSData dataWithContentsOfFile:output_path
                                              options:NSDataReadingMappedIfSafe
                                                error:&error];
        if (data == nil) return fail("read serialized archive", error);
        dispatch_data_t bytes = dispatch_data_create(
            data.bytes,
            data.length,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_NONE);
        id<MTLLibrary> reloaded = [device newLibraryWithData:bytes error:&error];
        if (reloaded == nil) return fail("reload serialized archive", error);
        id<MTLFunction> reloaded_function =
            [reloaded newFunctionWithName:@"poseidon2_hash_parent2"];
        if (reloaded_function == nil || reloaded_function.functionType != MTLFunctionTypeKernel) {
            fprintf(stderr, "reloaded parent2 entry is not a kernel\n");
            return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:reloaded_function error:&error];
        if (pipeline == nil) return fail("lower reloaded parent2 pipeline", error);
        return 0;
    }
}
"#;

fn main() {
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metal");
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2_parent2.metal");

    let output = Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(AUX_METALLIB);
    // `include_bytes!` needs an artifact on every build host. Empty means the
    // worker retains the promoted one-level parent schedule.
    fs::write(&output, []).expect("cannot initialize auxiliary Metal library output");

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        if let Err(archive_error) = build_aux_archive(&output) {
            println!("cargo:warning=parent2 GPU archive unavailable: {archive_error}");
            fs::write(&output, []).expect("cannot restore auxiliary Metal placeholder");
            if let Err(air_error) = build_aux_metallib(&output) {
                println!("cargo:warning=parent2 AIR library unavailable: {air_error}");
                fs::write(&output, [])
                    .expect("cannot restore empty auxiliary Metal library output");
            }
        }
    }
}

/// Builds and executes a tiny Foundation/Metal helper on the native build host.
/// `MTLBinaryArchive` emits an applegpu slice, moving parent2 pipeline lowering
/// out of every scored worker and into the untimed Cargo build.
fn build_aux_archive(output: &Path) -> Result<(), String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?;
    let source_dir = Path::new(&manifest_dir).join("src/hash/poseidon2");
    let base = fs::read_to_string(source_dir.join("poseidon2.metal"))
        .map_err(|error| format!("cannot read base Metal source: {error}"))?;
    let extension = fs::read_to_string(source_dir.join("poseidon2_parent2.metal"))
        .map_err(|error| format!("cannot read parent2 Metal source: {error}"))?;
    let extension = extension
        .strip_prefix("#include \"poseidon2.metal\"")
        .ok_or("parent2 source lost its expected include header")?;

    let out_dir = output.parent().ok_or("auxiliary output has no parent")?;
    let combined = out_dir.join("poseidon2_parent2_combined.metal");
    let helper_source = out_dir.join("poseidon2_parent2_archiver.m");
    let helper = out_dir.join("poseidon2_parent2_archiver");
    fs::write(&combined, format!("{base}\n{extension}"))
        .map_err(|error| format!("cannot write combined Metal source: {error}"))?;
    fs::write(&helper_source, ARCHIVER_SOURCE)
        .map_err(|error| format!("cannot write archive helper: {error}"))?;

    let compiled = Command::new("/usr/bin/clang")
        .args([
            "-fobjc-arc",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
        ])
        .arg(&helper_source)
        .arg("-o")
        .arg(&helper)
        .output()
        .map_err(|error| format!("cannot launch clang for archive helper: {error}"))?;
    if !compiled.status.success() {
        return Err(command_failure("clang parent2 archiver", &compiled));
    }

    clear_if_present(output)?;
    let archived = Command::new(&helper)
        .arg(&combined)
        .arg(output)
        .output()
        .map_err(|error| format!("cannot launch parent2 archiver: {error}"))?;
    if !archived.status.success() {
        return Err(command_failure("parent2 archiver", &archived));
    }
    let length = fs::metadata(output)
        .map_err(|error| format!("cannot inspect parent2 GPU archive: {error}"))?
        .len();
    if length == 0 {
        return Err("parent2 GPU archive is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte parent2 GPU archive");
    Ok(())
}

fn build_aux_metallib(output: &Path) -> Result<(), String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?;
    let source_dir = Path::new(&manifest_dir).join("src/hash/poseidon2");
    let source = source_dir.join("poseidon2_parent2.metal");
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

    fs::remove_file(output)
        .map_err(|error| format!("cannot replace auxiliary library placeholder: {error}"))?;
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
        .map_err(|error| format!("cannot inspect auxiliary metallib: {error}"))?
        .len();
    if length == 0 {
        return Err("auxiliary metallib is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte portable parent2 Metal library");
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
