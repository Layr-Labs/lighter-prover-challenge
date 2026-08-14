//! Builds an optional device archive for the threadgroup Merkle subtree kernel.

use std::path::Path;
use std::process::{Command, Output};
use std::{env, fs};

const AUX_METALLIB: &str = "poseidon2_subtree.metallib";
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
        if (argc != 3) return 2;
        NSString *source_path = [NSString stringWithUTF8String:argv[1]];
        NSString *output_path = [NSString stringWithUTF8String:argv[2]];
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:source_path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (source == nil) return fail("read source", error);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return 1;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (library == nil) return fail("compile source", error);
        id<MTLFunction> function =
            [library newFunctionWithName:@"poseidon2_hash_subtree128"];
        if (function == nil) return 1;

        id<MTLBinaryArchive> archive = [device
            newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
            error:&error];
        if (archive == nil) return fail("create archive", error);
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        if (![archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error]) {
            return fail("add subtree pipeline", error);
        }
        if (![archive serializeToURL:[NSURL fileURLWithPath:output_path] error:&error]) {
            return fail("serialize archive", error);
        }

        NSData *data = [NSData dataWithContentsOfFile:output_path
                                              options:NSDataReadingMappedIfSafe
                                                error:&error];
        if (data == nil) return fail("read archive", error);
        dispatch_data_t bytes = dispatch_data_create(
            data.bytes,
            data.length,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_NONE);
        id<MTLLibrary> reloaded = [device newLibraryWithData:bytes error:&error];
        if (reloaded == nil) return fail("reload archive", error);
        id<MTLFunction> archived_function =
            [reloaded newFunctionWithName:@"poseidon2_hash_subtree128"];
        if (archived_function == nil || archived_function.functionType != MTLFunctionTypeKernel) {
            return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:archived_function error:&error];
        if (pipeline == nil) return fail("create archived pipeline", error);
        if (pipeline.maxTotalThreadsPerThreadgroup < 128) return 1;
        return 0;
    }
}
"#;

fn main() {
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metal");
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2_subtree.metal");

    let output = Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(AUX_METALLIB);
    fs::write(&output, []).expect("cannot initialize subtree Metal archive");
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        if let Err(error) = build_aux_archive(&output) {
            println!("cargo:warning=subtree GPU archive unavailable: {error}");
            fs::write(&output, []).expect("cannot restore empty subtree Metal archive");
        }
    }
}

fn build_aux_archive(output: &Path) -> Result<(), String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?;
    let source_dir = Path::new(&manifest_dir).join("src/hash/poseidon2");
    let base = fs::read_to_string(source_dir.join("poseidon2.metal"))
        .map_err(|error| format!("cannot read base Metal source: {error}"))?;
    let extension = fs::read_to_string(source_dir.join("poseidon2_subtree.metal"))
        .map_err(|error| format!("cannot read subtree Metal source: {error}"))?;
    let extension = extension
        .strip_prefix("#include \"poseidon2.metal\"")
        .ok_or("subtree source lost its include header")?;
    let out_dir = output.parent().ok_or("archive output has no parent")?;
    let combined = out_dir.join("poseidon2_subtree_combined.metal");
    let helper_source = out_dir.join("poseidon2_subtree_archiver.m");
    let helper = out_dir.join("poseidon2_subtree_archiver");
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
        .map_err(|error| format!("cannot launch clang: {error}"))?;
    if !compiled.status.success() {
        return Err(command_failure("clang subtree archiver", &compiled));
    }
    fs::remove_file(output)
        .map_err(|error| format!("cannot replace archive placeholder: {error}"))?;
    let archived = Command::new(&helper)
        .arg(&combined)
        .arg(output)
        .output()
        .map_err(|error| format!("cannot launch subtree archiver: {error}"))?;
    if !archived.status.success() {
        return Err(command_failure("subtree archiver", &archived));
    }
    let length = fs::metadata(output)
        .map_err(|error| format!("cannot inspect subtree archive: {error}"))?
        .len();
    if length == 0 {
        return Err("subtree archive is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte subtree GPU archive");
    Ok(())
}

fn command_failure(command: &str, output: &Output) -> String {
    format!(
        "{command} exited with {}; stdout: {}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim()
    )
}
