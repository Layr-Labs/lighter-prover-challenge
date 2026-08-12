//! Builds an optional portable Metal library with a threadgroup-tiled
//! Range/U32 quotient kernel. The checked-in library remains the exact
//! fallback on non-Mac builders and whenever Apple's offline compiler is
//! unavailable.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::{env, fs};

const AUX_METALLIB: &str = "poseidon2_range_tile.metallib";
const AUX_SOURCE: &str = "poseidon2_range_tile.metal";
const KERNEL_NAME: &str = "range_check_gate_quotient";
const TILED_KERNEL_NAME: &str = "range_check_gate_quotient_tiled";
const TILE_ROWS: usize = 32;
const TILE_COLS: usize = 64;

fn main() {
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2.metal");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let output = out_dir.join(AUX_METALLIB);
    let generated_source = out_dir.join(AUX_SOURCE);
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let source_dir = manifest_dir.join("src/hash/poseidon2");
    let source_path = source_dir.join("poseidon2.metal");
    let source = fs::read_to_string(&source_path).expect("cannot read Poseidon2 Metal source");
    let tiled = specialize_range_kernel(&source).expect("cannot specialize Range/U32 kernel");
    fs::write(&generated_source, tiled).expect("cannot write tiled Metal source");

    // `include_bytes!` needs an artifact on every build host. Empty means the
    // worker uses the promoted device-memory kernel without any other change.
    fs::write(&output, []).expect("cannot initialize tiled Metal library output");

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        if let Err(error) = build_aux_metallib(&generated_source, &source_dir, &output) {
            println!("cargo:warning=Range/U32 tiled Metal library unavailable: {error}");
            fs::write(&output, []).expect("cannot restore empty tiled Metal library output");
        }
    }
}

/// Produces a second kernel without editing the checked-in source/metallib
/// pair. The 32 threads of one Apple SIMD-group cooperatively stage the first
/// 64 wire columns (16 KiB) and every later reference uses the tile when its
/// column is resident. Higher columns retain the original device load.
fn specialize_range_kernel(source: &str) -> Result<String, String> {
    let signature = format!("kernel void {KERNEL_NAME}(");
    let kernel_start = source
        .find(&signature)
        .ok_or_else(|| format!("missing {KERNEL_NAME} kernel"))?;
    let open_brace = source[kernel_start..]
        .find('{')
        .map(|offset| kernel_start + offset)
        .ok_or("Range/U32 kernel has no body")?;
    let kernel_end = matching_brace(source, open_brace)? + 1;
    let mut kernel = source[kernel_start..kernel_end].to_owned();

    replace_once(
        &mut kernel,
        &format!("kernel void {KERNEL_NAME}("),
        &format!("kernel void {TILED_KERNEL_NAME}("),
    )?;
    replace_once(
        &mut kernel,
        "const device ulong* wires [[buffer(0)]]",
        "const device ulong* device_wires [[buffer(0)]]",
    )?;
    replace_once(
        &mut kernel,
        "uint gid [[thread_position_in_grid]])",
        "uint gid [[thread_position_in_grid]],\n    uint tid [[thread_position_in_threadgroup]])",
    )?;
    replace_once(
        &mut kernel,
        "    uint source_row = gid * step;\n",
        &format!(
            "    uint source_row = gid * step;\n\
             \n\
             \tthreadgroup ulong tiled_wires[{tile_len}];\n\
             \tfor (uint column = 0; column < {tile_cols}u; ++column) {{\n\
             \t    tiled_wires[(ulong)column * {tile_rows}u + tid] =\n\
             \t        device_wires[(ulong)column * lde_rows + source_row];\n\
             \t}}\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n",
            tile_len = TILE_ROWS * TILE_COLS,
            tile_cols = TILE_COLS,
            tile_rows = TILE_ROWS,
        ),
    )?;

    // Route the block-selection helper through a tiled overload as well; its
    // 8--64 item reads are some of the densest reuse in the promoted families.
    replace_once(
        &mut kernel,
        "random_access_select_8(\n                        wires, lde_rows, source_row,",
        "random_access_select_8_tiled(\n                        device_wires, tiled_wires, lde_rows, source_row, tid,",
    )?;
    kernel = rewrite_wire_subscripts(&kernel)?;

    let helper = format!(
        "\ninline ulong range_tiled_wire(\n\
         \tconst device ulong* device_wires,\n\
         \tthreadgroup const ulong* tiled_wires,\n\
         \tuint lde_rows,\n\
         \tuint source_row,\n\
         \tuint tid,\n\
         \tulong column) {{\n\
         \tif (column < {tile_cols}u) {{\n\
         \t    return tiled_wires[column * {tile_rows}u + tid];\n\
         \t}}\n\
         \treturn device_wires[column * lde_rows + source_row];\n\
         }}\n\n\
         inline ulong random_access_select_8_tiled(\n\
         \tconst device ulong* device_wires,\n\
         \tthreadgroup const ulong* tiled_wires,\n\
         \tuint lde_rows,\n\
         \tuint source_row,\n\
         \tuint tid,\n\
         \tulong list_base,\n\
         \tulong bit_base,\n\
         \tuint block) {{\n\
         \tulong items[8];\n\
         \tfor (uint i = 0; i < 8u; ++i) {{\n\
         \t    ulong column = list_base + (ulong)block * 8u + i;\n\
         \t    items[i] = range_tiled_wire(\n\
         \t        device_wires, tiled_wires, lde_rows, source_row, tid, column);\n\
         \t}}\n\
         \tuint level_size = 8u;\n\
         \tfor (uint level = 0; level < 3u; ++level) {{\n\
         \t    ulong b = range_tiled_wire(\n\
         \t        device_wires, tiled_wires, lde_rows, source_row, tid, bit_base + level);\n\
         \t    for (uint k = 0; k < level_size / 2u; ++k) {{\n\
         \t        ulong x = items[2u * k];\n\
         \t        ulong y = items[2u * k + 1u];\n\
         \t        items[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));\n\
         \t    }}\n\
         \t    level_size /= 2u;\n\
         \t}}\n\
         \treturn items[0];\n\
         }}\n\n",
        tile_cols = TILE_COLS,
        tile_rows = TILE_ROWS,
    );

    let mut specialized = String::with_capacity(source.len() + kernel.len() + helper.len());
    specialized.push_str(&source[..kernel_start]);
    // Retain the promoted kernel and all other functions in the auxiliary
    // library; append the uniquely named tiled variant beside it.
    specialized.push_str(&source[kernel_start..kernel_end]);
    specialized.push_str(&helper);
    specialized.push_str(&kernel);
    specialized.push_str(&source[kernel_end..]);
    Ok(specialized)
}

fn matching_brace(source: &str, open: usize) -> Result<usize, String> {
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).ok_or("unbalanced closing brace")?;
                if depth == 0 {
                    return Ok(open + offset);
                }
            }
            _ => {}
        }
    }
    Err("unterminated Range/U32 kernel".to_owned())
}

fn replace_once(haystack: &mut String, from: &str, to: &str) -> Result<(), String> {
    let count = haystack.matches(from).count();
    if count != 1 {
        return Err(format!(
            "expected one occurrence of {from:?}, found {count}"
        ));
    }
    *haystack = haystack.replacen(from, to, 1);
    Ok(())
}

fn rewrite_wire_subscripts(kernel: &str) -> Result<String, String> {
    let needle = "wires[";
    let mut output = String::with_capacity(kernel.len() + 8_192);
    let mut cursor = 0usize;
    let mut replacements = 0usize;

    while let Some(relative) = kernel[cursor..].find(needle) {
        let start = cursor + relative;
        // `device_wires[` is intentionally not a candidate: the preceding
        // identifier character distinguishes it from the old `wires[` token.
        if start > 0 {
            let previous = kernel.as_bytes()[start - 1];
            if previous.is_ascii_alphanumeric() || previous == b'_' {
                output.push_str(&kernel[cursor..start + needle.len()]);
                cursor = start + needle.len();
                continue;
            }
        }

        let expression_start = start + needle.len();
        let close = kernel[expression_start..]
            .find(']')
            .map(|offset| expression_start + offset)
            .ok_or("unterminated wires subscript")?;
        let compact: String = kernel[expression_start..close]
            .chars()
            .filter(|ch| !ch.is_ascii_whitespace())
            .collect();
        let column = if compact == "source_row" {
            "0".to_owned()
        } else {
            compact
                .strip_suffix("*lde_rows+source_row")
                .ok_or_else(|| format!("unexpected wire subscript {compact:?}"))?
                .to_owned()
        };

        output.push_str(&kernel[cursor..start]);
        output.push_str(
            "range_tiled_wire(device_wires, tiled_wires, lde_rows, source_row, tid, (ulong)(",
        );
        output.push_str(&column);
        output.push_str("))");
        cursor = close + 1;
        replacements += 1;
    }
    output.push_str(&kernel[cursor..]);

    if replacements != 72 {
        return Err(format!(
            "expected 72 Range/U32 wire loads, transformed {replacements}"
        ));
    }
    Ok(output)
}

fn build_aux_metallib(source: &Path, include: &Path, output: &Path) -> Result<(), String> {
    let air = output.with_extension("air");
    clear_if_present(&air)?;
    let first = run_metal(source, include, &air)?;
    let compiled = if first.status.success() {
        first
    } else if optional_toolchain_missing(&first) {
        install_metal_toolchain()?;
        let _ = Command::new("/usr/bin/xcrun").arg("--kill-cache").status();
        clear_if_present(&air)?;
        run_metal(source, include, &air)?
    } else {
        return Err(command_failure("metal", &first));
    };
    if !compiled.status.success() {
        return Err(command_failure("metal retry", &compiled));
    }

    fs::remove_file(output)
        .map_err(|error| format!("cannot replace tiled library placeholder: {error}"))?;
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
        .map_err(|error| format!("cannot inspect tiled metallib: {error}"))?
        .len();
    if length == 0 {
        return Err("tiled metallib is empty".to_owned());
    }
    println!("cargo:warning=embedded {length}-byte portable Range/U32 tiled Metal library");
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
