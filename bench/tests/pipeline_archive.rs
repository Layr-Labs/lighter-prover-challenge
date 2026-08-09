//! The committed `MTLBinaryArchive` must cover every kernel the shipped shader
//! defines, and the check must be able to FAIL — otherwise it is decoration.
//!
//! This lives in `bench` rather than beside the code it tests because `bench`
//! is a workspace member and `plonky2` is not: `cargo test -p plonky2` refuses
//! with "requires dev-dependencies and is not a member of the workspace", which
//! is why the equivalent metallib guard in that crate has never run.
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use plonky2::hash::poseidon2::verify_pipeline_archive;

/// POSITIVE ARM: every kernel compiled from the shipped shader must be present
/// in the committed archive under `FailOnBinaryArchiveMiss`. Fails loudly when
/// someone edits `poseidon2.metal` without regenerating the archive — the
/// failure mode that would otherwise cost 1.6 s per worker in silence.
#[test]
fn committed_archive_covers_every_shipped_kernel() {
    match verify_pipeline_archive(None) {
        Ok(n) => assert!(n >= 9, "expected at least 9 kernels verified, got {n}"),
        Err(error) => panic!("{error}"),
    }
}

/// NEGATIVE ARM / positive control: a shader that differs by one arithmetic
/// operation compiles to a different function hash, so the archive must MISS.
/// If this passes, `FailOnBinaryArchiveMiss` is not discriminating and the
/// positive arm above proves nothing.
#[test]
fn archive_misses_a_modified_shader() {
    let shader = include_str!(
        "../../vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal"
    );
    // Change emitted code, not just source text: a comment leaves each
    // kernel's AIR byte-identical and would still HIT, which would make this
    // control lie. `gl_add` and `gl_sub` have identical signatures
    // (`ulong(ulong, ulong)`), so swapping one call site is guaranteed to
    // compile and guaranteed to change the function hash.
    let modified = shader.replacen("gl_add(", "gl_sub(", 1);
    assert_ne!(modified, shader, "the substitution must actually apply");

    assert!(
        verify_pipeline_archive(Some(&modified)).is_err(),
        "a modified shader hit the committed archive: FailOnBinaryArchiveMiss \
         is not discriminating, so the positive arm proves nothing"
    );
}
