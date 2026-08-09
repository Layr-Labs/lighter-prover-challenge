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

/// The library-choice POLICY, exhaustively: **compile from source only when the
/// metallib is unavailable.**
///
/// This replaces an earlier test that asserted the OPPOSITE ordering and passed
/// happily while the shipped code did something else — a test pinned to a policy
/// that is no longer implemented is worse than no test, because green reads as
/// coverage.
///
/// The rule exists because compiling speculatively in order to probe a
/// source-AIR archive costs ~185 ms/worker on any host where the metallib would
/// have loaded and the archive misses anyway. That quadrant is UNREACHABLE on
/// macOS 15.x and is exactly where a macOS 26.x ranked host lives, so it has to
/// be closed by ordering rather than by checking reachability on this box.
#[test]
fn source_compile_is_never_speculative() {
    use plonky2::hash::poseidon2::compiles_from_source;

    assert!(
        !compiles_from_source(true),
        "the metallib loaded, so a source compile is pure loss: the archive is \
         source-AIR and cannot serve a metallib-AIR library anyway"
    );
    assert!(
        compiles_from_source(false),
        "the metallib is unavailable, so the source compile was unavoidable and \
         costs nothing extra -- this is the quadrant the archive pays off in"
    );
}

/// LAYER 1 of the attach guard: the committed archive must carry a stamp of the
/// shader it was generated from, and that stamp must match the shader shipped.
///
/// The cheap half — a string compare, no Metal — and the half that catches the
/// failure we actually hit: the tip rewrote `poseidon2.metal` under an archive
/// built from the previous revision. It matters because attaching a
/// non-matching archive is NOT free; it measured ~+140 ms per worker for
/// nothing, so "detect and do not attach" beats "attach and hope".
#[test]
fn committed_archive_is_stamped_with_the_shipped_shader() {
    assert!(
        plonky2::hash::poseidon2::pipeline_archive_matches_shader(),
        "the committed pipeline archive was built from a different poseidon2.metal. \
         Regenerate it: cargo run --release --example gen_pipeline_archive"
    );
}
