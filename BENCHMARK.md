# Lighter prover benchmark

This bootstrap benchmark measures verified proving throughput against the
Lighter circuit imported from the `lighter-auto` `d43c6e7` snapshot. Its
public-equivalent circuit source is `lighter-prover` `5bbb307`, and the trusted
CPU verifier source is pinned to that same public revision. This is bootstrap
infrastructure validation, not the final competition baseline. The candidate is
built directly from the upstream workspace; there is no separate challenge
crate or file overlay.

## Editable surface

Candidates may edit the root `Cargo.toml` and `Cargo.lock`, `circuit/`, `bench/`,
and `vendor/`. The `bench` crate produces the candidate `prove` worker, and the
vendored plonky2 workspace provides the Metal backend.

`benchmark-tools/`, fixtures, scripts, workflows, and `benchmark.json` are
protected. The trusted CPU verifier source is pinned to circuit revision
`5bbb307` and the frozen CPU plonky2 backend; after review it is published as a
separate prebuilt arm64 executable. The published verifier launches the
candidate under macOS Seatbelt, owns the timer, verifies both proofs and their
public outputs, and is the only process that writes a score.

For ranked runs, protected `benchmark.sh` creates the same Seatbelt profile used
by the Flock challenge and passes its path to the trusted verifier. Only the
candidate worker enters the sandbox. Its environment is cleared, networking and
child processes are denied, and filesystem writes are limited to fresh private
scratch. Ranked setup and execution fail closed if `sandbox-exec` is unavailable.
The trusted verifier stays outside the sandbox so candidate code cannot control
the clock, proof checks, or score output.

This is process containment rather than a VM boundary. The self-hosted runner
must be dedicated, disposable, and contain no unrelated credentials or secrets.

## Proof compatibility

Candidate circuit and backend changes must continue producing proofs accepted
by the protected verifier. Constraint-system changes that alter verifier data
require a reviewed benchmark-version update and a newly published verifier.

## Fixture status

The benchmark reuses the repository's existing 500-transaction
`bench/bench_test.json` blob as protected
`benchmark-tools/fixtures/bench.json`. The trusted verifier remains the
correctness authority: its exact SHA-256 must match the protected bootstrap
fixture, it must deserialize under the pinned bootstrap circuit, and both
candidate proofs and their public outputs must verify before a score is written.

## Local use

```bash
./setup.sh
./benchmark.sh
```

The setup builds only the candidate `prove` binary. It verifies the trusted
verifier's checksum and code signature before compiling candidate code.

`benchmark-tools/build-trusted-verifier.sh` is an author-only publication tool.
After the protected source diff is reviewed and committed, set its
`REVIEWED_COMMIT`; it creates a clean detached `.trusted-benchmark` worktree at
that exact commit, builds the locked harness there, then copies, signs, and
checksums the verifier in the main worktree. Ranked setup never invokes it.
