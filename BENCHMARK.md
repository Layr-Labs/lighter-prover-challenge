# Lighter prover benchmark

This benchmark measures verified proving throughput against the current Lighter
circuit baseline (`23d1596`). The candidate is built directly from the upstream
workspace; there is no separate challenge crate or file overlay.

## Editable surface

Candidates may edit the root `Cargo.toml` and `Cargo.lock`, `circuit/`, `bench/`,
and `vendor/`. The `bench` crate produces the candidate `prove` worker, and the
vendored plonky2 workspace provides the Metal backend.

`benchmark-tools/`, fixtures, scripts, workflows, and `benchmark.json` are
protected. The trusted verifier is a separate prebuilt arm64 executable linked
against pristine current-main circuit code and the frozen CPU plonky2 backend.
It launches the candidate under macOS Seatbelt, owns the timer, verifies both
proofs and their public outputs, and is the only process that writes a score.

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

Current upstream `main` changed the witness and state-root format for universal
cross margin without updating the historical `bench/bench_test.json`. That file
does not deserialize/prove against `23d1596` and is not a correctness anchor.

Before ranked runs are enabled, export a complete 500-transaction fixture from
the current Lighter prover pipeline to
`benchmark-tools/fixtures/bench-current-500.json`, then validate a CPU proof and
a Metal proof against the protected CPU verifier.

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
