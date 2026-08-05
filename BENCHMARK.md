# Lighter prover benchmark

This benchmark measures verified proving throughput against the mixed-block
Lighter circuit imported from
`elliottech/lighter-prover@381fd529eb61dfff9ad245d94fce214a0a64d927`.
The trusted CPU verifier source is pinned to that same public revision. The
candidate is built directly from the upstream workspace; there is no separate
challenge crate or file overlay.

The checked-in all-empty witness is an end-to-end smoke case, not a ranked
workload. Its output is explicitly marked `public-synthetic-provisional`,
noncompetitive, and cacheable. It must never be compared with
`private-active-ranked` scores.

## Editable surface

Candidates may edit the root `Cargo.toml` and `Cargo.lock`, `circuit/`, `bench/`,
and `vendor/`. The `bench` crate produces the candidate `prove` worker, and the
vendored Plonky2 workspace provides the editable prover baseline.

`benchmark-tools/`, fixtures, scripts, workflows, and `benchmark.json` are
protected. The trusted CPU verifier source is pinned to circuit revision
`381fd52` and the frozen CPU plonky2 backend; after review it is published as a
separate prebuilt arm64 executable. The published verifier launches the
trusted `prove-via-bench.sh` wrapper, owns the timer, verifies the final proof and
its public outputs, and is the only process that writes a score. The wrapper
uses the fixed root bridge, which copies and launches the candidate worker as
the disposable `lighter-prover-bench` identity under macOS Seatbelt.

For local/default runs, protected `benchmark.sh` continues to create the shared
Seatbelt profile and pass it to the trusted verifier. In ranked bridge mode it
passes no profile: the root-owned bridge renders and applies the worker profile
after dropping execution to uid 560. Only the copied candidate worker enters
Seatbelt. Its environment is cleared, networking and child processes are
denied, and filesystem writes are limited to a private bridge run directory.
The trusted verifier stays outside Seatbelt so candidate code cannot control the
clock, proof checks, or score output.

This is process containment rather than a VM boundary. The self-hosted runner
must be dedicated, disposable, and contain no unrelated credentials or secrets.

## Proof compatibility

Candidate circuit and backend changes must continue producing proofs accepted
by the protected verifier. Constraint-system changes that alter verifier data
require a reviewed benchmark-version update and a newly published verifier.

The standalone upstream `bench` binary exposes transaction count and
heavy/light transactions-per-proof as runtime parameters. That tool builds the
corresponding circuits and matching verifier data together, so changing a width
is internally consistent there. The challenge uses a different trust boundary:
the candidate and protected verifier build independently. It therefore pins
chain ID 304, heavy width 4, and light width 10 on both sides. Changing either
width changes the transaction circuits, recursive chains, final block circuit,
and verifier data, so a tuned-width proof is rejected by the fixed challenge
verifier. Transaction count and witness contents may change the workload
without changing those circuit shapes.

## Scope of assurance

This benchmark assures prover performance only. It measures verified proving
throughput; it does not assess whether the circuit faithfully proves the DEX
state transition. That soundness property is inherited from the pinned upstream
`elliottech/lighter-prover@381fd52` and the pinned `elliottech/plonky2@e1c2d35`
backend, which are trusted by assumption.

Candidate submissions cannot weaken it. `benchmark-tools/harness/Cargo.toml`
pins `circuit` and `plonky2` to those fixed git revisions, so the trusted
verifier is built against upstream circuit definitions regardless of what a
candidate does to the editable local `circuit/`. A constraint-system change
alters the circuit digest, so a proof produced against modified constraints
fails verification against the pinned verifier data. The verifier checks the
final `BlockCircuit` proof, which recursively verifies pre-execution and both
heavy/light transaction chains, then compares every final public block output
against a block witness it recomputes from the protected fixture.

## Fixture status

Public and local runs consume the exact imported `bench/bench_test.json` bytes
(SHA-256
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`).
Upstream parsing expands that synthetic witness to 10 heavy and 490 light
logical transactions in 3 heavy and 49 light chunks. Because every transaction
is empty, the final proof does not bind those recursion counts or the configured
500-transaction numerator. Public output therefore uses
`public-synthetic-smoke` mode with a provisional, noncompetitive, cacheable
score and records its transaction-count source as
`synthetic-configuration-unbound-by-proof`.

Compatible active private witnesses are parsed without synthetic expansion.
Private ranked mode evaluates every direct-child JSON fixture in one
checksum-pinned private bundle. Each fixture must contain exactly 500 active
transactions, and the configured fixture count determines the aggregate
transaction and proof counts. Only that set may use `official-throughput` mode
and `private-active-ranked` metadata; its trusted numerator is the aggregate
parsed count of non-empty transactions. Every final proof and all public
outputs must verify before either kind of score is written.

## Local use

### Recommended local hardware

- macOS on Apple Silicon (M1–M4)
- 32 GB RAM
- 10 GB free disk space

Systems with 24 GB RAM may work but have not been validated; 16 GB RAM is not
recommended. Local runs use the public fixture. Private witnesses are evaluated
only by the official workflow.

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

## Trusted verifier authenticity

The trusted verifier's authenticity rests entirely on the SHA-256 pin in
`benchmark-tools/trusted/SHA256SUMS`, which is only as trustworthy as the
reviewed commit that carries it. The binary is **ad-hoc** code-signed
(`codesign --force --sign -`, no identity and no Team ID), so the
`codesign --verify --strict` checks in `setup.sh`, both workflows, and
`.github/scripts/test-trusted-verifier.sh` are tamper-evidence: they prove the
file has not been altered since signing, and prove nothing about who signed it.
`benchmark.sh` re-checks the SHA-256 pin on every run.

`benchmark-tools/check-trusted-verifier-reproducibility.sh` rebuilds the
verifier from `REVIEWED_COMMIT` and compares the full SHA-256 against the pin,
so the pin can in principle be checked against something other than the commit
that introduced it. That check currently fails: the dependency graph pulls in
`const-random`, which bakes fresh compile-time entropy into every build, so the
published binary is not reproducible by anyone including its author.
`benchmark-tools/trusted/README.md` records the evidence, the residual risk, and
the recipe a future republication needs to become reproducible.

## Ranked private fixture

Ranked runs use the official Apple M4 Pro host with 48 GB RAM. They use two
sequential one-job JIT registrations selected by the sole existing
`lighter-prover-challenge-m4` runner label. The first job checks out the
dispatched SHA directly, creates a regular Git tar archive with protected
workflow tooling, refuses to continue if that archive carries Cargo
configuration of its own, and asks the root bridge to extract and build it as
disposable uid 560. No candidate code executes as the Actions runner. This job
has no Environment or R2 credentials and uploads exactly the regular bridge
output `prove` plus `prove.sha256` under a run-ID/run-attempt-scoped artifact
name.
The second job checks out the trusted default-branch harness, downloads and
verifies that two-file artifact, installs it as `target/release/prove-bin`, and
installs the trusted wrapper as `target/release/prove`. It never compiles
candidate code. It binds score provenance directly to `github.sha`, then
re-hashes `prove-bin` immediately before execution.

The archive is built from the whole dispatched tree, so the build inherits the
`editablePaths` boundary that upstream submission validation enforces. The
workflow re-checks the one build input that boundary would otherwise be the sole
protection for: a repo-root `.cargo/config.toml` or `.cargo/config`, which Cargo
reads from the build directory upwards and which can carry `rustflags`, a
`[target.*]` `linker` or `runner`, or a `[source]` replacement. If the archive
root contains any `.cargo` entry, the run fails rather than the file being
stripped, so a validation bypass shows up in the job log instead of being
silently swallowed. The check is scoped to the archive root: Cargo never reads
configuration from a subdirectory, so `bench/.cargo/config.toml` and the like
are inert and remain allowed. The rest of the in-tree build configuration is
already neutralized by the host bridge, which exports a job-private `CARGO_HOME`
cloned from a root-owned template that contains no `config`/`config.toml`/`bin`,
sets `RUSTC`, `RUSTFLAGS` and `CARGO_TARGET_DIR` explicitly, and invokes the
pinned root-owned toolchain's own `cargo` binary rather than a rustup shim, with
no rustup on the build `PATH`. Nothing therefore reads the in-tree
`rust-toolchain`; the toolchain is pinned entirely from the trusted side.

The second job obtains the ranked bundle from
`fixtures/ranked-v1.tar.gz` in a protected private object bucket. Candidate
build jobs never receive the bucket credentials or private bytes. The trusted
benchmark job verifies the bundle's Environment-secret checksum, validates and
extracts only generic direct-child JSON files into a private directory, removes
the archive and credentials, then rechecks the candidate before execution. The
development Environment is currently configured for five 500-transaction
blocks (2,500 aggregate transactions); fixture and transaction counts remain
Environment variables so a future bundle can change the set without a code
change. Neither per-fixture identifiers nor checksums enter scores or logs.
Solvers do not need to configure or access this bucket; its deployment and
credential policy are maintained outside the benchmark repository.

The score artifact is uploaded only after every fixture succeeds. The bundle,
raw fixture bytes, candidate proof output, stdout/stderr, and failure artifacts
are never uploaded. Local `./benchmark.sh` continues to use the checked-in
fixture file and requires no R2 configuration, but its score is always the
provisional public smoke case described above. Local private mode instead
accepts only the dedicated extracted fixture directory and requires
`LIGHTER_EXPECTED_FIXTURE_COUNT`.

## Runner and sandbox requirements

The focused host design has two identities. `lighter-prover-challenge`
(uid 561, gid 20) runs the trusted Actions runner and verifier.
`lighter-prover-bench` (uid/gid 560) is disposable and is the only identity that
extracts/builds the submitted archive or executes `prove-bin`. The root bridge
accepts only absolute paths strictly below
`/opt/lighter-prover-challenge/work`; build and proof descendants start with a
clean fixed environment, so workflow secrets are not forwarded to candidate
code.

Build mode is also sandboxed, in two phases with different network stances.
Dependency resolution (`cargo fetch --locked`) is the only step that may reach
the network, and it executes no candidate `build.rs` or proc-macro. Compilation
runs candidate build scripts and proc-macros and is network-denied and offline;
it needs no egress because the dependency set is already local. Both build
profiles deny writes to every root-owned path the benchmark job later trusts,
deny writes to the Darwin per-user folders that the janitor cannot purge, and
deny reads of the host homes and GitHub App material. The host package can be
switched to a fully network-denied build once every dependency a submission can
name is present in the host Cargo cache template; until then candidates may add
dependencies, so `cargo fetch` keeps egress for that one step. Unlike proof
mode, the build phases must fork and exec (`rustc`, build scripts, the linker),
so process spawning is bounded by `ulimit -u` rather than denied.

The root supervisor runs the janitor after every one-job JIT runner exit, before
the next registration. The janitor terminates uid-560 processes, verifies that
none survive, purges its writable state, verifies the locked fixed identity,
and rebuilds its home from a root-owned template. Runner-state cleanup
separately resets uid 561 processes and workspace state between registrations.
Cleanup, quarantine, and signal handling fail closed, and the JIT wallclock is
four hours. Do not enable this workflow until the focused host package and its
on-host build/prove, Seatbelt, sudo-grant, and janitor checks pass.

The credentialed job restricts `PATH` to root-owned system directories, disables
user and system Git configuration, disables AWS user configuration and metadata
lookup, and calls the fixed root-owned AWS CLI by absolute path. The trusted
verifier clears the wrapper environment; the wrapper invokes only the absolute
root bridge and forwards exactly the worker, fixture, and proof paths. Proof
mode copies those inputs into a private bridge directory and applies the
root-generated Seatbelt profile while executing the worker as uid 560.

This is intentionally a focused build/prove isolation layer. It performs no
LLM, prompt, or security evaluation, and it does not implement a full signed
manifest or PF firewall system.

The existing published-verifier integration test remains authoritative for
trusted timeout enforcement, malformed and oversized proof rejection,
process-tree cleanup, and local/default Seatbelt non-scratch write denial. CI
additionally runs the shared sandbox probe, which checks permitted scratch
writes and denies non-scratch writes, network access, process forks, and
mDNSResponder resolver IPC. The benchmark workflow contract checks the
two-job, bridge, artifact, credential, provenance, and success-only upload
boundaries.
