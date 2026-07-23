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
trusted `prove-via-bench.sh` wrapper, owns the timer, verifies both proofs and
their public outputs, and is the only process that writes a score. The wrapper
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

## Ranked private fixture

Ranked runs use two sequential one-job JIT registrations on the sole existing
`lighter-prover-challenge-m3` runner label. The first job checks out the
dispatched SHA directly, creates a regular Git tar archive with protected
workflow tooling, and asks the root bridge to extract and build it as disposable
uid 560. No candidate code executes as the Actions runner. This job has no
Environment or R2 credentials and uploads exactly the regular bridge output
`prove` plus `prove.sha256` under a run-ID/run-attempt-scoped artifact name.
The second job checks out the trusted default-branch harness, downloads and
verifies that two-file artifact, installs it as `target/release/prove-bin`, and
installs the trusted wrapper as `target/release/prove`. It never compiles
candidate code. It binds score provenance directly to `github.sha`, then
re-hashes `prove-bin` immediately before execution.

Only the second job uses the protected GitHub Environment
`benchmark-private-data`. Configure that Environment exactly as follows:

- deployment branches and tags: **Selected branches and tags**;
- branch rules: the repository default branch and `submissions/*`; no tag rule;
- required reviewers: the trusted ranked-benchmark approvers;
- prevent self-review: enabled;
- secret `R2_ACCESS_KEY_ID`: the bucket-scoped token access-key ID;
- secret `R2_SECRET_ACCESS_KEY`: the corresponding secret;
- secret `R2_BUCKET_ENDPOINT`:
  `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`, with no bucket or object
  suffix; and
- variable `LIGHTER_PRIVATE_FIXTURE_SHA256`:
  `d014c969a88bcb0f1673acc410c9e75d1cac53d575463514855050226759c23f`.

The focused host-isolation package must be installed before enabling ranked
runs. It provides root-owned `bench-exec.sh` and `janitor.sh`, the narrow
passwordless sudo grant for the Actions identity, the disposable uid 560
account lifecycle, and the root-owned pinned Rust toolchain. After that host
prerequisite, no manual GitHub changes beyond the existing Environment
configuration above are expected.

The non-secret workflow constants are bucket `lighter-prover-private` and key
`fixtures/ranked-v1.json`. The R2 bucket must deny anonymous reads, and the API
token must have object-read access only to that bucket. The downloader uses the
root-owned AWS CLI at `/usr/local/bin/aws`, writes through a private temporary
file, checks the pinned SHA-256, applies mode `0600`, and atomically installs the
fixture. The workflow then unsets all three R2 credential variables before
starting `benchmark.sh`.

The current R2 object is temporary bootstrap data: it is the compatible public
fixture uploaded privately to exercise the complete production path, and must
be replaced by the generated private ranked fixture before the competition
baseline is finalized. The live production-downloader E2E completed
successfully with SHA-256
`d014c969a88bcb0f1673acc410c9e75d1cac53d575463514855050226759c23f`,
49,948,018 bytes, and a byte-for-byte comparison to the source. No credential
material was stored in this repository. There is intentionally no downloader
unit test; this live R2 E2E is its validation.

The score artifact is uploaded only after a successful benchmark. Raw fixture
bytes, candidate proof output, stdout/stderr, and failure artifacts are never
uploaded. Local `./benchmark.sh` continues to use the checked-in fixture and
requires no R2 configuration.

## Runner and sandbox requirements

The focused host design has two identities. `lighter-prover-challenge`
(uid 502) runs the trusted Actions runner and verifier.
`lighter-prover-bench` (uid/gid 560) is disposable and is the only identity that
extracts/builds the submitted archive or executes `prove-bin`. The root bridge
accepts only absolute paths strictly below
`/opt/lighter-prover-challenge/work`; build and proof descendants start with a
clean fixed environment, so workflow secrets are not forwarded to candidate
code.

The root supervisor runs the janitor after every one-job JIT runner exit, before
the next registration. The janitor terminates uid-560 processes, verifies that
none survive, purges its writable state, and deletes/recreates the account and
home. Runner-state cleanup separately resets uid 502 between registrations.
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
