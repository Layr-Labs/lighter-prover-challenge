# Benchmark assessment — 2026-07-17

## Verdict: READY WITH CONDITIONS

The local process/FIFO timing escape is fixed and regression-tested in this
worktree. The current setup is suitable for a bounded beta if Yukon enforces
the declared editable scope before dispatch; promotion still requires the
planned private fixture and the remaining operational checks below.

## Resolved in this worktree

### Process/FIFO timing escape

- **Local reproduction:** With the original sandbox profile, a temporary
  contestant prover created `proof.bin` as a FIFO, forked a child to generate
  the valid proof, and exited the timed parent. The full 32-transaction
  benchmark passed with `proving_seconds = 5.875` and score `5.014`, while the
  wall-clock run took `39.690` seconds.
- **Fix:** The profile now denies `process-fork`
  (`.github/scripts/prover.sb:4`), so contestant code cannot leave a child
  process computing or writing after `prove-bin` exits.
- **Regression coverage:** `.github/scripts/test-sandbox-prove.sh` verifies
  that a sandboxed `prove-bin` cannot leave a child writing in scratch; CI runs
  that test before the public benchmark
  (`.github/workflows/ci.yml:19-23`).
- **Post-fix validation:** The same FIFO candidate was rejected by the full
  benchmark (`prover failed: exit status: 101`). The normal prover then passed
  the same sandboxed 32-transaction benchmark with
  `proving_seconds = 28.954` and score `1.017`.

### Prover wall-clock timeout

- **Local reproduction:** A local sibling `prove` executable that only slept
  for two seconds kept `bench` waiting until it exited; the parent then failed
  only because `proof.bin` was missing.
- **Fix:** `bench` now polls its direct child, kills and reaps it at the
  15-minute default deadline, and produces no score
  (`challenge/src/bin/bench.rs:52-75,111-118`). The trusted
  `LIGHTER_PROVE_TIMEOUT_SECONDS` environment override allows short local
  timeout tests without changing the ranked limit.
- **Regression coverage:** `.github/scripts/test-prover-timeout.sh` runs the
  real parent against a two-second sibling child with a one-second local
  override; it passes only when the parent reports the timeout. The workflow
  runs it after building the binaries (`.github/workflows/ci.yml:25-29`).
- **Post-fix validation:** The local timeout regression passed, and the normal
  32-transaction benchmark passed with `proving_seconds = 28.474` and score
  `1.034`.

## Positive controls verified

- Candidate content is overlaid onto a trusted `master` checkout only after
  ancestry and path checks
  (`.github/scripts/run-ranked-benchmark.sh:7-25,73-79`).
- The contestant code is isolated to the `prove` child; the trusted parent owns
  timing, proof decoding, fresh circuit construction, verification, and score
  output (`challenge/src/bin/bench.rs:44-91`).
- Both proofs are verified and key fixture-bound outputs are checked before a
  score is emitted (`challenge/src/bin/bench.rs:94-120`).
- Toolchain, dependency revisions, and candidate build inputs are pinned/locked
  (`rust-toolchain:1`; `challenge/Cargo.toml:11-18`;
  `.github/scripts/run-ranked-benchmark.sh:27-33,76-79`).

## External platform control

`benchmark.json` declares `challenge/submission` as the sole editable path
(`benchmark.json:7-9`). Yukon dispatches the candidate ref, so GitHub evaluates
that ref's workflow YAML; this is safe only because Yukon materializes the
candidate with edits restricted to the declared path before dispatch. Under that
platform guarantee, `.github/workflows/benchmark.yml` remains trusted and the
repository's own ancestry/path check is defense in depth. Confirm this Yukon
behavior once as part of runner integration; it cannot be proven from this
repository alone.

## Release gates

1. Add a memory limit.
2. Verify runner access, Yukon editable-scope enforcement, and private-fixture
   provenance operationally.

## Deferred promotion-evaluation work

The current ranked path uses the known public fixture
`bench/bench_test.json` (`.github/scripts/run-ranked-benchmark.sh:53-82`;
`benchmark.sh:13-16`). A private fixture will be used later for promotion
evaluation. That fixture must remain secret and be evaluated by the trusted
harness; until then, public-fixture scores are not evidence of general prover
performance.

The current paired measurement uses one baseline-first sample
(`.github/scripts/run-ranked-benchmark.sh:69-82`). Repeat measurements and
cache/thermal characterization are intentionally deferred to promotion
evaluation, where their additional runner cost is justified.

## Scope of validation

Shell syntax, benchmark JSON, sandbox and timeout regressions, a full
32-transaction FIFO escape, its post-fix rejection, and normal
32-transaction proving were validated locally. The repository's only changes
are this report, the untracked handoff prompt, and the benchmark hardening
files.
