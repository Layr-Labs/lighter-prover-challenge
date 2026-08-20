# `bench/` agent instructions

<!-- LIGHTER-FRONTIER-STATUS:START -->
- Source: `github.prs + yukon.submissions`; observed: `2026-08-20T15:31:03+00:00`; freshness: **stale**
- Official frontier: **33.004420 TPS**
  (source `github.pr.current-best`, commit `506be5f46e5e95e17d9770b594b191bda893de93`)
- Our best known official result: **unknown**
  (source `yukon.submissions.filtered-solver-users`, commit `None`)
- Gap to frontier: **unknown**
- Frontier proving time: `unknown`;
  our best proving time: `unknown`
- PRs observed: `200`; pending validation: `17`;
  latest PR: `7579`
- CI observations: `5` (passed `21`, failed `0`, pending `4`); last loop fingerprint: `2d78baf83038fd5a`
- GitHub Status: `fresh` / `none` — All Systems Operational; active incidents `1`
- Solver cohort: `7` accounts; pooled FAST target `2`; scope `pooled-across-accounts`; coverage `round-robin-initial-observation`
- Account bests: `heathcliffeth7`=unknown, `joelcrypto21`=unknown, `basingamarket-ctrl`=unknown, `barangunay0`=unknown, `joelchristianai3-jpg`=unknown, `nathanethx`=unknown, `homalenderrr`=unknown
- Experiment draws (fast / slow / regression):
  - `exp-quarter-l2`: `0 / 0 / 0` of `2` pooled fast over `0` attempts; coverage `0/7`; equal-account median `None`; best `unknown`; decision `resubmit/evaluate`
- Warning: **Yukon CLI refresh failed; check Yukon DNS/network access or YUKON_API_URL; GitHub Status incident: All Systems Operational**
<!-- LIGHTER-FRONTIER-STATUS:END -->

The `bench` crate owns worker orchestration and the end-to-end proving pipeline:
pre-execution, heavy/light transaction proofs, recursive chain folds, and the
final block proof.

## Work here

- Profile the stage that is actually on the critical path before changing it.
- Keep scheduling and allocation experiments isolated. A window, QoS,
  exclusive-GPU, Rayon, prewarm, or lifetime change is one experiment unless
  the note explicitly defines a tested stack.
- Preserve transaction routing, recursive input order, public outputs, and
  circuit construction parameters. Changes in `bench/src/prover.rs` or
  `bench/src/bin/prove.rs` must remain compatible with the pinned verifier.
- Keep diagnostics under `cfg(feature = "diagnostic_profile")` or an
  equivalent non-default gate. Release output must not depend on logging,
  probes, or debug-only timing.
- Treat M5/16 GB local timings as directional and correctness-only. The ranked
  host is the official M4 Pro/48 GB environment; compare same-class official
  draws and do not infer a mechanism from one noisy draw.

## Required gates

```sh
cargo test --release -p bench --bin prove
./setup.sh
./benchmark.sh
```

The local score must show the trusted verifier passed and all expected proofs
verified. It is still labelled `public-synthetic-smoke` and is not a ranking
claim.

## Existing knowledge

Read the relevant reports under `findings/` before reopening a direction. In
particular, do not assume that more outer parallelism, a deeper proof window,
GPU NTT, a dedicated chain pool, spin-waiting, or a prewarm is beneficial just
because the dependency graph suggests overlap. The official result and the
recorded host class decide.

## Code Review Rules

- Review every PR against the current frontier snapshot and its exact base/head
  commit, not against a stale note or local smoke score.
- Require an isolated hypothesis, matching experiment ID, complete verifier
  evidence, and same-host/same-parent comparisons before accepting a throughput
  claim.
- Treat window-depth, scheduler, prewarm, and GPU-admission changes as separate
  mechanisms unless the PR explicitly records a controlled stack experiment.
