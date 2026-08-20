# Vendored Plonky2 agent instructions

<!-- LIGHTER-FRONTIER-STATUS:START -->
- Source: `github.prs + yukon.submissions`; observed: `2026-08-20T01:04:08+00:00`; freshness: **fresh**
- Official frontier: **33.004420 TPS**
  (source `yukon.submissions`, commit `None`)
- Our best known official result: **unknown**
  (source `yukon.submissions.filtered-solver-users`, commit `None`)
- Gap to frontier: **unknown**
- Frontier proving time: `75.747 s`;
  our best proving time: `unknown`
- PRs observed: `200`; pending validation: `10`;
  latest PR: `7453`
- CI observations: `5` (passed `24`, failed `0`, pending `1`); last loop fingerprint: `8238ad562d17377c`
- GitHub Status: `fresh` / `none` — All Systems Operational; active incidents `0`
- Solver cohort: `7` accounts; pooled FAST target `2`; scope `pooled-across-accounts`; coverage `round-robin-initial-observation`
- Account bests: `heathcliffeth7`=unknown, `joelcrypto21`=unknown, `basingamarket-ctrl`=unknown, `barangunay0`=unknown, `joelchristianai3-jpg`=unknown, `nathanethx`=unknown, `homalenderrr`=unknown
- Experiment draws (fast / slow / regression):
  - `exp-quarter-l2`: `0 / 0 / 0` of `2` pooled fast over `0` attempts; coverage `0/7`; equal-account median `None`; best `unknown`; decision `resubmit/evaluate`
<!-- LIGHTER-FRONTIER-STATUS:END -->

This is the highest-leverage and highest-risk implementation surface. Changes
must improve prover execution while preserving the challenge's fixed proof
compatibility boundary.

- Do not alter circuit widths, FRI/verifier parameters, transcript ordering,
  circuit digest, or trusted-verifier code.
- Prefer value-exact changes that remove redundant allocation/copying, preserve
  polynomial order, improve overlap, or reduce backend work without changing
  the represented constraints.
- For Metal, preserve CPU fallback, command-buffer lifetime, synchronization,
  queue ownership, buffer-set limits, and the exclusive-phase invariants.
  A shader or generated metallib change needs a separate measured experiment
  and must be tested on the official GPU class.
- Never leave release-only probes, debug prints, stale generated binaries, or
  accidental formatting churn in a candidate archive.
- Run focused unit tests plus the release build and trusted verifier after each
  change. Record exactly which files changed and whether proof verification
  passed.

Do not stack an unmeasured vendor change on top of an unclassified scheduler
change. If a result is neutral or negative, record it as useful evidence and
make the next agent able to bisect it.

## Code Review Rules

- Require a value-exact or verifier-backed argument for hashing/backend changes;
  a timing claim alone is insufficient.
- Reject changes that alter transcript bytes, digest semantics, circuit widths,
  or the trusted verifier contract under the label of a performance tweak.
- Review PR evidence by exact commit and official result; do not promote a
  mechanism from one noisy draw.
