# Circuit agent contract

This directory is part of the protected prover/verifier interface. A circuit
change is not an ordinary micro-optimization: it can invalidate the trusted
verifier or make a local speedup incomparable with the official benchmark.

<!-- LIGHTER-FRONTIER-STATUS:START -->
- Source: `github.prs + yukon.submissions`; observed: `2026-08-20T16:20:47+00:00`; freshness: **stale**
- Official frontier: **unknown**
  (source `github.pr.current-best`, commit `None`)
- Our best known official result: **unknown**
  (source `yukon.submissions.filtered-solver-users`, commit `None`)
- Gap to frontier: **unknown**
- Frontier proving time: `unknown`;
  our best proving time: `unknown`
- PRs observed: `0`; pending validation: `0`;
  latest PR: `None`
- CI observations: `0` (passed `0`, failed `0`, pending `0`); last loop fingerprint: `bd0d23788529000a`
- GitHub Status: `fresh` / `none` — All Systems Operational; active incidents `1`
- Solver cohort: `7` accounts; pooled FAST target `2`; scope `pooled-across-accounts`; coverage `round-robin-initial-observation`
- Account bests: `heathcliffeth7`=unknown, `joelcrypto21`=unknown, `basingamarket-ctrl`=unknown, `barangunay0`=unknown, `joelchristianai3-jpg`=unknown, `nathanethx`=unknown, `homalenderrr`=unknown
- Experiment draws (fast / slow / regression):
  - `exp-quarter-l2`: `0 / 0 / 0` of `2` pooled fast over `0` attempts; coverage `0/7`; equal-account median `None`; best `unknown`; decision `resubmit/evaluate`
- Warning: **GitHub refresh failed: str; check DNS/network access or configure the corresponding API URL override; Yukon CLI refresh failed; check Yukon DNS/network access or YUKON_API_URL; GitHub Status incident: All Systems Operational**
<!-- LIGHTER-FRONTIER-STATUS:END -->

## Rules

- Preserve the fixed chain width `304`, heavy width `4`, light width `10`, and
  the verifier/circuit digest unless the challenge owner explicitly changes the
  protocol.
- Do not alter constraints, serialization, transcript order, public inputs, or
  fixture semantics merely to improve a local smoke result.
- Keep performance work outside the mathematical protocol whenever possible:
  scheduling, allocation, batching, backend kernels, and data movement are the
  preferred first targets.
- Every circuit experiment needs an explicit parent commit, a local correctness
  result, and an authoritative official result before it can be called a win.
- Never record proof bytes, private fixture contents, credentials, or raw
  verifier output in `findings/`.
- Read-only subagents may inspect this directory but may not edit it. A single
  isolated implementation worktree owns any candidate change.

## Code review rules

- Reject a performance claim that lacks `passed: true` and complete proof
  verification.
- Separate public synthetic smoke timing from private ranked throughput.
- Treat one fast draw as evidence, not a promoted invariant; use the shared
  experiment ledger and the five-fast policy before recommending promotion.
