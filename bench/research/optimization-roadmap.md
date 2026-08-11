# Lighter prover optimization roadmap

This file tracks the ordered optimization campaign separately from
`promoted-options.md`, which is the dataset of public promoted submissions.
Update this roadmap after every meaningful implementation, benchmark,
verification, rejection, submission, or change in priority.

Last updated: 2026-08-11, America/Chicago

## Status legend

- **Queued** — not implemented or measured yet.
- **Drafted** — source exists, but required compilation or measurement is incomplete.
- **Testing** — an isolated correctness or performance experiment is active.
- **Partial** — useful evidence exists, but the current frontier/M4 transfer case is unproven.
- **Rejected** — measured evidence crossed the rejection threshold and the change was reverted.
- **Operational** — an experiment-selection or submission policy rather than a code change.

## Ordered ideas

| # | Optimization angle | Status | Current evidence and update | Next action |
|---:|---|---|---|---|
| 1 | Binary BaseSum-63 CPU fallback | **Rejected** | Same release binary, Metal-backed `C-B-B-C-C-B` runs: candidate median `28.2697 s`, frontier-control median `27.1270 s`; candidate was `+4.21%` slower (`-4.04%` throughput). Trusted verification passed, but chain proofs raised the CPU constraint-row ceiling from 26 to 64. | Do not submit or retest the blanket fallback; retain only as evidence for selective admission. |
| 2 | Gate-by-gate CPU/Metal admission matrix | **Rejected (BaseSum-63)** | Selective row-free admission routed BaseSum-63 to CPU only in degree-16 transaction/final shapes and retained Metal in degree-14 chain shapes. Same-binary candidate median `37.31 s` versus frontier `37.02 s` (`+0.78%` runtime); mean was `+0.13%`. | Do not submit the selective BaseSum policy. Profile a different gate family only when counters show a queue-tail contribution exceeding its CPU survivor cost. |
| 3 | Metal threadgroup-size sweep by kernel | **Partial; blanket sweep rejected** | Same-binary blanket 1-D caps were non-repeatable: cap-64 median `30.135 s` versus cap-128 `30.220 s`, but means favored 128 by `1.89%` and confirmation reversed the early signal; cap-256 also showed no win. | Add per-kernel GPU execution-time counters and test quotient, leaf, parent, and absorb kernels separately; do not repeat a whole-pipeline cap sweep. |
| 4 | Column-store pool capacity and reuse policy | **Keep locally; submit hybrid** | Best-fit made 36 oversized loans totaling 1.80 GiB of excess mapping. Pure exact reuse avoided those loans but forced terminal allocations and was rejected. The hybrid exact-below-256-MiB policy cut miss-request bytes from 10.72 GB to 9.13 GB and 544 MiB misses from 9 to 6. Same-binary eight-run candidate median `33.80 s` vs control `35.21 s` (`-4.00%` runtime), means `33.34 s` vs `34.08 s` (`-2.18%`); pairwise signs split 2-2. Trusted verification passed 1/1. | Submit as an M4 transfer candidate because counters and both aggregates are positive; do not call it proven until the private-active result resolves. |
| 5 | Earlier buffer retirement at proof-stage boundaries | **Queued** | Existing frontier code retires several circuit extensions early, but no new last-consumer lifetime audit has been completed. | Map quotient, permutation, FFT, Merkle, and circuit-data last consumers and test one safe release point at a time. |
| 6 | FRI delayed-reduction tile tuning | **Queued** | The frontier includes a successful delayed-reduction lineage; local tile/block-size tuning is untested. The separate initial-oracle gathering experiment was `0.218%` slower and was reverted. | Sweep coefficient block sizes and reduction intervals around the current implementation. |
| 7 | Adaptive proof-pipeline window | **Queued** | No new adaptive window policy has been tested; prior watcher batching was a different mechanism and was `3.639%` slower. | Measure GPU occupancy, join waits, and resident memory for small fixed windows before attempting adaptation. |
| 8 | Rayon pool and Apple performance-core scheduling | **Rejected (global thread count)** | `RAYON_NUM_THREADS=8` regressed median runtime by `1.64%`; nine threads had a `0.39%` nominal median gain but a `0.13%` mean regression and split pairwise signs 2-2. | Do not hard-code a worker count. Revisit only with phase-specific QoS/affinity plus counters, especially on the actual M4 Pro topology. |
| 9 | Allocator configuration | **Partial; eight arenas rejected** | `MALLOC_CONF=narenas:8` initially appeared `2.85%` faster but failed reverse confirmation; four-sample median was `+0.46%` runtime and mean `+0.83%`. Default decay remains inherited and correct. | Do not encode the M1 core-count arena heuristic. Measure fragmentation/lock statistics before testing background thread, tcache, or large-allocation settings. |
| 10 | Final-block circuit embedding on the current frontier | **Rejected on Metal** | Six-circuit semantic identity and trusted verification passed, but current-frontier Metal `B-C-C-B` measured embedded median `27.735 s` versus runtime build `26.625 s`, a `+4.17%` regression. The older CPU-only `14.46%` win did not transfer. | Do not submit the compact final blob. Revisit only if decode can avoid reconstructing constants/sigmas on the active serialized Metal queue. |
| 11 | Generator data-layout locality | **Rejected (tested variants)** | Post-seed watcher-template same-binary `B-C-C-B` measured baseline median `183.36 s` versus candidate `186.58 s` (`+1.76%` runtime); trusted verification passed 1/1. Dense scatter was officially rejected at `-33.46%`, and batched watcher accounting was `3.639%` slower. | Require new phase evidence before another generator-layout rewrite; do not recombine these rejected variants. |
| 12 | Merkle/FRI query zero-copy paths | **Queued** | No true zero-copy query-path implementation has been tested. Initial-oracle sorted gathering was related but distinct and regressed by `0.218%`. | Audit allocation/copy counts in query extraction and path materialization, then remove one temporary buffer at a time. |
| 13 | Proof serialization into one pre-sized output buffer | **Queued** | The frontier already uses a 2 MiB `BufWriter`, explicit flush, and fast process exit; no direct single-buffer serialization experiment has been measured. | Measure serialization/output spans first and proceed only if they exceed expected run noise. |
| 14 | M4-aware static admission policy | **Queued / dependent** | Blanket BaseSum-63 regressed `4.21%` and structurally selective BaseSum-63 regressed `0.78%` by median. No per-shape policy has positive repeatable evidence yet. | Accumulate counter-backed results on the actual M4 Pro before hard-coding any architecture-specific admission decision. |
| 15 | Leaderboard redraw for statistically credible candidates | **Operational** | No new redraw has been submitted from this campaign. The policy is to redraw only verified candidates with repeatable local evidence or an already-strong executable, never to treat marker-only promotions as algorithmic evidence. | Apply only after a candidate survives correctness, local A/B, and frontier refresh checks. |

## Related completed experiments

These experiments overlap the roadmap but are not substitutes for the exact
queued ideas above.

| Experiment | Result | Decision |
|---|---|---|
| CPU-survivor static alpha reduction | Candidate approximately `0.387%` slower | Reverted |
| FRI initial-oracle leaf gathering | Candidate approximately `0.218%` slower | Reverted |
| Batched initial generator-watcher accounting | Focused timing approximately `3.639%` slower | Reverted |
| Dense transaction-value scatter | Official score `28.6955` TPS versus `29.7608` frontier (`-33.46%` challenge delta) | Rejected |
| Pre-resolved transaction-input seed metadata | Correctness checks passed, but no qualifying private-active timing evidence | Reverted / not submitted |
| Blanket BaseSum-63 CPU fallback | `4.21%` slower by Metal median | Reverted |
| Selective row-free BaseSum-63 CPU fallback | `0.78%` slower by Metal median; `0.13%` slower by mean | Reverted |
| Blanket Metal 1-D threadgroup cap sweep | Cap 64/128 result changed sign; cap 256 showed no win | Reverted |
| Jemalloc `narenas:8` | `0.46%` slower by median; `0.83%` slower by mean | No source change |
| Rayon 8/9 workers | Eight regressed `1.64%`; nine was aggregate noise | No source change |
| Final-block compact embedding | Older CPU-only `14.46%` win became `4.17%` slower on current-frontier Metal | Reverted; do not submit |
| Hybrid exact-bin Metal column-store pool | `4.00%` lower local median and `2.18%` lower mean; 1.58 GB less fresh shared-buffer demand; pairwise signs 2-2 | Keep locally; submit for M4 validation |

## Experiment update checklist

For every idea tested:

1. Record the base commit, candidate commit or exact dirty diff, release-binary SHA-256, hardware, and environment.
2. Verify the intended path with diagnostics before trusting timing results.
3. Run proof compatibility through the pinned trusted verifier when possible.
4. Use a predeclared alternating same-binary order and report every sample, median, mean, and throughput-equivalent delta.
5. Mark the idea **Rejected** and revert it when it crosses its failure threshold; do not keep ambiguous code in the frontier.
6. Refresh `promoted-options.md` and the live submission queue before committing or submitting a surviving candidate.
7. Update this roadmap and `experiment-results.tsv` after every keep/reject decision, trusted verification result, submission creation, and final promoted/rejected/failed submission outcome.
8. For an official submission, record its full ID, candidate commit, frontier score at submission time, official score/delta, status, and note reference in both logs before considering the experiment closed.
