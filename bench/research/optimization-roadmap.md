# Lighter prover optimization roadmap

This file tracks the ordered optimization campaign separately from
`promoted-options.md`, which is the dataset of public promoted submissions.
Update this roadmap after every meaningful implementation, benchmark,
verification, rejection, submission, or change in priority.

Last updated: 2026-08-11, America/Chicago

Current public frontier: promotion #138, `e268c13`,
`29.9399105848455 tx/s`. Local experiment rows retain their historical frontier
references so submitted and rejected deltas remain reproducible.

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
| 4 | Column-store pool capacity and reuse policy | **Officially rejected; reverted** | Best-fit made 36 oversized loans totaling 1.80 GiB of excess mapping. The hybrid exact-below-256-MiB policy cut miss-request bytes from 10.72 GB to 9.13 GB and measured `-4.00%` local median runtime, but pairwise signs split 2-2. Candidate `b727fb0`, official submission `1d69b9c2-8554-47af-a40c-fce715581b57` (Yukon commit `c9f47d6`) scored `29.4067300861804 tx/s` against `29.8785698468374`, a `-0.4718397606570 tx/s` delta. | Keep the diagnostic evidence only. Best-fit is restored; do not stack this M1-positive/M4-negative policy into later candidates. |
| 5 | Earlier buffer retirement at proof-stage boundaries | **Kept locally; ready for official validation** | Commit `82635d8` moves the three per-proof initial FRI Merkle trees into query construction, copies every challenged initial leaf/path, then drops their Metal-backed stores before fold-tree query proofs. Combined with exact reuse below 256 MiB, same-binary `B-C-C-B` measured control `37.153416166,37.027430625 s` versus candidate `36.090781292,35.907613125 s`; candidate mean improved `2.94%`, both pairings won, trusted verification passed 5/5, release SHA-256 `8b6b4c6ed8a3982fa818e2c314a587753a593f4ca784c10d72f3db67caab1d5a`. | Submit the combined stack and record its full terminal result. Do not attribute the whole delta to retirement alone; exact-bin pool policy and the earlier lifetime boundary were toggled together. Continue local work incrementally while validation runs. |
| 6 | FRI delayed-reduction tile tuning | **Rejected (coefficient-slot block); other parameters untested** | Same-binary symmetric sweep `2048-4096-1024-1024-4096-2048` passed trusted verification 6/6. Means were 2,048 `37.418893958 s`, 4,096 `42.996962541 s` (`+14.9071%`), and 1,024 `46.379565646 s` (`+23.9469%`). Large endpoint drift makes the exact deltas noisy, but neither alternative had a positive aggregate signal; release SHA-256 `3f12ff6e3da450d8cd85de2048667f153bccabf477b54dec1df6c0b40727b650`. | Retain 2,048 and do not repeat a blind M1 power-of-two sweep. Other reduction intervals or shape-aware tiles require per-reduction counters or direct M4 access before implementation. |
| 7 | Adaptive proof-pipeline window | **Queued** | No new adaptive window policy has been tested; prior watcher batching was a different mechanism and was `3.639%` slower. | Measure GPU occupancy, join waits, and resident memory for small fixed windows before attempting adaptation. |
| 8 | Rayon pool and Apple performance-core scheduling | **Rejected (global thread count)** | `RAYON_NUM_THREADS=8` regressed median runtime by `1.64%`; nine threads had a `0.39%` nominal median gain but a `0.13%` mean regression and split pairwise signs 2-2. | Do not hard-code a worker count. Revisit only with phase-specific QoS/affinity plus counters, especially on the actual M4 Pro topology. |
| 9 | Allocator configuration | **Partial; eight arenas rejected** | `MALLOC_CONF=narenas:8` initially appeared `2.85%` faster but failed reverse confirmation; four-sample median was `+0.46%` runtime and mean `+0.83%`. Default decay remains inherited and correct. | Do not encode the M1 core-count arena heuristic. Measure fragmentation/lock statistics before testing background thread, tcache, or large-allocation settings. |
| 10 | Final-block circuit embedding on the current frontier | **Rejected on Metal** | Six-circuit semantic identity and trusted verification passed, but current-frontier Metal `B-C-C-B` measured embedded median `27.735 s` versus runtime build `26.625 s`, a `+4.17%` regression. The older CPU-only `14.46%` win did not transfer. | Do not submit the compact final blob. Revisit only if decode can avoid reconstructing constants/sigmas on the active serialized Metal queue. |
| 11 | Generator data-layout locality | **Rejected (tested variants)** | Post-seed watcher-template same-binary `B-C-C-B` measured baseline median `183.36 s` versus candidate `186.58 s` (`+1.76%` runtime); trusted verification passed 1/1. Dense scatter was officially rejected at `-33.46%`, and batched watcher accounting was `3.639%` slower. | Require new phase evidence before another generator-layout rewrite; do not recombine these rejected variants. |
| 12 | Merkle/FRI query zero-copy paths | **Rejected (tested grain variants)** | `with_min_len(4)` regressed `9.14%`. Max grain one initially measured `2.28%` faster while the now-rejected pool policy was present, but the isolated promotion-137 `B-C-C-B / C-B-B-C` rerun reduced that to `0.42%` by mean and `0.32%` by median, with pairwise signs split 2-2. Exact isolated SHA-256 `e391abdded403ddbb0246b70848dbac74252fc8f1351265e699cfdc77c4248ff`; source checkpoint `530fafd`. | Max grain one crossed its reverse-confirmation rejection rule and was reverted. Continue with query-data layout/copy elimination, not global iterator grain. |
| 13 | Proof serialization into one pre-sized output buffer | **Rejected by profile ceiling** | Three diagnostic public workers measured the complete `serialize_and_flush_proof` span at `0.357`, `0.374`, and `0.468 ms` inside `30.05–33.03 s` processes: roughly `0.001–0.002%` of runtime. The existing 2 MiB `BufWriter` already makes this path immaterial. | Do not implement a single-buffer serializer unless the official fixture or proof format changes enough to make output exceed `0.1%` of runtime. |
| 14 | M4-aware static admission policy | **Queued / dependent** | Blanket BaseSum-63 regressed `4.21%` and structurally selective BaseSum-63 regressed `0.78%` by median. No per-shape policy has positive repeatable evidence yet. | Accumulate counter-backed results on the actual M4 Pro before hard-coding any architecture-specific admission decision. |
| 15 | Leaderboard redraw for statistically credible candidates | **Operational** | No new redraw has been submitted from this campaign. The policy is to redraw only verified candidates with repeatable local evidence or an already-strong executable, never to treat marker-only promotions as algorithmic evidence. | Apply only after a candidate survives correctness, local A/B, and frontier refresh checks. |
| 16 | Four-state Poseidon2 FRI proof-of-work search | **Rejected** | Profiled PoW spans totaled `1.991 s`. Existing x4 was `1.177x` faster than four scalar permutations and passed a four-state differential, but same-binary eight-run end-to-end means regressed `0.21%`, medians nominally improved `0.44%`, and pairwise signs split 2-2. | Reverted and not submitted. Revisit PoW only if a dependency/profile change puts it on the proof-window critical path. |
| 17 | Skip finished prover-state teardown before `_exit` | **Officially rejected; reverted** | A trace exposed `43.146 ms` of return-time teardown. Same-binary mean improved `0.82%`, median `0.056%`, pairwise 2-2, and trusted verification passed 1/1. Candidate `d1d839a`; submission `7244133b-9f3a-4072-b84d-89993282260e` (Yukon `23f3ca0`) scored `25.8191460929401 tx/s` against `29.8785698468374`, a `-4.0594237538973 tx/s` delta. | The official run landed in a broad low-service band and is inconclusive about a `~0.26%` effect, but it crossed the predeclared failure rule. Source is reverted; retain only the trace/safety evidence. |
| 18 | Start `light_tx` embedded decode before the pre proof | **Rejected; reverted** | Startup tracing showed a `96.099 ms` remaining-loader tail and `light_tx` was the longest remaining cold decode at `367.2 ms`. A dedicated early loader nevertheless lost both `B-C-C-B` pairings: control `27.30,27.63 s`, candidate `27.71,27.97 s`; candidate mean/median regressed `1.365%`. Same binary SHA-256 `a5d5c466acc0f0861e71b0c0d0f944edbed9065a19d724a50105bdb709c530a7`. | Do not start the other remaining decodes at launch. The extra decode competes with the pre load/proof; revisit only with phase-aware scheduling or a noncontending codec/data path. |
| 19 | Dependency-striped delayed-reduction opening dot product | **Rejected; reverted** | Four 160-bit lanes per quadratic limb preserved field values in focused boundary tests, but the production-degree release kernel screen was decisively negative: frontier single `7.483208,7.465583 ms` versus striped `8.898250,8.896291 ms`, `+19.05%` mean runtime, both pairings lost. | Keep the promoted single accumulator. Do not try two/eight lanes without assembly or hardware-counter evidence that register pressure/spills can be avoided. |
| 20 | Retire ephemeral commitment coefficients before FRI | **Rejected; reverted** | Wire/Z/quotient coefficients were dropped after reduced-polynomial construction while Merkle trees/caps and reusable constants remained intact. Trusted verification passed 5/5, but same-binary `B-C-C-B` control mean `30.117623313 s` beat candidate `30.495082562 s`; candidate was `1.2533%` slower and lost both pairings. | Do not eagerly free the inner vectors on the FRI critical path. Explore ownership transfer or pool reuse that avoids serialized destructor/deallocation work. |
| 21 | Bitmap-aware witness initialization and adjacent zero-fill audit | **Promoted externally in #138; inspect before extending** | The #138 public note reports deleting dense `PartitionWitness::new` zero-fill and making the sole unguarded materialization path bitmap-select zero for unset slots, avoiding roughly 71 MB of stores per transaction proof and 285 MB for the final proof. The promotion arrived inside a larger scheduling/Metal/memory/FFT stack, so the isolated official contribution is unknown. | Sync and inspect the exact #138 guards first. Do not duplicate its change; profile other large initialized witness/generator buffers and require a complete reader audit before testing any uninitialized allocation. |

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
| Hybrid exact-bin Metal column-store pool | `4.00%` lower local median, but official `29.4067300861804` vs `29.8785698468374 tx/s` (`-0.47184`) | Officially rejected; reverted |
| Pre-sized proof serialization buffer | Full serialize-and-flush span only `0.357–0.468 ms` (`~0.001–0.002%` of worker time) | Rejected before implementation |
| FRI query Rayon minimum grain of four | Control `30.10,29.41 s`; candidate `33.21,31.74 s`, `9.14%` slower by mean; both pairings lost | Reverted without confirmation |
| FRI query Rayon maximum grain of one | Isolated control `26.90,26.98,28.07,25.98 s`; candidate `27.32,26.39,26.31,27.46 s`; candidate only `0.42%` faster by mean and `0.32%` by median, pairwise 2-2 | Reverted; not submitted |
| Four-state Poseidon2 PoW batching | Control mean/median `27.050/27.140 s`; candidate `27.1075/27.020 s`; mean `0.21%` slower, median `0.44%` nominally faster, pairwise 2-2 | Reverted; not submitted |
| Finished prover-state destructor elision | Local trace-derived fixed cost `43.146 ms`; official `25.8191460929401` vs `29.8785698468374 tx/s` (`-4.0594237538973`) | Officially rejected; reverted |
| Early `light_tx` embedded decode | Control `27.30,27.63 s`; candidate `27.71,27.97 s`; candidate `1.365%` slower and lost both pairings | Reverted without confirmation |
| Four-lane opening dot-product accumulators | Single `7.483208,7.465583 ms`; striped `8.898250,8.896291 ms`; striped `19.05%` slower and lost both pairings | Reverted before full proving |
| Early retirement of per-proof coefficient vectors | Control `29.915660042,30.319586584 s`; candidate `30.211574167,30.778590958 s`; candidate `1.2533%` slower, trusted verification 5/5 | Reverted without confirmation |
| FRI coefficient-slot block sweep | 2,048 mean `37.418893958 s`; 4,096 mean `42.996962541 s` (`+14.9071%`); 1,024 mean `46.379565646 s` (`+23.9469%`); trusted verification 6/6 | Retain 2,048; selector reverted without confirmation |
| Exact-size Metal reuse plus FRI initial-tree retirement | Control `37.153416166,37.027430625 s`; candidate `36.090781292,35.907613125 s`; candidate `2.94%` lower by mean, both pairings won, trusted verification 5/5 | Kept as `82635d8`; ready for official validation |

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
