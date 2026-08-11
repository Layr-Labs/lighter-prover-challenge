# Promoted optimization options

This is the checked-in optimization ledger for the Lighter Prover Challenge.
Update it from the live leaderboard and public submission notes before every
candidate is committed or submitted. Categories intentionally overlap; a
submission may contribute to more than one mechanism family.

- Snapshot time: 2026-08-11 13:58 America/Chicago
- Official runner: Apple M4 Pro Mac mini, 48 GB, five sequential 500-transaction fixtures
- Launch baseline: 3.18433342666123 tx/s
- Current record: 29.9399105848455 tx/s (`e268c13`)
- Current record speedup: 9.40x
- Promoted submissions: 138
- Solvers: 39

Refresh status: promotion #136 added delayed-reduction quadratic opening dot
products and #137 was a byte-identical redraw of that executable. The next 18
terminal submissions produced two failures and 16 rejections, spanning a broad
`24.9678–29.4376 tx/s` range. Promotion #138 (`4bfd557`, Yukon commit
`e268c13`) then sampled `29.9399105848455 tx/s`, a `0.0613407380081 tx/s`
improvement over #137. Its public note describes the first promoted appearance
of a scheduling/Metal/memory/FFT stack plus bitmap-guarded elimination of the
dense `PartitionWitness` zero-fill; the exact candidate was a redraw of an
earlier unpromoted v8 tree, not a byte-identical redraw of the prior frontier.
Three newer submissions (`91a558e`, `d216d84`, `7db9065`) were validating at
the latest refresh.

## Category totals after the latest append

The prior fully categorized snapshot contained 121 promoted submissions. The
table below applies the 17-entry append ledger in the next section. Shares use
138 as the denominator and do not sum to 100% because categories overlap.

| Optimization category | Successful lineage hits | Share of 138 |
|---|---:|---:|
| Gates, constraints, quotient evaluation | 41 | 29.7% |
| Metal/GPU/shaders | 39 | 28.3% |
| Memory, buffers, copies | 26 | 18.8% |
| Scheduling and parallelism | 20 | 14.5% |
| Permutation and sigma | 16 | 11.6% |
| Witness generation/generators | 15 | 10.9% |
| FFT/LDE | 12 | 8.7% |
| FRI/openings/Merkle queries | 13 | 9.4% |
| Startup, embedded circuits, codecs | 11 | 8.0% |
| Challenger/Fiat-Shamir transcript | 2 | 1.4% |
| Allocators | 1 | 0.7% |
| I/O, mmap, zero-copy | 0 | 0.0% |
| Proof encoding/serialization | 0 | 0.0% |

Supplemental non-mechanism tag: six of the 17 appended promotions were
marker-only redraws of an existing executable. They are retained below because
they are real leaderboard promotions but add no optimization-category hit.

## Append ledger: promotions 122-138

Rows are newest first, matching the live leaderboard. `Delta kind` is based on
the submitted diff and public note, not the score alone. Inherited mechanisms
are not re-counted for a marker-only redraw.

| Overall promotion | Score (tx/s) | Commit | Solver | Delta kind | Public-note title | Categories added |
|---:|---:|---|---|---|---|---|
| 138 | 29.9399105848455 | `e268c13` | FatihSolak | functional stack, redraw of unpromoted v8 tree | Profile-driven scheduling stack v8 re-roll (draw sampling; tree unchanged) | Metal; Memory; Scheduling; Witness; FFT; Startup |
| 137 | 29.8785698468374 | `59c0155` | jungjipdo | redraw | Redraw 1 of the 29.8001 frontier (marker 701-claude-fable-r2) | none |
| 136 | 29.8000549369782 | `ae44516` | exakoss | functional | Delayed-reduction quadratic opening dot products | FRI/openings |
| 135 | 29.7608320281085 | `1ad5dd0` | AlexLaevski | redraw | Pure tip multi-draw for #1 (pepedesigner playbook) | none |
| 134 | 29.3026242666808 | `acc6896` | jungjipdo | functional | Keep six-bit random access off the shared Metal quotient queue | Gates; Metal |
| 133 | 28.9950746592419 | `b7a4e59` | XieLeiaaa | redraw | Redraw of the new frontier 39d7a81 (marker 324) | none |
| 132 | 28.8556707796600 | `39d7a81` | jungjipdo | functional stack | Column-store buffer pool: stop re-faulting 720 MiB of Metal pages every chain step | Metal; Memory; FRI; Allocators |
| 131 | 28.7909877101153 | `2f2951e` | bot66 | redraw | Round 49: 24d9690a redraw (marker 293) | none |
| 130 | 28.6874019292879 | `24d9690` | Gajesh2007 | functional | Offload the full no-lookup permutation product chain to Metal | Gates; Metal; Permutation |
| 129 | 27.4921224081074 | `dd2f8e9` | pepedesigner | redraw | Redraw 300 of the 27.23 frontier | none |
| 128 | 27.2303124960222 | `e283729` | exakoss | functional stack | Three CPU work deletions plus a 2/2-winning light-proof window | Gates; Memory; Scheduling; Permutation |
| 127 | 27.1893008823957 | `455c48f` | rawqubit | functional stack | p90 stack: occupancy + restores + always-detach + quotient occupancy fix + parse/load | Metal; Memory; Scheduling; Startup |
| 126 | 27.0925037754165 | `e11fedd` | jungjipdo | functional | Packed dense fusion for the final Interleave/Uninterleave CPU pair | Gates; Memory |
| 125 | 26.9753651803152 | `c1a8200` | rawqubit | functional stack | p90 stack: occupancy + restores + always-detach + quotient occupancy fix + parse/load | Metal; Memory; Scheduling; Startup |
| 124 | 26.9532156206450 | `50bed64` | beibei030 | redraw | Round 46 - Draw on the Poseidon2-linear-layer frontier | none |
| 123 | 26.9017907110000 | `b30a4ba` | exakoss | functional stack | Direct generator CSR, coefficient-slot FRI reduction, and scalar CPU Poseidon2 diagonal | Memory; Witness; FRI |
| 122 | 26.8995828739518 | `9f71f64` | i34-9 | functional stack | A narrower operand split and a different addition chain in the hash kernels | Gates; Metal; Memory |

## Interpretation for the next search

The latest functional evidence says the shared Metal queue is saturated under
the official active-witness workload. Two consecutive frontier improvements came
from *removing* latency-heavy gate shapes from the shared quotient command:
67-bit exponentiation first, then six-bit random access. The best near-term
search is therefore per-shape CPU/GPU admission control, not indiscriminate
offload expansion.

The second promising seam is allocation and first-touch behavior. The first
allocator-category hit restored sequential-worker jemalloc decay, while the
column-store pool stopped repeatedly faulting roughly 720 MiB of shared Metal
pages per chain step. Buffer lifetime, exact-size reuse, detach policy, and
cross-proof resident ownership should be evaluated under the actual sequential
one-worker-at-a-time harness, while remembering that each process has a 48 GB
machine available to itself.

The newest functional promotion shows that opening evaluation still contains
profitable base/extension field arithmetic: accumulating each quadratic limb
in 160 bits and reducing once per polynomial deleted repeated Goldilocks
reductions without changing opening values. Treat opening dot products and
other exact delayed-reduction seams as live, but do not count the following
marker-only redraw as a second mechanism hit.

Promotion #138 adds a distinct low-level lesson: the dense partition-witness
constructor was paying a serial zero-fill before setting only the values guarded
by its bitmap. Eliminating those stores while making the one formerly
unguarded materialization path return zero for unset slots is value-exact and
attacks both serial witness time and DRAM bandwidth. Its larger inherited stack
also reinforces portable unconditional QoS/window/prefault/readback/FFT changes,
but the note explicitly warns that queue-depth-triggered routing regressed on
the official host. Treat the zero-fill deletion as the clearest new mechanism;
do not infer that every state-conditional scheduler in the stack transferred.

Do not treat marker-only redraws as algorithmic evidence. They show that the
official score has enough variance for a strong unchanged tree to promote, so
any small candidate should be screened against repeated frontier-exact control
draws and should not be kept on a single favorable result.

## Local hypothesis outcomes

These rows are negative or provisional local evidence, not promoted-category
hits. Keep them here so later candidates do not repeat a disproven admission
decision and so M1/M4 transfer assumptions remain reviewable.

| Date | Hypothesis | Reproducible evidence | Outcome |
|---|---|---|---|
| 2026-08-11 | Combine exact-size reuse for recurring Metal column stores below 256 MiB with ownership transfer of the three ephemeral initial FRI Merkle trees, materializing their challenged leaves/paths first and returning their stores to the pool before fold-tree query construction | Starts from promotion #138 (`e268c13`) plus research ledger `581bac9`; focused exact-pool test and Cargo check passed. Same release SHA-256 `8b6b4c6ed8a3982fa818e2c314a587753a593f4ca784c10d72f3db67caab1d5a`; candidate-default cold gate and fixed `B-C-C-B` all passed trusted verification (5/5). B restored best-fit plus retained trees: `37.153416166,37.027430625 s`; C enabled exact small-bin reuse plus early tree retirement: `36.090781292,35.907613125 s`. Candidate mean `35.999197209 s` versus control `37.090423396 s`, `2.94%` lower runtime (`~3.03%` throughput-equivalent), and both mirrored pairings won. | **Keep as `82635d8`.** This is evidence for the combined reuse+lifetime stack, not an isolated estimate for either component. The fixed last-use boundary is portable; no transcript, proof ordering, coefficients, caps, or serialization changed. Submit the verified stack, then keep incremental experiments isolated while official validation runs. |
| 2026-08-11 | Retune the promoted FRI coefficient-slot Rayon block from 2,048 to 1,024 or 4,096 to trade task granularity against Apple-Silicon cache locality | Same-binary selector in both generic and Goldilocks extension-degree-two reducers; `cargo check` and release build passed; release SHA-256 `3f12ff6e3da450d8cd85de2048667f153bccabf477b54dec1df6c0b40727b650`; trusted verification passed 6/6. Symmetric order `2048-4096-1024-1024-4096-2048` produced 2,048 `29.671916625,45.165871291 s` (mean `37.418893958`), 4,096 `44.087350875,41.906574208 s` (mean `42.996962541`, `+14.9071%`), and 1,024 `37.227142208,55.531989084 s` (mean `46.379565646`, `+23.9469%`). Endpoint drift was large, so deltas are directional rather than precise; neither alternative had a positive aggregate signal. | **Reject and revert without confirmation or submission.** Retain 2,048. Smaller blocks add scheduling/full-polynomial traversal overhead; larger blocks reduce useful parallelism or exceed the favorable working set. Revisit only with per-shape counters or direct M4 access, not another blind M1 power-of-two sweep. |
| 2026-08-11 | Drop per-proof wire, Z/partial-product, and quotient coefficient vectors immediately after the reduced FRI opening polynomial is constructed, retaining their Merkle trees/caps and the reusable constants/sigmas oracle | Ownership refactor kept oracle zero intact and cleared only the three ephemeral `PolynomialBatch::polynomials` vectors at their exact last read; `PLONKY2_RETAIN_EPHEMERAL_COEFFICIENTS=1` restored frontier lifetime in the same binary. Cargo check passed; release SHA-256 `4891dc4dea1cf362a427d96cc2ececfb58cecdb01a9a29fa6c6fca16f0cd5923`; trusted verification passed every run (5/5 overall, 3/3 candidate). Fixed `B-C-C-B`: control `29.915660042,30.319586584 s`; candidate `30.211574167,30.778590958 s`. Control mean `30.117623313 s`, candidate `30.495082562 s`: candidate `1.2533%` slower and both pairings lost. | **Reject and revert without confirmation or submission.** The last-use proof is sound, but eagerly dropping hundreds of inner allocations serializes allocator/destructor work directly before the FRI FFT/query phase; that cost exceeds any reduced memory pressure. Keep the current scope-end destruction and pursue buffer reuse/move ownership rather than eager freeing. |
| 2026-08-11 | Stripe each limb of the newly promoted Goldilocks quadratic opening dot product across four independent 160-bit accumulators, then merge before the same two reductions, to expose multiply/carry instruction-level parallelism on Apple cores | The candidate preserved arbitrary zip lengths and the existing `u32::MAX` chunk bound; focused differentials passed all zero, unequal-length, boundary, noncanonical, and 4097-element cases. A same-process release microbenchmark used `2^17` extension/base pairs for 64 rounds in single-striped-striped-single order. Frontier single-accumulator times were `7.483208,7.465583 ms`; four-lane times were `8.898250,8.896291 ms`. The striped kernel was `19.05%` slower by mean and lost both pairings. | **Reject and revert before full proving.** Eight live low/high accumulator chains plus loop unrolling and final merges increase register pressure/code cost more than they hide scalar multiply latency on the M1. Do not test two/eight lanes without disassembly or counters showing spills can be avoided; the promoted single-chain delayed reducer remains best. |
| 2026-08-11 | Start decoding the longest remaining embedded circuit (`light_tx`) on a dedicated large-stack thread before joining the pre-circuit loader/prover, then consume that result in the existing remaining-circuit loader | Existing startup trace put pre load at `132.802 ms`, pre proof from `134.638` to `728.725 ms`, and the remaining four loads from `134.404` to `824.824 ms`, exposing a `96.099 ms` load tail. The ignored loader harness measured cold sequential pre/heavy-tx/heavy-chain/light-tx/light-chain at `94.9/346.5/84.7/367.2/83.6 ms`; all-five overlap was `633.1 ms`, sequential `976.8 ms`, rebuild `1.1 s`, warm `437.1 ms`. Same release binary SHA-256 `a5d5c466acc0f0861e71b0c0d0f944edbed9065a19d724a50105bdb709c530a7`, control `LIGHTER_EARLY_LIGHT_TX_LOAD=0`, fixed `B-C-C-B`: control `27.30,27.63 s`; candidate `27.71,27.97 s`. Candidate mean/median `27.840 s` versus control `27.465 s`, a `1.365%` runtime regression, and both pairings lost. | **Reject and revert without confirmation or submission.** Starting another heavyweight decode at process launch competes with the higher-priority pre load/proof and lengthens the critical path despite reducing the nominal remaining-loader tail. Do not start all four remaining decodes early; revisit startup only with phase-aware scheduling or a decode path proven not to contend for the shared Metal/CPU/memory resources. |
| 2026-08-11 | Suppress recursive destruction of finished circuit/proof input graphs after the final proof is owned, letting the worker's existing post-flush `_exit(2)` reclaim them wholesale | Existing trace: `prove_block_after_pre` ended at `34,215,961.875 us`, while its outer `block_pipeline` ended at `34,259,107.541 us`, exposing `43.146 ms` of return-time teardown immediately before serialization. Same release binary SHA-256 `2cf8789508fcaee07de9a14dfd8fe8aa61af1d81d26bd06b73ddfcd3b38695f1`, `B-C-C-B / C-B-B-C`, with B setting `LIGHTER_DROP_FINISHED_PROVER_STATE=1`: control `27.71,27.07,26.71,26.53 s`; candidate `26.29,27.04,27.09,26.71 s`. Candidate mean `26.7825` vs `27.005 s` (`0.82%` lower), median `26.875` vs `26.890 s` (`0.056%` lower), pairwise 2-2. Cargo check passed; trusted verification 1/1, public smoke `28.500265542 s` provisional only. Candidate checkpoint `d1d839a`; official submission `7244133b-9f3a-4072-b84d-89993282260e` (Yukon commit `23f3ca0`) scored `25.8191460929401 tx/s` against `29.8785698468374`, a `-4.0594237538973 tx/s` delta. | **Officially rejected and reverted.** The score landed in a broad low-service band and therefore does not resolve the predicted `~0.26%` mechanism, but the predeclared rule required an official frontier improvement. Retain the trace and safety evidence; do not keep or stack the deletion after its terminal rejection. |
| 2026-08-11 | Search four consecutive FRI proof-of-work witnesses per Rayon item with the existing AArch64 Poseidon2 x4 permutation, instead of one scalar permutation per item | Diagnostic profile: 106 PoW spans totaled `1.990932 s`, mean `18.782 ms`, max `298.942 ms`. Focused four-state differential passed 1/1. A 100,000-group microbenchmark measured four scalar permutations at `369.468 ms` versus x4 at `313.932 ms` (`1.177x`). Same release binary SHA-256 `b3b87b7811df5b306b9a11c4eabedc2eda3cbb16030b89faf1a9421e9855724a`, order `B-C-C-B / C-B-B-C`, with B setting `PLONKY2_FRI_POW_BATCH_WIDTH=1`: control `27.33,26.28,26.95,27.64 s`; candidate `27.08,27.98,26.96,26.41 s`. Candidate mean `27.1075` vs `27.050 s` (`0.21%` slower), median `27.020` vs `27.140 s` (`0.44%` nominally faster), pairwise signs 2-2. | **Reject and revert without submission.** The faster primitive is overlapped or offset by coarser `find_any` work; mixed aggregates and split signs show no critical-path gain. Do not try width two without new critical-path evidence. |
| 2026-08-11 | Force every FRI query round to be an independently stealable Rayon unit with `with_max_len(1)`, reducing long query-phase waits while several proofs share the global pool | The first eight-run test with the now-rejected pool candidate present measured `2.28%` lower mean runtime and 3-1 pairwise signs. After restoring promotion-137 best-fit, isolated binary SHA-256 `e391abdded403ddbb0246b70848dbac74252fc8f1351265e699cfdc77c4248ff` ran `B-C-C-B / C-B-B-C`: control `26.90,26.98,28.07,25.98 s`; candidate `27.32,26.39,26.31,27.46 s`. Candidate mean `26.870` vs `26.9825 s` (`0.42%` lower), median `26.855` vs `26.940 s` (`0.32%` lower), pairwise signs 2-2 and reverse-confirmation signs 1-1. Earlier trusted verification passed 1/1; source checkpoint `530fafd`. | **Reject and revert without submission.** The isolated effect is noise-sized and crossed the predeclared reverse-confirmation rule; the earlier larger signal depended on an officially rejected stack. Continue with query data-layout or copy removal rather than global iterator grain. |
| 2026-08-11 | Give each FRI query Rayon task a minimum grain of four queries, reducing per-proof query-task fan-out while several transaction/chain proofs already compete in the global pool | A diagnostic-only outer span measured 106 `build_query_proofs` phases totaling `1.627600 s`, mean `15.355 ms`, median `1.035 ms`, and max `232.077 ms`; recurring light transaction/chain contexts contained `186–232 ms` outliers, confirming a material tail. Same release binary SHA-256 `5616edc764fb6dc29ac26e92fd2a120096638182812b71c9872648ffb7a6e1c1`, screen `B-C-C-B`, with B setting `PLONKY2_FRI_QUERY_MIN_LEN=1`: control `30.10,29.41 s`, candidate-default-four `33.21,31.74 s`. Candidate mean/median `32.475 s` vs `29.755 s`, a `9.14%` runtime regression; both pairings lost. | **Reject and revert without confirmation.** Fine-grained within-proof query parallelism is valuable even under multi-proof concurrency; the observed tail is not fixed by coarsening every proof's tasks. Continue only with tree-major sibling materialization or copy/allocation removal, not a global minimum grain. |
| 2026-08-11 | Serialize the final proof into one pre-sized contiguous buffer instead of the existing 2 MiB `BufWriter` | Reused three diagnostic public-fixture traces from the column-pool campaign. `serialize_and_flush_proof` was `468.458`, `374.083`, and `357.250` microseconds, while corresponding process spans were `31.157`, `30.046`, and `33.030` seconds. The entire output path is only about `0.001–0.002%` of process time, including flush and close. | **Reject before implementation.** Even deleting the whole measured span cannot move official throughput measurably; proof encoding/serialization remains an uncrowded category because the current writer path is already below the noise floor. |
| 2026-08-11 | Preserve exact Metal column-store buffer sizes below 256 MiB, but retain best-fit for large/startup/final requests, so a 20 MiB borrower cannot cannibalize a recurring 64 MiB buffer and force later page faults | Diagnostic best-fit: 281 requests, 255 hits, 26 misses, 36 oversized loans, 10,716,446,720 miss-request bytes, nine 544 MiB misses. Pure exact reuse reduced 544 MiB misses to six but forced terminal 256/320 MiB allocations and was rejected. Hybrid: 26 misses, 9,133,096,960 miss-request bytes, six 544 MiB misses, no terminal 256/320 MiB misses. Release SHA-256 `20a1e68f5a22c7057d32395e795a91854bf88d23b3a60ac641c3f0d2cc52e4e9`; same-binary `B-C-C-B / C-B-B-C`, with B restoring best-fit by environment: control `35.65,29.96,35.95,34.77` s, candidate `32.77,30.73,34.83,35.03` s. Candidate median `33.80` vs `35.21` s (`4.00%` lower runtime); mean `33.34` vs `34.0825` s (`2.18%` lower); pairwise signs 2-2. Workspace check and trusted verification 1/1 passed. Candidate commit `b727fb0`; official submission `1d69b9c2-8554-47af-a40c-fce715581b57` (Yukon commit `c9f47d6`) scored `29.4067300861804 tx/s` against `29.8785698468374`, a `-0.4718397606570 tx/s` delta. | **Officially rejected and reverted.** The M1 structural counters and favorable aggregates did not transfer into a frontier gain on the M4 Pro; retain the evidence, but restore best-fit and do not stack this policy into later candidates. |
| 2026-08-11 | Restrict Rayon to the M1 Max's eight performance cores, or nine workers, instead of the default ten logical cores to reduce efficiency-core scheduling interference with Metal | Environment-only same-binary test using release SHA-256 `5589f6b9cd716d9f88a8a6a9a9cd7b44f4df44325d6490e1d8978f01d518d495`, with `LIGHTER_BUILD_BLOCK_CIRCUIT=1` forced in every arm so the rejected embedded-final candidate's extra blob/code was inert and identical. Eight-thread `D-C-C-D`: candidate `28.46, 26.74` s versus default `27.09, 27.22` s; medians `27.60` versus `27.155` s (`+1.64%` runtime). Nine-thread screen plus reverse confirmation: candidate `28.03, 26.26, 27.30, 26.95` s versus default `27.22, 27.61, 26.33, 27.24` s. Nine-thread median `27.125` s versus `27.230` s was a noise-sized `0.39%` nominal gain, while means were `27.135` versus `27.100` s (`0.13%` slower) and pairwise signs split 2-2. | **Reject; no source change.** Removing both efficiency-core workers clearly lengthened the critical path, while removing one produced no repeatable effect. Rayon worker count alone does not bind work to Apple core classes, and the official M4 Pro has a different core topology. Revisit only with QoS/affinity control plus phase counters, not a global thread-count heuristic. |
| 2026-08-11 | Build and compact-embed the fixed final `BlockCircuit` in the untimed Cargo job, then decode it on the existing block lane instead of constructing it during proving | Reconstructed on promotion #137 as a five-file same-binary candidate with original runtime construction selectable by `LIGHTER_BUILD_BLOCK_CIRCUIT=1`. Release SHA-256 `5589f6b9cd716d9f88a8a6a9a9cd7b44f4df44325d6490e1d8978f01d518d495`; generated `block.embed` was 4.46 MiB. The six-circuit identity oracle passed after comparing commitment semantics independently of CPU-vs-Metal backing representation: targets, coefficient polynomials, parameters, leaf counts, full caps, verifier data, sigma data, generator streams, common data, and every remaining prover-only field matched. Trusted public smoke verification passed 1/1 at `29.0962` s. Same-binary out-of-sandbox Metal `B-C-C-B`: runtime-builder control `26.40, 26.85` s; embedded candidate `27.23, 28.24` s. Candidate median `27.735` s versus `26.625` s is `+1.110` s or `+4.17%` runtime (`-4.00%` throughput). | **Reject and revert.** The earlier CPU-fallback experiment (`184.99` s runtime build vs `158.24` s embedded, `14.46%` faster) does not transfer to the current Metal frontier. Compact decoding reconstructs the final constants/sigmas commitment while the transaction pipeline is active, adding work/queue contention that the existing concurrent builder hides more effectively. Reconsider only if the final blob can retain or reconstruct proof data without entering the serialized Metal commitment queue, or if decode is delayed to an actually idle phase. |
| 2026-08-11 | Reduce jemalloc arenas to eight (`MALLOC_CONF=narenas:8`) to match the M1 Max performance-core count and improve large-extent reuse | Environment-only same-binary test using release SHA-256 `ce5db429ceeb2c9632d0cce575c3acaa93fa1a382b6228a89a200ed1c10a4062`, with the Metal threadgroup cap fixed at the frontier's 128 in every run. Fixed screen `D-C-C-D` followed by reverse confirmation `C-D-D-C`; direct `/usr/bin/time -p` samples were candidate `30.01, 28.88, 31.50, 30.76` s and default `30.34, 30.28, 30.21, 29.32` s. Candidate median `30.385` s versus default `30.245` s is `+0.46%` runtime; means were `30.2875` s versus `30.0375` s (`+0.83%`). The initial two-sample screen appeared `2.85%` faster, but both confirmation pairings favored default and erased the signal. | **Reject; no source change.** Eight arenas did not improve repeatably and slightly regressed both aggregate statistics. Do not encode an M1 core-count heuristic for the M4 Pro; arena count should be revisited only with allocator statistics showing fragmentation or lock contention, or as an official-host matrix rather than whole-proof local drift. |
| 2026-08-11 | Tune the blanket 1-D Metal dispatch threadgroup cap from the frontier's 128 threads to 64 or 256 for Apple GPU occupancy | Same release binary SHA-256 `ce5db429ceeb2c9632d0cce575c3acaa93fa1a382b6228a89a200ed1c10a4062`, clean-environment Metal screening in fixed `128-256-64-64-256-128` order followed by alternating 64/128 confirmation. Direct `/usr/bin/time -p` samples: cap 64 `25.90, 27.97, 32.30, 35.43` s; cap 128 `27.16, 31.74, 29.69, 30.75` s; cap 256 screening `27.45, 28.40` s. The 64 median was `30.135` s versus `30.220` s for 128 (a noise-sized `0.28%` nominal throughput gain), but its mean was `30.400` s versus `29.835` s (`1.89%` slower), and it won the first two pairings but lost the two confirmation pairings by `2.61` and `4.68` s. | **Reject and revert.** The sign followed run order and machine state rather than threadgroup width; neither 64 nor 256 showed a repeatable advantage over 128. A blanket cap is too coarse because quotient, NTT-adjacent hashing, leaves, parents, and absorb kernels have different register/occupancy behavior. Revisit only with per-kernel caps and GPU execution-time counters on the M4 Pro, not whole-proof wall time on a thermally drifting M1 Max. |
| 2026-08-11 | Return `BaseSumGate<2>(63)` to CPU only when an existing CPU-only exponentiation or interleave gate already makes its 64 constraint rows and wires free | Same release binary SHA-256 `f198f3c23e3c282e5af82557e4e710ef508722958bf2c44f58ebb8063ff424bd`, clean-environment Metal runs with the frontier restored by `PLONKY2_METAL_BINARY_BASE_SUM=1`. After discarding two samples whose wrapper timing was incomplete, direct `/usr/bin/time -p` samples were candidate `38.26, 37.31, 34.48` s and frontier control `39.28, 37.02, 33.61` s. Candidate median `37.31` s versus control `37.02` s is `+0.29` s or `+0.78%` runtime (`-0.78%` throughput); means were `36.6833` s versus `36.6367` s (`+0.13%` runtime). A routing diagnostic confirmed BaseSum63 stayed on Metal in degree-14 pre/chain circuits and moved to CPU only in the degree-16 transaction/final shapes, without increasing their existing CPU row maxima. | **Reject and revert.** The structurally selective policy removed candidate 1's clear chain-circuit penalty, but it did not produce a repeatable gain: one pair favored the candidate and two favored the frontier, with both aggregate statistics slightly negative. The extra CPU work merely traded against Metal tail work on the local M1 Max. Reconsider only with M4-specific counters or a broader scheduling change that overlaps this CPU work rather than serializing it. |
| 2026-08-10 | Return `BaseSumGate<2>(63)` from the shared Metal Range/U32 quotient command to its packed CPU evaluator | Same release binary SHA-256 `97401a67a84343d3a716bfc32d1bea2d5405055879ac90f3403f13650302c342`, clean-environment Metal runs in predeclared `C-B-B-C-C-B` order: candidate `29.4954, 28.2697, 27.7576` s; frontier control `28.3905, 27.1270, 26.8526` s. Candidate median `28.2697` s versus control `27.1270` s: `+1.1427` s or `+4.21%` runtime (`-4.04%` throughput). The default candidate also produced one final recursive proof accepted by the pinned trusted verifier; its `29.9498` s public-synthetic smoke time is noncompetitive and is not compared with official TPS. | **Reject and revert.** Diagnostics confirmed the gate moved as intended, but chain proofs raised the CPU constraint-row ceiling from 26 to 64; that extra dense evaluation outweighed the shorter Metal tail on the local M1 Max. Do not submit this shape alone. |

## Update checklist

1. Read the live promoted count and record from `https://lighter.fast/`.
2. Run `yukon submissions --all` and inspect each new promoted note.
3. Append one row per promotion; label marker-only or behavior-identical changes
   as `redraw` and add no inherited category hits.
4. Recompute category counts and shares using the new promoted total.
5. Update the snapshot metadata and commit this file with the candidate.
