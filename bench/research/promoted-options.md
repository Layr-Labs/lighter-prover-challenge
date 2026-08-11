# Promoted optimization options

This is the checked-in optimization ledger for the Lighter Prover Challenge.
Update it from the live leaderboard and public submission notes before every
candidate is committed or submitted. Categories intentionally overlap; a
submission may contribute to more than one mechanism family.

- Snapshot time: 2026-08-11 11:42 America/Chicago
- Official runner: Apple M4 Pro Mac mini, 48 GB, five sequential 500-transaction fixtures
- Launch baseline: 3.18433342666123 tx/s
- Current record: 29.8785698468374 tx/s (`59c0155`)
- Current record speedup: 9.38x
- Promoted submissions: 137
- Solvers: 39

Refresh status: promotion #136 adds delayed-reduction quadratic opening dot
products; #137 is a byte-identical redraw of that executable. A live refresh
at 11:42 found no newer promotion; six unrelated submissions were validating.

## Category totals after the latest append

The prior fully categorized snapshot contained 121 promoted submissions. The
table below applies the 16-entry append ledger in the next section. Shares use
137 as the denominator and do not sum to 100% because categories overlap.

| Optimization category | Successful lineage hits | Share of 137 |
|---|---:|---:|
| Gates, constraints, quotient evaluation | 41 | 29.9% |
| Metal/GPU/shaders | 38 | 27.7% |
| Memory, buffers, copies | 25 | 18.2% |
| Scheduling and parallelism | 19 | 13.9% |
| Permutation and sigma | 16 | 11.7% |
| Witness generation/generators | 14 | 10.2% |
| FFT/LDE | 11 | 8.0% |
| FRI/openings/Merkle queries | 13 | 9.5% |
| Startup, embedded circuits, codecs | 10 | 7.3% |
| Challenger/Fiat-Shamir transcript | 2 | 1.5% |
| Allocators | 1 | 0.7% |
| I/O, mmap, zero-copy | 0 | 0.0% |
| Proof encoding/serialization | 0 | 0.0% |

Supplemental non-mechanism tag: six of the 16 appended promotions were
marker-only redraws of an existing executable. They are retained below because
they are real leaderboard promotions but add no optimization-category hit.

## Append ledger: promotions 122-137

Rows are newest first, matching the live leaderboard. `Delta kind` is based on
the submitted diff and public note, not the score alone. Inherited mechanisms
are not re-counted for a marker-only redraw.

| Overall promotion | Score (tx/s) | Commit | Solver | Delta kind | Public-note title | Categories added |
|---:|---:|---|---|---|---|---|
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
| 2026-08-11 | Preserve exact Metal column-store buffer sizes below 256 MiB, but retain best-fit for large/startup/final requests, so a 20 MiB borrower cannot cannibalize a recurring 64 MiB buffer and force later page faults | Diagnostic best-fit: 281 requests, 255 hits, 26 misses, 36 oversized loans, 10,716,446,720 miss-request bytes, nine 544 MiB misses. Pure exact reuse reduced 544 MiB misses to six but forced terminal 256/320 MiB allocations and was rejected. Hybrid: 26 misses, 9,133,096,960 miss-request bytes, six 544 MiB misses, no terminal 256/320 MiB misses. Release SHA-256 `20a1e68f5a22c7057d32395e795a91854bf88d23b3a60ac641c3f0d2cc52e4e9`; same-binary `B-C-C-B / C-B-B-C`, with B restoring best-fit by environment: control `35.65,29.96,35.95,34.77` s, candidate `32.77,30.73,34.83,35.03` s. Candidate median `33.80` vs `35.21` s (`4.00%` lower runtime); mean `33.34` vs `34.0825` s (`2.18%` lower); pairwise signs 2-2. Workspace check and trusted verification 1/1 passed. | **Keep locally and submit for M4 validation.** Structural allocation evidence and both aggregates are positive, but the split pairwise signs make the local magnitude uncertain. Treat official private-active validation as decisive; pure exact reuse remains rejected. |
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
