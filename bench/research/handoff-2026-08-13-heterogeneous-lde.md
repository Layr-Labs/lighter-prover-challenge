# Lighter Prover Challenge handoff: structural CPU/GPU frontier

Last refreshed: **2026-08-13 13:47 America/Chicago**.

This is the standalone context file for continuing the optimization campaign in
a new chat. Read it before inspecting or changing source. It summarizes the
live baseline, benchmark rules, retained and rejected mechanisms, measured CPU
and GPU station maps, M1-to-M4 caveats, and the exact next experiment.

## Executive state

| Item | Current truth |
|---|---|
| Official frontier | Promotion **#145**, source `7477de7`, submission `a955674`, **30.7380852237325 tx/s** |
| Frontier refresh | `origin/master` is still `7477de7`; no post-#145 promotion at 13:47 CDT |
| Validation queue | 13 submissions validating; newest terminal near-frontier draw was `9501c24` at `30.6186674443975 tx/s`, rejected |
| Official machine | Apple M4 Pro Mac mini, 48 GB; five sequential private 500-active-transaction fixtures |
| Local machine | Apple M1 Pro MacBook Pro, 32 GB |
| Research branch | `codex/ra6-neon-fold`, pushed to `ethduke`; one concurrent temporary diagnostic edit appeared during handoff |
| Research head | `ceb7213` (`Record production-shape LDE routing result`) |
| Heavy lock | `/private/tmp/lpc-heavy.lock` absent at handoff |
| Best current research lead | Conditional heterogeneous CPU/GPU LDE split, gated on same-phase overlap measurement |
| Active production candidate | **None**; temporary `cpu_lde/lde_values` trace instrumentation is in progress, not a candidate |

The corrected top-level conclusion is:

> The M1 is neither “mostly waiting” nor fully saturated. It averages about
> 6.03 compute cores, while its GPU is busy about 50.8% of the profiled span.
> Straight GPU NTT offload is slower than CPU LDE, but their degree-16 rates are
> close enough that processing disjoint columns concurrently could be the first
> structural, potentially multi-percent station mechanism—**only if GPU idle is
> present during CPU LDE phases rather than elsewhere**.

Do not implement the split before answering that timing question. The existing
trace already contains the necessary CPU and GPU intervals.

## Benchmark and safety rules

`benchmark.json` is schema version 1: this is one standalone benchmark, not a
track benchmark.

Editable paths:

- `Cargo.toml`
- `Cargo.lock`
- `circuit/`
- `bench/`
- `vendor/`

Protected harness, fixture, workflow, verifier, and benchmark-manifest files
must not be edited. The fixed trusted verifier rejects any change to circuit
semantics/verifier data. Heavy/light widths are pinned to 4/10 and chain ID to
304.

Official ranked scores come only from the private active fixture. The checked-in
`bench/bench_test.json` is a synthetic smoke fixture; its local/public score is
provisional, cacheable, and not comparable with leaderboard throughput.

Standard local entry points:

```sh
./setup.sh
./benchmark.sh
```

Before any build, test, profile, or proof run:

1. Check `git status --short --branch`.
2. Check whether `/private/tmp/lpc-heavy.lock` exists.
3. Acquire that lock atomically before heavy work; release it immediately when
   the process exits.
4. Never overlap builds, tests, profiling, or prover runs with another task.
5. Use the pinned trusted verifier for any candidate proof.

For every meaningful success or failure, update all three ledgers:

- `bench/research/optimization-roadmap.md`
- `bench/research/promoted-options.md`
- `experiment-results.tsv`

Record base/candidate commits, exact executable SHA-256, selectors, raw times,
hardware, correctness checks, trusted verification, decision, and official
submission outcome.

## Branch warning: not a clean submission base

The current research branch differs substantially from `origin/master`. It
contains the knowledge base, the retained Coset experiment, ignored benchmark
harnesses, and diagnostic source additions. Important source deltas include:

- `vendor/plonky2/field/src/goldilocks_extensions.rs`
- `vendor/plonky2/plonky2/src/gates/coset_interpolation.rs`
- `vendor/plonky2/plonky2/src/gates/cpu_survivor_bench.rs`
- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`

Therefore:

- use the current branch for trace analysis and reading notes;
- do **not** submit it as-is;
- if an LDE split survives the measurement gate, create a short clean branch
  from current `origin/master` such as `codex/lde-split`;
- isolate the LDE mechanism first; stack the Coset change only after the primary
  mechanism independently wins.

At 13:47 a concurrent task had an uncommitted temporary edit in
`vendor/plonky2/plonky2/src/fri/oracle.rs`: a diagnostic-profile span named
`cpu_lde/lde_values`. The same diff also exposed line-ending-only noise around
nearby imports and `coset_fft_zero_tail`. Do not overwrite, stage, format, or
commit that file from another chat. First check whether its owner has completed,
recorded the overlap result, and restored/committed the source. Re-check the
heavy lock before doing anything expensive.

Do not run `yukon sync` on this knowledge branch without preserving it: normal
sync is destructive to editable paths.

## What the promoted frontier already contains

Understanding inherited work prevents duplicate experiments.

### Promotion #138

The first promoted appearance of a broad scheduling/Metal/memory/FFT/witness
stack included, among other things:

- bitmap-guarded deletion of dense `PartitionWitness` zero-fill;
- moved/gathered wire IFFT ownership rather than redundant clones;
- coefficient scaling directly into retained shared Metal storage;
- CPU LDE fill overlapping Metal absorbs for qualifying streamed commitments;
- four-wide FFT kernels;
- detached digest readback slots;
- buffer reuse/prewarm and pipeline scheduling mechanisms.

The promoted note attributes roughly 71 MB of avoided zero stores per
transaction proof and 285 MB for the final proof to the witness change, but
the stack was not isolated mechanism by mechanism.

### Promotion #139

- Opened the light-proof ramp from one to two.
- Corrected the exclusive-GPU claim boundary.
- Moved final-block Metal-store prewarm before its page walk.
- Public measurements reported 89.6% steady GPU utilization and 95.9% buffer-set
  occupancy on the M4 lineage.

### Promotion #143

Three functional mechanisms:

- strength-reduced/two-way packed `ExponentiationGate<67>` evaluation;
- deferred alpha-weighted Goldilocks reduction in the Metal Range/U32 quotient
  kernel;
- corrected chain-spine backlog accounting.

### Promotions #144 and #145

- #144 is a marker-only redraw of #143.
- #145 changes the artifact, not Rust/MSL algorithms: it ships a fat
  `poseidon2.metallib` containing an M4-compatible `applegpu` slice.
- Public cold library plus ten-pipeline construction fell from about `240.2 ms`
  to `1.5 ms`.

## Metallib status: fixed, not an open blocker

The current #145 artifact is byte-exact at:

```text
39c066b3c3ffa6e4518cd069156085b8c9ab60f22ec22c2b463af46a4c452574
```

Current local state:

```text
makeLibrary(data:)  ~0.1 ms, 10 functions
test metallib_matches_shader_source                 ok
test metallib_loads_and_exposes_every_kernel        ok
```

The earlier “Metal language version 4.0 unsupported” claim is obsolete. Rust
instrumentation or host scheduling that reuses existing kernels does **not**
require a metallib rebuild. Any edit to `poseidon2.metal` requires regenerating
and validating the artifact before submission.

## Corrected whole-worker CPU map

Source: `bench/research/cpu-decomposition-note.md`.

Method: release worker, symbols retained without changing code generation,
`/usr/bin/sample` at 1 ms for 28 seconds. Raw file currently exists at
`/tmp/cpu.txt` (258 MB; temporary and not checked in).

409,039 leaf-attributed samples:

| Category | Samples | % all thread samples | % compute samples |
|---|---:|---:|---:|
| WAIT / idle | 240,111 | 58.7% | — |
| FFT / NTT | 70,145 | 17.1% | **41.5%** |
| Poseidon2 CPU hashing | 43,090 | 10.5% | **25.5%** |
| Gate constraint evaluation | 24,684 | 6.0% | **14.6%** |
| Other | 14,311 | 3.5% | 8.5% |
| Memory / copies | 9,184 | 2.2% | 5.4% |
| Witness generation | 5,588 | 1.4% | 3.3% |
| Permutation argument | 1,350 | 0.3% | 0.8% |
| FRI / openings | 576 | 0.1% | 0.3% |

Important leaves:

- `fft_classic_simd_single_layer_neon`: 28,323 samples;
- `fft_classic_simd_two_layers_neon_w4`: 24,077;
- `prepare_zero_padded_fft`: 13,891;
- `poseidon2_x4`: 7,123 plus 3,887;
- `platform_memmove`: 7,839.

### The “58.7% waiting” headline was wrong

The initial interpretation counted threads rather than cores. At 28,000
samples per thread, 168,928 compute samples correspond to **6.03 cores busy on
average**:

- about 75% of the eight M1 Pro performance cores;
- about 60% of all ten cores.

This is respectable utilization with some headroom, not a mostly stalled
machine.

Wait attribution:

| Blocking site | Samples | Share of wait |
|---|---:|---:|
| Main/orchestration thread joining workers | 110,181 | 46.2% |
| Condvar / parked Rayon workers | 91,674 | 38.5% |
| Unattributed | 32,973 | 13.8% |
| Rayon join/steal | 3,335 | 1.4% |
| **Metal buffer-set acquisition** | **26** | effectively **0%** |

Consequences:

- main-thread `pthread_join` is normal orchestration waiting, not reclaimable
  compute;
- parked Rayon workers do not prove dependency starvation;
- widening the single Metal buffer set is not supported on M1: only 26 of
  409,039 samples blocked there;
- the public M4 occupancy number does not imply CPU threads are blocked on
  buffer acquisition.

## Production GPU map

Source: `bench/research/merkle-production-gap-trace-note.md`.

The existing `diagnostic_profile` trace captured 644 command buffers. Raw trace
currently exists at `/tmp/merkle_trace.json` (4.3 MB; temporary). It contains
9,331 Chrome-trace events, including CPU IFFT/FFT spans, proof context/shape,
Metal submission events, and GPU start/end timestamps.

| Metal command | Count | Summed GPU seconds | Share summed GPU |
|---|---:|---:|---:|
| `merkle_tree` | 274 | 13.704 | 50.5% |
| `range_u32_quotient` | 106 | 6.941 | 25.6% |
| `poseidon_quotient` | 106 | 3.675 | 13.5% |
| `merkle_absorb` | 63 | 1.422 | 5.2% |
| `permutation_quotient` | 86 | 1.161 | 4.3% |
| `merkle_parents` | 8 | 0.160 | 0.6% |

Merkle totals 15.286 seconds, 56.3% of summed GPU duration.

| Quantity | Value |
|---|---:|
| GPU wall span | 35.962 s |
| Sum of command-buffer durations | 27.154 s |
| Busy-time union | 18.258 s, 50.8% |
| GPU idle | 17.704 s, 49.2% |
| Command-buffer overlap factor | **1.487x** |

Caveat: `diagnostic_profile` inflates CPU time, so 49.2% is not a ranked-M4
utilization estimate. The 1.487x overlapping submission pattern is still real.

### Merkle tail cascade is closed

Seven small parent levels over 274 `merkle_tree` buffers total only about
`0.199 s` of GPU time before overlap discount. Other command buffers fill much
of those stalls. This is below the earlier isolated-harness projection.

- Do not implement the tail cascade.
- Leaf/first-parent fusion is also closed at about 0.2% of a worker and high
  correctness/occupancy risk.
- Merkle remains large in aggregate, but the tested local work-deletion seams
  do not have comparable wall-clock leverage.

## Pipeline and serial-tail measurements

- Full diagnostic trace: 10,196 events in a 28.590-second process.
- The light transaction queue retired at 22.715 seconds with the serial light
  chain only through step 18.
- Steps 19–48 occupied another **3.649 seconds**; individual uncontended late
  chain proofs were roughly 102–128 ms.
- A separate public decomposition described roughly 34 uncontended steps at
  60.1 ms each and about 0.46 seconds of aggregate station-hold excess, but the
  local whole-worker sample shows buffer-set acquisition is not what parks CPU
  threads.
- The post-predecessor chain feed totaled only **234.631 ms** across 30 late
  successors, mean 7.821 ms/step.
- The final-block tail was measured around 1.36 seconds in older traces and
  2.38 seconds in the instrumented run; instrumentation and host state make
  absolute cross-run comparison unsafe.

Scheduling implications:

- the chain tail is real;
- adding global queue locks or static window changes did not advance it;
- any future tail candidate must remove a concrete dependency or move
  meaningful critical work, not infer contention from wait time.

## CPU gate measurements and retained mechanism

Production-shape CPU survivor microbenchmarks:

| Gate | ns/row | Constraints | ns/constraint |
|---|---:|---:|---:|
| `ExponentiationGate<67>` | 270 | 68 | 3.97 |
| `CosetInterpolationGate<bits=4,deg=6>` | 251 | 12 | 20.92 |
| `RandomAccessGate<bits=6>` | 141 | 10 | 14.10 |
| `MulExtensionGate<13 ops>` | 115 | 26 | 4.42 |
| `ArithmeticExtensionGate<10 ops>` | 113 | 20 | 5.65 |
| `ArithmeticGate<20 ops>` | 79 | 20 | 3.95 |

Gate evaluation is only 6.0% of all thread samples, so row-weighted station
ceilings matter more than `ns/row` rankings.

### Retained Coset delayed reduction

Commits: implementation `ae99e3a`; corrected production ceiling `f792b49`.

- Specialized median: 233 ns/row.
- Generic median: 258 ns/row.
- Evaluator speedup: `1.1073x`, 9.7%, with disjoint sample arms.
- Production rows: 8,912,896 (`52 * 2^17 + 2^21`).
- Aggregate CPU removed: 0.2228 seconds.
- Approximate eight-way wall saving: **28 ms, ~0.09%** of a 30-second worker.
- Differential tests and trusted verification passed.

Decision: keep as a correct stackable mechanism, never submit alone, and do not
spend a full protected A/B trying to resolve a 28 ms effect under multi-second
host drift.

### RandomAccess bits=6

Stack-scratch expansion:

- 140 ns/row stack versus 142 heap;
- overlapping arms;
- only ~7 ms / 0.02% wall ceiling;
- rejected and reverted.

Packed/fused fold:

- scalar 130 ns/row;
- packed 113–114 ns/row;
- packed/fused 107 ns/row;
- direct saving 23 ns/row, about 0.628 aggregate CPU seconds or 78 ms / 0.26%
  optimistic wall;
- tests and trusted verification 5/5 passed;
- protected controls `29.598131208, 30.218835167 s` versus candidates
  `28.582630792, 30.876317000 s`;
- nominal candidate mean improved 0.5985%, but pairings split
  `-3.430961%/+2.175735%` and candidate drift was 2.294 seconds—29 times the
  mechanism ceiling.

Decision: rejected/reverted in `dec9697`; recoverable implementation at
`941f7e1` only for direct M4 or low-noise station measurement.

## CPU LDE versus GPU NTT

Source: `bench/research/cpu-lde-versus-gpu-ntt-note.md`.

Ignored focused benchmark added in `47f9fd2`:
`benchmark_cpu_lde_versus_gpu_ntt`.

Median of five alternating arms, 136 columns, rate bits 3, cap height 4:

| Shape | CPU coset-LDE | GPU NTT + Merkle | Estimated Merkle | Estimated GPU NTT | CPU/GPU ratio |
|---|---:|---:|---:|---:|---:|
| degree 14, LDE `2^17` | 34.33 ms | 46.10 ms | 2.94 ms | ~43.16 ms | 0.795x |
| degree 16, LDE `2^19` | 153.94 ms | 180.36 ms | 11.74 ms | ~168.62 ms | 0.913x |

Conclusions:

- straight whole-commitment GPU reassignment is closed at both production
  shapes;
- this confirms the earlier exclusive degree-14 GPU-NTT end-to-end regression
  of 3.14%;
- at degree 16, CPU and GPU per-column rates are nevertheless within roughly
  10%;
- no prior experiment split one commitment's columns across both devices.

Idealized degree-16 split model:

| GPU capacity available during the CPU LDE phase | CPU/GPU columns | Modeled stage time | Versus CPU-only |
|---|---:|---:|---:|
| 100% idle | 71 / 65 | 80.5 ms | 1.91x |
| 49% available | 94 / 42 | 106.4 ms | 1.45x |
| 25% available | 111 / 25 | 125.3 ms | 1.23x |

These are not worker projections. They ignore unified-memory contention,
submission CPU cost, and existing cross-proof CPU/GPU overlap.

## Immediate next action: analyze phase-aligned overlap

Do this before changing source or running another prover.

The existing `/tmp/merkle_trace.json` contains:

- 316 `FFT + blinding` spans;
- 112 `IFFT` spans;
- 106 contextual `compute wire polynomials (IFFT)` spans;
- 106 proof-shape `lde_rows` counters;
- 106 `perform final FFT` spans;
- every Metal command's submit sequence and GPU start/end host timestamps;
- proof contexts including degree bits, wires, parent context, and instance.

### Question

For production CPU LDE/FFT intervals—especially degree-16, 136-column wire
commitments—what fraction of their time has no GPU command executing, and does
their completion gate the dependent commitment on the critical path?

### Required analysis

1. Parse Chrome-trace complete events (`ph = "X"`) for CPU LDE/FFT spans.
2. Reconstruct GPU intervals from `metal_gpu/start_host_ns` and
   `metal_gpu/end_host_ns`, pairing them by queue sequence/context.
3. Put both clocks in the same host-time domain; validate alignment against the
   corresponding `metal_submit_to_completed` events.
4. For each FFT/LDE interval, calculate:
   - duration;
   - GPU busy union inside it;
   - GPU idle duration/fraction inside it;
   - overlapping command names;
   - proof degree/shape/parent phase;
   - time from FFT end to dependent Metal submission/start.
5. Aggregate separately for degree 14, degree 16, degree 18, wire commitment,
   quotient/FRI work, steady transaction pipeline, chain drain, and final block.
6. Distinguish GPU idle during CPU LDE from aggregate GPU idle elsewhere.

### Decision gate

Proceed to implementation only if all are true:

- a material share of degree-16 CPU LDE time has same-phase GPU capacity
  (rough guide: at least ~25%, enough for a modeled ~1.23x stage effect);
- the affected LDE completion is on or near a dependent commitment's critical
  path, rather than fully overlapped with another proof;
- the idle is repeated across many production proofs, not concentrated in
  startup/final tails;
- the expected worker ceiling remains comfortably above local noise after
  accounting for proof counts and CPU/GPU contention.

Close the angle if GPU idle during relevant LDE spans is small, belongs to
unrelated phases, or the LDE is already overlapped by other critical work.

### If the gate passes

Create a clean `codex/lde-split` branch from `origin/master` and design a
same-binary candidate/control experiment:

- keep the first eight CPU-produced columns and first absorb latency exactly as
  promoted;
- assign only later, disjoint column groups to GPU NTT;
- let CPU and GPU write distinct ranges of the final shared LDE store;
- join before any hash pass that requires a not-yet-ready group;
- do not increase global buffer-set count or retained-memory policy;
- start with degree-16 136-column shapes only;
- use an environment selector for exact control;
- compare every LDE value, complete Merkle tree/cap/path, proof bytes where
  applicable, and trusted verifier output;
- measure unified-memory contention and time-to-first-absorb, not only isolated
  NTT kernel duration.

This should reuse existing NTT Metal kernels. A Rust-only split does not require
a metallib rebuild; an MSL edit does.

## Structural fallbacks if heterogeneous LDE closes

### 1. CPU FFT call-site decomposition

FFT is 41.5% of CPU compute, but the current sample aggregates different
callers and criticalities. Attribute the three hot leaves by proof shape and
stage before changing algorithms:

- `fft_classic_simd_single_layer_neon`;
- `fft_classic_simd_two_layers_neon_w4`;
- `prepare_zero_padded_fft`.

Look for structural work deletion or layout fusion, not block-size sweeps. The
frontier already contains four-wide FFT kernels and a zero-tail shortcut, so
inspect inherited implementations before proposing another SIMD width or zero
fill deletion.

### 2. Poseidon2 CPU hashing call-site decomposition

Poseidon2 CPU hashing is 25.5% of compute, but it mixes small-tree routing,
proof-of-work, and other callers. First separate critical versus overlapped
callers. Existing x4 PoW was 1.177x faster in isolation but neutral end-to-end,
showing why leaf cost alone is insufficient.

### 3. Direct M4 confirmation of small retained CPU mechanisms

Only after direct M4 access exists, reconsider the Coset + packed RandomAccess
stack as a low-risk additive candidate. Its projected combined wall ceiling is
only about 0.35%; local full-worker variance cannot resolve it.

## Closed or strongly deprioritized paths

Do not repeat these under a new name without new hardware counters or a
materially different mechanism.

### Scheduling and parallelism

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| Proof window 5/6/7 | Depth 5 +2.95%; depth 7 +3.76%, split | Keep 6; no static neighbor sweep |
| Rayon 8/9 workers | 8 +1.64%; 9 mean +0.13%, split | No global thread-count tuning |
| Witness fanout 16→8 | +1.1367%, split | Keep 16 |
| Chain quotient priority burst | +0.6827%, pairings `-7.66%/+9.10%` | Global queue lock rejected |
| Parallel chain predecessor feed | 234.6 ms ceiling; pairings split | Existing implicit ready queue sufficient |
| Parallel permutation Z chains | nominal mean win but pairings `+2.61%/-5.06%` | Keep serial chains |
| Exclusive completion spin waits | public 30.4206/30.3974 vs 30.7381 | Wakeup-only idea rejected |

### GPU, Merkle, and commitments

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| Merkle parent-tail cascade | 0.199 s GPU before 1.487x overlap discount | Closed before implementation |
| Leaf/first-parent fusion | ~0.2% worker ceiling, high risk | Closed |
| Full paired wire absorb | nominal -6.55% runtime mean, reversed warm pairing | Rejected; batching harmed overlap |
| First-absorb-preserving tail pair | first pairing +1.637% runtime | Rejected |
| Range/U32 16×136 tile | +1.416%, split | Rejected |
| Range/U32 32×120 SIMD tile | +0.087%, split | Rejected |
| Poseidon2 RC fold | isolated 12.857 vs 12.650 ms, +1.638% | Register-pressure regression |
| Blanket threadgroup caps | 64/128 changed sign; 256 no win | No blanket sweep |
| Exclusive d14 GPU NTT | +3.1395% first pairing | Whole-commitment offload closed |
| Streamed descriptor stack | only 2,752 bytes removed; +8.78%, both lost | Not a payload copy |

The promoted streamed boundary already:

- moves non-routed witness columns into IFFT;
- gathers routed values directly into the required buffer/order;
- writes scaled LDE values directly into the final shared Metal store;
- hashes that store without a separate payload upload/transpose;
- overlaps CPU fill and GPU absorb in eight-column groups.

Wire-LDE traffic was measured around 36.79 GiB aggregate, but the only leftover
streamed host allocation found was 22 tiny slice-descriptor vectors.

### Memory, allocation, and lifetime

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| Hybrid exact-bin Metal pool | local -4.00% median; official 29.4067 vs 29.8786 | M1-positive/M4-negative, reverted |
| Exact pool + initial-tree retirement | local -2.94%, both pairs; official 29.3561 vs 29.9399 | Rejected/reverted |
| Early coefficient retirement | +1.2533%, both lost | Destructor work landed on critical path |
| Final buffer prewarm | +6.318%, split | 400 MiB first-touch contention harmful |
| Jemalloc eight arenas | +0.46% median, +0.83% mean | No allocator sweep |
| Buffer-set widening | 26/409,039 blocked samples | Not supported on M1 |

### Witness/generator work

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| Late sparse zero fill | +2.4077%, both lost | Keep bitmap guard |
| Unchecked gather | +0.6202%, both lost | Safe loop already good |
| Demand-zero allocation | isolated +2.0449%, both lost | First touch costs more |
| Four-cell gather unroll | nominal mean win, split under 2.303 s drift | Keep compiler loop |
| Watcher template | +1.76% | Rejected |
| Batched watcher accounting | +3.639% focused | Rejected |
| Dense scatter | official -33.46% challenge delta | Rejected |

### FRI/openings and CPU micro-optimizations

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| FRI query min grain 4 | +9.14%, both lost | Fine stealing matters |
| FRI query max grain 1 | only 0.42% mean, 0.32% median, pairwise 2-2 | Rejected |
| FRI block 1024/4096 | +23.95%/+14.91% means | Keep 2048 |
| Four-lane opening dot | +19.05%, both lost | Register pressure dominates |
| Initial-oracle leaf gathering | +0.218% | Rejected |
| PoW x4 | primitive 1.177x; worker mean +0.21%, split | Overlapped, rejected |
| CPU static alpha reduction | +0.387% | Rejected |
| RandomAccess stack allocation | ~0.02% ceiling | Closed |

### Startup, serialization, and teardown

| Experiment | Terminal evidence | Conclusion |
|---|---|---|
| Final block embedded circuit | CPU-only -14.46%; Metal frontier +4.17% | Rejected on active pipeline |
| Early `light_tx` decode | +1.365%, both lost | Startup contention |
| Pre-sized proof serialization | 0.357–0.468 ms, ~0.001–0.002% worker | Closed |
| Skip finished-state teardown | 43.146 ms ceiling; official rejected | Reverted |

## Dataset perspective

The promoted dataset now has 145 rows. Categories overlap:

| Category | Successful lineage hits | Share of 145 |
|---|---:|---:|
| Gates/constraints/quotient | 42 | 29.0% |
| Metal/GPU/shaders | 42 | 29.0% |
| Memory/buffers/copies | 27 | 18.6% |
| Scheduling/parallelism | 22 | 15.2% |
| Permutation/sigma | 16 | 11.0% |
| Witness/generators | 15 | 10.3% |
| FRI/openings/Merkle queries | 13 | 9.0% |
| Startup/embedded/codecs | 13 | 9.0% |
| FFT/LDE | 12 | 8.3% |
| Challenger/transcript | 2 | 1.4% |
| Allocators | 1 | 0.7% |
| I/O/mmap/zero-copy | 0 | 0% |
| Proof serialization | 0 | 0% |

Low category count is not sufficient evidence: serialization was measured
negligible, and allocator/I/O-like changes repeatedly moved page or destructor
work onto the critical path. The heterogeneous LDE split is attractive because
it follows measured station rates, not because FFT has fewer promoted hits.

## Statistical and M1→M4 lessons

- Local control endpoints often drift by 2–5 seconds, while many mechanisms
  have 10–250 ms ceilings.
- A favorable mean with either mirrored pairing losing is not evidence.
- The official system has runner/service classes around 25–26 and 29–30+ tx/s;
  a single official draw is not a mechanism measurement.
- M1-positive memory policy results have already failed on M4.
- Use focused same-binary station tests and production-count ceilings before a
  full proof A/B.
- Count rows/calls/bytes first; convert aggregate CPU time to wall time by the
  actual parallel width.
- For architecture-sensitive GPU work, require either repeatable M1 evidence
  with a large ceiling or direct M4 counters.
- Submit only a verified candidate with a credible critical-path mechanism;
  do not redraw marker-only trees from this campaign merely because score noise
  exists.

## Relevant commits

| Commit | Meaning |
|---|---|
| `7477de7` | Official #145 promoted source |
| `ae99e3a` | Coset delayed-reduction implementation |
| `f792b49` | Corrected Coset production row count and ~0.09% wall ceiling |
| `941f7e1` | Recoverable packed RandomAccess implementation |
| `dec9697` | Reverted packed RandomAccess production code |
| `25757e6` | Refuted encoder reuse; corrected Merkle cascade to ~0.4% isolated ceiling |
| `5a4e9f9` | Production GPU dispatch trace; closed Merkle tail |
| `d41d15a` / `95ca3e0` | CPU decomposition and corrected wait interpretation |
| `47f9fd2` | Production-shape CPU LDE versus GPU NTT benchmark/note |
| `ceb7213` | Ledger update for conditional heterogeneous LDE research |

## Primary references

- `bench/research/optimization-roadmap.md` — authoritative ordered status.
- `experiment-results.tsv` — machine-readable experiment log.
- `bench/research/promoted-options.md` — 145-promotion categorized dataset and
  local outcomes.
- `bench/research/cpu-decomposition-note.md` — corrected whole-worker CPU map.
- `bench/research/cpu-lde-versus-gpu-ntt-note.md` — production-shape device-rate
  comparison.
- `bench/research/merkle-production-gap-trace-note.md` — production GPU union
  and overlap.
- `bench/research/merkle-dispatch-split-note.md` — isolated Merkle decomposition
  and metallib correction.
- `bench/research/coset-interpolation-delayed-reduction-note.md` — retained CPU
  mechanism and corrected ceiling.
- `bench/research/cpu-survivor-gate-ranking-note.md` — gate census and ranking.
- `bench/research/full-paired-absorb-fusion-handoff.md` — streamed commitment
  architecture and failed overlap changes.
- `bench/research/exclusive-d14-wire-ntt-note.md` — rejected straight GPU NTT
  route.

## Suggested opening instruction for the next chat

> Read `bench/research/handoff-2026-08-13-heterogeneous-lde.md` completely,
> then verify the branch/frontier, inspect the ownership/status of the temporary
> `cpu_lde/lde_values` instrumentation in `fri/oracle.rs`, and inspect
> `/tmp/merkle_trace.json`. Do not implement or start a competing heavy run.
> First compute phase-aligned GPU idle inside production CPU LDE/FFT spans,
> broken down by proof shape and phase. Decide whether a disjoint-column CPU/GPU
> LDE split has a credible worker-level ceiling. Update the roadmap,
> promoted-options ledger, and TSV with the result.
