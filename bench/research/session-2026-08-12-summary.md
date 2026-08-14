# Session summary — 2026-08-12

## Baseline

| | Score | Commit |
|---|---:|---|
| Previous local reference (#141) | `30.4758937588950 tx/s` | `0a470b3` |
| **Current frontier (#145)** | **`30.7380852237325 tx/s`** | `7477de7` / `a955674` |

Local research branch was three promotions stale. `yukon sync` hard-resets the
working branch, so the pre-sync tree was checkpointed first at
`checkpoint/pre-144-sync` (also pushed to `ethduke`).

Promotions since #141:

| # | Submission | Solver | Score | Kind |
|---|---|---|---:|---|
| 142 | `c90e02f` | AlexLaevski | `30.5343213569663` | markers only |
| 143 | `9a57cdd` | exakoss | `30.6448190741910` | **functional** |
| 144 | `81fcf73` | AlexLaevski | `30.6618694127846` | markers only |
| 145 | `a955674` | jungjipdo | `30.7380852237325` | **functional metallib artifact** |

#143 carries three mechanisms, all confirmed present in the synced tree:
packed `ExponentiationGate` strength reduction, deferred Goldilocks reduction
in the Metal Range/U32 quotient kernel, and a `SPINE_BACKLOG` accounting repair
(three spawn sites, two increments).

#145 changes no Rust or MSL mechanism: it packages the same ten Poseidon2
kernels in a fat metallib with an M4-compatible `applegpu` slice. Its public
component measurement reports cold library plus pipeline creation falling from
about `240.2 ms` to `1.5 ms`; this is now the source baseline for every new
candidate.

Local M1 smoke after sync: `46.444411542 s` / `10.765557865833795 tx/s`. This
is a single cold post-sync run on M1 Pro against the M4 Pro ranked host and is
not a clean local baseline — do not compare it to the ledger's `~30 s` runs
without a warm repeat.

## Experiments

| Angle | Result | Decision |
|---|---|---|
| Poseidon2 external round-constant fold (GPU, Merkle kernels) | `12.857 ms` vs `12.650 ms` control on isolated `2^19 x 8` Merkle build; candidate `+1.638%` | **Rejected, reverted** |
| CPU-routed gate evaluator ranking | `CosetInterpolationGate` at `251 ns/row`, `20.92 ns/constraint` | Target identified |
| Merkle GPU dispatch decomposition (#145 base) | leaf `46%` of tree; all large levels `10.7`-`12.2 ns/hash` so kernels are permutation-bound; tail levels `<=4096` hashes cost `~940 us` for `~92 us` of work | **Both thin: tail collapse `~0.4%` of a worker, fusion `~0.2%`; encoder reuse refuted at `0.20%`** |
| Coset interpolation delayed reduction (three reductions) | `233 ns/row` vs `258 ns/row` same-binary, `1.1073x`, arms do not overlap; `8,912,896` rows/worker so `0.2228 s` aggregate CPU = **`~28 ms` wall, `~0.09%`** | **Kept; do not submit alone** |
| Random-access bits=6 stack scratch | `140 ns/row` stack vs `142 ns/row` heap, `1.0143x`, arms overlap; `~0.055 s` aggregate CPU = **`~7 ms` wall, `~0.02%`** | **Rejected, reverted before full proving** |
| Random-access bits=6 packed selector fold | `130 ns/row` scalar vs `107 ns/row` packed/fused in two focused confirmations; protected control mean `29.908483188 s`, candidate mean `29.729473896 s`, nominal `0.60%` runtime win but pairings split `-3.43%/+2.18%` | **Rejected by pairing rule; reverted in `dec9697`** |

Fresh measurement after merging the exact #145 source and fat metallib gives
`RandomAccessGate<bits=6> = 141 ns/row` (`14.10 ns/constraint`) and
`MulExtensionGate = 123 ns/row` (`4.73 ns/constraint`). The next admitted
candidate is therefore the fixed-shape 63-selection RandomAccess fold, packed
across its existing 32-row batch with AArch64 NEON/fused multiply-accumulate.
It cleared that gate at `23 ns/row` with non-overlapping arms, but the full
protected `B-C-C-B` reversed sign. The candidate's two endpoints moved by
`2.294 s` against a `~78 ms` wall-clock ceiling, so the favorable `0.60%` mean
cannot be attributed to the rewrite. Production code was reverted; commit
`941f7e1` retains the implementation for a future M4 or lower-noise direct
measurement. The unchanged #145 metallib was used throughout.

### Why the GPU angle failed

The MDS/linear-layer deferral it was meant to extend is **already promoted** —
`external_linear_layer` is fully lazy and `internal_linear_layer` uses fused
`gl_mul_add`. The remaining unreduced operation was round-constant injection;
folding 84 of 118 `gl_add`s into the lazy accumulator was byte-identical and
provably inside the bound, but lost to register pressure. The same harness
shows reference arithmetic at `16.539 ms` versus `12.868 ms` (`1.285x`), so the
station is arithmetic-sensitive and the negative is real, not measurement
blindness. Detail in [[poseidon2-rc-fold-note]].

### Why the CPU angle is promising

`CosetInterpolationGate` costs `251 ns/row` against the `270 ns/row` of the
already-optimized `ExponentiationGate` that carried a promotion, and `5x` more
per constraint than any other CPU survivor. The cost is entirely in
`partial_interpolate`: 48 quadratic-extension multiplications per row, each
fully reduced. The delayed-reduction primitives to fix it already exist and are
promoted (`u160_add_product`, `u160_times_7`, `reduce160`), as does the
`TypeId`-based Goldilocks specialization pattern. Detail and the three
separable reductions in [[cpu-survivor-gate-ranking-note]].

### Coset interpolation outcome

All three reductions identified in [[cpu-survivor-gate-ranking-note]] were
implemented and measured together: fused delayed reduction across the summed
product pair, one `u160_times_7` per output limb, and the loop-invariant second
limb of `term`. Result `1.1073x` on the evaluator with no overlap between arms,
trusted verifier passed, generic fallback preserved for non-Goldilocks fields.
A row counter then measured the true production
scale: `8,912,896` rows per worker (`52 * 2^17 + 2^21`, matching the proof
census), so the `25 ns/row` saving is `0.2228 s` of aggregate CPU and, divided
by the Rayon width, about `28 ms` of wall clock — `~0.09%`. An earlier estimate
of `0.1`-`0.2 s` was wrong because it treated CPU time as wall time.

The same arithmetic reproduces exakoss's independently reported `~5.1` sampled
CPU-seconds for `ExponentiationGate` (`27.3M` rows x `270 ns` = `7.36 s`),
which validates the model and shows their promoted mechanism was also worth
only a few tenths of a percent. Keep the change and stack it; do not submit it
alone and do not spend a `B-C-C-B` on it. Detail in
[[coset-interpolation-delayed-reduction-note]].

## Standing ceilings after the Merkle decomposition

Every vein now has a measured ceiling, and none is a multi-percent prize:

| Vein | Ceiling |
|---|---:|
| CPU gate arithmetic, all six survivors | `9%` total, `0.1`-`0.3%` per mechanism |
| Merkle tail-level collapse | **Closed.** Production ceiling `0.199 s` GPU (`1.3%` of Merkle), discounted hard by `1.487x` command-buffer overlap |
| Merkle leaf/first-parent fusion | **Closed.** `~0.2%` of a worker |

A production GPU trace confirmed Merkle dominance (`merkle_tree` is `50.5%` of
summed GPU time, Merkle overall `56.3%`) but also that the GPU is idle `49.2%`
of the span locally with `1.487x` command-buffer overlap, so GPU work reduction
has poor wall-clock leverage. Detail in [[merkle-production-gap-trace-note]].

The next real gain likely needs a structural change — fewer or larger
commitments, different tree arity where the protocol allows, or work deleted
rather than rescheduled — not another micro-optimization of an existing
station. Detail in [[merkle-dispatch-split-note]].

## Metal toolchain (resolved during this session)

Two problems were hit and both were fixed; recorded so the symptoms are
recognizable if they recur, not as open items.

1. The Metal Toolchain component was absent, so `poseidon2.metallib` could not
   be regenerated. Installed — Apple metal version 32023.864.
2. The frontier's committed metallib then still failed to load here ("language
   version 4.0 which is not supported on this OS"), so the prover
   source-compiled the shader every process. That inflated the local absolute
   timings recorded above, including the slow smoke runs.

On #144 both metallib tests passed after the fix, and the `-std=metal3.2`
rebuild reproduced the committed artifact byte for byte
(`c9a9edaf4bed715b1d36f0c1b684ae8b6e0cbd7fcb9e0d22122a76f9b60c6929` in both the
worktree and `HEAD`). Shader work is unblocked and regeneration is a
verified-reproducible step. A regenerated artifact must still load on the
ranked M4 host, so check the emitted language version before submitting one.

**This recurs per promotion.** #145 ships a freshly compiled metallib
(`39c066b3`) at language version 4.0 and the load test fails here again. The
`-std=metal3.2` rebuild has to be repeated after any promotion that changes the
artifact. It does not affect the focused harnesses, which compile from source,
nor `metallib_matches_shader_source`, which pins the source rather than the
artifact bytes.

## Methodology note

Count rows before implementing: `ns/row` from a focused harness times the
production row count gives a wall-clock ceiling in minutes, with no idle
machine and no proving run required.

Experiments 22–36 in the ledger are fourteen consecutive rejections, nearly all
with mechanism ceilings of `10 ms`–`250 ms` against M1 control drift of
`2`–`5 s`. Both angles this session were resolved by isolated station-level
measurement in seconds, without spending a protected `B-C-C-B`. Keep gating
sub-1% mechanisms on isolated measurement first.

The #143 note also documents two ranked runner classes (`25–26` vs
`29–30.5 tx/s`) confirmed by simultaneous cross-solver draws, so a single
official score is not mechanism evidence.
