# Merkle GPU dispatch decomposition

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`), merged into
the research branch with the delayed-reduction `CosetInterpolationGate` change
on top. Local host Apple M1 Pro, 32 GB.

Measurement only — no production code changed. Harness:
`hash::poseidon2::metal::tests::benchmark_metal_merkle_dispatch_split`.

## Why

The constraint map attributes `9.73 s` of the `18.56 s` GPU kernel budget to
Merkle work, and the roadmap's standing high-risk idea is to fuse leaf hashing
with the first parent level to delete a digest write-then-read round trip.
Nothing said how that `9.73 s` splits by dispatch, so the size of that prize
was unknown. GPU time matters roughly `8x` more than CPU time here because the
Metal queue is the serialized station (occupancy `95.9%`) while CPU work is
Rayon-parallel.

## Method, and one discarded first attempt

The first attempt submitted each stage in its own command buffer. It reported
every level below 4096 hashes at a flat `~100 us` and a *negative* overhead
versus the fused path, which is the tell: `gpu_duration` reads
`GPUStartTime`/`GPUEndTime`, so a separate command buffer per level charges the
per-command-buffer cost 16 times while production pays it once. Those numbers
are not production-representative and were discarded.

The reported measurement instead sweeps *prefixes* of the tree in a single
command buffer — leaf only, leaf + 1 parent level, leaf + 2, and so on — and
differences consecutive totals. Each marginal therefore includes the level's
kernel and its real encoder barrier, exactly as production pays them.
`2^19 x 8` leaves, cap height 4, median of 5, three independent runs.

## Result

Full tree `12.68`-`13.05 ms`. Leaf is `46.1`-`46.8%` of it.

| Level | Hashes | Marginal | ns/hash |
|---|---:|---:|---:|
| leaf | 524,288 | `5875.7 us` | `11.21` |
| parent | 262,144 | `2800.5 us` | `10.68` |
| parent | 131,072 | `1511.1 us` | `11.53` |
| parent | 65,536 | `799.2 us` | `12.19` |
| parent | 32,768 | `363.0 us` | `11.08` |
| parent | 16,384 | `232.8 us` | `14.21` |
| parent | 8,192 | `164.0 us` | `20.02` |
| parent | 4,096 … 16 | 8,176 total | `~940 us` combined |

## Two conclusions, and they invert the roadmap's priority

**1. Leaf/first-parent fusion is a `~1%` idea, not a `22%` one.**

The first parent level costs `2800 us`, `22%` of the tree, but that is a loose
upper bound and the per-hash column shows why: every large level costs
`10.7`-`12.2 ns/hash`, statistically indistinguishable from the leaf level's
`11.21`. If the digest round trip were a material share, the parent levels
would be measurably cheaper or dearer per hash than the leaf level. They are
not — these kernels are **permutation-bound, not memory-bound**.

Fusion does not delete the first level's 262,144 permutations. It deletes only
the intermediate digest traffic: `524,288 x 32 B` written and read back is
`33.6 MB`, about `168 us` at M1 Pro bandwidth, or **`1.3%` of the tree**. That
is the honest ceiling, and it does not justify the threadgroup/grid-barrier and
digest-residency risk the roadmap flags.

**2. The unglamorous target is `~5x` bigger: collapse the tail levels.**

The nine levels at or below 4096 hashes perform 8,176 permutations, about
`92 us` of real work, but cost `~940 us` — `6.8`-`8.3%` of the tree across the
three runs. Roughly `850 us` is per-encoder and barrier overhead on levels that
do almost nothing; the last level hashes **16** nodes and still costs on the
order of `100 us`.

The fix avoids the barrier hazard entirely. Levels with at most 1024 parents —
1024, 512, 256, 128, 64, 32, 16, seven levels and 2,032 hashes — fit in a
**single threadgroup**, where `threadgroup_barrier` between levels is
well-defined because that one threadgroup *is* the whole grid for the dispatch.
No grid-wide barrier is assumed, which is precisely the assumption the roadmap
warns against.

## Follow-up 1: the floor is a dispatch cost, not an encoder cost (refuted)

`run_merkle` creates a new compute command encoder per level. A compute encoder
defaults to `MTLDispatchTypeSerial`, so consecutive dispatches inside one
encoder are already ordered and memory-coherent. If the `~104 us` floor were an
encoder cost, reusing one encoder for every parent level would delete it from
all 15 levels at once — `~1.5 ms`, `12%` of the tree — with no kernel change and
no barrier risk.

Measured, same binary, median of 7 alternating:

| Arm | Median |
|---|---:|
| One encoder for all parents | `12709.1 us` |
| One encoder per level | `12735.0 us` |

`25.9 us`, `0.20%`, `1.0020x`, samples fully overlapping. **Refuted.** The GPU
drains between dependent dispatches whether or not they share an encoder, so
the floor is a dispatch/barrier cost. Encoder restructuring is worthless here;
only deleting dispatches can help.

## Follow-up 2: the cascade prize is `~0.4%`, not `1.6%`

The first estimate of this note ignored that **a single threadgroup occupies one
GPU core of sixteen**. Collapsing levels into one dispatch serializes their work
onto `1/16` of the machine, which claws back most of the barrier saving.

At the measured whole-GPU rate of `11.2 ns/hash`, a single threadgroup runs at
roughly `179 ns/hash`:

| Collapse | Levels | Hashes | Now | Cascade | Saved | Share of tree |
|---|---:|---:|---:|---:|---:|---:|
| `<= 1024` | 7 | 2032 | `774.6 us` | `468.1 us` | `306.5 us` | `2.42%` |
| `<= 512` | 6 | 1008 | `653.1 us` | `284.6 us` | `368.5 us` | `2.91%` |
| `<= 256` | 5 | 496 | `570.6 us` | `192.9 us` | `377.7 us` | `2.98%` |

Best case is about `378 us` per tree. Every tree walks down to the same cap, so
across the 324 Merkle build spans that is roughly `122 ms`, or **`0.41%` of a
30 s worker** — not the `1.6%` first projected here.

### And that `0.41%` is an upper bound with a live threat to it

This harness times one tree on an otherwise idle GPU. Production reports GPU
busy at `89.6%` with the shared buffer set `95.9%` occupied, which means other
command buffers may already be filling these barrier stalls. To the extent they
are, removing the stalls saves nothing at all. Nothing measured here can
distinguish the two cases.

## Next action

**Do not write the cascade kernel yet.** At `0.41%` optimistic, with a credible
path to `0%` if the stalls are already overlapped, it no longer clears the bar
that justified moving off the CPU vein. The deciding measurement is a
*production* trace of inter-dispatch gaps during Merkle builds — whether the
GPU is genuinely idle across those tail barriers — not another isolated-harness
experiment or a kernel.

Do **not** implement leaf/first-parent fusion either. Its ceiling is `1.3%` of
the tree, roughly `0.2%` of a worker, for the highest-risk kernel change on the
board.

### Standing conclusion

The three measured veins now have ceilings: CPU gate arithmetic `9%` total at
`0.1`-`0.3%` per mechanism, Merkle leaf/parent fusion `0.2%`, Merkle tail
collapse `0.4%`. None is a multi-percent prize. The next real gain likely needs
a structural change — fewer or larger commitments, a different tree arity where
the protocol allows it, or work deleted rather than rescheduled — rather than
another micro-optimization of an existing station.

## Metallib load status on #145 (resolved 2026-08-13)

The #145 fat `poseidon2.metallib` remains byte-exact at SHA-256
`39c066b3c3ffa6e4518cd069156085b8c9ab60f22ec22c2b463af46a4c452574`, and
the local Metal runtime/toolchain issue is fixed. `makeLibrary(data:)` now
loads it in about `0.1 ms` and exposes all ten functions;
`metallib_matches_shader_source` and
`metallib_loads_and_exposes_every_kernel` both pass. The earlier language-4.0
failure claim is obsolete and must not be used to interpret current local
worker timings.

The measurements in this note remain valid because the focused harness built
its pipelines from source and did not depend on the committed artifact. Future
diagnostic-only Rust instrumentation also does not require regenerating the
metallib; only a change to `poseidon2.metal` does.

## Related

- [[cpu-survivor-gate-ranking-note]] — the CPU vein this supersedes; its whole
  ceiling is `9%` of a worker with per-mechanism yields of `0.1`-`0.3%`.
- [[coset-interpolation-delayed-reduction-note]] — the one kept CPU mechanism.
- [[poseidon2-rc-fold-note]] — earlier rejected work in these same kernels.
