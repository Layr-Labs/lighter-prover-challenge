# Poseidon2 external round-constant fold into the lazy linear layer

Base: promoted frontier `5287cfe` (submission `81fcf73`, `30.6618694127846 tx/s`),
synced from #144. Local host Apple M1 Pro, 32 GB.

## Correction to the incoming plan

The handoff brief proposed widening exakoss's deferred-reduction trick to "the
Poseidon2 permutation/MDS layer inside the Merkle kernels." Inspection of the
synced frontier shows that work is already done and promoted:

- `external_linear_layer` lifts all twelve states into `lazy_t`, runs `mat4`
  three times plus the three-way `sums` fold carry-free, and calls
  `lazy_materialize` exactly once per element;
- `sum_state` is lazy across all twelve operands;
- `internal_linear_layer` uses the fused `gl_mul_add`, which absorbs the addend
  into the 128-bit product before a single reduction.

There is no unreduced MDS accumulation left to defer.

## Mechanism actually tested

The adjacent unharvested cost in the same kernels is round-constant injection.
Each external round still pays a standalone `gl_add` per state element (118 per
permutation counting the internal rounds), and `gl_add` is the generic add with
`add_epsilon_u32`'s conditional two-round correction. Because the linear layer
materializes its output anyway, the constant can ride the accumulator instead:

```
state[i] = lazy_materialize(lazy_add(lazy_add(lazy[i], sums[i & 3]), lazy_of(rc[i])));
```

`external_linear_layer_rc(state, rc)` folds the constants for the layer that
produces the operand, so 84 of the 118 injections cost two 64-bit adds rather
than a full `gl_add`. Round order, constants, and all field values are
unchanged. The 22 internal rounds and external round 4 keep explicit adds
because `internal_linear_layer` materializes through `gl_mul_add`, which has no
accumulator to fold into.

### Bound

Each input half is at most `2^32 - 1`. `mat4` gives every output a coefficient
sum of 7, the three-way `sums` fold reaches 21, and `lazy[i] + sums[i & 3]`
reaches 28. One additional constant half takes that to `29 * (2^32 - 1)`, so
`lazy_materialize`'s extracted `lh` is at most 28 and `e = hh + carry` at most
29 — both inside the `< 2^5` / `<= 32` window its correctness argument
requires, with three units of margin.

## Correctness

`metal_poseidon2_rc_fold_matches_unfolded` builds a complete `2^16 x 8` Merkle
tree with both permutations in one process and compares every node word. All
digests were byte-identical, not merely congruent.

## Result

Both arms compiled from the same shader text in the same process via
`#define POSEIDON2_RC_FOLD 0`, timed on a `2^19 x 8` Merkle build with true GPU
durations, median of 7 alternating samples:

| Arm | Median | Samples |
|---|---:|---|
| Folded (candidate) | `12.857375 ms` | `12.813–12.974 ms` |
| Unfolded (control) | `12.650125 ms` | `12.592–12.952 ms` |

The candidate is **1.638% slower**. Samples are tight and the arms do not
overlap at the median.

## Why this is not a measurement artifact

The same harness measures the reference-arithmetic build of the identical
kernel at `16.539 ms` versus `12.868 ms` for the promoted limb arithmetic, a
`1.285x` spread. The station responds strongly to arithmetic changes, so the
negative result is a real cost of the fold rather than an insensitive
measurement.

The mechanism removes roughly seven ALU ops per element but extends the live
range of twelve `lazy_t` accumulators by one more value each. That is the same
register-pressure cliff exakoss documented when they deliberately excluded the
two register-heavy quintic families from their own deferral.

## Decision

Reject and revert before any end-to-end A/B; the isolated station measurement
is decisive and cheaper than a protected `B-C-C-B`. Do not retry constant
folding, wider lazy accumulation, or full round unrolling in these kernels on
M1 — the shader's existing comment that compact round loops beat full unrolling
points at the same constraint. Revisit only with direct M4 occupancy/register
counters.

## Toolchain state at the time of this experiment (since resolved)

While this experiment ran, `poseidon2.metallib` could not be regenerated on this
machine: the Metal Toolchain component was absent
(`xcodebuild -downloadComponent MetalToolchain`) and `xcode-select` pointed at
CommandLineTools. Both were fixed later the same day — see
[[coset-interpolation-delayed-reduction-note]] for the resolved state and the
verifying test output. Shader work is no longer blocked.

Two claims made here at the time were wrong and should not be carried forward:

- That a stale-but-valid metallib would load silently and mask a shader edit.
  In this window the committed artifact did not load **at all** ("language
  version 4.0 which is not supported on this OS"), so every process took the
  source-compile path and a shader edit measured through the prover would have
  taken effect. The silent-staleness hazard is real in principle — the load path
  probes only for missing kernels — but it was not what was happening here.
- That the measurements above were affected. They were not: this experiment was
  measured entirely through `new_library_with_source` in the focused harness,
  which compiles both arms from the same shader text regardless of the committed
  artifact. The `+1.638%` result stands.

## Related

- [[cpu-survivor-gate-ranking-note]] — the CPU-side follow-up run in the same session.
