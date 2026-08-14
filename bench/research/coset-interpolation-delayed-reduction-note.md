# Delayed-reduction barycentric interpolation for `CosetInterpolationGate`

Base: promoted frontier `5287cfe` (submission `81fcf73`,
`30.6618694127846 tx/s`). Local host Apple M1 Pro, 32 GB.

Selected by measurement, not by inspection: see
[[cpu-survivor-gate-ranking-note]], where `CosetInterpolationGate` ranked at
`251 ns/row` against the already-optimized `ExponentiationGate`'s `270 ns/row`,
and `5x` the per-constraint cost of any other CPU survivor.

## Mechanism

All the gate's remaining cost is in `partial_interpolate`, whose inner loop runs
once per interpolation point (16 per row at the production shape):

```rust
let term = x - x_i.into();
let next_eval = eval * term + val * terms_partial_prod;
let next_terms_partial_prod = terms_partial_prod * term;
```

`ext2_mul` already delays reduction *within* one multiplication — one
`reduce160` per limb and one `u160_times_7`. Three separable reductions remain,
all implemented together in `ext2_partial_interpolate`
(`vendor/plonky2/field/src/goldilocks_extensions.rs`):

1. **Fuse the two products that are immediately summed.** Reduction mod `p` is a
   ring homomorphism, so `eval * term` and `val * prod` can share 160-bit
   accumulators: four `reduce160`s instead of six, and the canonicalizing
   extension add disappears. This is #136's FRI-opening trick applied to gate
   evaluation.
2. **`W`-multiply once per output limb instead of once per multiplication.**
   Grouping the `W`-weighted products into a single accumulator drops one
   `u160_times_7` per iteration.
3. **`term`'s second limb is loop-invariant.** It is `x`'s second limb for every
   point, so it is passed as `x1` rather than rebuilt, and only the first limb
   is subtracted — the generic path lifts `x_i` to `(x_i, 0)` and runs a full
   extension subtraction.

Net: eight reductions per iteration become six, plus one fewer `times_7`, one
fewer canonicalizing add, and one fewer base subtraction.

### Bounds

Every raw limb is below `2^64`, so each product is below `2^128`. The plain part
of `c0` holds two products and the `W`-weighted part holds two before
`u160_times_7`, giving `c0 < 2^129 + 7 * 2^129 = 2^132`; `c1` holds four
products, `c1 < 2^130`; `d0 < 2^131` and `d1 < 2^129`. All are far below
`reduce160`'s `2^160 - 2^128 + 2^96` precondition, and the `u32` high
accumulators stay in single digits.

### Specialization

`partial_interpolate` stays generic. `goldilocks_partial_interpolate` gates on
`TypeId` for exactly `GoldilocksField` / `QuadraticExtension<GoldilocksField>`
and returns `None` otherwise, so the generic fold remains authoritative for
every other field. This mirrors `crate::util::reducing`, which reaches the same
`u160` helpers for FRI openings.

Like the other delayed-reduction helpers, the returned representative need not
match reduce-per-term evaluation bit for bit; it is the same field element,
which is all the quotient pass observes.

## Correctness

- `goldilocks_partial_interpolate_matches_generic_fold`: differential against a
  spelled-out generic fold over slice lengths 1..=16, eight trials each,
  covering random and boundary accumulators (`(0,1)`, `(1,0)`, `(0,0)`) and
  domain points that coincide with the evaluation point so `term == 0`.
- Existing `eval_fns` (base evaluation against the recursive circuit
  constraints), `low_degree`, and `test_accumulate_matches_default_across_batch`
  all pass — 10/10 in the module.
- Full `plonky2 --lib` suite: 209 passed, 1 failed at the time of the run. The
  failure was `metallib_loads_and_exposes_every_kernel`, pre-existing and
  unrelated to this change — only three files were touched, none of them Metal.
  That failure has since been fixed on this machine and the test passes; see the
  toolchain history below.
- Trusted verifier smoke run passed (`passed: true`) with release worker
  SHA-256 `224806ba3d4687212733e76710d8ab979c286e8c2475daf2546ba40bd996bf43`.
  Its `96.951711625 s` is not a usable timing sample: full test suites and
  builds were running concurrently. Do not compare it to anything.

## Result

Same-binary A/B via the `#[cfg(test)]`-only `FORCE_GENERIC_INTERPOLATE` switch,
which is a `const fn` returning `false` outside tests so the production path
keeps no branch and no atomic. Median of 7 alternating samples, 32-point
production batch:

| Arm | Median | Samples |
|---|---:|---|
| Specialized | `233 ns/row` | `228, 229, 232, 233, 235, 235, 239` |
| Generic | `258 ns/row` | `258, 258, 258, 258, 259, 261, 262` |

**`1.1073x` — a 9.7% reduction in the evaluator's cost, with no overlap between
arms** (specialized max `239 ns`, generic min `258 ns`).

The five untouched gates in the same run act as an environment control and
returned to their pre-change values (`ExponentiationGate` `273` vs `270`,
`RandomAccessGate` `141` vs `141`, `MulExtensionGate` `114` vs `115`,
`ArithmeticExtensionGate` `111` vs `113`, `ArithmeticGate` `82` vs `79`). An
earlier cross-build comparison was discarded because those same controls had
drifted `+5%` to `+14%`.

## Measured end-to-end contribution

An earlier revision of this note estimated `0.1`-`0.2 s` end-to-end. **That was
wrong**: it treated aggregate CPU time as wall-clock time. The corrected figure
is roughly an order of magnitude smaller.

A temporary counter in `eval_unfiltered_base_batch_accumulate` measured the rows
this evaluator actually sees in a full 500-transaction run:

```text
[coset-rows] 8912896
```

The count decomposes exactly as `52 * 2^17 + 1 * 2^21` — 52 chain proofs at a
`2^17` quotient domain plus the one `degree_bits=18` final proof — which matches
the boundary trace's proof census, so it is the true production row count and
not a sampling estimate.

| Quantity | Value |
|---|---:|
| Rows evaluated per worker | `8,912,896` |
| Saved per row (measured) | `25 ns` |
| **Aggregate CPU saved** | **`0.2228 s`** |
| Wall clock at 8-way parallelism | `~28 ms` |
| Share of a `30 s` run | **`~0.09%`** |

The quotient pass is Rayon-parallel across the P-cores, so aggregate CPU saving
divides by the parallel width. `0.223 s` of single-threaded work removed is
about `28 ms` of wall clock, not `223 ms`.

### Methodology cross-check

Applying the same arithmetic to the gate exakoss optimized: the two
`degree_bits=16` shapes give `52 * 2^16 * 8 = 27,262,976` rows, and at the
measured `270 ns/row` that is `7.36 s` of aggregate CPU for
`ExponentiationGate`. Their public note independently reports about `5.1`
*sampled* CPU-seconds from Time Profiler, which is the same order and expected
to undercount inlined work. The row-count model therefore reproduces a number
obtained by completely different means, which is what makes the `0.2228 s`
figure above trustworthy.

It also reframes their result: an `11`-`14%` improvement on `7.36 s` is roughly
`0.8`-`1.0 s` of aggregate CPU, about `0.1 s` of wall clock. Their promoted
arithmetic mechanism was itself worth only a few tenths of a percent, bundled
with two others and surfaced by a favorable draw.

## Decision

**Keep, but do not submit alone.** The change is correct, value-identical,
strictly less work, and free to carry: no scheduling or memory behavior changes,
and the generic fallback is preserved for every non-Goldilocks field. But at
`~0.09%` of wall clock it is far below both the local noise floor and the
ranked host's draw variance, so a solo submission would be indistinguishable
from a redraw and would consume a submission slot to learn nothing.

Carry it in a stack with other mechanisms, in the way #143 bundled three.

Do **not** spend a protected `B-C-C-B` on it: the ledger's control endpoints
drift `2`-`5 s`, which is roughly two orders of magnitude above this mechanism's
`28 ms` ceiling. The isolated same-binary result plus the row count are the
strongest evidence obtainable here, and together they cost seconds rather than
four full proving runs.

### General lesson for this campaign

Count rows before implementing. `ns/row` from a focused harness times the
production row count gives the wall-clock ceiling in minutes, without an idle
machine and without a single proving run. Several of the ledger's fourteen
consecutive rejections had ceilings that this arithmetic would have exposed as
unresolvable before any code was written.

## Toolchain history (resolved)

Two Metal problems were hit during this session and both were fixed before it
ended. They are recorded only so the symptoms are recognizable if they recur;
neither is a current limitation.

1. The Metal Toolchain component was initially absent, so `poseidon2.metallib`
   could not be regenerated at all (`xcode-select` pointed at CommandLineTools;
   `xcrun metal` reported "missing Metal Toolchain"). Fixed by installing it —
   Apple metal version 32023.864.
2. With the toolchain installed, the frontier's committed metallib still failed
   to load here: `metallib_loads_and_exposes_every_kernel` reported "This
   library is using language version 4.0 which is not supported on this OS," so
   the prover fell back to compiling the shader from source in every process.
   That inflated local absolute timings, including the slow smoke runs recorded
   above.

Both tests pass on this machine now:

```text
test metallib_matches_shader_source ............ ok
test metallib_loads_and_exposes_every_kernel ... ok
makeLibrary(data:)  0.1 ms  [10 fns]
```

The `-std=metal3.2` rebuild reproduced the committed artifact **byte for byte**
— worktree and `HEAD` both hash to
`c9a9edaf4bed715b1d36f0c1b684ae8b6e0cbd7fcb9e0d22122a76f9b60c6929` — so the
frontier metallib is exactly what the shipped source compiles to, and
regenerating after a shader edit is a verified-reproducible step rather than a
risk. Keep checking the emitted language version, since the artifact must load
on the ranked M4 host as well as here.

This also corrected [[poseidon2-rc-fold-note]], which had assumed a stale
metallib would load and silently mask a shader edit. During the window when the
artifact did not load, the source-compile path was always taken, so a shader
edit measured through the prover would in fact have taken effect.
