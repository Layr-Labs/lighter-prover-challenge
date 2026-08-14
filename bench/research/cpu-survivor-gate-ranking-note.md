# CPU-routed gate evaluator census and ranking

Base: promoted frontier `5287cfe` (submission `81fcf73`,
`30.6618694127846 tx/s`). Local host Apple M1 Pro, 32 GB.

Motivation: #143's promoted note reports roughly `5.1` sampled CPU-seconds in
`ExponentiationGate`'s packed evaluator alone, and their strength reduction of
it was one of three mechanisms in a promoted submission. This asks which other
CPU-routed evaluator is worth the same treatment, using measurement rather than
constraint counts.

## Census

`PLONKY2_GPU_POSEIDON_DIAGNOSTICS=1 ./target/release/prove bench/bench_test.json`
emits `[gate-census]` for each distinct shape. Note the verifier swallows worker
stderr, so `benchmark.sh` does not surface it — run the worker directly.

Five shapes appear. Gates with `off=0` still evaluate on the CPU quotient pass:

| Shape | CPU survivors |
|---|---|
| `degree_bits=14`, 18 gates, `excluded=[]` | all 18, including `Poseidon2Gate` and three `RangeCheckGate`s |
| `degree_bits=16`, 25 gates | Noop, PublicInput, `ArithmeticGate`, `MulExtensionGate`, `ExponentiationGate<67>`, `RandomAccessGate<bits=6>` |
| `degree_bits=16`, 28 gates | same six |
| `degree_bits=14`, 16 gates (chain) | Noop, Constant, PublicInput, `ArithmeticExtensionGate`, `ArithmeticGate`, `MulExtensionGate`, `CosetInterpolationGate` |
| `degree_bits=18`, 27 gates (final) | the chain set plus `U32InterleaveGate` and `UninterleaveToU32Gate` |

The first `degree_bits=14` shape reporting `excluded=[]` is worth a separate
look: an 18-gate shape with 136 gate constraints running its entire quotient on
the CPU, including a degree-7 `Poseidon2Gate` and 396 range-check constraints,
is either a genuine offload gap or a shape that is not on the hot path. Not
investigated here.

## Ranking

`gates::cpu_survivor_bench` times `eval_unfiltered_base_batch_accumulate`
directly at each gate's production parameters, 32-point batch (the production
`BATCH_SIZE`), median of 9 timed groups of 256 iterations after 64 warm-up
iterations:

```
  270.0 ns/row   68 constraints    3.97 ns/constraint  ExponentiationGate<67 bits>
  251.0 ns/row   12 constraints   20.92 ns/constraint  CosetInterpolationGate<bits=4,deg=6>
  141.0 ns/row   10 constraints   14.10 ns/constraint  RandomAccessGate<bits=6,copies=1>
  115.0 ns/row   26 constraints    4.42 ns/constraint  MulExtensionGate<13 ops>
  113.0 ns/row   20 constraints    5.65 ns/constraint  ArithmeticExtensionGate<10 ops>
   79.0 ns/row   20 constraints    3.95 ns/constraint  ArithmeticGate<20 ops>
```

The `ExponentiationGate` figure is already exakoss's optimized version, since
this measurement is taken on the synced frontier.

## Conclusion

`CosetInterpolationGate` is the clear unharvested target: within 8% of the
per-row cost of the gate that carried a promotion, and `5x` more expensive per
constraint than any other survivor. It is CPU-routed in the chain
(`degree_bits=14`, 53 proofs per worker) and final (`degree_bits=18`) shapes.
Weighting by rows, `ExponentiationGate` still sees more total work because the
`degree_bits=16` shapes have four times the rows, but no one has touched
interpolation and the final-block tail is a measured `1.36 s`.

## Identified mechanism, not yet implemented

`eval_unfiltered_base_batch_accumulate` is already batched, hoists the cached
two-adic subgroup and the values buffer out of the point loop, and accumulates
through `batch_multiply_add_inplace`. The remaining cost is entirely inside
`partial_interpolate`, whose inner loop runs 16 times per row:

```rust
let term = x - x_i.into();
let next_eval = eval * term + val * terms_partial_prod;
let next_terms_partial_prod = terms_partial_prod * term;
```

Three quadratic-extension multiplications per point, 48 per row, each doing its
own full Goldilocks reductions. Three separable reductions:

1. **Delayed reduction on the `eval * term + val * terms_partial_prod` pair.**
   This is a two-term ext2 dot product. Today it costs four `reduce128` plus a
   canonicalizing extension add; with 160-bit accumulation it costs two
   `reduce160`. The primitives already exist and are promoted:
   `u160_add_product`, `u160_times_7`, and `reduce160` in
   `vendor/plonky2/field/src/goldilocks_extensions.rs`, with
   `ext2_dot_product_arity16` as the shape to copy. This is #136's trick applied
   to gate evaluation instead of FRI openings.
2. **`term = x - x_i.into()`** lifts a base element to `(x_i, 0)` and does a
   full extension subtraction; only limb 0 changes.
3. **`term`'s second limb is loop-invariant** — it is `x`'s second limb for all
   16 iterations — so the `W`-multiply against it can be hoisted.

`partial_interpolate` is generic over `F: Field + Extendable<D>` while the
primitives are Goldilocks-specific. The established specialization pattern is
`vendor/plonky2/plonky2/src/util/reducing.rs:233`, which gates on
`TypeId::of::<BF>() != TypeId::of::<GoldilocksField>()` and keeps a generic
fallback.

Validate with `gates::cpu_survivor_bench` for the isolated delta and the
existing `coset_interpolation` gate tests plus `gate_testing` for value
identity, before spending a protected `B-C-C-B`.

## Related

- [[poseidon2-rc-fold-note]] — the GPU-side angle run in the same session, rejected.
