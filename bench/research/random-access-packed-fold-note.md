# RandomAccess bits=6 packed selector fold

Status: rejected by protected pairing rule; production code reverted in
`dec9697`; implementation remains recoverable at `941f7e1`.

## Mechanism

Starting from promoted #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
retained CosetInterpolation rewrite, the candidate specialized the only
RandomAccess shape that survives on CPU: bits=6 over a fixed 32-point batch.
Its 63 selections per row were packed through `F::Packing`, traversed
selector-major so a packed bit stayed live across each level, loaded directly
from the contiguous 64-column wire block, and evaluated as fused
`x.multiply_accumulate(b, y - x)`. Every other shape retained the scalar path.

This was CPU Rust only. No MSL source, embedded metallib, Metal pipeline, or GPU
admission policy changed. The verified #145 fat metallib remained untouched.

## Focused screen

Two rotated same-binary confirmations on M1 Pro:

| Arm | Confirmation 1 | Confirmation 2 |
|---|---:|---:|
| Scalar | `130 ns/row` | `130 ns/row` |
| Packed, original operation order | `113 ns/row` | `114 ns/row` |
| Packed, fused multiply-accumulate | `107 ns/row` | `107 ns/row` |

Every fused sample was below every scalar sample. The `23 ns/row` reduction
across roughly `27.3M` rows projects to `~0.628 s` aggregate CPU, or about
`78 ms / 0.26%` optimistic wall time at eight-way parallelism. An attempted
extension that packed the surrounding constraint terms worsened `112` to
`114 ns/row` and was reverted before the protected worker build.

All four RandomAccess module tests passed. The strongest differential compared
raw noncanonical Goldilocks words to the independent materialized evaluator
for bits 0, 1, 2, 3, 4, and 6 across packing-boundary batch sizes.

## Protected result

The exact release worker SHA-256 was:

```text
7de015f21f64a08b1eff76c9632887741222c7bbbf0d228134d0e346059befb0
```

The candidate-default gate plus all four `B-C-C-B` proofs passed the pinned
trusted verifier (5/5). `PLONKY2_RANDOM_ACCESS_PACKED=0` selected scalar control
in the same binary.

| Arm | First | Second | Mean |
|---|---:|---:|---:|
| Scalar control | `29.598131208 s` | `30.218835167 s` | `29.908483188 s` |
| Packed candidate | `28.582630792 s` | `30.876317000 s` | `29.729473896 s` |

The candidate nominally improved mean runtime `0.598523%` and equivalent
throughput `0.602127%`, but pairings split `-3.430961%` then `+2.175735%`.
Candidate endpoints drifted `2.294 s`, about 29 times the projected `78 ms`
mechanism ceiling. The predeclared success rule required both pairings, so a
reverse confirmation and submission were not justified.

## Decision

Reject locally and revert without submission. The focused evaluator gain is
credible and the implementation remains useful for a direct M4 or lower-noise
station-level measurement, but the protected M1 run cannot attribute an
end-to-end improvement of this size.
