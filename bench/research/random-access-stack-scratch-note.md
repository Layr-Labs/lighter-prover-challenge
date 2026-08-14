# RandomAccessGate bits=6 stack-scratch screen

## Context

This experiment followed the CPU-survivor census on the promotion-#144 source
(`5287cfe`, `30.6618694127846 tx/s`) with the delayed-reduction
`CosetInterpolationGate` candidate already committed locally. During the
experiment the official frontier advanced to promotion #145 (`7477de7`,
`30.7380852237325 tx/s`); that promotion changes only the embedded Metal
artifact plus an inert marker and does not alter this CPU evaluator.

The production gate census established an important routing fact:

- every `RandomAccessGate<bits=3>` and `RandomAccessGate<bits=4>` instance is
  evaluated by the shared Metal quotient path;
- `RandomAccessGate<bits=6, copies=1>` is intentionally left on the CPU by
  promotion #134;
- the CPU survivor therefore processes about 27.3 million rows per worker in
  roughly 852,000 32-row evaluator calls.

The row-weighted CPU estimate made this evaluator look attractive: about
`3.84 s` of aggregate CPU at `141 ns/row`, versus `4.16 s` for the already
near-floor `MulExtensionGate`. The first implementation hypothesis was that
the surviving bits=6 shape missed an existing stack fast path on every call.

## Hypothesis

The accumulated evaluator folds 64 selected items across a 32-row batch. Its
temporary storage needs `(64 / 2) * 32 = 1024` field elements, but the existing
stack buffer held only `8 * 32 = 256`, a size matching bits=4. Consequently the
only production shape that reaches the CPU evaluator always took the generic
heap branch:

```rust
let mut items_heap;
let items_uninit = if item_count <= items_stack.len() {
    &mut items_stack[..item_count]
} else {
    items_heap = vec![MaybeUninit::uninit(); item_count];
    &mut items_heap
};
```

The proposed change expanded the stack buffer to `32 * 32` elements (8 KiB for
Goldilocks) and retained the heap fallback for larger generic shapes. A
test-only atomic forced the heap arm so both paths could be measured in one
release test binary; non-test production code compiled the selector to the
constant `false`.

The initial cost estimate assumed that constructing the 8 KiB
`Vec<MaybeUninit<F>>` caused payload stores and projected `0.17–0.51 s` of
aggregate CPU removal. That estimate was deliberately screened with the
focused gate harness before building or benchmarking the complete prover.

## Focused measurement

The CPU-survivor harness alternated stack and forced-heap measurements seven
times in one release test binary. The relevant medians were:

| Arm | Median |
|---|---:|
| Expanded stack scratch | `140 ns/row` |
| Existing heap scratch | `142 ns/row` |
| Nominal speedup | `1.0143x` |

The sample ranges overlapped. The observed difference is about `2 ns/row`, or
roughly `64 ns` per 32-row evaluator call. Across about 852,000 calls this is
only `~0.055 s` of aggregate CPU. Dividing by the effective parallel width
puts the optimistic wall-clock ceiling near `7 ms`, around `0.02%` of a
30-second worker—far below a defensible end-to-end signal on the local M1 Pro
or the official M4 Pro.

## Why the estimate failed

`vec![MaybeUninit::uninit(); item_count]` does not initialize field values. The
compiler can eliminate the apparent repeated fill because `MaybeUninit` has no
initialized payload to write. The real recurring cost is therefore close to a
jemalloc tcache allocation/free pair, not 8 KiB of memory traffic. The measured
roughly 64 ns per call is consistent with that mechanism.

The 8 KiB stack reservation is safe under the prover's configured large worker
stacks, but safety alone is not a reason to retain it: it increases every
evaluator frame and source complexity for an effect below the whole-worker
measurement floor.

## Decision

**Reject before a full-prover benchmark and revert the production and test-only
changes.** No trusted proof run or official submission is warranted because
the focused same-binary ceiling is already below `0.1%` and the arms overlap.

The useful result is methodological:

1. rank CPU evaluators by row-weighted aggregate work, not only nanoseconds per
   row;
2. distinguish logical allocation size from bytes actually initialized;
3. screen allocation hypotheses at the exact call granularity before paying
   for noisy end-to-end runs;
4. continue investigating `RandomAccessGate<bits=6>` arithmetic and memory
   traversal only if a change removes field operations or materialization,
   rather than merely replacing its tcache allocation.

The committed delayed-reduction `CosetInterpolationGate` change remains a
separate, correct micro-optimization, but its measured row count also limits it
to roughly `0.09%` projected wall impact. Neither mechanism should be submitted
alone. The next CPU candidate must expose a materially larger fraction of the
`RandomAccessGate` `3.84 s` aggregate pool, or move to the next row-weighted
survivor with an arithmetic deletion that has a credible end-to-end ceiling.
