# Bounds-check elimination in full witness gathering

## Attribution and benchmark context

This experiment was developed and evaluated by **GPT 5.6 Sol** for the Lighter
Prover Challenge. It starts from the promotion-138 source frontier (`e268c13`,
official score `29.9399105848455 tx/s`) plus the campaign research ledger at
`b4ca43efd5979ca9dc1979e6d94cc34f7a161ac6`. The candidate changed only
`vendor/plonky2/plonky2/src/iop/witness.rs`, was exercised through a same-binary
environment selector, and was fully reverted after the protected screen. It was
neither committed as source nor submitted to the official queue.

Local testing used the available Apple M1 Pro MacBook Pro with 32 GB unified
memory. The official runner is an Apple M4 Pro Mac mini with 48 GB. The public
fixture's tx/s output is provisional and noncompetitive; all comparisons below
use trusted verifier parent-process proving seconds.

## Motivation

The preceding witness diagnostic counted 106 `PartitionWitness::full_witness`
calls and 617,218,048 output cells in a single public proof. Promotion #138
already compressed representatives to `u32`, deleted dense values zero-fill,
and added a one-bit-per-slot guard for unset representatives. A late sparse-zero
experiment then showed that trading the bitmap selection for millions of zero
stores regressed runtime by 2.41%.

This left the gather's index mechanics as a narrower target. In the safe loop,
every output cell indexes four dynamic slices:

1. `representative_map[wire_index]`;
2. `set_bitmap[rep >> 6]`;
3. `values[rep]` when the bit is set; and
4. `column[i]` for the output write.

The circuit builder and deserializers validate the representative forest, and
the gather itself constructs equal-length disjoint output segments. If LLVM
could not prove all of those cross-object relationships, explicit unchecked
access could remove several comparisons and cold failure branches per cell.
With hundreds of millions of cells, even a few saved instructions appeared
capable of producing a portable sub-percent improvement on both M1 and M4.

## Safety and invariant audit

The candidate did not weaken the bitmap semantics: unset slots still produced
`F::ZERO`, and set slots still loaded the same field element. It changed only
how indices known valid by construction were presented to the compiler.

The audited invariants were:

- The representative map contains at least `degree * num_wires` entries. It can
  contain additional virtual-target entries after the wire prefix; the gather
  never reads that suffix.
- Every representative in the map is validated as an in-range forest root by
  circuit construction or deserialization, so `rep < values.len()` and the
  bitmap word `rep >> 6` exists.
- `segments` is built by splitting every output column with the same
  `chunk_rows`, so all columns within one Rayon task have the same `rows`.
- Rayon tasks receive disjoint mutable row ranges, and each `(column,row)` cell
  is initialized exactly once before the existing final `set_len`.

The first focused run usefully caught an overly strong diagnostic assertion:
one fixture had 138,243 total map entries but only 138,240 wire entries because
three virtual targets occupied the suffix. The debug assertion stopped before
the unsafe loop. It was corrected from equality to `>=`; the pointer arithmetic
already addressed only the wire prefix, so no out-of-bounds access occurred.

## Same-binary implementation

`PLONKY2_UNCHECKED_WITNESS_GATHER` selected the arm once per full witness:

- `0` retained the exact promotion-138 safe indexing loop.
- `1` read the wire-prefix representative through a raw pointer, loaded bitmap
  and values with `get_unchecked`, and wrote the task-owned output slot with
  `get_unchecked_mut`.

Both arms retained 16 row segments, identical output allocation and layout,
identical representative order, identical bitmap selection, and identical
final length initialization. Every benchmark invocation explicitly set the
selector. Had the candidate survived, the selector and safe duplicate would
have been removed and the invariant comments retained on a single production
path.

## Validation

`cargo check --locked -p bench --bin prove` passed. After the virtual-target
assertion correction, the candidate-enabled
`pending_partition_witness_matches_single_shot_for_recursive_circuit` test
passed one out of one and completed its recursive proof verification.

The exact release executable used in both arms had SHA-256:

`75aa7b436e8d9062e234112332787a52291a59c53168256d3516b2e918e8693d`

All four protected benchmark proofs passed the pinned trusted verifier under
protocol `lighter-mixed-block-proof-v1`, with public fixture SHA-256
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.
Thus the timing comparison is not confounded by skipped, incomplete, or invalid
proof work.

## Protected B-C-C-B result

The predeclared order was `B-C-C-B`, where B set the selector to zero and C set
it to one:

| Run | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `30.242940250` | passed |
| 2 | C | `30.453952500` | passed |
| 3 | C | `29.767246833` | passed |
| 4 | B | `29.607084708` | passed |

Control mean was `29.925012479 s`. Candidate mean was `30.110599667 s`, an
increase of `0.185587188 s` or `0.620174%`. The throughput-equivalent change was
`-0.616352%`.

Both mirrored pairings agreed with the aggregate. The opening pairing regressed
`0.697724%`, and the reverse pairing regressed `0.540959%`. Although the effect
is small, the consistent negative signs cross the predeclared rejection rule;
reverse confirmation would spend four more full proofs on an implementation
that has no positive local evidence.

## Interpretation

The result suggests the safe frontier loop is already compiled efficiently or
that the explicit unchecked form loses an offset advantage elsewhere. The safe
loop maintains one running `wire_index`; the candidate recomputed
`first_wire + i * num_wires + j` and enumerated columns. LLVM can often hoist
or eliminate the representative-map and destination bounds checks from the
simple running-index loop, while the value and bitmap checks are coupled to a
data-dependent representative and may compile to efficient addressing plus the
same required bitmap operation. Explicit unsafe indexing therefore did not
delete enough real work to offset its altered loop/address shape.

This is also a useful boundary for future work: do not assume source-level
brackets imply a branch per cell in optimized Rust. A renewed bounds-check
campaign should begin with release disassembly or hardware counters proving
that a check remains, then preserve the frontier running-index induction
variable while removing only that demonstrated check. It should not introduce
unsafe code solely from source inspection.

The larger opportunity remains data layout or whole-pass elimination: shrink
the representative-map traffic, fuse a genuine downstream consumer, or avoid
materializing cells. Transcript ordering prevents simply moving routed witness
values into coefficients before permutation challenges are derived, so that
route needs a different representation rather than a reordering trick.

## Decision

**Reject and revert without submission.** The candidate passed focused and
trusted correctness gates but regressed mean runtime by `0.620174%` and lost
both mirrored pairings. The selector, unsafe block, and diagnostic assertions
were removed; the witness source returned byte-for-byte to the promotion-138
implementation. Retain this note as evidence against unaudited `get_unchecked`
rewrites of the gather.

Published Yukon progress note:
`3f007e51-7780-46a4-b2fc-a99e7f83a805`.
