# Late sparse-zero materialization for `PartitionWitness`

## Attribution and benchmark context

This experiment was developed and evaluated by **GPT 5.6 Sol** for the Lighter
Prover Challenge. It starts from the promotion-138 source frontier (`e268c13`,
official score `29.9399105848455 tx/s`) plus this campaign's checked-in research
history at `17a14d346d2661e7c2a1cab7c0e47d37b15ff433`. The candidate was a single
dirty-file experiment in `vendor/plonky2/plonky2/src/iop/witness.rs`; it was not
committed or submitted and was fully reverted after the protected screen.

Local measurements were made on the available Apple M1 Pro MacBook Pro with
32 GB unified memory. The ranked host is an Apple M4 Pro Mac mini with 48 GB.
The checked-in public fixture is synthetic and its reported tx/s is explicitly
provisional and noncompetitive, so the decision below uses alternating trusted
proving seconds from the verifier parent, not public-smoke tx/s.

## Motivation

Promotion #138 removed the dense `F::ZERO` initialization of every
`PartitionWitness::values` slot. Values are now allocated uninitialized, and a
one-bit-per-slot bitmap says whether each representative was written. The sole
full-witness materialization path tests that bitmap for every output cell and
returns zero when the selected representative remains unset.

The immediately preceding diagnostic campaign measured 106 full-witness
materializations and 617,218,048 output cells in one public proof. Representative
set density was only 41.51-50.93%, so roughly half of the representative slots
remain unset. This suggested a different balance from the promoted design:
materialize zeros only into unset representative slots once at `full_witness`
entry, then gather every output cell without the bitmap lookup and branch.

For a degree-16 proof, the proposed trade was roughly 4.5 million sequential
zero stores against removal of about 8.9 million bitmap loads, shifts, masks,
and selections in the much larger row-major gather. Because the transformation
changes unconditional memory work rather than queue-depth policy, it also
appeared more portable from M1 to M4 than another scheduler threshold.

## Same-binary implementation

The experiment added `PLONKY2_LATE_ZERO_WITNESS` as a same-executable selector.
Every benchmark arm set it explicitly:

- `0` was the promotion-138 control: leave unset representative storage
  uninitialized and bitmap-guard every gathered output cell.
- `1` was the candidate: scan the bitmap once, write `F::ZERO` only to valid
  unset representative slots using `trailing_zeros`, then gather without a
  per-cell bitmap test.

The environment lookup happened once per `full_witness`, outside the cell loop.
The gather was factored into a const-generic helper and runtime-dispatched once
to separately monomorphized guarded and unguarded loops. This avoided measuring
a selector branch in every cell. The existing 16 disjoint row segments,
column-major output shape, `MaybeUninit` ownership argument, representative-map
order, and final `set_len` operation were unchanged.

The zero-fill scan masked the final partial bitmap word, visited only unset
bits, and never wrote beyond `values.len()`. Set representatives retained their
generated values. The candidate therefore intended to be value-equivalent to
the existing bitmap-select path while changing where zero materialization cost
was paid.

## Build and focused correctness evidence

`cargo check --locked -p bench --bin prove` passed. The exact release executable
used for both arms was built with the repository release profile and had
SHA-256:

`4de3e0720759bc60cbd5e5641bfdd6759e3959045935dfe076527e09fd6c6595`

The candidate-enabled
`pending_partition_witness_matches_single_shot_for_recursive_circuit` focused
test passed. The candidate-enabled direct-seeded differential stopped earlier,
before proof construction, at its existing assertion that counts raw
`PartitionWitness.values` differences: 384,727 positions exceeded 131 random
generators. The identical test passed when rerun with the selector disabled.

That result is quarantined as a frontier test-harness defect rather than counted
as either candidate proof evidence or a candidate regression. The selector is
not read until `full_witness`, while the assertion iterates `values` before
`full_witness`; therefore the candidate cannot affect the failing comparison.
After #138, unset `values` storage is intentionally uninitialized, so iterating
all raw slots in that test is not a valid deterministic oracle. A future test
repair should compare `set_bitmap` first and compare values only for set
representatives, or compare materialized witnesses. No test-only repair was
mixed into this performance candidate.

The protected benchmark was the decisive correctness gate. Its pinned verifier
accepted one proof in every arm, four out of four total, under protocol
`lighter-mixed-block-proof-v1` and public fixture SHA-256
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.

## Protected B-C-C-B result

The predeclared order was `B-C-C-B`, with B setting
`PLONKY2_LATE_ZERO_WITNESS=0` and C setting it to `1`. All four runs used the
same executable and the trusted verifier's parent-process timer.

| Run | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `30.014395583` | passed |
| 2 | C | `30.407384250` | passed |
| 3 | C | `31.086269542` | passed |
| 4 | B | `30.033468791` | passed |

Control mean was `30.023932187 s`. Candidate mean was `30.746826896 s`, an
increase of `0.722894709 s` or `2.407728%`. The throughput-equivalent change
was `-2.351120%`. The first mirrored pairing regressed `1.309334%`; the second
regressed `3.505425%`. Thus the candidate lost both pairings as well as the
aggregate.

The result is larger and more internally consistent than the expected sub-1%
noise band. No reverse confirmation or official M4 submission was warranted.

## Interpretation

The promotion-138 design is better on this workload: deleting the constructor
zero-fill and paying compact bitmap tests during gather beats restoring even a
sparse late fill. The likely explanation is that the candidate adds a complete
serial bitmap traversal plus millions of write-allocate stores immediately
before a large parallel read/gather. Those stores consume memory bandwidth and
dirty cache lines while the unguarded gather still must perform the irregular
representative-map read and values load. The saved bitmap operation is compact,
predictable, and often cache-resident; it is not the dominant cost of the cell
gather.

This negative result also narrows the next witness-memory search. Do not restore
zeros globally, late, sparsely, or with a wider store loop unless a new profile
shows the bitmap selection itself on the critical path. Better candidates must
delete output-cell traffic, fuse a true downstream consumer, shrink the
representative map, or change the memory layout so both the mapping and value
load become more sequential. Merely exchanging bitmap tests for zero stores is
the wrong direction on the M1 and is unlikely to improve on the M4's stronger
cores without also reducing unified-memory traffic.

## Decision

**Reject and revert without submission.** The candidate passed four trusted
proof verifications but regressed mean runtime by `2.407728%` and lost both
mirrored pairings. The source patch and selector were removed; the working tree
returned byte-for-byte to the promotion-138 witness implementation. Retain the
focused-test oracle warning and the performance evidence, but do not stack this
late-zero scheme with another optimization.

Published Yukon progress note:
`0741afd8-fd14-41a3-be72-0192941e0c6e`.
