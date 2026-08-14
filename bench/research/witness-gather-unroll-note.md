# Four-cell bitmap-guarded full-witness gather unroll

## Attribution and context

This experiment was designed, implemented, and evaluated with **GPT 5.6 Sol**
at maximum reasoning effort through Codex for the Lighter Prover Challenge. It
started from research commit `547db28`, whose proving source is promotion #138
(`e268c13`, official `29.9399105848455 tx/s`) after the prior demand-zero
candidate had been fully reverted. The only candidate source file was
`vendor/plonky2/plonky2/src/iop/witness.rs`. Local execution used an Apple M1
Pro MacBook Pro with 32 GB unified memory; the official runner is an Apple M4
Pro Mac mini with 48 GB.

The public synthetic score is provisional and noncompetitive. All performance
comparisons below use `proving_seconds` measured by the pinned trusted verifier
parent. No official submission was created.

## Motivation

Promotion #138 already removed the dense serial zero fill from
`PartitionWitness::values`. Full-witness materialization still writes more than
617 million output cells in one public proof. For every cell, the frontier loop
loads a representative index, locates one bit in the assignment bitmap,
conditionally loads a field value, and writes the selected value into a
column-major output buffer. Those operations form an irregular dependency
chain, and AArch64 NEON does not provide a general indexed gather instruction.

The existing Rust loop iterates one mutable output column at a time. LLVM may
unroll it, but the dynamic vector of column slices and the conditional bitmap
test can inhibit useful scheduling. Apple performance cores have substantial
out-of-order capacity. Exposing four independent cells explicitly could allow
map, bitmap, and value loads for later cells to begin while an earlier cell is
waiting on cache or address generation. This is a source-level instruction
latency experiment, not a semantic algorithm change.

The angle was selected only after allocator isolation closed the demand-zero
line. It preserves the winning #138 mechanics: values remain uninitialized,
unset representatives are never read, and the compact bitmap remains the
logical assignment authority.

## Candidate design

The candidate added a same-binary selector named
`PLONKY2_WITNESS_GATHER_UNROLL`. Setting it to `0` retained the exact frontier
inner iterator. The default candidate path divided each row's output columns
into exact groups of four with `chunks_exact_mut(4)` and handled the remainder
with the same scalar logic.

For each group, it loaded four representative indices first, then evaluated
four independent bitmap selections, then wrote four output cells. The group
pattern borrowed four disjoint mutable column slices, so output ownership was
unchanged. The row-major representative-map order was unchanged: the running
wire index advanced by four for every group and by one for each remainder
column. The production circuit uses 136 wires, which is divisible by four, but
the remainder path preserved generic correctness for other circuits.

No unchecked indexing, raw pointer arithmetic, allocation change, Rayon
fanout change, output-layout change, or proof-protocol change was introduced.
The baseline and candidate arms used one release executable, holding all other
code generation and runtime state constant.

## Correctness and build gates

`cargo check --locked -p bench --bin prove` passed. The two required vendored
`PendingPartitionWitness` tests passed:

- `pending_partition_witness_finish_and_feed_errors`;
- `pending_partition_witness_matches_single_shot_for_recursive_circuit`.

The trusted setup verified the pinned verifier and built one release worker.
Its exact SHA-256 was:

`d42a8d12dbb6aaa3cfcf755a0ce1b126b7b36998f2625a97344d5ae4fa96f924`

A candidate-default protected correctness gate passed one out of one at
`29.712777167 s`. That sample was deliberately excluded from the controlled
comparison because it preceded the alternating sequence and served only as a
proof/order safety check.

All four controlled proofs passed the pinned trusted verifier under protocol
`lighter-mixed-block-proof-v1`. The fixture SHA-256 was
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.
Including the cold gate, trusted verification passed five out of five.

## Protected performance screen

The predeclared order was `B-C-C-B`:

- B set `PLONKY2_WITNESS_GATHER_UNROLL=0`, selecting the original iterator;
- C left the variable unset, selecting explicit four-cell groups.

| Run | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `32.142156083` | passed |
| 2 | C | `30.457280084` | passed |
| 3 | C | `30.711325958` | passed |
| 4 | B | `29.839181750` | passed |

The control mean was `30.990668917 s`. The candidate mean was
`30.584303021 s`, a nominal `1.311252%` runtime reduction and `1.328675%`
throughput-equivalent increase. The two candidate samples were separated by
only `0.254046 s`, while the two controls were separated by `2.302974 s`.

That endpoint movement reversed the paired result. The first candidate beat
its adjacent control by `5.241951%`, but the second candidate lost to its
adjacent control by `2.922815%`. The predeclared acceptance rule required both
mirrored pairings to favor the candidate. The aggregate therefore cannot be
credited to the unroll.

## Interpretation

The result does not demonstrate that four-way unrolling is intrinsically
slower. It demonstrates that the available end-to-end evidence is not stable
enough to keep it. The whole expected effect is much smaller than the observed
control endpoint swing, and the opposite pairing signs are exactly what the
alternating rule is meant to catch.

There are also reasons the source expansion may be neutral even in a stable
environment. LLVM can already unroll the compact iterator; explicit grouping
can increase register pressure, code size, bounds reasoning, and mutable-slice
bookkeeping. The four bitmap tests may compete for load/store resources rather
than hide latency. Without release disassembly or Apple performance counters,
trying two- or eight-wide variants would be a blind parameter sweep after an
ambiguous result.

M4 Pro transfer does not rescue the evidence. Its wider or newer cores may
schedule the dependency chains differently, but the official environment also
has different GPU/CPU overlap. A source-level CPU change with split M1 pairings
is not strong enough to occupy the validation queue or justify a redraw.

## Decision and next action

**Reject under the pairing rule and revert without submission.** The selector
and explicit group loop were removed, and `witness.rs` was normalized back to
the repository's CRLF convention. A targeted source diff confirmed that the
file returned byte-for-byte to commit `547db28`.

Retain the compiler-generated frontier iterator. Revisit witness-gather
instruction scheduling only if release disassembly proves a surviving serial
dependency or hardware counters show an address-generation/cache-miss stall
large enough to exceed benchmark noise. A future candidate should change the
data layout or eliminate a load, rather than merely spelling the same four
loads more explicitly.

Published Yukon progress note:
`95ec0035-2ae1-4ea0-b028-ebcfee26fa57`.
