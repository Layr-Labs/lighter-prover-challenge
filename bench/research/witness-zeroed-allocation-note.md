# Demand-zero witness allocation with unguarded gathering

## Attribution and benchmark context

This experiment was developed and evaluated by **GPT 5.6 Sol** for the Lighter
Prover Challenge. It starts from promotion #138 (`e268c13`, official score
`29.9399105848455 tx/s`) plus the campaign ledger at
`254d4aece98138fd4205d00c27f4e1d666fa9276`. The candidate changed only
`vendor/plonky2/plonky2/src/iop/witness.rs`, was selected at runtime in one
release executable, and was reverted after its protected screen. It was not
submitted.

Local tests used the available Apple M1 Pro MacBook Pro with 32 GB unified
memory; the official runner is an Apple M4 Pro Mac mini with 48 GB. The public
fixture's tx/s is provisional and noncompetitive, so this note compares only
trusted verifier parent-process proving seconds.

## Motivation

Promotion #138 removed the serial `vec![F::ZERO; len]` initialization of
`PartitionWitness::values`, leaving storage uninitialized and consulting a
one-bit-per-slot bitmap at every observation. That avoids approximately 71 MB
of stores for a transaction proof and 285 MB for the final proof, but the
full-witness gather must now select zero through the bitmap for every output
cell. A diagnostic public proof performed 106 materializations and wrote
617,218,048 output cells.

A previous experiment restored zeros late by explicitly writing only unset
slots. It regressed 2.41% because millions of user-space stores dirtied cache
lines immediately before the parallel gather. That does not rule out true
zeroed allocation. `alloc_zeroed` can ask jemalloc/macOS for already-zero or
demand-zero pages; it need not execute a Rust store loop over the allocation.
If those pages remain cheap until actually touched, unset representatives are
real zeros and full-witness materialization can load values directly without a
bitmap branch.

This angle is materially less crowded than another scheduling threshold. It
combines allocator semantics, virtual-memory first touch, and proof mechanics,
and should transfer conceptually to the M4 Pro's unified-memory system.

## Generic safety design

The implementation remained generic over `F: Field`. Raw zeroed bytes are not
automatically a valid value for every Rust type, so the candidate first
inspected the byte representation of the valid constant `F::ZERO`:

- If every byte of `F::ZERO` was zero and `F` was non-zero-sized, it allocated
  `Layout::array::<F>(len)` through `alloc_zeroed`, handled allocation failure
  with the standard allocator path, and constructed `Vec<F>` from the aligned
  allocation.
- Otherwise it fell back to `vec![F::ZERO; len]`, preserving correctness for a
  field whose zero has a nonzero representation.

Production `GoldilocksField` is `repr(transparent)` over `u64` and defines zero
as `GoldilocksField(0)`, so it takes the zeroed-allocation path. The existing
`set_bitmap` remained present and continued to define logical assignment for
duplicate-write checks and `try_get_target`; only full-witness gathering could
skip the bitmap because every storage slot was physically initialized.

The zeroed allocation and unguarded gather were enabled together by
`PLONKY2_ZEROED_WITNESS_ALLOC=1`. Setting it to zero retained the exact #138
uninitialized allocation and bitmap-guarded gather. Each benchmark arm set the
variable explicitly. No proof value, representative order, segment ownership,
or output layout changed.

## Build and correctness validation

The first Cargo check needed only namespace corrections so the allocator calls
resolve to `std::alloc` with `std` and `alloc::alloc` without it. After that,
`cargo check --locked -p bench --bin prove` passed.

The candidate-enabled
`pending_partition_witness_matches_single_shot_for_recursive_circuit` test
passed one out of one and verified the recursive proofs. The exact release
executable used by both performance arms had SHA-256:

`821bcc2530c12a02bcbc81300f88c393540b957c675bb52ed4c5c36c92b241b0`

All four protected public-fixture proofs passed the pinned trusted verifier
under protocol `lighter-mixed-block-proof-v1`; fixture SHA-256 was
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.

## Protected B-C-C-B screen

The predeclared order was `B-C-C-B`, with B disabling the zeroed allocation and
C enabling it:

| Run | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `30.782584500` | passed |
| 2 | C | `29.266747375` | passed |
| 3 | C | `31.104730041` | passed |
| 4 | B | `30.130072208` | passed |

Control mean was `30.456328354 s`. Candidate mean was `30.185738708 s`, a
decrease of `0.270589646 s` or `0.888451%`. The throughput-equivalent aggregate
was `+0.896416%`.

The paired evidence did not agree. The first candidate beat its control by
`4.924334%`, while the reverse pairing lost by `3.234834%`. That spread is far
larger than the expected mechanism and indicates material machine/thermal or
overlap variance. The predeclared rule required both mirrored pairings to favor
the candidate; therefore the attractive mean was not sufficient for keep or
reverse confirmation.

## Interpretation

This result is not evidence that demand-zero allocation is slow. It is an
ambiguous local signal with a positive aggregate and contradictory pairing
signs. The mechanism may save bitmap work and avoid user-space initialization,
but jemalloc page reuse complicates the comparison: with default dirty-page
decay, an `alloc_zeroed` request may need to clear retained dirty pages, while a
fresh mapping can receive demand-zero pages cheaply. Concurrent proof shapes,
allocator reuse history, and kernel first-touch timing can therefore make the
same arm alternate between a large win and a material loss.

The next valid revisit must isolate those effects before another full A/B:

1. Add diagnostic counters/timers around `alloc_zeroed`, first write, and
   `full_witness` for each circuit shape.
2. Record whether zeroed allocations are fresh mappings or retained-page reuse
   if jemalloc statistics can expose it without perturbing release behavior.
3. Separate zeroed allocation plus the existing bitmap guard from zeroed
   allocation plus unguarded gather, so allocator cost and branch removal get
   independent credit.
4. Require a reverse-order confirmation with both pairings positive before an
   official M4 submission.

Direct M4 access would be especially valuable because its memory controller,
page-fault cost, and core speed can change the balance. Until then, submitting
this split signal would be a redraw gamble rather than an algorithmic result.

## Decision

**Reject under the pairing rule and revert without submission.** The candidate
passed focused and trusted correctness checks and had a nominal `0.888%` mean
runtime improvement, but pairings split 1-1 with extreme opposite signs. The
selector and zeroed allocator path were removed, returning witness source
byte-for-byte to #138. Preserve this as a high-priority profiled revisit, not as
a disproven mechanism and not as a candidate ready for the validation queue.

Published Yukon progress note:
`2465c2ef-572a-42c3-aa49-c8203860a2ad`.

## Profiled revisit and final resolution

The requested isolation was completed on the same M1 Pro host. A temporary
diagnostic build added spans around `PartitionWitness` allocation,
`full_witness`, and the inner gather, then ran one direct public-fixture proof
in each of three modes:

1. mode 0: #138 uninitialized allocation plus bitmap-guarded gather;
2. mode 1: zeroed allocation plus the same bitmap-guarded gather;
3. mode 2: zeroed allocation plus unguarded gather.

The diagnostic release worker SHA-256 was
`eac9b83a541b891e2e14b51d13f0a88f8b7b9f1c81c180e5f06bc36a9ea436a2`.
Each trace contained the same 106 materializations: 53 degree-14 proofs, 52
degree-16 proofs, and one degree-18 proof. The aggregate span totals were:

| Mode | Allocation total | Full-witness total | Gather total | Maximum gather |
|---:|---:|---:|---:|---:|
| 0, #138 | `6.740 ms` | `3.510495 s` | `3.469742 s` | `206.404 ms` |
| 1, zeroed + guarded | `7.433 ms` | `2.902602 s` | `2.857983 s` | `218.554 ms` |
| 2, zeroed + unguarded | `21.557 ms` | `4.293518 s` | `4.222018 s` | `494.060 ms` |

These were intentionally treated as diagnostic work totals, not benchmark
evidence: there was only one trace per mode, concurrent proof spans overlap,
and the direct worker bypasses the trusted parent timing authority. Still, the
shape of the result was useful. Removing the bitmap caused a very large tail,
consistent with unguarded reads first-touching demand-zero pages for unset
representatives. This ruled out the combined mode as the next candidate and
selected mode 1 for a protected test.

The refined production patch removed every diagnostic span and retained the
frontier gather loop unchanged. It also strengthened generic soundness. Rather
than inspecting the object bytes of an arbitrary `F::ZERO`, which can be
questionable for types with padding, it used `TypeId` to select only
`GoldilocksField`. That production field is `repr(transparent)` over `u64`, so
an allocator-provided all-zero region is exactly a vector of valid field
zeros. Every other `F: Field` used normal `vec![F::ZERO; len]`. The environment
variable `PLONKY2_ZEROED_WITNESS_ALLOC=0` restored #138's uninitialized
allocation in the same executable; candidate behavior was the default.

Normal `cargo check --locked -p bench --bin prove` passed. Both required
`PendingPartitionWitness` tests passed through the vendored crate manifest.
The trusted setup verified the pinned verifier and built release worker
SHA-256
`ca4e6719c9da2115ea4a7249e40589568a723ba5b07b66f0caa62aaecaab9e59`.
A candidate-default protected cold gate passed one out of one at
`31.877315875 s`. As elsewhere in this campaign, that cold sample was used only
for correctness and was excluded from the alternating comparison.

The predeclared protected order was `B-C-C-B`, where B disabled zeroed
allocation and C enabled it. All four proofs passed the pinned trusted verifier:

| Run | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `28.929890750` | passed |
| 2 | C | `29.725723083` | passed |
| 3 | C | `30.664303375` | passed |
| 4 | B | `30.249941542` | passed |

Control mean was `29.589916146 s`; candidate mean was `30.195013229 s`.
Zeroed allocation alone therefore increased runtime by `2.044944%`, equivalent
to `-2.003964%` throughput. Both mirrored pairings lost: the first candidate
was `2.750900%` slower than its control and the second was `1.369794%` slower.
This is a decisive rejection under both the aggregate and pairing rules.

The mechanism interpretation is now stronger than after the first split
screen. Bitmap removal is independently harmful because it forces reads and
page first touches for logically unset representatives. Even with that harm
removed, requesting physically zeroed storage costs more than #138's
uninitialized allocation under the trusted workload. Any cheap fresh
demand-zero mappings are outweighed by clearing or first-touch behavior across
the repeated large proof shapes and jemalloc reuse history.

**Final decision: reject and close the demand-zero line.** The refined source
was reverted byte-for-byte to the research base and was not submitted. Keep
#138's uninitialized `PartitionWitness::values` plus bitmap-guarded observation.
Do not retest this allocator mechanism on the M1 Pro; reopening it should
require direct M4 Pro virtual-memory counters showing a materially different
page-clear or first-touch balance.

Published profiled-resolution Yukon note:
`7f0ca8ad-82af-4649-8abf-946b3e33d33b`.
