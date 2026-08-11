# Four-lane delayed-reduction opening dot product

## Attribution and scope

This negative experiment was designed and evaluated with **GPT 5.6 Sol** at
maximum reasoning effort in Codex. It starts from promoted Lighter Prover
Challenge frontier 137, source `59c0155`, official score
`29.8785698468374 tx/s`. The relevant inherited optimization is promotion 136,
which specializes Goldilocks quadratic opening evaluation by accumulating each
extension limb in 160 bits and reducing once per polynomial.

The experiment changed only the CPU implementation of that exact dot product.
It did not change circuit constraints, polynomial values, evaluation points,
opening order, transcript order, proof encoding, Merkle hashing, FRI parameters,
or verifier code. It was rejected at the kernel gate and never submitted.

## Hypothesis

The promoted evaluator scans coefficient arrays of production degrees around
`2^17–2^19`. For each coefficient it performs two independent 64-by-64-bit
products, adding each 128-bit product and carry into one 160-bit accumulator per
quadratic limb. Delaying modular reduction removed the dominant per-term field
reductions, but left one serial add/carry dependency chain for each limb.

Apple performance cores have wide out-of-order execution resources. The
hypothesis was that splitting each limb over four independent accumulator lanes
would expose enough instruction-level parallelism to hide widening-multiply and
carry latency. After the scan, the four exact integer partial sums would be
merged and passed through the same two `reduce160` calls. The expected whole
prover gain was `0.2–1.0%`, conditional on a clear production-degree kernel win.

The predeclared gate was intentionally cheap and strict: field differentials
must pass, and both pairings plus the aggregate of a same-process release kernel
screen must favor striping. A failure at that level stops the candidate before
an expensive prover build or trusted proof run.

## Correctness construction

For one quadratic limb, the frontier computes the exact integer sum

```text
S = sum_i a_i * c_i
```

in a `(u128 low, u32 high)` accumulator and then reduces `S` modulo the
Goldilocks prime. The candidate partitioned indices by `i mod 4`:

```text
S0 = sum a_4k     * c_4k
S1 = sum a_(4k+1) * c_(4k+1)
S2 = sum a_(4k+2) * c_(4k+2)
S3 = sum a_(4k+3) * c_(4k+3)
S  = S0 + S1 + S2 + S3
```

This is exact integer reassociation, not field reassociation. The merged value
is identical to the frontier accumulator before reduction. The existing proof
that at most `u32::MAX` arbitrary `u64 × u64` products fit below `reduce160`'s
input bound therefore applies unchanged. Each lane is smaller than the total,
and merging cannot overflow the high word because the proved total cannot.

Arbitrary slice semantics were retained. The implementation used the shorter
input length, processed groups of four, and sent a zero-to-three-element tail
through lane zero. Inputs above the existing maximum terms per reduction kept
the historical chunk-and-field-add behavior.

## Implementation shape

The candidate held eight accumulator pairs live: four for limb zero and four
for limb one. Each group of four coefficients issued eight `u160_add_product`
operations against separate state. At the end of a chunk, three exact 160-bit
merges per limb reconstructed the original integer sum, followed by the same
two unsafe-but-bounded `reduce160` calls used by the promoted implementation.

For potential end-to-end confirmation, a hidden trait control and one
environment selection in `OpeningSet::new` allowed the release executable to
choose the promoted single-accumulator implementation with
`PLONKY2_OPENING_DOT_LANES=1`. This control code was never needed because the
kernel gate failed. All added source and the manual harness were reverted.

## Correctness results

The focused release suite passed all three non-timing tests:

- the generic default extension implementation matched scalar multiply/sum;
- the striped Goldilocks result matched the existing implementation across
  empty and unequal slices, full-u64/noncanonical representatives, powers of
  two and neighboring lengths through 4097;
- the exact BigUint proof of the `u32::MAX` `reduce160` bound remained valid.

The timing harness also asserted canonical equality between the striped and
single results on a deterministic production-degree random input before
measuring either arm.

## Performance protocol

The manual release harness allocated `131072` (`2^17`) quadratic extension
values and the same number of base scalars. Each timed block evaluated the full
dot product 64 times, consuming the result through `black_box`. Both variants
ran in the same optimized test executable. The fixed order was
single-striped-striped-single so that each variant occupied both an early and a
late position.

| Arm | First block | Second block | Mean |
|---|---:|---:|---:|
| Promoted single accumulator | `7.483208 ms` | `7.465583 ms` | `7.474396 ms` |
| Four-lane striped | `8.898250 ms` | `8.896291 ms` | `8.897271 ms` |

The striped candidate added `1.422875 ms` to the two-block mean, a `19.04%`
kernel regression. It lost both pairings, and the two measurements per arm were
internally stable to roughly two microseconds. This is far outside benchmark
noise and cannot be rescued by a whole-prover run.

## Interpretation

The dependency-chain intuition was incomplete because widening products were
not the only constrained resource. Four stripes keep eight low/high states
live, along with four coefficients, four scalars, loop indices, slice pointers,
and tail state. That footprint creates substantial register pressure on
AArch64. The compiler must also schedule a much larger unrolled loop and later
perform twelve word-level lane merges. Any spills, extra moves, instruction
cache cost, or reduced load scheduling flexibility can easily outweigh hidden
multiply latency.

The promoted loop already has two independent limb chains. Polynomial-level
Rayon parallelism supplies further outer concurrency. In that context, adding
four-way inner instruction-level parallelism may over-subscribe execution and
register resources without increasing useful throughput. The stable `19%`
loss indicates a structural code-generation cost, not thermal drift.

This local result is also likely to transfer directionally to the M4 Pro. The
M4 has a newer, wider core, but it does not make eight 160-bit accumulator
states free. A candidate that loses the isolated arithmetic kernel by `19%` on
the M1 cannot plausibly produce a small positive whole-prover gain solely from
M4 scheduling differences. Official queue capacity should not be spent on it.

## Decision and follow-up boundary

Reject and revert before building or benchmarking the full prover. The
promoted single-accumulator delayed-reduction implementation remains the best
known form.

Do not reflexively test two or eight lanes. Two lanes might reduce register
pressure but offers less latency hiding; eight worsens every identified cost.
Either is justified only after assembly or hardware counters demonstrate that
the four-lane regression came from an avoidable compiler artifact and that a
smaller state stays spill-free. More promising work should target a separate
pass, allocation, or cache miss around opening construction rather than
reassociating this already compact arithmetic loop.

No candidate source remains, no protected code changed, and no official
submission was created.
