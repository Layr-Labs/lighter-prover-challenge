# Four-state Poseidon2 proof-of-work batching

## Context and attribution

This note records a rejected optimization experiment for the Lighter Prover
Challenge. The implementation and analysis used **GPT 5.6 Sol**, effort
**max**, through Codex. The local screening machine is an Apple M1 Pro with
32 GB unified memory; ranked throughput is measured on an Apple M4 Pro Mac
mini with 48 GB. Local public-fixture times are used only for alternating
same-binary decisions and are not compared directly with ranked TPS.

The base was promoted frontier 137, source `59c0155`, with an official score
of `29.8785698468374 tx/s`. The previously rejected hybrid column-pool policy
and both rejected FRI query-grain changes were absent. The candidate changed
only the generic permutation interface, the Poseidon2 permutation adapter,
and the FRI proof-of-work search. It was reverted after its end-to-end failure
rule fired and was never submitted.

## Why examine proof-of-work

A diagnostic public-worker trace contained 106 `find proof-of-work witness`
spans. Their summed duration was `1,990,931.710 microseconds`, mean duration
`18,782.375 microseconds`, and maximum duration `298,941.875 microseconds`.
The sums include work from concurrently executing proofs and therefore are not
the worker critical path, but the phase was much larger than rejected output
serialization and occupied one of the least crowded successful categories:
challenger/transcript mechanics.

The existing scalar search clones a prepared challenger permutation state for
each candidate, writes the candidate witness, applies one Poseidon2
permutation, and checks the last squeezed element for the configured number of
leading zero bits. Rayon searches the candidate range with `find_any`, so proof
correctness requires only a valid witness; it does not require the smallest
witness or a stable witness across executions.

The current AArch64 Poseidon2 implementation already provides `poseidon2_x4`.
It interleaves four independent states so their instruction dependency chains
can overlap. Merkle leaf and parent code uses related pair/quad primitives
successfully, making the same idea plausible for PoW candidate search without
changing the permutation, transcript, difficulty, or verifier.

## Candidate design

The temporary candidate added a default `permute_batch_4` method to
`PlonkyPermutation`. The generic default performed four scalar permutations.
The Poseidon2 adapter overrode it by passing the four internal states to the
existing `poseidon2_x4` implementation. This kept non-Poseidon2 behavior exact
and reused the already-reviewed AArch64 arithmetic rather than adding another
SIMD kernel.

The FRI PoW search then treated each Rayon item as four consecutive witnesses.
It prepared four copies of the duplex intermediate state, inserted the four
candidate field elements, called `permute_batch_4`, and selected the first
valid lane within the group. The candidate range handled its terminal partial
group explicitly, although the search normally succeeds far earlier.

`PLONKY2_FRI_POW_BATCH_WIDTH=1` restored the historical scalar loop in the
same release binary. The official/default candidate used width four. Both arms
therefore had identical code layout, embedded circuits, compiler flags, Metal
kernels, allocator, and proof pipeline. Only the environment-selected PoW loop
changed.

The transformation can select a different witness from scalar `find_any`, but
that is already permitted: parallel `find_any` is nondeterministic. The prover
recomputed the chosen response through the ordinary challenger immediately
after the search and asserted that it met the same leading-zero threshold.
Verifier semantics were unchanged.

## Focused validation and microbenchmark

`cargo check --locked --offline -p plonky2 --lib` passed after the generic
associated permutation type was fully qualified. A focused release unit test
constructed four random Poseidon2 permutation states, ran four scalar
permutations and one batch-four permutation, and compared every output state.
It passed 1/1.

Before paying for a full worker build, a temporary timing-only test ran 100,000
groups of four states through each path on the M1 Pro:

| Kernel | Time |
|---|---:|
| Four scalar permutations | `369.468208 ms` |
| One interleaved x4 permutation | `313.931917 ms` |
| Direct primitive speedup | `1.177x` |

The timing-only test was removed before the release worker build. The
correctness differential would have remained if the candidate had survived.
The exact screening worker SHA-256 was
`b3b87b7811df5b306b9a11c4eabedc2eda3cbb16030b89faf1a9421e9855724a`.

## Alternating end-to-end result

The predeclared order was `B-C-C-B / C-B-B-C`, where B set batch width one and
C used the default batch width four. Direct `/usr/bin/time -p` samples were:

| Sample | Policy | Real | User | Sys |
|---:|---|---:|---:|---:|
| 1 | B, scalar | 27.33 | 180.13 | 11.78 |
| 2 | C, x4 | 27.08 | 180.33 | 11.11 |
| 3 | C, x4 | 27.98 | 177.11 | 11.58 |
| 4 | B, scalar | 26.28 | 176.36 | 10.23 |
| 5 | C, x4 | 26.96 | 174.34 | 11.75 |
| 6 | B, scalar | 26.95 | 177.76 | 10.65 |
| 7 | B, scalar | 27.64 | 180.43 | 12.72 |
| 8 | C, x4 | 26.41 | 176.26 | 10.70 |

Control mean was `27.050 s`; candidate mean was `27.1075 s`, a `0.2126%`
runtime regression. Control median was `27.140 s`; candidate median was
`27.020 s`, a conflicting `0.4422%` nominal runtime improvement. Adjacent
pairwise signs split 2-2. The first four-sample block had shown a `2.70%` mean
regression; reverse confirmation neutralized the magnitude but did not produce
a repeatable candidate direction.

The failure rule required both mean and median to improve and pairwise signs
to favor the candidate. The result failed both requirements. Trusted benchmark
verification was skipped after performance rejection because the focused
permutation equality test passed and the source was being removed rather than
submitted.

## Interpretation

The direct x4 primitive improvement was real, but it did not translate to the
worker critical path. Proof-of-work searches occur inside multiple proof tasks
sharing Rayon workers with witness, quotient, opening, and Merkle work. Four
times more work per `find_any` item reduces scheduler granularity, and much of
the measured PoW sum overlaps other proof work. Saving roughly 15% inside the
permutation kernel is therefore not equivalent to saving 15% of the summed
PoW spans or of worker time.

This is useful negative evidence for M4 transfer. The candidate was
architecture-aware without hard-coding a core count, but the local whole-worker
effect was below ordinary M1 drift and changed sign between mean and median.
Submitting it would amount to buying a leaderboard redraw with ambiguous code,
not validating a supported optimization.

## Decision and next work

Reject and revert batch-four PoW search. Do not retry width four without a
profile showing that PoW lies on the proof-window critical path rather than
under useful overlap. A width-two variant is not prioritized: its direct ILP
ceiling is lower and the width-four end-to-end result already shows that this
phase cannot convert its stronger primitive gain into repeatable worker time.

Continue with mechanisms that delete visible post-GPU passes, large memory
traffic, or waits rather than optimizing overlapped arithmetic. The public
note for the independently validating fused final quotient pair is especially
relevant evidence for that selection principle, but its exact mechanism should
not be duplicated while its official result is pending.
