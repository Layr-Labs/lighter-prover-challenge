# Retire ephemeral commitment coefficients before FRI

## Attribution and scope

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort in Codex. It starts from promoted Lighter Prover Challenge
frontier 137, source `59c0155`, official score `29.8785698468374 tx/s`; the
local clean checkpoint immediately before the experiment was `e0978d5`.

The candidate changed only the lifetime of coefficient-form polynomials owned
by one proof. It did not change polynomial values, commitments, Merkle trees,
caps, FRI inputs, transcript order, proof encoding, circuit constraints, or the
verifier. The change was correctness-valid but performance-negative and was
reverted without an official submission.

## Last-use observation

Every PLONK proof builds four FRI oracles:

1. constants/sigmas, stored in reusable circuit prover data;
2. wire polynomials, owned by the current proof;
3. permutation Z/partial-product polynomials, owned by the current proof;
4. quotient chunks, owned by the current proof.

Each `PolynomialBatch` contains coefficient-form polynomials plus a Merkle tree
over its low-degree evaluations. During `prove_openings`, the coefficient
vectors are read while reducing the requested polynomials into the final FRI
composition polynomial. After that loop, the final FFT, folding, proof of work,
and query construction access only the Merkle trees. The three per-proof
coefficient collections nevertheless remain allocated until the whole opening
proof returns and the commitment objects fall out of scope.

The constants/sigmas coefficients are different: later proofs of the same
circuit reuse them, so they cannot be retired. The candidate therefore targeted
only oracle indices one through three.

## Hypothesis

Production transaction and final circuits have degrees around `2^17–2^19` and
many wire columns. Their coefficient arrays account for hundreds of megabytes
per active proof. The local worker has previously reached roughly 6–8 GiB RSS,
and the ranked workload runs several active proof paths. Returning these pages
before the final FRI FFT/fold/query phases might reduce memory pressure, page
faults, and contention enough to improve whole-worker runtime by `0.3–1.5%`.

The contrary risk was explicit: each collection contains many inner vectors.
Clearing them performs serialized destructor and allocator work at the exact
phase boundary. If the allocator would otherwise recycle or cheaply drop them
at scope exit, moving that work earlier could lengthen the critical path without
meaningfully reducing resident pressure.

The predeclared failure rule required trusted proof compatibility and positive
aggregate timing with pairwise support. A `B-C-C-B` screen would stop without
reverse confirmation if the candidate lost both pairings and the mean.

## Ownership design

The original `PolynomialBatch::prove_openings` accepts an immutable slice of
oracle references. That API correctly prevents mutation but cannot express that
some coefficient vectors become dead in the middle of the operation.

The candidate factored the operation into three ownership phases:

- reduce the opening batches while all coefficient vectors are available;
- clear `polynomials` only for the three mutable ephemeral batches;
- finish the FFT, folding and query proof using immutable Merkle-tree views.

A production wrapper accepted oracle zero as a retained immutable reference and
the other oracles as mutable references. The temporary immutable borrow ended
before the clear loop, so the implementation used ordinary safe Rust ownership,
not reference casts, interior mutability, or unsafe aliasing. The existing
`reduce160`, FFT, Merkle, and proof code was unchanged.

At the caller, the three per-proof commitment variables became mutable solely
to permit this last-use transition. After `prove_openings` returned, their caps
were moved into the `Proof` exactly as in the frontier. All four Merkle trees
remained present throughout FRI query construction.

The same release executable contained a control. Setting
`PLONKY2_RETAIN_EPHEMERAL_COEFFICIENTS=1` selected the original immutable API and
historical scope-end lifetime. With the variable unset, the candidate retired
the three coefficient collections.

## Validation

`cargo check --locked -p bench --bin prove` passed. The exact release executable
had SHA-256:

`4891dc4dea1cf362a427d96cc2ececfb58cecdb01a9a29fa6c6fca16f0cd5923`

The pinned trusted verifier accepted every proof produced during the campaign:
five verified proofs out of five expected proofs overall. Three of those five
used candidate retirement, including the initial independent smoke and both
candidate arms in the alternating screen. No cap, transcript, proof shape, or
verification mismatch occurred.

The first candidate smoke completed in `28.762251584 s`. That single value was
used only for correctness and was not treated as comparative evidence.

## Same-binary performance screen

B was frontier lifetime (`PLONKY2_RETAIN_EPHEMERAL_COEFFICIENTS=1`); C was early
retirement. The fixed screen order was `B-C-C-B`:

| Run | Arm | Trusted proving time |
|---:|---|---:|
| 1 | B, retain | `29.915660042 s` |
| 2 | C, retire | `30.211574167 s` |
| 3 | C, retire | `30.778590958 s` |
| 4 | B, retain | `30.319586584 s` |

Control mean was `30.117623313 s`. Candidate mean was `30.495082562 s`, an
increase of `0.377459249 s` or `1.253284%`. The candidate lost both alternating
pairings. Because both the aggregate and pairwise criteria failed, the planned
reverse confirmation was correctly skipped.

All four screen proofs passed the trusted verifier, so the timing result is not
confounded by invalid or incomplete work.

## Interpretation

The last-use analysis was correct, but “dead” did not mean “free to retire on
the critical path.” Clearing the outer collections recursively drops every
inner coefficient allocation. Even though field elements have trivial
destructors, each vector still invokes allocator bookkeeping. Performing that
work in one serial boundary immediately before the FRI FFT and query phases
moves allocator traffic earlier and can evict useful metadata or working sets.

Historical scope-end destruction occurs after the opening proof has completed.
It may overlap differently with surrounding proof scheduling, return memory to
existing allocator caches, or be amortized by later object teardown. The
candidate reduced logical liveness but did not establish that physical pages
were promptly reclaimed or that peak memory was the active bottleneck on this
32 GiB test host. The stable two-pair loss shows that eager deallocation cost
dominated any benefit.

This is especially important for M4 transfer. The official host has 48 GiB, so
it is less likely than the local host to benefit from shaving a few hundred MiB
at the price of serialized allocation work. A local `1.25%` regression is not a
credible M4 promotion candidate.

## Decision and narrower follow-ups

Reject and revert without reverse confirmation or submission. Retain the
last-consumer map as useful evidence, but keep coefficient destruction at the
frontier scope boundary.

A future memory-lifetime candidate must avoid recursive deallocation on the
critical path. Plausible directions are moving the vectors into a reusable
arena/pool, transferring ownership to an asynchronous retirement queue with a
proved noninterference policy, or reusing coefficient storage for the final FRI
polynomial. Simply calling `clear`, assigning an empty vector, or dropping the
batch earlier repeats the disproven mechanism.

No candidate source remains and no official submission was created.
