# Exact-size Metal reuse with FRI initial-tree ownership retirement

## Attribution and benchmark context

This experiment was designed, implemented, and evaluated with **GPT 5.6 Sol**
at maximum reasoning effort through Codex. It targets the standalone
`lighter-prover-challenge` benchmark (`schemaVersion: 1`) and starts from the
promoted frontier commit `e268c13` plus the branch's research-ledger commit
`581bac9`. The live promoted submission observed before implementation was
`4bfd557`, with an official private-active score of
`29.9399105848455 tx/s`.

The current promoted stack already includes a proof window of six, QoS fixes,
backlog-aware chain scheduling, final-block store pre-faulting, two detached
digest-readback slots, four-wide FFT kernels, and deletion of the dense witness
zero-fill. This candidate intentionally preserves those mechanisms and does not
change the existing selective Metal/CPU admission thresholds.

## Motivation

An earlier hybrid exact-bin column-store experiment was the strongest local and
official attempt in this line. It reduced repeated Metal shared-buffer
allocation demand by about 1.58 GB in one diagnostic worker and improved local
mean/median wall time, but its official score of `29.4067300861804 tx/s` did not
beat the then-current `29.8785698468374 tx/s` frontier. The experiment showed
that allocation, page-fault, and residency costs are genuine, while also
showing that a pool policy alone was insufficient.

The follow-up hypothesis was therefore compositional: preserve recurring small
Metal buffer shapes, but also expose a materially earlier last-use boundary so
those exact-size buffers can be reused by another in-flight proof. The target
boundary is inside FRI query construction. Every PLONK proof begins FRI with
four initial Merkle trees: reusable constants/sigmas plus per-proof wires,
Z/partial-products, and quotient trees. Their full LDE leaf stores must remain
alive until the FRI transcript produces query indices, but after all challenged
initial leaves and Merkle paths have been copied into the proof, only the
smaller FRI commit trees are needed. Historically all initial trees remained
borrowed until every FRI-step query proof had also been constructed.

The proposed handoff gives FRI ownership of the three ephemeral initial trees.
It extracts every initial-tree query leaf and path in one parallel pass, drops
the owned trees immediately, and then constructs the FRI-step query proofs.
For Metal-backed leaves, dropping the tree returns the shared buffer to the
column-store pool. This is earlier retirement without eager allocator churn:
the buffer is retained for reuse rather than recursively freed to the system.

## Implementation

The production change is confined to four files:

- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`
- `vendor/plonky2/plonky2/src/fri/oracle.rs`
- `vendor/plonky2/plonky2/src/fri/prover.rs`
- `vendor/plonky2/plonky2/src/plonk/prover.rs`

The Metal column pool now requires exact matches for requests below 256 MiB.
Requests at or above 256 MiB retain smallest-fitting best-fit reuse, because
terminal/final shapes can safely consume larger idle buffers without stealing a
recurring transaction shape. `PLONKY2_METAL_COLUMN_BEST_FIT=1` restores the
historical policy in the same executable.

The PLONK prover clones the three small Merkle caps, moves the per-proof Merkle
trees into the opening-proof call, and continues borrowing all polynomial
coefficients in their original order. Constants/sigmas remain borrowed from
reusable circuit prover data. FRI builds its commit trees and PoW exactly as
before, derives the same query indices, and performs two per-query parallel
passes: initial trees first, then FRI commit trees. The first pass preserves
oracle order and performs the same `leaf_vec` and `prove` calls. Its completion
is the exact last-use boundary for the moved trees, so they are dropped before
the second pass. The second pass preserves reduction-round order and the same
index shifts. Proof structure, cap ordering, leaf values, sibling ordering,
transcript observations, and serialization are unchanged.

`PLONKY2_RETAIN_INITIAL_FRI_TREES=1` selects the historical borrowed-tree path
and original single-pass query construction in the same executable. Together,
the two environment controls make the baseline arm behaviorally equivalent to
the promoted frontier while holding code generation and all unrelated state
constant.

## Correctness and build gates

The following gates passed:

1. `cargo check --locked -p bench --bin prove` completed successfully.
2. The focused Metal pool unit test
   `column_store_pool_preserves_exact_small_shapes` passed.
3. `yukon setup` verified the pinned trusted verifier checksum and signature,
   then completed the native release build.
4. The exact release worker SHA-256 was
   `8b6b4c6ed8a3982fa818e2c314a587753a593f4ca784c10d72f3db67caab1d5a`.
5. A candidate-default protected benchmark run passed the pinned verifier at
   `38.678196375 s` (cold correctness gate only).
6. All four controlled A/B proofs passed the same trusted verifier.

The first attempted protected run exited with status 71 because the benchmark
was launched inside the coding sandbox and its generated macOS Seatbelt profile
could not be applied. Re-running the protected benchmark with permission to
execute outside the coding sandbox activated the benchmark's own Seatbelt and
Metal path and completed normally. This was an execution-environment failure,
not a candidate failure, and the failed launch is excluded from timing.

## Controlled same-binary performance screen

The predeclared order was `B-C-C-B`.

- **B (control):** `PLONKY2_METAL_COLUMN_BEST_FIT=1` and
  `PLONKY2_RETAIN_INITIAL_FRI_TREES=1`.
- **C (candidate):** neither variable set, enabling exact small-bin reuse and
  FRI ownership retirement.

Every run used the same release executable and the checked-in public synthetic
fixture. Public scores are provisional and noncompetitive; trusted proving
seconds are used only for local paired screening.

| Order | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | `37.153416166` | passed |
| 2 | C | `36.090781292` | passed |
| 3 | C | `35.907613125` | passed |
| 4 | B | `37.027430625` | passed |

Control mean was `37.090423396 s`; candidate mean was `35.999197209 s`.
Candidate runtime decreased by `1.091226187 s`, or approximately `2.94%`.
The throughput-equivalent improvement is approximately `3.03%`.

Both order-mirrored pairings favored the candidate:

- B1 versus C1: candidate was approximately `2.86%` faster.
- C2 versus B2: candidate was approximately `3.02%` faster.

The two candidate samples were also tightly grouped (about 183 ms apart), as
were the two controls (about 126 ms apart). This is substantially stronger than
the earlier pool-only screen, whose pairwise signs split 2-2 despite favorable
aggregate statistics.

## Decision and interpretation

**Keep.** The candidate passed trusted verification, improved both pairings,
and improved the aggregate by nearly three percent. It was committed as
`82635d8` with message `Retire Metal leaf stores at FRI query boundary`.

The result supports the combined mechanism rather than proving either component
in isolation. The earlier pool-only evidence establishes that exact-shape
preservation reduces allocation demand but was not sufficient to beat the
official frontier. The new ownership boundary gives those preserved shapes an
earlier opportunity to become available while other proofs remain active.
This is the intended cross-stage interaction: lifetime reduction creates
reusable supply, and exact-size selection prevents the supply from being
cannibalized by smaller requests.

The candidate deliberately does not eagerly clear coefficient vectors. A prior
experiment moved coefficient destruction earlier and regressed local mean time
by about 1.25%, because recursive allocator bookkeeping moved onto the critical
path. Here, logical lifetime ends through ownership transfer and pooled Metal
reuse; coefficient lifetime and destruction behavior remain unchanged.

## Caveats and next steps

The public fixture is synthetic and cannot predict the official private-active
score. The official M4 Pro host has different memory capacity, GPU throughput,
and scheduling behavior. The change is nevertheless value-exact and based on a
fixed last-use boundary rather than a machine-state threshold, reducing the
portability risk seen in earlier conditional scheduling experiments.

The next experiment should remain isolated from this commit and measure witness
fanout separately. If this candidate is submitted, the submission note should
state that the measured local improvement belongs to the combined exact-reuse
plus retirement stack, and should not claim that the retirement mechanism alone
accounts for the full delta.

## Official terminal result

Submission `9345ccf4-7264-4f4b-ab72-9bcb6755ec1e` (Yukon commit `89366c0`)
scored `29.3561409924708 tx/s` against the `29.9399105848455 tx/s` frontier, a
`-0.5837695923747 tx/s` delta. It was officially rejected. This score was in the
fast runner band rather than the contemporaneous 25.x cluster, so the terminal
result is meaningful negative transfer evidence rather than an obvious slow
service draw.

The combined exact-reuse plus retirement stack was reverted in `029af57`.
Because the same-binary candidate arm enabled both mechanisms together, neither
component receives an isolated official estimate. Do not keep or redraw this
stack without a new mechanism and new M4-side evidence.
