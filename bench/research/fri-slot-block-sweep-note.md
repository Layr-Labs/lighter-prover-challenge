# FRI coefficient-slot block-size sweep

## Attribution and scope

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort in Codex. It starts from promoted Lighter Prover Challenge
frontier 137, source `59c0155`, official score `29.8785698468374 tx/s`; the
local clean checkpoint immediately before the experiment was `ccae39a`.

The candidate added a same-binary selector around the existing coefficient-slot
partition size in `ReducingFactor::reduce_polys_base` and its Goldilocks
quadratic-extension fast path. It did not change field arithmetic, polynomial
order, the delayed-reduction interval, transcript data, proof encoding, circuit
constraints, Merkle construction, or verifier behavior. The tested alternatives
were performance-negative and the selector was reverted without an official
submission.

## Mechanism and hypothesis

The promoted FRI reduction path combines many coefficient polynomials into one
output polynomial. Its successful layout partitions output coefficient slots
into Rayon tasks; each task walks every input polynomial for a contiguous output
range. The frontier uses a fixed `SLOT_BLOCK` of 2,048 coefficients in both the
generic implementation and the specialized Goldilocks extension-degree-two
implementation.

That constant controls a concrete Apple-Silicon tradeoff:

- a smaller tile exposes more tasks and finer work stealing, but repeats Rayon
  scheduling and per-polynomial slice/bounds work more often;
- a larger tile amortizes scheduling overhead, but reduces available
  parallelism and grows each worker's output working set;
- 2,048 may be an inherited round number rather than the best point for the
  M1/M4 cache and performance-core topology.

The experiment therefore screened the nearest power-of-two alternatives, 1,024
and 4,096, against the promoted 2,048 value. A win would have been unusually
clean to transfer: the production patch could hard-code one constant, with no
new allocation and no proof-mechanics change. The expected whole-prover gain
was `0.2–1.0%` if the current tile was leaving cache locality or task parallelism
on the table.

The predeclared failure rule was to reject without confirmation when neither
alternative improved the symmetric two-sample aggregate and trusted proof
compatibility. Any apparent winner needed a separate reverse-order confirmation
before hard-coding or submission.

## Same-binary selector

The release executable read `PLONKY2_FRI_SLOT_BLOCK`, accepting power-of-two
values from 256 through 16,384 and falling back to 2,048. Both reduction paths
used the selected value for `par_chunks_mut` and the corresponding coefficient
start offset. Every benchmark arm explicitly set the environment variable, so
environment lookup and validation overhead was common to all arms.

The selector existed only to remove rebuild variance from the comparison. Had
an alternative survived, the production candidate would have replaced the
constant directly and deleted the environment lookup before its final benchmark
and official submission.

No arithmetic indexing changed beyond substituting the selected block size for
the fixed constant. Each output coefficient continued to receive the same
ascending sequence of base powers and polynomial coefficients. The existing
160-bit delayed-reduction dot-product primitive was unchanged.

## Build and validation

`cargo check --locked -p bench --bin prove` passed. A root invocation of
`cargo test -p plonky2 --lib reduce_polys_base` was not a valid workspace test
command because the vendored `plonky2` package is intentionally outside the
root workspace; it did not report a source, compilation, or test failure.

The exact release executable had SHA-256:

`3f12ff6e3da450d8cd85de2048667f153bccabf477b54dec1df6c0b40727b650`

The pinned trusted verifier accepted all six proofs produced by the timing
matrix: six verified proofs out of six expected proofs. Every result reported
the pinned verifier revision `381fd529eb61dfff9ad245d94fce214a0a64d927`,
protocol `lighter-mixed-block-proof-v1`, and the public synthetic fixture hash
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.

The experiment ran on the local M1 Pro MacBook Pro with 32 GB RAM. The official
runner is an M4 Pro Mac mini with 48 GB, so only robust algorithmic signals—not
small topology-specific noise—qualify for transfer.

## Balanced performance screen

The fixed symmetric order was `2048-4096-1024-1024-4096-2048`. This places the
control at both endpoints and each alternative in mirrored positions:

| Run | Slot block | Trusted proving time | Trusted result |
|---:|---:|---:|---|
| 1 | 2,048 | `29.671916625 s` | pass |
| 2 | 4,096 | `44.087350875 s` | pass |
| 3 | 1,024 | `37.227142208 s` | pass |
| 4 | 1,024 | `55.531989084 s` | pass |
| 5 | 4,096 | `41.906574208 s` | pass |
| 6 | 2,048 | `45.165871291 s` | pass |

The two-sample means were:

| Slot block | Mean | Runtime delta versus 2,048 |
|---:|---:|---:|
| 2,048 | `37.418893958 s` | control |
| 4,096 | `42.996962541 s` | `+14.907091%` |
| 1,024 | `46.379565646 s` | `+23.946918%` |

The final control was much slower than the opening control, proving that the
host experienced substantial thermal or system-state drift during this long
back-to-back matrix. Consequently the percentages are not treated as precise
effect estimates. The symmetric endpoints still make the decision safe: both
alternatives' means were materially worse than the control mean, 4,096 was
consistently slow in both samples (`44.087` and `41.907 s`), and 1,024 was both
slow and unstable (`37.227` and `55.532 s`). There is no positive signal to
justify another expensive confirmation matrix.

The public synthetic scores are provisional and noncompetitive by challenge
definition. They were used only to obtain trusted parent-process timing and
proof validation; no local synthetic tx/s value was compared to the official
leaderboard.

## Interpretation

The result supports the current 2,048-slot block as a reasonable compromise on
this code path. Halving it likely creates too many independently scheduled
chunks while repeating the full input-polynomial traversal for smaller output
ranges. The very unstable 1,024 samples are also consistent with a task-heavy
configuration becoming more sensitive to thermal throttling and competition
among CPU work, Metal command submission, and memory traffic.

Doubling to 4,096 reduces the number of available tasks and enlarges each
worker's contiguous output range. Its two samples were much more consistent
than the global run drift, yet both remained around 42–44 seconds. That is the
strongest negative evidence in the sweep: coarse tiles appear to starve useful
parallelism or exceed the favorable cache working set.

The M4 Pro has more performance cores and a different cache hierarchy than the
M1 Pro, but that does not rescue either candidate. A larger core count increases
the cost of exposing too few tiles, while a 24% local mean regression for the
smaller tile is far outside any plausible M4-only scheduling win. An M4-specific
retune should happen only with direct hardware access and a broader counter-led
sweep, not through an official submission of either losing constant.

The experiment tested coefficient-slot tile size only. It does not disprove
other delayed-reduction parameters: arithmetic chunk length before `reduce160`,
specialized vectorization, input-polynomial grouping, or phase-aware parallel
admission could have different optima and remain separate hypotheses.

## Decision and follow-ups

Reject 1,024 and 4,096, revert the runtime selector, and retain the promoted
2,048 constant. No reverse confirmation or official submission was warranted.
The worktree was restored to the exact frontier source after mechanically
preserving the vendored file's existing CRLF line endings.

Do not repeat a blind neighboring-power-of-two sweep on the M1. A future tile
experiment should first measure per-reduction input count, output length, Rayon
task count, and CPU time by circuit class. That evidence could motivate a
shape-aware tile or a direct M4-only sweep. The next local optimization should
move to a different mechanism seam rather than subdividing this decisively
negative result.

No candidate source remains and no official submission was created.
