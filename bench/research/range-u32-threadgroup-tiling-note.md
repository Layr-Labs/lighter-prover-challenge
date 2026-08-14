# Range-U32 wire-LDE threadgroup tiling

## Decision

Rejected and reverted on the Apple M1 Pro after a value-exact candidate passed
all focused Metal differentials and five protected trusted proofs but lost the
predeclared same-binary pairing rule. The implementation remains a plausible
M4-specific idea, but this exact 16-row design has no qualifying local signal.

## Baseline and hypothesis

The candidate was built on promoted submission `5a25029` / Yukon commit
`a67126a`, the 30.3111567697189 tx/s frontier. Its public note reports that
`range_u32_quotient` accounts for 21% of the worker's 18.56 s GPU-kernel budget
and rereads the same wire LDE 14.1 times across gate families. The proposed
remedy was a 16-row by 136-column Goldilocks tile: 17,408 bytes, within the
32 KiB threadgroup-memory budget reported for the M4 Pro.

## Isolated implementation

The scalar `range_check_gate_quotient` entry point remained available as the
control. A second `range_check_gate_quotient_tiled` entry point dispatched 32
loader threads per group, cooperatively staged up to 136 columns for 16
quotient rows, synchronized once, and let the first 16 lanes evaluate the exact
existing gate body. Columns beyond the tile used the original device read, and
tail rows were zero-filled before the barrier and skipped during evaluation.

The templated wire accessor compiled the scalar arm to direct device reads and
the tiled arm to threadgroup reads with an out-of-tile fallback. The existing
random-access helper was routed through the same accessor. No selector,
constraint, alpha-reduction, output layout, queue policy, retained buffer, or
proof scheduling semantics changed. `PLONKY2_RANGE_U32_TILE=0` selected the
scalar control; all other values selected the tiled candidate in the same
release executable.

Because the local machine has only Command Line Tools rather than the full
Metal offline toolchain, the changed shader intentionally failed the embedded
metallib function probe and compiled from source at worker startup. Both arms
used the same executable and paid that same cold compile path. This raised
absolute public-smoke runtime from the normal low-30-second band into the
mid/high-40-second band, but it does not explain the within-binary arm delta.
Offline metallib regeneration was deferred unless the candidate first passed
the protected comparison.

## Correctness and build gates

- `cargo check -p plonky2` passed.
- The focused RangeCheck differential passed for quotient steps one and four.
- The five-test gate-quotient set passed: Poseidon2, RangeCheck, U32, combined
  byte/quintic shapes, and the combined Metal/CPU quotient proof verifier.
- The exact release worker SHA-256 was
  `351e738e5fd2fe356924ed8e613c9b3a0682b107d419e0bca5246fea7672bfcb`.
- Candidate-default protected proof plus the four-run screen all passed the
  trusted verifier: five of five complete proofs.

## Protected B-C-C-B result

The fixed order used B = `PLONKY2_RANGE_U32_TILE=0` and C = tiled. Trusted
parent-process proving times were:

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 47.109057000 | passed |
| 2 | C | 46.737068125 | passed |
| 3 | C | 46.334087584 | passed |
| 4 | B | 44.662899292 | passed |

Control mean: `45.885978146 s`.

Candidate mean: `46.535577855 s`.

Candidate runtime delta: `+1.415682%`.

Throughput-equivalent delta: `-1.395921%`.

The opening pairing favored tiling by `0.789633%`, while the reverse pairing
lost by `3.741782%`. The candidate samples were stable within 0.403 s, but the
control endpoint improved by 2.446 s, so the aggregate and the required paired
signs do not support a keep.

## Interpretation and next work

This result rejects the exact M1 implementation, not the measured M4 bottleneck
reported by promotion #139. A future M4-only revisit would need direct GPU
kernel counters and should examine a 32-row or SIMD-group-native layout that
avoids spending half a 32-lane SIMD group on loading-only lanes. Locally, do
not spend more proofs on tile-size permutations without those counters.

Proceed instead to the next unconditional measured seam: prewarm the exact
buffers bound by the final block's first submissions, without changing pool
policy or steady-state ownership.

Standalone Yukon publication was attempted but blocked by the external-action
approval gate because this specific payload was not explicitly authorized for
public disclosure. No public note ID exists.
