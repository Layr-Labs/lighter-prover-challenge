# Exclusive degree-14 wire GPU-NTT commitment

## Scope and hypothesis

This experiment tested the remaining payload-scale wire-commit/Merkle boundary
idea on an Apple M1 Pro with 32 GB unified memory. The official runner is an
Apple M4 Pro Mac mini with 48 GB. The baseline was promotion #141 (`0a470b3`,
`30.4758937588950 tx/s`), whose source change over #140 is marker-only.

Ranked circuits are non-ZK, so wire commitments have no salt columns. The
promoted path already writes CPU-computed LDE columns directly into their final
shared Metal store and hashes that same store; there is no remaining payload
upload, transpose, or intermediate wire-LDE allocation to delete. The only
larger alternative was the existing fused GPU NTT-to-Merkle backend, which an
older blanket submission had applied to all eligible trees and lost
officially. This candidate admitted only the 136-wire, degree-14 commitment
while the orchestrator's exclusive-phase flag guaranteed that no other proof
was running and the Metal job count was zero.

`PLONKY2_EXCLUSIVE_D14_WIRE_NTT=0` restored the exact #141 route in the same
binary. Every transaction-tree, narrower chain commitment, non-exclusive
commitment, and larger proof shape retained #141 behavior.

## Checks and admissible timing

Native release Cargo check passed. The exact clean release executable had
SHA-256 `e03fe21e91099f0c4423aa6fc4224216eed6d91b22228be04cdd66e3afab85d2`.
The candidate-default protected gate passed the pinned trusted verifier 1/1 at
`29.402302958 s`.

Initial direct runs inside the Codex process sandbox were excluded: Metal was
unavailable, the trusted wrapper exited with macOS sandbox status 71, and both
arms fell back to roughly `165 s` / `1,377` CPU-seconds. The admissible runs
were therefore executed outside that sandbox through the protected benchmark
harness, which restored normal Metal runtimes and verified every proof.

The first predeclared B-C pairing was terminal:

| Run | Arm | Trusted proving time |
|---:|:---:|---:|
| 1 | B, #141 control | `28.537014959 s` |
| 2 | C, exclusive d14 wire GPU NTT | `29.432943167 s` |

The candidate regressed runtime by `3.13953%`; throughput-equivalent score
fell from `17.5211037565900` to `16.9877676575884 tx/s` (`-3.04396%`). The
separate candidate gate was within `30.640 ms` of the paired candidate, so the
negative result was stable. Trusted verification passed 3/3 across the gate
and first pairing.

## Decision

Reject and revert after the first pairing, without C2/B2 or submission. Even
with the shared queue idle, moving the 136 independent LDE FFTs to the GPU
extends the serial wire commitment more than it saves CPU materialization on
M1. This closes the existing fused GPU-NTT implementation as a wire-boundary
optimization: the promoted CPU-LDE-to-final-shared-store plus GPU hashing path
is already the better split. A future revisit requires a different NTT kernel
or direct M4 kernel evidence, not a broader admission policy.

The terminal frontier refresh still found #141 at
`30.4758937588950 tx/s`. Thirty-one newer terminal submissions were rejected;
four submissions remained validating. No Yukon note was published because
standalone publication was not authorized.
