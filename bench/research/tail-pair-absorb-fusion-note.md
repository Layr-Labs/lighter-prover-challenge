# First-absorb-preserving tail-pair fusion

## Decision

Rejected after the first protected pairing and reverted. Preserving the first
eight-column fill/absorb latency did not rescue paired later absorbs: the fused
tail candidate was 1.637445% slower than the exact eight-column control.

## Baseline and isolated hypothesis

The candidate was built on marker-only frontier #141 (`0a470b3`) plus research
ledger head `ce13339`; its functional prover source remains promotion #139.
The prior all-paired candidate filled up to 16 columns before submitting its
first Metal command. Its aggregate was favorable but its warm reverse pairing
lost, leaving a plausible latency explanation: fusion delayed the first absorb
and therefore shortened CPU/GPU overlap at the front of each streamed tree.

This revision kept group zero byte-for-byte on the promoted path: fill exactly
eight columns, submit `poseidon2_absorb_pass`, and immediately let that command
start. Only subsequent groups were paired (1+2, 3+4, and so on). A paired kernel
performed two sequential rate-eight Poseidon2 permutations while keeping the
intermediate 12-lane state private, deleting one later command and one shared
state write/read round trip per pair. A final unpaired group stayed on the
original kernel. `PLONKY2_WIRE_ABSORB_TAIL_PAIR=0` restored the exact promoted
eight-column loop in the same executable.

No digest arithmetic, permutation order, column layout, retained allocation,
buffer-pool policy, Merkle parent construction, or proof scheduling changed.

## Correctness and build gates

- `cargo check -p plonky2` passed.
- A 17-column streamed build forced the exact candidate shape: first normal
  eight-column pass, then a fused full-plus-one-column pair.
- Candidate and control full-tree, cap, and authentication paths matched the
  classic build exactly.
- Release worker SHA-256:
  `11a2c41ef4c2fd6030225ae4a40975f5ed7317ec708111e00aad8dbe64788306`.
- Candidate-default protected gate, scalar B1, and candidate C1 all passed the
  pinned trusted verifier: three of three.

## Protected first pairing

The predeclared screen was B-C-C-B, where B set
`PLONKY2_WIRE_ABSORB_TAIL_PAIR=0` and C enabled tail-pair fusion. The first
pairing was already a terminal failure signal:

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 30.949020167 | passed |
| 2 | C | 31.455793292 | passed |

Candidate runtime delta: `+1.637445%`.

Throughput-equivalent delta: `-1.611065%`.

The remaining C2/B2 proofs were intentionally not run: the experiment's rule
required both pairings to favor the candidate, so one loss makes success
mathematically impossible and further heavy runs cannot change the decision.

## Interpretation and frontier refresh

The regression persists even when the first absorb starts at the promoted
latency. The remaining cost is therefore the later 16-column fill granularity,
reduced overlap, larger kernel, or a combination of them; deleting command and
state traffic is insufficient. Close this fusion line on M1 and do not test
other pair offsets without direct M4 command/state-traffic counters.

The terminal Yukon refresh found no promotion newer than #141 at
`30.4758937588950 tx/s`. Seven submissions were still validating. No standalone
Yukon note was published for this local rejection.
