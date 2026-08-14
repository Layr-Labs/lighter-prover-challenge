# Exact final-block Metal set-buffer prewarm

## Decision

Rejected and reverted. Pre-faulting the final wires commitment's exact 272 MiB
coefficient-input and approximately 128 MiB Merkle-output buffers on the
existing utility thread passed correctness but regressed protected mean runtime
by 6.318399% and lost the first mirrored pairing by 15.363424%.

## Baseline and target

The experiment used promotion #139 (`5a25029`, commit `a67126a`) at
30.3111567697189 tx/s. That frontier already starts a utility-QoS thread after
the heavy path retires and pre-faults the final wires commitment's 136-column,
2^21-row retained Metal store (about 2.13 GiB).

The same commitment is the first large submission on the final-block tail. Its
single serialized `BufferSet` grows beyond recurring transaction-proof sizes:

- coefficient staging input: `136 * 2^18 * 8 = 285,212,672` bytes (272 MiB);
- Merkle output: `(2 * 2^21 - 2^4) * 4 * 8 = 134,217,216` bytes (approximately
  128 MiB).

The hypothesis was that moving those two allocations and their first touches
under the remaining light pipeline would remove the promoted note's reported
230-330 ms final-buffer stall.

## Isolated implementation

The existing utility prewarm allocated the retained store plus the exact input
and output buffers. It fully touched the two new transient buffers before
publishing them, preventing the page walker from racing the final proof's CPU
copy or GPU fill. The existing large-store stash and partial-walk behavior were
otherwise unchanged.

The two buffers lived in a separate one-shot stash and were consumed only when
the requested size matched exactly. Smaller light-pipeline commitments could
not borrow them. No `MAX_BUFFER_SETS`, column-store pool cap, best-fit policy,
queue order, proof value, or steady-state buffer ownership changed.
`PLONKY2_FINAL_SET_PREWARM=0` selected #139's column-store-only prewarm in the
same executable; all other values enabled the candidate.

## Correctness and build gates

- `cargo check -p bench` passed.
- `final_set_prewarm_requires_exact_sizes` passed on Metal, proving smaller and
  larger requests leave both stashes intact while exact requests consume them.
- Release worker SHA-256:
  `124031eb1719fb1094994e6a9a93f83b217b99d0a4780a47bce3767b3d18d98c`.
- Candidate-default protected proof and all four screen proofs passed the
  pinned trusted verifier: five of five.

## Protected B-C-C-B result

B set `PLONKY2_FINAL_SET_PREWARM=0`; C enabled exact transient prewarming.

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 36.780856334 | passed |
| 2 | C | 42.431655375 | passed |
| 3 | C | 39.706284625 | passed |
| 4 | B | 40.475705500 | passed |

Control mean: `38.628280917 s`.

Candidate mean: `41.068970000 s`.

Candidate runtime delta: `+6.318399%`.

Throughput-equivalent delta: `-5.942903%`.

The opening pairing regressed `15.363424%`; the reverse pairing nominally won
`1.900945%`. The aggregate is decisively negative and the pairing rule fails.

## Interpretation

The extra 400 MiB page walk is not free simply because it runs before the final
tail. It competes for unified-memory bandwidth and page-fault service with the
still-critical light pipeline, and the utility QoS only changes CPU scheduling,
not DRAM or VM contention. The final proof's demand-driven allocation has a
smaller critical-path cost than moving all first touches wholesale into that
phase.

Do not prewarm more final buffers or vary page stride/QoS without direct M4 VM
and bandwidth counters. The next candidate should attack work rather than move
it: fuse the final wire-commit fill/absorb granularity or remove a measured
field-element pass.

No standalone Yukon note was published because public disclosure of this
specific research payload was not authorized; there is no public note ID.
