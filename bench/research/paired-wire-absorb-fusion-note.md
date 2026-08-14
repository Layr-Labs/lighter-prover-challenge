# Paired wire-fill and fused Metal sponge absorbs

## Decision

Rejected by the predeclared pairing rule and reverted. Pairing adjacent
rate-eight streamed groups passed complete tree and proof correctness, and its
aggregate mean appeared 6.554958% faster, but the warm reverse pairing regressed
0.675228%. The aggregate was dominated by a 5.459-second drift between control
endpoints rather than a repeatable steady-state fusion gain.

## Baseline and mechanism

The experiment used functional promotion #139 (`a67126a`) at
30.3111567697189 tx/s. Its streamed commitment path fills eight retained LDE
columns on CPU, commits one `poseidon2_absorb_pass`, and fills the next group
while Metal absorbs the previous one. Every non-final pass writes 12 sponge
lanes per leaf to shared memory, and the next pass reads them back.

Candidate mode used 16-column groups only in exclusive trees with at least
2^20 leaves. One Rayon batch filled up to 16 columns. The existing Metal kernel
then performed two sequential rate-eight absorbs and Poseidon2 permutations
while retaining the 12-lane state in registers, eliminating one command buffer
and one state store/load round trip per pair. A remaining 1–8-column tail used
one permutation exactly as before. No payload-sized allocation, pool policy,
leaf order, bit reversal, parent level, cap, or authentication path changed.

`PLONKY2_WIRE_ABSORB_FUSION=0` selected the exact promoted eight-column host
path in the same executable; all other values enabled candidate mode.

## Correctness and build gates

- `cargo check -p bench` passed.
- The focused 17-column streamed Metal test passed in candidate and control
  modes. In each arm, retained columns, every level-order digest, the cap, and
  all authentication paths matched the classic tree.
- Release worker SHA-256:
  `7fc550d1bab4b3aa1263803b9d7ded8cd145b88370da9ec850607fda3241067b`.
- The candidate-default gate and all four protected screen proofs passed the
  pinned trusted verifier: five of five.

Changing the MSL source invalidated the checked-in metallib hash. This M1
therefore used source compilation; both selector arms lived in the same release
binary and used the same shader source. A positive candidate would still have
required regenerating the offline metallib before submission.

## Protected B-C-C-B result

B set `PLONKY2_WIRE_ABSORB_FUSION=0`; C enabled paired fill and fusion.

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 35.424763792 | passed |
| 2 | C | 30.936116792 | passed |
| 3 | C | 30.167929875 | passed |
| 4 | B | 29.965593667 | passed |

Control mean: `32.695178730 s`.

Candidate mean: `30.552023334 s`.

Nominal candidate runtime delta: `-6.554958%`.

Nominal throughput-equivalent delta: `+7.014774%`.

Pairing one favored candidate by `12.670930%`; pairing two regressed by
`0.675228%`. Candidate samples differed by 0.768 seconds, while control samples
differed by 5.459 seconds. Because either lost mirrored pairing was a declared
failure signal, no reverse-confirmation or official submission was warranted.

## Interpretation and next step

The arithmetic and ownership mechanism is sound, but doubling fill granularity
delays the first GPU absorb. On the warm M1 path, halving command/state traffic
did not reliably repay that lost overlap. The very favorable aggregate is a
cold/cache/order artifact and must not be treated as a 6.55% optimization.

Keep the promoted eight-column stream. Revisit fusion only with direct M4
command-buffer and state-traffic counters, or with a producer/consumer design
that can expose the first eight columns immediately while fusing later pairs.

The terminal leaderboard refresh found promotion #140 (`7b2d3a6`, commit
`5b329f4`) at 30.4408404924460 tx/s. Its diff contains only two marker comments,
and its public note calls it a marker-only multi-draw, so it adds no mechanism
category.

No standalone Yukon note was published because public disclosure of this
specific research payload was not authorized; there is no public note ID.
