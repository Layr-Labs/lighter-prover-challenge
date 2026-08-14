# Parallel production permutation Z chains

## Decision

Rejected by the predeclared pairing rule and reverted. Running the two
independent post-inversion Z accumulations through one Rayon join passed
correctness, but the first protected pairing regressed 2.610284% and the second
won 5.062752%. Those whole-proof swings are orders of magnitude larger than the
promoted note's 10–18 ms mechanism ceiling.

## Baseline and isolated mechanism

The source was synchronized through marker-only promotion #140 (`5b329f4`) at
30.4408404924460 tx/s. Its functional code is identical to #139.

The production permutation path already fuses two Fiat–Shamir challenges into
one witness/sigma traversal and computes both numerator/denominator quotient
product buffers in parallel batches. After both buffers are complete, however,
it calls `z_polynomials_from_quotient_chunk_products` twice in a serial `vec!`.
Each call is internally sequential because every row's Z value depends on the
previous row, but the two challenge chains are independent.

Candidate mode wrapped those two calls in one `plonky2_maybe_rayon::join` and
returned the results in the original challenge order. No field operation,
within-chain multiplication order, quotient product, allocation size, proof
value, or Metal work changed. `PLONKY2_PARALLEL_Z_CHAINS=0` restored the exact
serial construction in the same release executable.

## Correctness and build gates

- `cargo check -p bench` passed.
- `paired_fixed_mask_matches_symbolic_reference_with_zero_factor` passed in
  candidate and control modes. It exercises a production power-of-two shape,
  noncanonical limbs, fixed-wire cancellation, and a zero factor.
- The older `paired_two_challenge_path_is_limb_identical_to_general_loop` test
  fails identically in candidate and control on its pre-existing five-point
  synthetic case because `PolynomialValues::new` requires a power-of-two
  length. That unrelated test data was not changed for this experiment.
- Release worker SHA-256:
  `a149fbe83bce6d851d464d3af2add242187de1d358d478ebaa03a09ac81605b2`.
- The candidate-default gate and all four protected screen proofs passed the
  pinned trusted verifier: five of five.

## Protected B-C-C-B result

B set `PLONKY2_PARALLEL_Z_CHAINS=0`; C enabled the Rayon join.

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 30.587060542 | passed |
| 2 | C | 31.385469750 | passed |
| 3 | C | 29.611569541 | passed |
| 4 | B | 31.190676041 | passed |

Control mean: `30.888868292 s`.

Candidate mean: `30.498519646 s`.

Nominal candidate runtime delta: `-1.263719%`.

Nominal throughput-equivalent delta: `+1.279894%`.

Pairing one regressed `2.610284%`; pairing two nominally won `5.062752%`.
Candidate endpoints differed by 1.774 seconds and control endpoints by 0.604
seconds. Either lost pairing was a declared failure signal, so the favorable
aggregate cannot justify a reverse confirmation or official submission.

## Interpretation

The dependency proof is sound, but the tail is too small and too embedded in
global Rayon contention for whole-proof M1 timing to resolve. A join can also
make one chain compete with other proof-window work for a worker instead of
shortening the critical path. Preserve the simple serial construction.

Revisit only if direct span counters show the two chains are simultaneously
critical and an idle worker is reliably available, or if a phase-specific
dedicated pool can isolate them without starving concurrent proofs.

The terminal leaderboard refresh found no promotion newer than #140. No
standalone Yukon note was published because public disclosure of this specific
research payload was not authorized; there is no public note ID.
