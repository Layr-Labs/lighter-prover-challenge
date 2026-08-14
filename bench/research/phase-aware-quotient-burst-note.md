# Phase-aware Metal quotient submission burst

## Scope and hypothesis

This experiment used GPT-5.6 Sol at maximum reasoning effort in Codex on an
Apple M1 Pro with 32 GB unified memory. The official runner is an Apple M4 Pro
Mac mini with 48 GB. The baseline was promotion #141 (`0a470b3`,
`30.4758937588950 tx/s`), whose source change over #140 is marker-only.

The hypothesis was that the existing backlog-aware priority covers degree-14
chain Merkle buffer-set acquisition but not the three asynchronous quotient
commands. When the chain is at least three runnable steps behind, keeping its
Poseidon, Range/U32, and permutation quotient submissions adjacent and ahead
of transaction proofs which have not begun a quotient burst could advance the
serial dependency without changing GPU work or concurrency.

`PLONKY2_SPINE_QUOTIENT_BURST=0` restored unrestricted frontier submission in
the same executable. The stop rule required a positive mean and both mirrored
pairings.

## Diagnostic evidence

A `diagnostic_profile` release worker (SHA-256
`15e035f7fbeebcecc5bba3b6e566f4250a02217bb684c431f651175df8b48ed6`)
produced 10,196 events in a 28.590 s process. The trace retained logical proof
contexts for every Metal command plus submit-to-scheduled, submit-to-completed,
and GPU execution time.

The light transaction proof stage issued 407 profiled Metal commands and used
17.466 s of aggregate GPU execution. Its Metal completion spans included
11.891 s for classic Merkle commands, 7.577 s for streamed absorb commands,
7.514 s for permutation quotient, 6.828 s for Range/U32 quotient, and 3.685 s
for Poseidon quotient. These sums overlap and are not wall-time ceilings.

The actionable phase boundary was exact: the final light transaction proof
retired at 22.715 s, but the serial light chain had completed only step 18.
Steps 19 through 48 then occupied 3.649 s, ending at 26.364 s. Once the
transaction queue was gone, each late chain proof itself took only about
102–128 ms. This confirmed a real chain-tail problem and justified testing a
deterministic proof-stage priority rather than another static proof window.

## Implementation and correctness

The candidate added one process-global condition-variable guard around only
the three quotient command submissions in `compute_quotient_polys`. A
degree-14 proof was priority-eligible only when the existing
`SPINE_BACKLOG >= 3` signal was true. The guard dropped immediately after the
third submission, before CPU survivor evaluation or GPU completion waits. It
did not alter a kernel, command contents, transcript order, proof value,
buffer-set count, retained memory, or global pool policy.

Native Cargo check passed. The exact release executable had SHA-256
`fc9df0d96b9d3e11c5c43e5b87df9c74cee8f874fa26bd227aadfdc9d87cf945`.
The candidate-default protected benchmark passed the pinned trusted verifier
1/1 in protocol `lighter-mixed-block-proof-v1`; all four direct timing runs
produced the expected 196,008-byte proof artifact.

## Same-binary B-C-C-B result

| Run | Arm | Direct real seconds |
|---:|:---:|---:|
| 1 | B, unrestricted control | `30.17` |
| 2 | C, phase-aware burst | `27.86` |
| 3 | C, phase-aware burst | `32.61` |
| 4 | B, unrestricted control | `29.89` |

Control mean was `30.030 s`; candidate mean was `30.235 s`. The candidate
regressed runtime by `0.682651%` (`-0.678022%` throughput-equivalent).
Pairings split sharply: the first favored the candidate by `7.656613%`, while
the reverse pairing lost by `9.100033%`.

## Decision

Reject and revert without submission or confirmation. Serializing the start
of every quotient burst introduces a new host-side admission point; the chain
priority did not improve repeatably and the candidate mean was negative. Keep
the diagnostic result: the late 30-proof chain tail is real, but the next
candidate must remove or advance an actual dependency, not add another global
queue-order lock. A promising narrower direction is starting the next chain
step's predecessor-independent CPU work earlier with a bounded ready queue,
while preserving the existing single-step proof dependency and Metal policy.

The terminal frontier refresh still found #141 at `30.4758937588950 tx/s`;
four newer submissions were validating. No Yukon note was published because
standalone publication was not authorized.
