# Exclusive chain predecessor-feed parallel generator round

## Scope and measured ceiling

This experiment used promotion #141 (`0a470b3`, `30.4758937588950 tx/s`) as
the functional baseline and ran on an Apple M1 Pro with 32 GB unified memory.
The official runner is an Apple M4 Pro Mac mini with 48 GB.

The existing chain scheduler already prepares every successor's
predecessor-independent witness work early. In the saved 10,196-event trace,
all 30 light-chain steps after transaction drain had finished that phase and
were parked in `chain_predecessor_join` for `2.898–9.940 s`. Therefore a new
bounded ready queue would duplicate current behavior.

The uncovered dependency begins after each predecessor completes: feeding its
recursive proof wakes a generator worklist before `BlockTxChainProve` starts.
For late steps 19–48, the measured join-end to prove-start gap totaled
`234.631 ms`, averaging `7.821 ms` (`7.405–12.955 ms`). This is about a `0.8%`
whole-worker ceiling.

## Candidate and correctness

Only while the orchestrator's existing process-exclusive chain-drain flag was
set, the candidate wrapped that predecessor-triggered `feed_seeded` worklist in
the existing `ParallelWitnessGuard`. Its existing threshold of 64 generators
was unchanged. Pipelined chain feeds, transaction witnesses, proof dependency,
Metal concurrency, and proof values were unchanged.

`PLONKY2_PARALLEL_CHAIN_FEED=0` restored the exact #141 serial feed in the same
binary. Release Cargo check passed. The clean executable SHA-256 was
`d2ecb6e7dc381d4e3f08dea145c5d75160ab20bd044aff051efa0571b3d157ae`.
The candidate-default protected gate passed trusted verification at
`29.332086291 s`; all four alternating runs also verified, for 5/5 total.

## Protected B-C-C-B result

| Run | Arm | Proving time |
|---:|:---:|---:|
| 1 | B, serial feed | `32.425009041 s` |
| 2 | C, parallel exclusive feed | `29.765828500 s` |
| 3 | C, parallel exclusive feed | `29.694780875 s` |
| 4 | B, serial feed | `29.618962542 s` |

The candidate samples were tight (`71.048 ms` apart), but the controls drifted
by `2.806 s`, far beyond the `234.631 ms` mechanism ceiling. The first pairing
favored the candidate by `8.20102%`; the reverse pairing rejected it by
`0.255979%`. Means nominally favored the candidate (`29.730304688 s` versus
`31.021985792 s`, `-4.16376%` runtime), but that aggregate is dominated by the
slow first control and is not attributable to the change.

## Decision

Reject by the predeclared pairing rule and revert without submission. The one
quiet near-equal pairing shows a small negative signal, while global-pool task
dispatch cannot reliably save a 7.8 ms serial round. Do not add a bounded ready
queue—the trace proves one already exists implicitly—and do not parallelize
this feed again without a dedicated executor or a materially larger per-step
dependency.

The final live Yukon refresh attempt was unavailable, so the last successful
21:16 snapshot remains authoritative: marker-only #141 was the frontier at
`30.4758937588950 tx/s`. No Yukon note was published because standalone
publication was not authorized.
