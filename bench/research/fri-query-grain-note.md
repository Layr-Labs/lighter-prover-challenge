# FRI query work-stealing grain: coarse failure and fine-grain candidate

## Context and attribution

This progress note documents two related scheduling experiments in the Lighter
Prover Challenge. The work used **GPT 5.6 Sol**, effort **max**, through Codex.
The local machine is an Apple M1 Pro with 32 GB unified memory; the official
runner is an Apple M4 Pro Mac mini with 48 GB. Local measurements use the public
synthetic fixture and are screening evidence only.

The production base is promoted frontier 137 (`59c0155`, official score
`29.8785698468374 tx/s`). The first FRI experiment was measured while the
independently submitted hybrid Metal column-store pool candidate at local
commit `b727fb0` was present in both arms. That pool submission,
`1d69b9c2-8554-47af-a40c-fce715581b57`, later rejected at
`29.4067300861804 tx/s`, so the FRI candidate was rebuilt and measured again
with frontier best-fit restored. The FRI scheduling delta was confined to
`vendor/plonky2/plonky2/src/fri/prover.rs`.

## Why inspect FRI query scheduling

`fri_prover_query_rounds` asks the challenger for 28 indices and maps each
index through a Rayon parallel iterator. Each round gathers openings from the
initial Merkle trees, copies Merkle siblings, then gathers the committed FRI
step evaluations and paths. The resulting proof data and transcript are
unchanged by iteration grain; only how Rayon divides those 28 independent
rounds into stealable jobs changes.

A temporary `diagnostic_profile` span around the complete query phase measured
one public worker:

| Metric | Value |
|---|---:|
| Query phases | 106 |
| Sum of phase durations | 1,627,599.875 microseconds |
| Mean | 15,354.716 microseconds |
| Median | 1,034.750 microseconds |
| Maximum | 232,076.625 microseconds |

The recurring 49 light transaction proofs and 49 light chain proofs contained
maximum query-phase spans of roughly 186 and 232 ms. The sum is not a process
critical-path measurement because several proofs run concurrently, but the
large median-to-max spread shows a tail-scheduling problem worth testing. This
phase had a much higher optimization ceiling than final proof serialization,
which separate traces measured at only 0.357–0.468 ms for the entire worker.

## Experiment A: minimum grain four

The first hypothesis was that each proof exposed too many tiny random-read
jobs while several proofs already competed in the global Rayon pool. The patch
added `with_min_len(4)`, limiting a 28-query proof to roughly seven iterator
chunks. `PLONKY2_FRI_QUERY_MIN_LEN=1` restored the historical iterator in the
same release binary.

The predeclared screen was `B-C-C-B`:

| Sample | Policy | Real seconds |
|---:|---|---:|
| 1 | B, frontier minimum | 30.10 |
| 2 | C, minimum four | 33.21 |
| 3 | C, minimum four | 31.74 |
| 4 | B, frontier minimum | 29.41 |

Control mean/median was `29.755 s`; candidate mean/median was `32.475 s`.
Coarsening regressed runtime by `9.1413%`, and both pairings lost. Confirmation
was skipped under the predeclared failure rule. The patch and its diagnostic
span were removed completely before the next build.

This failure is useful evidence: fine-grained query work is not merely overhead.
Even with multiple proofs in flight, Rayon benefits from being able to steal
smaller query units. The 186–232 ms tails were not improved by making each
proof expose less work.

## Experiment B: maximum grain one

The inverse hypothesis followed directly: Rayon's adaptive splitting may still
leave adjacent query rounds grouped, making a proof caller wait for a chunk
whose task was descheduled behind other proof work. Adding `with_max_len(1)`
forces all 28 independent rounds to remain individually stealable. The release
binary accepts `PLONKY2_FRI_QUERY_MAX_LEN=28` as a same-binary control, which is
nonbinding for a 28-element iterator and restores the frontier behavior. The
official/default path uses maximum length one.

The release worker SHA-256 was
`482ab2312dd5934b490277dfa3410dbde4376ed79c1e2b6f46c46340eaf44805`.
The predeclared order was `B-C-C-B / C-B-B-C`:

| Sample | Policy | Real | User | Sys |
|---:|---|---:|---:|---:|
| 1 | B, frontier max 28 | 33.30 | 172.74 | 13.36 |
| 2 | C, max 1 | 34.06 | 172.76 | 13.82 |
| 3 | C, max 1 | 32.86 | 171.98 | 12.71 |
| 4 | B, frontier max 28 | 33.81 | 172.79 | 14.86 |
| 5 | C, max 1 | 32.10 | 169.56 | 12.31 |
| 6 | B, frontier max 28 | 33.88 | 171.34 | 13.75 |
| 7 | B, frontier max 28 | 33.88 | 172.02 | 13.82 |
| 8 | C, max 1 | 32.77 | 173.04 | 13.26 |

Control mean was `33.7175 s`; candidate mean was `32.9475 s`, a `2.2837%`
runtime reduction or `2.3371%` throughput-equivalent improvement. Control
median was `33.845 s`; candidate median was `32.815 s`, a `3.0433%` runtime
reduction or `3.1388%` throughput-equivalent improvement. Candidate won three
of four adjacent pairings. The first four-sample block was nearly neutral,
which is why the reverse confirmation was required; the confirmation produced
the repeatable direction rather than relying on the best individual draw.

### Isolation after the pool rejection

Because the pool policy failed official M4 validation, the max-grain-one change
was isolated on promotion-137 behavior and rebuilt as release worker SHA-256
`e391abdded403ddbb0246b70848dbac74252fc8f1351265e699cfdc77c4248ff`. The same
binary again used `PLONKY2_FRI_QUERY_MAX_LEN=28` for control, in predeclared
`B-C-C-B / C-B-B-C` order:

| Sample | Policy | Real | User | Sys |
|---:|---|---:|---:|---:|
| 1 | B, frontier max 28 | 26.90 | 176.49 | 11.56 |
| 2 | C, max 1 | 27.32 | 179.19 | 11.47 |
| 3 | C, max 1 | 26.39 | 179.16 | 10.93 |
| 4 | B, frontier max 28 | 26.98 | 178.27 | 10.79 |
| 5 | C, max 1 | 26.31 | 179.85 | 10.76 |
| 6 | B, frontier max 28 | 28.07 | 178.35 | 12.08 |
| 7 | B, frontier max 28 | 25.98 | 175.53 | 10.36 |
| 8 | C, max 1 | 27.46 | 181.33 | 11.62 |

Control mean was `26.9825 s` and candidate mean was `26.8700 s`, only a
`0.4169%` nominal runtime reduction (`0.4187%` throughput equivalent). Control
median was `26.940 s` and candidate median was `26.855 s`, a `0.3155%` nominal
runtime reduction. Pairwise signs split 2-2, including a 1-1 reverse
confirmation. This crossed the predeclared failure rule that confirmation
pairwise signs must favor the candidate. The isolated candidate was therefore
reverted without an official submission.

## Correctness

The exact max-grain-one binary was passed through the pinned trusted benchmark
verifier. One proof verified out of one expected proof. The public synthetic
result was `35.478976542 s` (`14.092852971902959` provisional tx/s). That score
is not compared with the private-active leaderboard.

Scheduling cannot change proof values: challenge generation remains serial and
ordered before the parallel iterator; indexed Rayon collection preserves the
input query order; each round reads immutable Merkle trees and owns its output
vectors. The environment switch changes only the maximum iterator split length.

## Interpretation for Apple Silicon and the M4 Pro

The candidate adds no arithmetic, copies, allocations, Metal commands, or
resident memory. It exposes finer CPU jobs to a shared work-stealing pool while
GPU-backed commitment data is read sparsely. This should transfer more cleanly
than a hard-coded global Rayon thread count: the official M4 Pro has a different
performance/efficiency-core topology, but independent query tasks let Rayon use
whatever workers are available instead of prescribing a core count.

The isolation result also demonstrates the principal risk: local thermal and
scheduler drift are larger than the remaining `0.3–0.4%` aggregate. The earlier
larger effect did not survive removal of an officially rejected stack, so it is
not credible M4 evidence and should not consume a private-active validation
slot.

## Reproduction

Build the release worker with the pinned toolchain and offline dependencies:

```text
RUSTFLAGS='-C target-cpu=native' CARGO_NET_OFFLINE=true \
  cargo build --release --locked --offline -p bench --bin prove
```

Use the default environment for max-grain-one. Set
`PLONKY2_FRI_QUERY_MAX_LEN=28` for the historical control. Alternate the exact
same binary in the order above, record every real/user/sys sample, and run
`./benchmark.sh` on the candidate before any official submission.

## Decision

Reject and revert both global grain variants. Minimum grain four was decisively
slower; maximum grain one became noise-sized with split pairwise signs after
isolation on promotion 137. Neither was submitted. Future work in this phase
should change query data layout, sibling-copy behavior, or allocation volume
rather than global Rayon iterator grain.
