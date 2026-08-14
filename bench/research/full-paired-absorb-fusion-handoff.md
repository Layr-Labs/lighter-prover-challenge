# Full paired Metal absorb fusion: findings and next-agent handoff

## Purpose

This note explains the full paired wire-fill / Metal absorb experiment, why its
apparently large local gain was rejected, and which follow-up designs are still
worth testing. It is written for an agent continuing the Lighter Prover
Challenge on an M1 Pro with 32 GB while targeting the official M4 Pro Mac mini
with 48 GB.

The important conclusion is nuanced: **the original implementation was
rejected, but register-resident sponge-state reuse remains a credible
mechanism**. The rejected result disproves “always delay the next submission
until sixteen columns are ready,” not the value of eliminating Metal command
and intermediate-state traffic.

## Baseline streamed commitment

The functional #139 lineage streams wide polynomial commitments in groups of
eight columns:

1. CPU workers compute eight retained LDE columns directly into a shared Metal
   column store.
2. The host immediately commits one `poseidon2_absorb_pass` command.
3. Metal loads the eight columns, absorbs them into a 12-lane Poseidon2 state,
   runs one permutation, and writes the state to a shared buffer unless this is
   the final pass.
4. While Metal handles group `g`, the CPU fills group `g + 1`.
5. The next Metal pass reloads the 12-lane state and continues.

This is a producer/consumer pipeline. Its key advantage is latency: the GPU can
begin as soon as the first eight columns exist. Its cost is that every boundary
between absorb passes stores and reloads 12 `u64` lanes per leaf and requires a
separate command buffer/encoder/dispatch.

For `N` leaves, one eliminated state boundary avoids approximately:

```
12 lanes * 8 bytes * N leaves * (one store + one load) = 192 * N bytes
```

At `N = 2^20`, that is approximately 192 MiB of logical shared-state traffic
per eliminated boundary. A 136-column tree has 17 rate-eight groups, so pairing
eight adjacent group pairs can remove eight command submissions and roughly
1.5 GiB of logical state traffic for that tree. These are logical bytes; Apple
GPU cache behavior determines how much reaches unified memory.

## Original full-paired experiment

### Mechanism

The candidate changed the producer granularity from eight to sixteen columns
for qualifying large exclusive trees. After all sixteen columns were ready, a
new kernel:

- loaded the previous 12-lane state once;
- absorbed columns 0–7 and ran Poseidon2;
- kept the state in registers;
- absorbed columns 8–15 and ran Poseidon2 again;
- stored the state once, or wrote the final digest.

An incomplete final group still performed exactly one final permutation. The
same release executable contained the exact eight-column control selected by
`PLONKY2_WIRE_ABSORB_FUSION=0`.

No leaf order, field operation order, bit reversal, parent construction,
authentication path, cap, allocation policy, or proof transcript changed.

### Correctness

The implementation passed:

- Cargo/build checks;
- a 17-column candidate/control streamed-tree differential;
- retained-column equality;
- every level-order digest and cap comparison;
- every authentication-path comparison;
- the candidate-default protected proof and all four alternating protected
  proofs, five trusted verifications out of five.

Release worker SHA-256:

```
7fc550d1bab4b3aa1263803b9d7ded8cd145b88370da9ec850607fda3241067b
```

### B-C-C-B timing

`B` was the exact eight-column stream and `C` was full paired fusion.

| Run | Arm | Proving time | Trusted verification |
|---:|:---:|---:|:---:|
| 1 | B | 35.424763792 s | passed |
| 2 | C | 30.936116792 s | passed |
| 3 | C | 30.167929875 s | passed |
| 4 | B | 29.965593667 s | passed |

Means:

- control: `32.695178730 s`;
- candidate: `30.552023334 s`;
- nominal runtime delta: `-6.554958%`;
- nominal throughput-equivalent delta: `+7.014774%`.

This aggregate is not decision-grade because the mirrored comparisons disagree:

- `B1 -> C1`: candidate wins `12.670930%`;
- `C2 -> B2`: candidate loses `0.675228%`.

The two control endpoints moved by 5.459 seconds, while the candidate endpoints
moved by only 0.768 seconds. The attractive mean therefore comes mainly from a
cold/slow first baseline rather than a repeatable optimization. Under the
predeclared rule, either lost mirrored pairing requires rejection.

### Why it probably split

Full pairing couples two effects with opposite signs:

1. **Positive:** fewer Metal commands and fewer state-buffer store/load
   boundaries.
2. **Negative:** the first GPU command is delayed until sixteen CPU-produced
   columns exist instead of eight.

On a warm pipeline, the second effect can destroy CPU/GPU overlap. The first
baseline run was cold enough to exaggerate the aggregate, while the warm
reverse pair showed that the steady-state candidate did not repay the delayed
first submission.

The correct conclusion is not “fusion cannot work.” It is “do not fuse by
delaying the first absorb.”

## Strengthened follow-up: preserve the first absorb

The most direct follow-up keeps group zero completely unchanged:

1. Fill columns 0–7.
2. Immediately commit the original single-absorb kernel.
3. Only then fill columns 8–23 and fuse those two later absorbs.
4. Continue pairing later groups; use a single pass for an unmatched tail.

For 136 columns, this changes 17 commands into 9 while retaining the original
time-to-first-GPU-command. It still removes eight intermediate state boundaries,
but the CPU has the duration of the first GPU pass in which to produce the
first sixteen-column tail pair.

At the parent-thread boundary, this strengthened design had:

- passed Cargo check;
- passed the 17-column full-tree/cap/all-path differential in candidate mode;
- passed the same differential in exact eight-column control mode;
- built release worker SHA-256
  `11a2c41ef4c2fd6030225ae4a40975f5ed7317ec708111e00aad8dbe64788306`;
- passed a candidate-default protected proof at `29.414588750 s`.

The candidate-default time is only a correctness/smoke result, not an A/B
result. A fixed `B-C-C-B` screen had begun, but its first control run was still
in progress when the parent turn was interrupted. **The next agent must inspect
the current process, lock, source, and score files before assuming this
experiment is terminal or restarting it.**

Suggested selector:

```
PLONKY2_WIRE_ABSORB_TAIL_PAIR=0   # exact eight-column control
unset PLONKY2_WIRE_ABSORB_TAIL_PAIR  # tail-pair candidate
```

Keep only if the aggregate is positive and both adjacent mirrored pairings favor
the candidate. Do not rescue another split result using the mean alone.

## M1 Pro to M4 Pro transfer

Local M1 results are useful for correctness and for rejecting large regressions,
but this mechanism is unusually architecture-sensitive:

- The M4 Pro has a larger/faster GPU and different cache/bandwidth balance, so
  the cost of a 12-lane state round trip may scale differently.
- A faster GPU can make CPU LDE production the bottleneck, increasing the cost
  of waiting for sixteen columns. This is why preserving the first submission
  matters even more than raw kernel speed.
- Conversely, higher unified-memory bandwidth may reduce the value of deleting
  state traffic, while command-encoding/submission overhead may not fall by the
  same factor.
- More available memory on the 48 GB official host does not automatically help:
  this candidate is about traffic and overlap, not retained capacity.
- Source-edited MSL invalidates the checked-in metallib. Same-binary local A/B
  remains fair because both arms pay the same source path, but an official
  submission must regenerate the metallib and update its source hash. Otherwise
  every official worker can pay avoidable shader compilation/lowering cost.

Therefore use a two-level decision:

1. M1 must pass exact correctness and show no robust regression.
2. A qualifying candidate should be confirmed on M4 with per-command GPU
   timings or submitted only if both M1 pairings win clearly enough to survive
   the transfer uncertainty.

Current live frontier at the handoff was marker-only promotion #141, commit
`0a470b3`, at `30.4758937588950 tx/s`. Its functional source remains #139, so it
changes the score to beat but not the absorb mechanism baseline.

## Out-of-the-box variants worth considering

### 1. Readiness-aware tail pairing

Pair later groups only when the CPU producer has already completed both groups
before the GPU needs its next command; otherwise submit one group immediately.
This attacks the real tradeoff directly: reuse state when it is free, preserve
overlap when it is not.

Prefer deterministic signals such as circuit shape, proof phase, and producer
queue state over wall-clock heuristics. Timing-dependent routing can introduce
run-to-run proof scheduling variance even though proof values remain exact.

### 2. Shape-specific fusion

Do not apply one policy to every streamed tree. Census the actual group count,
leaf count, exclusive/non-exclusive phase, CPU fill time, and Metal absorb time.
Fuse only shapes where the GPU is expected to remain busy long enough for the
next pair to become ready. A 136-column final/exclusive tree can behave very
differently from a 16–24-column transaction tree.

### 3. Fuse three or four tail groups only behind a proven backlog

The 12-lane state does not grow when more sequential Poseidon2 calls are kept in
one kernel, so triple/quad fusion can delete more boundaries without a larger
state array. The risk is producer latency, kernel duration, instruction-cache
pressure, and reduced scheduling flexibility. Test this only after counters
show that two complete future groups are already ready; do not blindly enlarge
the static fill batch.

### 4. Persistent consumer kernel with ready flags

A high-risk design launches one kernel after the first eight columns and keeps
each leaf's state live while CPU producers publish later-column readiness via
shared-memory atomics. This could remove every inter-pass state round trip and
preserve first-submit latency.

Risks are substantial: unified-memory coherence semantics must be explicit,
GPU threads may spin and occupy the device, forward progress can be fragile,
long kernels can trigger watchdog behavior, and polling traffic can contend
with the CPU fill. Treat this as an isolated research prototype, not a direct
submission candidate.

### 5. Move LDE production to Metal, then absorb immediately

The largest redesign is to run the existing NTT stages for column groups on
Metal and feed their resident results directly into the absorb kernel. It can
remove the CPU producer bottleneck rather than merely schedule around it. A
practical first experiment should target only the largest exclusive tree and
reuse the existing `ntt_prepare`, `ntt_stage`, and finalize pipelines.

Separate dispatches will still be needed for global NTT synchronization, but
they can share one command-buffer sequence and avoid CPU/Metal ownership
handoffs. This has a higher ceiling on M4 than command-count tuning, but also a
much larger correctness and occupancy surface.

### 6. Fuse final absorb with the first parent level

The final leaf pass currently writes all four-lane digests, then a parent kernel
reads them. A specialized final kernel could attempt to produce the first
parent level directly for aligned leaf pairs, deleting one digest write/read
level and one dispatch. Bit-reversed output order, cross-thread coordination,
and cap/path layout must remain exact. Start with a tiny full-tree differential;
this is more invasive than tail-pair fusion but attacks another payload-scale
boundary.

### 7. Exact-lifetime buffer-role handoff

The broad Metal buffer-pool experiments transferred poorly to M4, so avoid
raising global retention. Instead, at a proven last use, move an exact-size
buffer directly into the next compatible role. This can eliminate allocation
and page-fault work without changing the global best-fit policy or peak retained
memory.

Do not optimize short descriptor vectors or Objective-C handle clones again;
the boundary audit showed those account for only kilobytes versus tens of GiB
of aggregate field traffic.

### 8. Command batching without arithmetic fusion

Two adjacent kernels can sometimes be encoded into one command buffer. This
reduces host submission overhead but does not preserve registers between
kernels and therefore does not remove the state-buffer round trip. It is a
lower-risk diagnostic that can separate “command cost” from “state traffic,”
but it cannot be committed until all CPU-produced inputs in that command buffer
are ready, so it can still harm overlap.

### 9. Do not prioritize argument-buffer micro-reuse

Reusable metadata/argument buffers, pipeline objects, and encoders may trim
small host overhead, but pipeline state is already retained and the measured
ceiling is dominated by field payload and GPU queue behavior. Investigate these
only after command/state counters show a meaningful host-side gap.

## Required measurements for the next serious attempt

Add diagnostic counters without changing candidate/control behavior:

- number of streamed trees by `(leaf_count, leaf_width, phase)`;
- CPU fill start/end for every eight-column group;
- host time of each Metal commit;
- GPU scheduled/start/end time of each absorb command;
- time from first column fill start to first command commit;
- estimated state boundaries and logical state bytes removed;
- queue occupancy when each command is committed;
- whether the next one/two groups were already complete when the prior GPU pass
  finished;
- peak retained Metal bytes and pool hits/misses, to prove no hidden memory
  expansion.

The key question is not simply whether a fused kernel is faster. It is:

> At each group boundary, was Metal waiting for the CPU, was the CPU waiting for
> Metal, or were both busy—and did fusion move that critical boundary?

Without that timeline, whole-proof variance can easily turn a cold baseline
outlier into a false multi-percent win.

## Correctness and benchmarking checklist

1. Preserve an exact eight-column environment-selected control in the same
   release binary.
2. Differential-test widths 17, 24, 25, 31, 32, 33, 136, and 137 when feasible;
   these exercise full pairs and one-column tails.
3. Compare retained columns, every digest, the cap, and every authentication
   path, not only the root.
4. Run the pinned trusted verifier on candidate-default before timing.
5. Use fixed `B-C-C-B`; require both mirrored pairings and the aggregate to win.
6. If the signal is small, run a reverse `C-B-B-C` confirmation rather than
   submitting on one favorable order.
7. Refresh the live frontier and public notes before committing/submitting.
8. Regenerate and validate the offline metallib before any official submission.
9. Record the release SHA-256, hardware, selector, every raw time, verifier
   result, and official M4 outcome in all three research ledgers.

## Related files

- `bench/research/paired-wire-absorb-fusion-note.md` — terminal record for the
  rejected full-paired implementation.
- `bench/research/optimization-roadmap.md` — experiment queue and decisions.
- `bench/research/promoted-options.md` — live promoted-submission dataset.
- `experiment-results.tsv` — machine-readable experiment ledger.
- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs` — streamed host
  scheduling and Metal command encoding.
- `vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal` — absorb kernels.

## Bottom line

Full paired fusion was correctly rejected because its +7.01% nominal aggregate
did not survive mirrored ordering. The likely defect was delayed first-GPU
submission, not incorrect arithmetic or useless state reuse. The
first-absorb-preserving tail-pair design is the right strengthened test; after
that, the highest-ceiling directions are deterministic readiness-aware fusion,
GPU-side LDE-to-absorb flow, and exact-lifetime buffer-role handoff. Every one
of them must be judged as a CPU/GPU pipeline change, not merely as a faster
Metal kernel.
