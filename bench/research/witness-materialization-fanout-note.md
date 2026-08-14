# Pipeline witness-materialization fanout

## Attribution and scope

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort in Codex. It follows promoted frontier #138 (`e268c13`,
`29.9399105848455 tx/s`) and was screened incrementally on the locally kept
exact-size Metal reuse plus FRI initial-tree-retirement stack (`82635d8`).

The candidate changed only the number of Rayon row tasks used by
`PartitionWitness::full_witness` for the degree-14 and degree-16 pipeline proof
shapes. The degree-18 final block retained the promoted 16-task fanout. Field
values, bitmap semantics, representative-map order, output matrix layout,
proof construction, transcript, Metal policy, FRI ownership, and serialization
were unchanged. The candidate was performance-negative and was reverted.

## Why this path was tested

Promotion #138 removed the dense `PartitionWitness::values` zero-fill. Unset
representative slots are now uninitialized storage guarded by a one-bit bitmap;
`full_witness` maps every wire cell through `representative_map`, tests the
bitmap, and emits either the assigned field value or `F::ZERO`. It constructs
one column-major output vector per wire and partitions the row dimension into
16 Rayon tasks.

The promoted scheduling stack also raises the light proof window to six. Many
degree-14 chain and degree-16 transaction proofs can therefore reach
materialization while sharing the global Rayon pool. The hypothesis was that
16 tasks per proof create enough queueing and work-stealing overhead to produce
large tails. Reducing the degree-14/16 fanout to eight might preserve useful
parallelism while halving task pressure. The final proof runs after the busy
pipeline and has 35.7 million output cells, so its fanout stayed at 16.

This is a static shape rule, not a queue-depth or machine-state trigger. That
choice was deliberate: recent official evidence showed that conditional
scheduling policies can fire at very different rates on the local M1 Pro and
official M4 Pro.

The predeclared failure rule required trusted proof compatibility, positive
mean runtime, and both mirrored pairings to favor eight tasks. Any loss on the
mean or either pairing meant reject and revert without confirmation.

## Diagnostic evidence

A temporary `diagnostic_profile` build added one span and three counters around
`PartitionWitness::full_witness`. Its release worker SHA-256 was:

`40374201c823dd8ad084df4a9f31af8b6f17f4d7a4b22e3c03731eac40b81af8`

The trusted verifier accepted the diagnostic binary 1/1. A direct public-fixture
run then emitted 10,369 trace events. The additional counters appeared once for
each of 106 proof materializations:

- total output cells: `617,218,048`;
- set-slot density across shapes: `41.51–50.93%`;
- summed materialization spans: `9,544.850 ms`;
- average span: `90.046 ms`;
- minimum/maximum: `1.612/1,286.267 ms`.

The most important comparison was within the same degree-16 output shape.
Three heavy transaction proofs averaged `16.200 ms` (`8.520–27.580 ms`), while
49 light transaction proofs averaged `164.735 ms` and contributed
`8,072.006 ms` of summed spans, with a `1,286.267 ms` maximum. Degree-14 light
chain materializations averaged `26.612 ms` with a `323.075 ms` maximum. The
same arithmetic shape therefore became much slower under the busy light
pipeline, making global-pool contention a credible mechanism.

These are nested/concurrent spans, so their sum is not an end-to-end wall-time
ceiling. They identify queueing and tail behavior; only the protected
same-binary screen can decide whether lower fanout improves the critical path.

## Isolation incident

An initial 16-vs-8 screen was accidentally built while another task had placed
an uncommitted FRI initial-tree-retirement candidate in four shared source
files. The collision was detected immediately after the matrix. Those timings
were quarantined: they were not entered as isolated evidence, not used for the
decision, and not published as a result.

Only `witness.rs` was reverted at that point; the other task's four files were
left untouched. The owner subsequently completed its own trusted `B-C-C-B`,
kept the 2.94%-faster reuse/retirement stack as `82635d8`, and released the
shared heavy-work lock. Fanout was then reimplemented and rebuilt from that
clean committed baseline. This note reports only the clean incremental screen.

## Same-binary implementation

The release executable accepted `PLONKY2_PIPELINE_WITNESS_CHUNKS` values
`1,2,4,8,16`, falling back to 16. Every benchmark arm explicitly set the
variable, so environment lookup and validation overhead was common.

For `degree <= 2^16`, the selected value controlled the number of row segments
created before `segments.par_iter_mut()`. Larger degrees used 16 regardless of
the variable. The segment ownership proof was unchanged: each task owned one
disjoint row range of every output column and initialized each
`MaybeUninit<F>` cell exactly once before column lengths were set.

Had eight survived, the production patch would have hard-coded the static
shape decision and removed the environment lookup. No such code remains.

`cargo check --locked -p bench --bin prove` passed. The exact clean release
executable used by both arms had SHA-256:

`3e3a990ea8fbd9e5eb745e6a45ee9f7938d0732a0bd0f452397bd856f287482b`

## Clean protected performance screen

The fixed order was `16-8-8-16`:

| Run | Pipeline chunks | Trusted proving time | Verification |
|---:|---:|---:|---|
| 1 | 16 | `29.272534541 s` | pass |
| 2 | 8 | `30.294703875 s` | pass |
| 3 | 8 | `28.723772166 s` | pass |
| 4 | 16 | `29.082638459 s` | pass |

Control mean was `29.177586500 s`. Eight-task mean was `29.509238020 s`, an
increase of `0.331651520 s` or `1.136665%`; throughput-equivalent delta was
`-1.123890%`.

The first mirrored pairing regressed `3.491906%`. The reverse pairing nominally
improved `1.233954%`, so signs split 1-1. Because the mean was negative and the
candidate lost one pairing, it crossed the predeclared rejection rule. All four
proofs passed the pinned trusted verifier, ruling out incomplete or invalid work
as the explanation.

## Interpretation

The diagnostic tail was real, but static task-count reduction was the wrong
control. Sixteen tasks give each proof more opportunities to use workers that
become briefly available. Halving that supply increases each segment's serial
span and can leave the owning proof waiting on a few larger stragglers. Under a
quiet interval, 16 also improves ordinary memory-level parallelism across the
representative map, bitmap, values, and output columns.

The large light-path tails likely measure competition with other high-priority
Rayon work rather than overhead from task creation itself. Reducing every
degree-14/16 materialization's fanout cannot express which proof is on the
critical chain or which work should yield. It therefore sacrifices useful
parallelism without reliably resolving starvation.

This result should transfer conservatively to M4. The official host has more
performance cores, making a blanket reduction from 16 to eight even less
attractive. The candidate's `1.14%` local mean regression and split pairings do
not justify an official draw.

## Decision and narrower follow-ups

Reject and revert without reverse confirmation or submission. Retain 16 tasks
for all shapes.

Do not test four or one as blind global fanouts; the direction from 16 to eight
already loses. A future scheduling candidate needs a different primitive, such
as a dedicated materialization pool, proof-criticality-aware admission with a
portable static rule, or performing enough work inline on the owning proof
thread before exposing optional remainder tasks. Any such design must first
show that it reduces the observed light-path tail without lengthening the quiet
degree-16 path.

No candidate source remains and no official submission was created.
