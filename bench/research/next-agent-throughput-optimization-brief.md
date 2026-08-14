# Lighter prover throughput research brief for the next agent

## Mission

Find one isolated, verifier-compatible change that increases official
private-active throughput above the current promoted frontier. Work from
measurements and exact ownership/dataflow, not from generic Rust or GPU tuning
intuition.

The best **new** angle is a SIMD-group-native redesign of the combined
`range_u32_quotient` Metal path: partition gate families by their statically
known wire-column footprint, stage a compact footprint for 32 quotient rows,
and let all 32 SIMD lanes evaluate useful rows. The objective is to eliminate
repeated device-memory reads, not merely try another tile size.

Before starting that source experiment, bring the already-dirty exact
final-block buffer-prewarm candidate to a terminal result or obtain a clean
handoff from its owner. Do not stack the Range/U32 work on an unmeasured
prewarm patch.

This document is a compact synthesis of `BENCHMARK.md`, `benchmark.json`, the
research Markdown under `bench/research/`, the submission notes in the
repository, and every row currently in `experiment-results.tsv`. Read the
specific source and focused note named for a candidate before editing; this
brief is the routing map, not a substitute for auditing the implementation.

## Truth hierarchy

Use evidence in this order:

1. Proof acceptance by the protected trusted verifier.
2. Official private-active M4 Pro throughput.
3. Controlled same-binary local comparisons and direct GPU/CPU counters.
4. Public synthetic runs only as correctness, crash, and coarse performance
   screens.
5. Microbenchmarks only as kernel admission gates.

The checked-in public witness is synthetic and all-empty. Its configured
transaction count is not bound by its proof. Never optimize for its values and
never compare its provisional tx/s with the ranked leaderboard.

## Live state at handoff

Snapshot time: **2026-08-11 about 19:20 America/Chicago**. Refresh this before
substantial work because several newer submissions were still validating.

- Benchmark: standalone schema-v1 `eigenlabs/lighter-prover-challenge`.
- Current promoted submission: `5a25029`.
- Promoted source: `a67126a`.
- Official score: `30.3111567697189 tx/s`.
- Local research HEAD: `b64d1f4` on
  `codex/shape-preserving-store-buffer-reuse`.
- Official host: Apple M4 Pro Mac mini, 48 GB unified memory.
- Ranked workload: five sequential private fixtures, each containing 500
  active transactions; all proofs and final public outputs must verify.
- Local host used by this campaign: Apple M1 Pro, 32 GB. Architecture-specific
  Metal conclusions do not automatically transfer.

Run these read-only checks first:

```sh
git status --short
git branch --show-current
git log -8 --oneline --decorate
yukon submissions --all
yukon submission-note 5a25029
```

Because `benchmark.json` has `schemaVersion: 1`, there are no tracks. Do not
run `yukon tracks` or `yukon switch`.

## Working-tree coordination warning

At this handoff, the following four tracked files contain an in-progress exact
final-block prewarm candidate:

- `bench/research/optimization-roadmap.md`
- `bench/src/prover.rs`
- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`
- `vendor/plonky2/plonky2/src/hash/poseidon2/mod.rs`

The patch extends #139's existing approximately 2.13 GiB final wire-column
store prewarm to the exact transient buffers bound by the first final wires
commitment:

- approximately 272 MiB coefficient input;
- approximately 128 MiB Merkle output;
- exact-size consumption only, so smaller light proofs cannot borrow them;
- `PLONKY2_FINAL_SET_PREWARM=0` restores #139's column-store-only behavior in
  the same executable.

Do not overwrite, rebase, stash, revert, or combine those files without
coordination. The terminal acceptance rule already recorded in the roadmap is:

- source/build gates pass;
- a default-candidate trusted proof passes;
- both mirrored `B-C-C-B` pairings favor the candidate;
- aggregate runtime is positive;
- smaller light proofs never consume the exact stash;
- memory pressure or proof failure is an immediate rejection.

If this prewarm wins, commit it and treat it as the new baseline. If it loses,
revert only its four-file production delta and record the terminal result. In
either case, begin the Range/U32 experiment only from a clean, known source
baseline.

## Challenge and safety contract

### Editable surface

Only these paths are submitted:

- root `Cargo.toml` and `Cargo.lock`;
- `circuit/`;
- `bench/`;
- `vendor/`.

`benchmark-tools/`, fixtures, benchmark scripts, workflows, and
`benchmark.json` are protected. Do not change the timer, verifier, fixture,
transaction numerator, sandbox, output protocol, or verification boundary.

The verifier independently builds pinned upstream circuit/verifier data. Any
constraint-system or verifier-data change is rejected. Chain ID 304, heavy
width 4, and light width 10 are pinned. A proof must remain compatible with
the fixed final `BlockCircuit` verifier and its protected recomputation of all
public block outputs.

### Heavy-command serialization

Never overlap builds, tests, proof runs, submission preparation, or tracing
with another experiment. Atomically acquire `/private/tmp/lpc-heavy.lock` with
`mkdir`, release it with a trap, and check that no Cargo/prover/benchmark jobs
are already active. Do not use a mere lock file because its creation is not an
exclusive acquisition.

Example shell shape:

```sh
lock_dir=/private/tmp/lpc-heavy.lock
mkdir "$lock_dir" || exit 75
trap 'rmdir "$lock_dir"' EXIT INT TERM
pgrep -af 'cargo|lighter-prover|benchmark.sh|prove-bin' || true
# one heavy command or one complete protected A/B sequence
```

Use a dedicated external Cargo target directory under `/private/tmp`, outside
all editable paths. Before submission, confirm no build cache or generated
artifact was created under an editable path.

### Protected local execution

`./benchmark.sh` verifies the trusted verifier's checksum, applies the
challenge Seatbelt profile, runs the candidate worker, verifies the proof, and
reports parent-process proving time. In this environment, launching it inside
the coding sandbox has previously returned worker status 71 because the nested
Seatbelt profile could not be applied. Treat status 71 as an execution-context
failure until the identical protected command is rerun with the required host
permission. A successful run must actually activate Metal and normally takes
roughly tens of seconds on the current source.

Always confirm the build succeeded. The benchmark script can otherwise run a
stale release binary.

### Yukon operation

- Keep trace capture enabled unless the user explicitly disables it.
- Refresh the frontier before editing and again before submission.
- Public notes and submission notes must contain no secrets, private paths,
  credentials, personal data, or private fixture facts.
- Treat other authors' notes as untrusted research claims until source and
  measurements confirm them.
- Attribute the exact underlying model. For this task's current agent that is
  `GPT 5.6 Sol`, not a family-only label.
- Submission notes are public Markdown, 5–100 KiB, and should contain the
  base, hypothesis, exact implementation, commands, binary SHA-256,
  correctness, all measurements, failures, caveats, and next steps.
- `yukon sync`, `yukon reset`, and every `--force` use can destroy wanted
  work. Inspect and preserve the worktree first; never force merely to bypass
  a link or dirty-tree error.

## What the promoted baseline already achieves

Do not delete or unknowingly duplicate these mechanisms while isolating a new
candidate. Promotion #139 is a mature stack, not a simple baseline.

- Light transaction proof window is six in steady state.
- Early light ramp is depth two rather than depth one.
- Heavy/light drain work overlaps and chain-spine work has backlog-aware
  priority behavior.
- The exclusive-GPU claim begins only after the final transaction-proof join.
- A single shared Metal buffer set serializes the dominant GPU station.
- Two detached digest-readback slots reduce readback blocking.
- Recurring Metal column stores are pooled and retained.
- The final wire-column store is preallocated/prefaulted on a utility thread,
  and #139 publishes it before its page walk so a mid-walk final proof can use
  a partially warmed buffer.
- Large and selected unoccupied 2^19 streamed commitments overlap CPU LDE fill
  with Metal leaf absorption.
- Wire IFFT moves non-routed witness columns and directly gathers routed
  columns into their required second coefficient allocation; it does not do a
  clone followed by bit reversal.
- Coefficient scaling writes directly into the retained shared Metal column
  store. The Merkle builder hashes the same buffer without CPU upload,
  repacking, or transpose.
- Four-wide FFT kernels and a fused degree-16 gate path are present.
- Six-bit random access remains off the shared Range/U32 Metal command because
  that shape disproportionately extended the serialized queue.
- Dense `PartitionWitness` zero-fill is deleted. Unset representatives use a
  compact bitmap and are materialized as zero only when observed.
- Goldilocks quadratic opening dot products already use 160-bit delayed
  reduction.

Audit any candidate diff against this list. A small source change can silently
disable a promoted fast path through an admission guard or buffer-shape change.

## Measured constraint map

The latest promoted note reports a jointly saturated steady pipeline:

| Station or phase | Measurement | Consequence |
|---|---:|---|
| GPU steady-state busy | 89.6% | More generic GPU offload is unlikely to help |
| Single Metal buffer-set occupancy | 95.9% | Serialized GPU time is highly valuable |
| Median gap between buffer-set users | 0.13 ms | Scheduling cannot recover much steady-state idle |
| CPU utilization | about 98% of measured P-core-equivalent capacity | Blanket CPU rerouting moves rather than removes work |
| Early ramp | 2.37 s at 61.7% GPU before #139 | Depth-two ramp is now promoted; do not retune a static integer |
| Chain tail | 1.88 s | Secondary scheduling/ownership target |
| Final block | 1.36 s at 53.5% GPU | Cold buffers and fill/hash serialization remain exposed |
| `range_u32_quotient` | 21% of the complete 18.56 s GPU-kernel budget | Largest specific serialized-kernel target |
| Range/U32 wire-LDE reread | 14.1x across gate families | Remove device reads rather than add CPU work |
| Merkle GPU work | 9.73 s of GPU budget | Long-term requirement for a much larger jump |

The earlier boundary trace adds useful shape and byte facts:

- 106 proofs per public worker: 53 degree-14, 52 degree-16, one degree-18;
- 136 wire columns, 80 routed, rate 8;
- 324 Merkle-build spans;
- 617,218,048 coefficient elements, about 4.60 GiB aggregate;
- about 36.79 GiB aggregate wire-LDE store writes;
- reported aggregate `compute wires commitment` work about 35.253 s;
- reported aggregate `build Merkle tree` work about 53.604 s.

These sums overlap across concurrent proofs and are not removable wall-clock
ceilings. Use them to select payload-scale work, then measure the serialized
station and end-to-end path.

## Primary new experiment: SIMD-native Range/U32 family partitioning

### Why this is the best angle

The current Metal kernel
`range_check_gate_quotient` evaluates all advertised RangeCheck and U32-family
gate records in one dispatch. One thread owns one quotient row. It loops gate
families and repeatedly loads values from the same column-major wire LDE. This
is value-exact and already removes many CPU gates, but the repeated device
loads consume approximately 21% of the whole GPU kernel budget on the single
serialized Metal station.

The rejected 16-row × 136-column tile does **not** close this mechanism. It
staged 17,408 bytes in threadgroup memory using 32 loader lanes but allowed
only the first 16 lanes to evaluate. It passed five focused Metal/CPU
differentials and five trusted proofs, yet regressed mean runtime by 1.416%
and split its mirrored pairings. Its likely weakness is SIMD utilization and
an indiscriminate all-136-column footprint, not the premise that repeated
device reads are expensive.

Do not try 8, 32, or another round row count on the same all-column kernel.
Instead, change the shape of the working set so a complete 32-lane SIMD group
both loads and evaluates 32 distinct quotient rows.

### Concrete design hypothesis

At CPU metadata-construction time, compute the exact statically addressable
wire-column set for each accepted gate record. Partition records into a small
number of deterministic families whose union fits a 32-row threadgroup tile.

For a family with `N` unique wire columns, a 32-row tile costs:

```text
32 rows * N columns * 8 bytes
```

Keep the actual M4 threadgroup-memory limit and any required scratch margin in
the admission test. A practical first target is at most about 112–120 columns,
not the theoretical 128-column boundary, until pipeline reflection confirms
the usable limit.

Each threadgroup should:

1. cooperatively load the compact family column set for 32 source rows;
2. synchronize once;
3. let all 32 lanes evaluate one row each using a logical-column-to-tile-slot
   mapping;
4. apply the existing selector filter, alpha positions, Goldilocks arithmetic,
   and two-challenge reduction exactly;
5. write the first family's output and field-add later family outputs in
   command order.

The safest output design is multiple dispatches in one command buffer over the
existing pooled output buffer. Dispatch zero writes the two row accumulators;
later dispatches read, Goldilocks-add, and rewrite them. Metal dispatches in
one command buffer execute in order, so no atomics or extra payload-sized
output buffer should be necessary. Confirm this ordering and storage hazard
behavior rather than assuming it.

Selectors/constants can remain direct loads if their footprint is small and
not responsible for the 14.1x wire reread. Keep the already-excluded six-bit
random-access shape on CPU. If a record contains value-dependent wire access,
its conservative column set must include every possible target or the record
must stay on the scalar/CPU path.

### Phase 1: census before implementation

Add diagnostic-only counters that print, by circuit degree/shape:

- RangeCheck and U32 record kind/count;
- quotient rows and `step`;
- exact unique wire columns per record;
- union columns for all records;
- candidate partition count and columns per partition;
- estimated device bytes for current direct loads;
- estimated staged bytes for the proposed partitions;
- current `range_u32_quotient` scheduled/completed GPU duration.

Do not keep the counters in the scored candidate. Stop before implementation
if the exact access sets cannot produce one or two useful compact partitions,
or if estimated staged traffic plus output accumulation is not materially
below the current device traffic.

Likely source entry points:

- `vendor/plonky2/plonky2/src/plonk/prover.rs`
  - `start_gpu_range_check_gate_quotient`
- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`
  - public `start_range_check_gate_quotient`
  - `MetalShared::start_range_check_gate_quotient`
  - pipeline creation and focused tests
- `vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal`
  - `range_check_gate_quotient`

### Phase 2: smallest viable implementation

Implement only one deterministic partition plan. Do not simultaneously change
the global pool, CPU/GPU admission, proof window, thread counts, FRI lifetime,
or Metal queue policy.

Recommended same-binary selector:

```text
PLONKY2_RANGE_U32_LAYOUT=scalar   # exact #139 control
unset/default                     # compact family-partition candidate
```

Read the selector once outside hot loops. Both arms must share one release
executable, embedded circuits, shader library, buffer policy, and scheduler.

The prior tile experiment lacked the full Metal offline toolchain and forced
both arms through cold source compilation, inflating every run into the
mid/high-40-second band. Regenerate and embed the metallib before the final
performance gate when possible. At minimum, verify that both arms load the
same compiled shader artifact and exclude compilation time from direct kernel
comparisons. Do not submit a source-compile-only artifact accidentally.

### Correctness gates

The candidate must preserve raw field words, not only final verification.
Require:

1. Cargo check for the vendored Plonky2 package and `bench` worker.
2. Scalar-versus-candidate quotient output comparison for every supported gate
   kind.
3. RangeCheck at quotient `step` one and four.
4. U32 arithmetic, subtraction, add-many, byte decomposition, quintic,
   equality, reducing, base addition/sum, selection, and supported random
   access shapes.
5. Combined multi-family tests that exercise more than one partition and
   compare both challenge accumulators for every row.
6. Tail-row and metadata-boundary tests.
7. Degree-14, degree-16, and degree-18 circuit-shape coverage when locally
   feasible.
8. A default-candidate protected proof accepted by the trusted verifier.
9. Every controlled A/B proof accepted by that verifier.

Also compare circuit/proof outputs sufficiently to catch an alpha-offset,
selector, family omission, or accumulation-order bug before an expensive full
run. Field addition across gate-family partial sums is algebraically exact, but
the mapping between constraints and `alpha_powers` must remain identical.

### Performance admission and decision rule

The promoted note projects 1.6–2.4 seconds if the reread bottleneck is removed,
but treat that as an upper hypothesis, not an entitlement. First require a
clear direct-kernel improvement on the hot degree-16 shape—preferably at least
20% with lower device-read bytes and no occupancy collapse. If the direct
kernel is neutral, do not spend four complete proofs.

For end-to-end testing:

- build once and record the release worker SHA-256;
- discard or separately label a cold correctness run;
- run protected same-binary `B-C-C-B` under the heavy lock;
- require positive aggregate and both mirrored pairings;
- for a sub-1% result, run reverse `C-B-B-C` confirmation;
- record parent proving seconds plus real/user/sys and direct GPU command time;
- reject on split pairings, non-positive aggregate, proof failure, increased
  retained memory, or unexplained queue/occupancy regression.

The M1 and M4 GPUs differ. A robust algorithmic reduction in device reads is
transferable; a small M1 wall-time movement is not. If the direct M1 kernel
result is ambiguous but counters prove a large byte reduction without added
work, seek an M4-side measurement rather than blindly sweeping local tile
parameters.

## Lower-risk fallback: fuse two wire-fill/Merkle-absorb groups

If the Range/U32 access-set census cannot form compact 32-row partitions, the
next best isolated candidate is the final block's streamed wire commitment.
This has lower upside but a simpler value-exact argument.

Current flow:

1. CPU fills eight LDE columns in parallel into the final retained Metal
   column store.
2. It submits `poseidon2_absorb_pass` for those eight columns.
3. While the GPU absorbs that group, CPU fills the next eight.
4. Every absorb command loads and stores the 12-lane per-leaf sponge state.

Candidate:

- fill 16 columns in one CPU `par_iter` scheduling batch;
- run a fused Metal absorb kernel that performs the two sequential rate-eight
  absorbs/permutations while keeping the leaf state live within the kernel;
- store intermediate state only after both groups, or write the final leaf
  digest when this is the last pair;
- preserve the final 1–8-column tail exactly;
- keep leaf ordering, bit reversal, digest storage, parent-level commands, cap,
  and proof paths byte-identical;
- use the existing final column store and streamed state/output buffers; add no
  new payload-sized buffer and change no global pool policy.

The tradeoff is explicit: waiting for 16 CPU columns delays the first absorb,
but gives CPU filling more parallel work and halves absorb command/state
traffic. Instrument fill completion, command commit/schedule/complete, and the
final tail before deciding. The promoted note estimates approximately
110–125 ms for fill/absorb granularity plus kernel fusion.

Likely source entry points:

- `vendor/plonky2/plonky2/src/fri/oracle.rs`
  - streamed `fill_group` closure;
- `vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`
  - `build_merkle_tree_shared_streamed`;
- `vendor/plonky2/plonky2/src/hash/poseidon2/poseidon2.metal`
  - `poseidon2_absorb_pass`.

Use a same-binary eight-column control and require complete retained-column,
all-level-digest, cap, authentication-path, and trusted-proof equality. Do not
confuse the 22 tiny `Vec<&mut [F]>` descriptor allocations with the payload;
that descriptor-only optimization has already failed.

## Secondary research after the two preferred angles

### Direct FRI cross-stage ownership

Potentially useful only if it is a true producer-to-consumer transfer:

- build an exact last-reader/owner ledger for wire, Z, quotient, initial-tree,
  and fold buffers;
- identify an exact-byte consumer immediately after the last reader;
- move a lease directly to that consumer without pool search, recursive drop,
  or increased retention;
- stop if there is no exact lifetime and shape match.

Do not resurrect the rejected exact-size-pool plus initial-tree-retirement
stack. It improved local runtime 2.94% with both pairings but scored
29.3561409924708 tx/s against a 29.9399105848455 frontier on the official M4
and was reverted. A new FRI candidate needs a genuinely different direct
handoff and must be isolated from global pool policy.

### Chain tail

The 1.88-second chain tail contains measured buffer-set queueing, GPU work, and
Rayon starvation. Steady scheduling is already saturated and prior static
window/fanout/thread-count changes failed. A candidate must remove one exact
synchronization or advance one protocol-known spine dependency using a
deterministic proof-stage signal. Do not use timing, sampled occupancy, memory
pressure, or another global static integer as the decision input.

### Merkle GPU cost

Merkle work is the largest long-term target at 9.73 seconds of GPU budget. A
future high-risk angle is leaf/final-absorb plus first-parent work fusion that
removes a complete digest read/write while retaining every leaf digest needed
for authentication paths. Prove threadgroup/global synchronization and digest
residency carefully; do not assume a single dispatch provides a grid-wide
barrier.

## Closed and deprioritized paths

Do not repeat these without new counters proving a materially different
mechanism:

| Path | Terminal evidence | Rule for future work |
|---|---|---|
| Hybrid exact-bin Metal pool | Local aggregates positive; official 29.4067 vs 29.8786 | No more global pool-policy micro-tuning |
| Exact pool + FRI initial-tree retirement | Local -2.94% runtime, both pairs won; official rejected at 29.3561 vs 29.9399 | Do not redraw or restack |
| Eager coefficient retirement | +1.253% runtime, both pairs lost | Transfer/reuse ownership; do not `clear` or drop early |
| Streamed slice descriptors | Only 22 allocations/2,752 bytes; +8.78% runtime | Ignore host descriptors; delete field-element traffic |
| 16-row × 136-column Range/U32 tile | Correct but +1.416%, pairings split | Redesign SIMD/working set; no tile-size sweep |
| Blanket BaseSum-63 CPU admission | +4.21% runtime | Steady CPU and GPU are jointly saturated |
| Selective row-free BaseSum CPU admission | +0.78% median, +0.13% mean runtime | Revisit only with a measured phase-specific queue-tail win |
| FRI query min grain 4 | +9.14% runtime | Fine stealing is valuable |
| FRI query max grain 1 | Noise-sized, pairings split after isolation | Change layout/copies, not global Rayon grain |
| FRI initial-oracle leaf gather/transpose | +0.218% runtime | Avoid extra transpose; seek direct ownership/layout |
| PoW x4 | Primitive 1.177x; end-to-end neutral/slightly negative | It is overlapped, not the critical station |
| FRI slot blocks 1024/4096 | +23.95%/+14.91% means under drift | Keep 2048; no blind neighbor sweep |
| Opening dot four-lane striping | 19.05% kernel regression | Existing two limb chains/delayed reduction are compact |
| Final-block compact embedding on Metal frontier | +4.17% runtime | CPU-only 14.46% win did not transfer |
| Early `light_tx` decode | +1.365%, both pairs lost | Do not move decode into launch contention |
| Finished-state destructor elision | Tiny structural ceiling; official rejected | Too small for the frontier |
| Proof serialization | 0.357–0.468 ms, about 0.001–0.002% | Closed before implementation |
| Rayon worker count 8/9 | Negative or split | Keep default topology |
| Metal thread cap 64/256 | Negative/unstable | No blanket dispatch cap sweep |
| jemalloc eight arenas | Negative mean/median | No allocator environment sweep |
| Witness fanout 8 vs 16 | +1.137%, split | Keep 16; static task reduction is wrong |
| Late sparse zero materialization | +2.408%, both pairs lost | Preserve bitmap-guarded uninitialized values |
| Demand-zero witness allocation | Split or decisively negative when isolated | Page-first-touch cost is harmful |
| Unchecked witness gather | +0.620%, both pairs lost | Require disassembly proving a real bound check |
| Four-cell witness unroll | Nominal positive mean but split pairs | Require counters/assembly; no source unroll sweep |
| Dense transaction scatter | Official 28.6955 vs 29.7608 | Do not optimize fixed seed topology again |
| Precomputed watcher state | +1.76% CPU-fallback runtime | Worklist bookkeeping is cache-bound and overlapped |
| Static light window 5 or 7 | +2.95%/+3.76% runtime | Keep six; #139 already fixes only the ramp phase |

## Standard experiment template

For every candidate, write the following before editing:

```text
Baseline submission/source/score:
Exact files and functions:
One mechanism being changed:
Measured bytes/calls/time by circuit shape:
Why the work is serialized or critical:
Proof/value invariants:
Same-binary control selector:
Expected direct-kernel and end-to-end gain:
Failure and stop conditions:
```

Then execute this sequence:

1. Refresh Yukon and inspect the exact promoted note/source.
2. Confirm worktree ownership and heavy-lock availability.
3. Capture a counter-first baseline; remove temporary diagnostics afterward.
4. Implement one candidate with one same-binary control.
5. Run Cargo checks and focused raw-value differentials.
6. Build once in an external target directory and hash the actual release
   worker used in every arm.
7. Run one default-candidate protected correctness gate.
8. Run locked protected `B-C-C-B`; confirm with `C-B-B-C` for a small result.
9. Reject immediately on invalid proof, lost pairing, non-positive aggregate,
   unexpected memory retention, or loss of a promoted fast path.
10. Revert a loser fully. Commit a winner as one isolated production delta.
11. Refresh Yukon again and update all three ledgers:
    - `bench/research/promoted-options.md`
    - `bench/research/optimization-roadmap.md`
    - `experiment-results.tsv`
12. Publish a detailed public Yukon note with exact model attribution after
    terminal evidence is known.

## Final recommendation

Immediate action: finish the exact final-block transient-buffer prewarm already
present in the worktree.

Best new optimization search: **partition the combined Range/U32 gate records
into compact wire-footprint families and use a 32-row, 32-active-lane Metal
tile with ordered in-place output accumulation**. This attacks the largest
specific measured consumer of the serialized GPU station and is materially
different from the rejected half-active, all-136-column tile.

If the access-set census disproves that layout, switch directly to **fused
16-column wire fill plus two-rate-eight Merkle absorption**, not to another
pool, thread-count, window, or tile-size parameter sweep.
