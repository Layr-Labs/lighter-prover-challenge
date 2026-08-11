# Early `light_tx` embedded-circuit decode

## Attribution and scope

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort in Codex. It tests one narrow startup-scheduling hypothesis on
the promoted Lighter Prover Challenge frontier. It does not change circuit
semantics, the transcript, witness values, proof encoding, or the verifier.

Base frontier: promotion 137, source `59c0155`, official score
`29.8785698468374 tx/s`. The experiment was run after the finished-state
teardown checkpoint `d1d839a`, but the early-load switch affected only process
startup and was compared inside one identical release executable. The teardown
candidate was present in both arms, so it cannot explain the A/B difference.

## Hypothesis

The prover has five compact embedded circuit blobs. The pre-processing circuit
is needed first, while the other four circuits are loaded on a background
thread and consumed later. Existing trace data showed that the background load
did not quite finish by the end of the pre proof. The remaining loader ended
about 96 milliseconds after the pre proof, creating a small startup tail before
the main proof pipeline could use all circuit data.

The longest cold decode among those four remaining circuits was `light_tx`.
The candidate therefore started only `light_tx` earlier, on its own named
large-stack thread, before waiting for the pre loader and pre proof. The normal
remaining-circuit loader consumed the early thread result instead of decoding
`light_tx` itself. The other three decodes retained their historical order and
location. An environment switch, `LIGHTER_EARLY_LIGHT_TX_LOAD=0`, restored the
frontier schedule in the same binary.

The expected improvement was deliberately small: roughly `0.2–0.7%` if moving
the longest remaining decode earlier removed the measured 96-millisecond tail
without harming the higher-priority pre load or proof. The predeclared failure
rule was strict: reject if either alternating pairing lost, if aggregate runtime
did not improve, or if any circuit/proof correctness check failed.

## Prior timing evidence

The existing startup trace reported these boundaries:

- pre circuit load completed in `132.802 ms`;
- the pre proof ran from `134.638 ms` to `728.725 ms` after process start;
- the remaining four circuit loads ran from `134.404 ms` to `824.824 ms`;
- the remaining loader therefore extended `96.099 ms` past the pre proof.

An ignored timing harness also measured individual cold embedded loads:

| Circuit | Cold load time |
|---|---:|
| pre | `94.9 ms` |
| heavy transaction | `346.5 ms` |
| heavy chain | `84.7 ms` |
| light transaction | `367.2 ms` |
| light chain | `83.6 ms` |

In that harness, sequentially loading all five took `976.8 ms`; launching all
five cold loads concurrently took `633.1 ms`; rebuilding took about `1.1 s`;
and a warm repeated load took `437.1 ms`. These data made `light_tx` the natural
single-blob candidate, while also warning that concurrent cold decoding was
limited by shared resources rather than perfectly scalable CPU work.

## Implementation

The candidate added an optional early `light_tx` loader next to the existing
early pre-circuit loader. It used the same large stack size required by the
embedded codec and returned the ordinary decoded `CircuitData` through the
thread join handle. The later remaining-circuit loader accepted that handle,
joined it at the historical `light_tx` position, and otherwise followed the
unchanged loading sequence.

This organization preserved a clean same-binary control. With
`LIGHTER_EARLY_LIGHT_TX_LOAD=0`, no early `light_tx` thread was started and the
remaining loader decoded it exactly where the frontier did. With the variable
unset, the candidate schedule started the extra decode at launch. No decoded
value or blob bytes differed between arms.

The release executable SHA-256 was:

`a5d5c466acc0f0861e71b0c0d0f944edbed9065a19d724a50105bdb709c530a7`

## Benchmark protocol and result

The fixed screen order was `B-C-C-B`, where B was the frontier schedule selected
by `LIGHTER_EARLY_LIGHT_TX_LOAD=0` and C was the early `light_tx` schedule. The
same release executable, fixture, machine, and surrounding teardown policy were
used for all four runs.

| Run | Arm | Runtime |
|---:|---|---:|
| 1 | B, frontier schedule | `27.30 s` |
| 2 | C, early `light_tx` | `27.71 s` |
| 3 | C, early `light_tx` | `27.97 s` |
| 4 | B, frontier schedule | `27.63 s` |

The frontier control mean and median were both `27.465 s`. The candidate mean
and median were both `27.840 s`. Early loading added `0.375 s`, or `1.365%`, to
runtime. It lost both direct alternating pairings: `27.71 > 27.30` and
`27.97 > 27.63`.

The regression was larger than the entire measured 96-millisecond tail the
candidate hoped to remove. It also crossed the failure rule after the initial
screen, so no reverse confirmation or official submission was warranted. The
candidate source was reverted immediately.

## Interpretation

The startup tail was real, but its existence did not imply that moving the
largest decode to time zero would shorten the critical path. Compact circuit
decoding is not an isolated sleep-like latency. It allocates and initializes a
large object graph, reconstructs prover data, touches substantial memory, and
may invoke work that competes with Metal-backed commitment setup. Starting that
work at launch overlaps it with the most urgent startup operations: loading the
pre circuit, starting the pre proof, initializing shared runtime state, and
bringing the first GPU/CPU working sets resident.

The individual-loader measurements also support this interpretation. The five
cold loads took `976.8 ms` sequentially but still took `633.1 ms` when all were
started together. Concurrent decoding therefore recovers only part of the
serial sum. Memory bandwidth, allocator activity, cache pressure, GPU queue
serialization, or some combination of them constrains overlap. The single
early `light_tx` result shows that even a less aggressive two-loader launch can
move contention onto a higher-priority phase and cost more than the tail it
removes.

This result also explains why starting all four remaining blobs at process
launch is not a sensible follow-up. That variant would multiply the same launch
contention and is contradicted by both the all-five harness and the end-to-end
single-blob regression. A narrower decode was chosen precisely to avoid that
failure mode, and it still lost decisively.

## Decision

Reject and revert without confirmation or submission. The implementation was
semantically conservative and the same-binary switch made the timing comparison
clean, but the performance result is unambiguously negative: `+1.365%` runtime
and two lost pairings.

The useful retained evidence is about scheduling mechanics:

1. The 96-millisecond tail should not be attacked by indiscriminately moving
   compact decode work earlier.
2. Pre load and pre proof have higher critical-path priority than remaining
   circuit decode; startup policies must protect them from memory, allocator,
   CPU, and Metal contention.
3. Starting the other three remaining decodes early is ruled out absent a new
   noncontending codec or materially different phase schedule.
4. A future startup experiment should measure per-load CPU time, allocation and
   Metal-command activity, and the pre-proof completion time—not just the final
   loader join—before choosing an overlap window.
5. More promising variants would reduce decode work or move it into an actually
   idle phase, rather than merely increasing launch concurrency.

No code from this candidate remains in the frontier tree.
