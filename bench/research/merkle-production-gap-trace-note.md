# Production GPU dispatch trace: closing the Merkle tail cascade

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
committed delayed-reduction `CosetInterpolationGate` change. Local host Apple
M1 Pro, 32 GB. Measurement only — no production code changed.

Method: existing `diagnostic_profile` instrumentation, which already records
`GPUStartTime`/`GPUEndTime` per command buffer via
`profile_command_buffer`. No new instrumentation was needed.

```sh
cargo build --release --locked --offline -p bench --bin prove --features diagnostic_profile
LIGHTER_PROFILE_PATH=/tmp/merkle_trace.json ./target/release/prove bench/bench_test.json /tmp/proof.bin
```

644 command buffers captured. GPU busy union `18.258 s`, which lands close to
the public constraint map's `18.56 s` GPU kernel budget — an independent check
that the trace is measuring the right thing.

## The question this had to answer

The isolated harness put the tail-collapse cascade at `~0.4%` of a worker, but
that assumed the tail barrier stalls are real GPU idle. If production overlaps
them with other command buffers, the cascade is worth nothing. The harness
could not distinguish the two cases; this trace can.

## GPU time by command buffer

| Command buffer | n | GPU s | Share of summed GPU |
|---|---:|---:|---:|
| `merkle_tree` | 274 | `13.704` | `50.5%` |
| `range_u32_quotient` | 106 | `6.941` | `25.6%` |
| `poseidon_quotient` | 106 | `3.675` | `13.5%` |
| `merkle_absorb` | 63 | `1.422` | `5.2%` |
| `permutation_quotient` | 86 | `1.161` | `4.3%` |
| `merkle_parents` | 8 | `0.160` | `0.6%` |

Merkle work totals `15.286 s`, `56.3%` of summed GPU time. Merkle's dominance
is confirmed. `merkle_tree` averages `50.01 ms` per command buffer.

## Verdict: the tail cascade is closed

Two independent findings kill it.

**1. The ceiling is `1.3%` of Merkle GPU time, not `4.9%` of a tree.**

274 `merkle_tree` command buffers times seven tail levels at the measured
`~104 us` floor is `0.199 s` of GPU time — against `15.286 s` of Merkle GPU
work, that is `1.3%`. The per-tree figure looked larger only because a single
`2^19` tree is far smaller than the production aggregate.

**2. Command buffers overlap, so the stalls are already being filled.**

| Quantity | Value |
|---|---:|
| GPU wall span | `35.962 s` |
| Sum of command buffer durations | `27.154 s` |
| Union (GPU busy) | `18.258 s` (`50.8%` of span) |
| GPU idle | `17.704 s` (`49.2%` of span) |
| **Overlap factor** | **`1.487x`** |

An overlap factor of `1.487x` means that whenever the GPU is busy, roughly 1.5
command buffers are in flight. A barrier stall inside one `merkle_tree` command
buffer is therefore substantially filled by another command buffer's work, and
the `0.199 s` ceiling is discounted further — toward zero, by an amount this
trace cannot bound tightly but which is clearly large.

Worse for the premise: the GPU is **idle `49.2%` of the span** on this run. The
`104 us` per-level floor was measured on an otherwise idle GPU; under
production concurrency those levels do not even occupy the GPU for that long,
because it switches to other work.

## The assumption that drove us here was wrong locally

The move from the CPU vein to the GPU rested on the GPU being the serialized,
saturated station — `89.6%` busy, buffer set `95.9%` occupied. On this machine
the GPU is **half idle**, so reducing GPU work has poor wall-clock leverage
here regardless of which GPU mechanism is chosen.

The honest caveat: this is the `diagnostic_profile` build, whose instrumentation
inflates CPU time and therefore exaggerates GPU idle. The public `89.6%` figure
is for the promoted build on the ranked M4 Pro. So the true local figure sits
somewhere between, and the ranked host is certainly more GPU-bound than this.
But `1.487x` overlap is a property of the submission pattern rather than of CPU
speed, and it transfers.

## Decision

Do not implement the tail cascade. Its production ceiling is `0.199 s` of GPU
time before discounting for overlap, and the overlap discount is severe.
Together with the already-rejected leaf/first-parent fusion (`1.3%` of a tree,
`~0.2%` of a worker), **the Merkle vein is closed** by measurement rather than
by intuition.

## Where the remaining GPU time actually is

`range_u32_quotient` at `6.941 s` is `25.6%` of summed GPU time, matching the
public map's "21% of the GPU kernel budget". #143 already applied delayed
reduction to its alpha accumulation and deliberately left the two
register-heavy quintic families on strict per-product reduction. That exclusion
is the one identified, unharvested arithmetic target left inside the largest
non-Merkle GPU consumer — but note it was excluded for a measured reason
(a register-pressure cliff), so it is not free ground.

## Standing ceilings

| Vein | Ceiling | Status |
|---|---:|---|
| CPU gate arithmetic, six survivors | `9%` total, `0.1`-`0.3%` each | one mechanism kept (`0.09%`) |
| Merkle leaf/first-parent fusion | `~0.2%` of a worker | closed |
| Merkle tail cascade | `0.199 s` GPU, heavily discounted | **closed** |

No measured vein now offers a multi-percent gain. The next real move is
structural — deleting work rather than rescheduling it — or accepting that the
executable is near a local optimum and competing on draws.

## Related

- [[merkle-dispatch-split-note]] — the isolated harness decomposition this tests.
- [[cpu-survivor-gate-ranking-note]] — the CPU vein and its `9%` ceiling.
- [[coset-interpolation-delayed-reduction-note]] — the one kept mechanism.
