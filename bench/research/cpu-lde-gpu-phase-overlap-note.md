# Phase-aligned CPU LDE / GPU overlap

Analysis date: 2026-08-13. Analysis head: `7483c65`. Raw artifact:
`/tmp/merkle_trace.json` (9,331 Chrome-trace events, 4,553,830 bytes,
mtime 2026-08-13 12:52:27 CDT). The trace was produced immediately before the
measurement note commit `5a4e9f9`, from the #145 + retained-Coset research
lineage. No build or prover run was performed for this analysis.

Provenance limitation: the exact SHA-256 of the diagnostic executable was not
recorded when the trace was captured, and the current `target/release/prove`
is newer than the trace. The raw trace, source lineage, commands, hardware, and
event counts are retained, but the executable hash cannot be reconstructed.

Analyzer:

```sh
python3 bench/research/analyze_lde_gpu_overlap.py /tmp/merkle_trace.json
```

## Clock alignment and validation

The profiler's complete events use nanoseconds since its `Instant` epoch;
Metal's `GPUStartTime`/`GPUEndTime` use host time since boot. The analyzer pairs
each of the 644 command buffers by queue sequence, submit thread, proof
identity, and command name. It derives the constant host-to-trace offset from
the minimum non-negative completion-callback lag.

- all 644 GPU intervals paired;
- mapped interval duration versus `execution_ns`: maximum error 0 ns;
- GPU starts before its `metal_submit_to_completed` span: 0;
- GPU ends after that span: 0;
- completion-callback lag: minimum 0 us, median 796 us, p95 84.148 ms,
  maximum 206.495 ms.

The large tail is callback/scheduling latency after GPU completion, not a clock
alignment error. Moving the constant offset by a plausible few microseconds
does not affect any result below.

## The requested degree-16 wire-LDE result

Fifty of 52 degree-16 transaction proofs expose a contextual
`FFT + blinding` span inside `compute wires commitment`. Two commitments took a
path without that named inner span, so whole-stage results are also reported.

| Quantity | Degree-16 wire LDE |
|---|---:|
| Contextual intervals | 50 |
| Summed CPU-span duration | 18.029758 s |
| GPU-idle share, duration-weighted | **67.37%** |
| Per-interval GPU-idle median | **69.82%** |
| Per-interval p25 / p75 | **39.60% / 92.83%** |
| Union of CPU intervals | 12.553967 s |
| GPU idle inside that union | **7.692852 s** |

This is repeated steady-pipeline capacity, not a startup or final-tail
artifact. The four chronological quartiles have duration-weighted idle shares
of 67.95%, 79.60%, 54.33%, and 49.49%. The union contains 85 idle gaps; their
median is 28.27 ms. Forty-two gaps are at least 30 ms and contain 7.364 s of
the 7.693 s idle total.

Commands overlapping these LDE spans are primarily work from other proofs:
76 `merkle_tree`, 31 `range_u32_quotient`, 30 `poseidon_quotient`, 24
`permutation_quotient`, eight `merkle_absorb`, and one `merkle_parents` command
buffer. This directly distinguishes same-phase idle from the trace's aggregate
49.2% GPU-idle headline.

All 52 degree-16 whole wire-commitment stages total 44.563210 s of summed span
time. Their weighted GPU-idle share is 50.70%; the stage union is 24.177603 s
with 11.587006 s of GPU idle. This broader interval includes Merkle building
and waiting and is not used as the LDE rate.

## Other phase cuts

| CPU interval | n | GPU idle, weighted |
|---|---:|---:|
| degree-14 wire LDE | 53 | 59.48% |
| degree-16 partial-product LDE | 51 | 36.96% |
| degree-16 quotient LDE | 50 | 40.91% |
| degree-16 FRI final FFT | 52 | 43.54% |
| late chain wire LDE, steps 19+ | 30 | 100.00% |
| final degree-18 wire commitment, whole stage | 1 | 34.18% |

The 100% late-chain figure does not rescue the already-rejected degree-14
exclusive GPU-NTT route; those short wire LDEs total only 636.7 ms and the
existing GPU NTT is slower at that shape.

## Dependency and critical-path qualification

The CPU wire LDE is a real dependency. For all 50 contextual degree-16 spans,
`build Merkle tree` starts after the FFT ends: median gap 4.146 us, p25
3.386 us, p75 5.083 us, maximum 2.273 ms.

It is not proven to be the worker critical path. The first dependent final
Merkle command is submitted a median 60.146 ms after FFT end and actually
starts on the GPU a median 170.103 ms after FFT end. Only 21/50 submissions
occur within 20 ms; p75 submit delay is 249.268 ms and the maximum is 3.145 s.
Thus the LDE gates the host commitment immediately, but much of its downstream
GPU work is already behind other proof work. A GPU NTT inserted here can fill
real gaps, but it can also extend the queue that the dependent hash must drain.

## Ceiling model

The isolated degree-16 rates were 153.94 ms for 136 CPU columns and an
estimated 168.62 ms for 136 GPU-NTT columns. With no CPU/GPU contention:

| Assumed usable GPU capacity during LDE | CPU / GPU columns | Stage time | Saving/proof |
|---|---:|---:|---:|
| measured 67.37% | 84 / 52 | 95.3 ms | 58.6 ms |
| conservative 25% gate | 111 / 25 | 125.3 ms | 28.6 ms |
| ranked-M4 aggregate-free proxy, 10.4% | 124 / 12 | 140.6 ms | 13.3 ms |

Across 52 transaction proofs, the 25% model removes 1.487 s of summed LDE
span. Discounting by the observed 18.030/12.554 = 1.436 CPU-interval overlap
factor leaves an optimistic ~1.036 s worker ceiling, about 3.4% of a 30.7 s
worker. Using all measured M1 capacity gives an optimistic ~2.123 s / 6.9%
ceiling. Both ignore unified-memory contention, submission CPU cost, and the
dependent Metal queue delay.

## Decision

**The M1 phase-alignment sub-gate passes, but the implementation gate does
not. Do not create `codex/lde-split` yet.**

Positive findings:

- same-phase degree-16 GPU idle is materially above 25%;
- it repeats across production transaction proofs and all timeline quartiles;
- gaps are long enough to contain useful NTT work;
- CPU LDE completion immediately gates the dependent commitment's host work.

Remaining blockers:

- the ranked M4 report is 89.6% GPU busy, so this M1 diagnostic trace does not
  establish 25% usable capacity on the target machine;
- dependent GPU work is often queued tens to hundreds of milliseconds after
  LDE, so local dependency is not the same as worker criticality;
- the conservative ~1.0 s ceiling is below the observed 2–5 s protected local
  endpoint drift and is before unified-memory contention;
- whole-commitment GPU NTT failed officially and the later exclusive degree-14
  route regressed 3.14%, so adding unpreemptible NTT work to the ranked GPU
  stream has a demonstrated downside.

The earlier statement that streamed CPU-FFT/GPU-hash overlap already implements
the proposed split is too strong: hashing already-computed columns is not GPU
NTT on disjoint columns. The heterogeneous mechanism remains technically
distinct, but it now requires a phase-aligned M4 trace (or equally direct M4
station counters) before production implementation. A local full-worker A/B
cannot resolve the remaining transfer question.
