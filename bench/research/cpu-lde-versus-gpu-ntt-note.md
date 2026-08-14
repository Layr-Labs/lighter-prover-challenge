# CPU coset-LDE versus GPU NTT at production shapes

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
committed `CosetInterpolationGate` change. Local host Apple M1 Pro, 32 GB.
Measurement only. Harness:
`hash::poseidon2::metal::tests::benchmark_cpu_lde_versus_gpu_ntt`.

## Why

The CPU decomposition ([[cpu-decomposition-note]]) put FFT/NTT at `41.5%` of
CPU compute — about `11.6 s` of a `28 s` sample window once converted at the
measured `6.03` busy cores — while the GPU trace
([[merkle-production-gap-trace-note]]) showed the GPU `49.2%` idle. That is the
largest single CPU consumer meeting the largest idle resource.

The one prior attempt to route LDE work to the GPU
([[exclusive-d14-wire-ntt-note]]) tested a single degree-14 wire commitment
inside an exclusive phase with zero jobs in flight, and lost at `+3.14%`. This
sweeps the real shapes instead.

## Method

Both arms build the same commitment. CPU is `PolynomialCoeffs::lde(rate_bits)`
then `coset_fft_with_options`, per column, across the Rayon pool — the
production path. GPU is `build_from_coeffs`, which performs NTT **and** the
Merkle build, so the Merkle component is estimated at the separately measured
`~11.2 ns/hash` and subtracted to isolate the NTT.

Median of 5, arms alternated. `rate_bits = 3`, `cap_height = 4`.

## Result

| Shape | CPU coset-LDE | GPU NTT+Merkle | Merkle est. | GPU NTT alone | Ratio |
|---|---:|---:|---:|---:|---:|
| `2^14 x 136` (LDE `2^17`) | `34.33 ms` | `46.10 ms` | `2.94 ms` | `~43.16 ms` | `0.795x` |
| `2^16 x 136` (LDE `2^19`) | `153.94 ms` | `180.36 ms` | `11.74 ms` | `~168.62 ms` | `0.913x` |

**The GPU NTT is slower than parallel CPU LDE at both production shapes.** The
earlier single-shape rejection was correct and generalizes. Straight
CPU-to-GPU reassignment of LDE work is not available.

Note the trend: the deficit narrows with size, `0.795x` to `0.913x`. Degree 18
was omitted here (2.3 GiB of coefficients) but is the shape where the GPU is
most likely to reach parity.

## The interesting consequence (superseded — see Closed below)

The two paths are now *comparable*, not far apart — `1.132 ms/column` on CPU
versus `1.240 ms/column` on GPU at the degree-16 shape. Production currently
uses only one of them at a time for a given commitment.

Splitting a commitment's columns across both, sized so each finishes together:

| GPU capacity available | Split (CPU / GPU cols) | Time | Versus CPU-only |
|---|---|---:|---:|
| fully idle | 71 / 65 | `80.5 ms` | `1.91x` |
| `49%` free (measured) | 94 / 42 | `106.4 ms` | `1.45x` |
| `25%` free | 111 / 25 | `125.3 ms` | `1.23x` |

This reframes the question from "which device is faster" — where the GPU loses
— to "can both run at once", where the answer is plausibly yes, because the
rates are within `10%` of each other and the GPU is measurably idle.

## What is NOT established

Deliberately not projecting a worker-level number from this. The model above
assumes:

- CPU and GPU LDE can run concurrently without contending. They share unified
  memory bandwidth, and the CPU arm already saturates the Rayon pool, so
  submitting and draining GPU work costs some CPU.
- The measured `49.2%` GPU idle is available *during the LDE phases
  specifically*, rather than concentrated in other parts of the run.
- Production does not already overlap one commitment's CPU LDE with another
  commitment's GPU work, which would mean the idle is not really there.

The third assumption is the one that would invalidate the whole idea, and it is
directly checkable.

## Closed: the answer was already in the source

Two facts found while locating the CPU LDE call site close this angle without
needing the overlap check.

**1. Whole-commitment GPU NTT was already tried and officially rejected.**
`vendor/plonky2/plonky2/src/fri/oracle.rs` carries the verdict as a constant:

```rust
/// Official ranked A/B: submission 644c4257 (this on, over the 8.0011
/// frontier) scored 6.2323 despite a +4.6% controlled local win — the NTT
/// stages extend each tree's exclusive occupancy of the serialized GPU
/// stream, which is the ranked critical path.
const GPU_NTT_COMMITMENTS: bool = false;
```

A `+4.6%` controlled *local* win became a `-22%` official result. The stated
failure mechanism is exactly the risk flagged above: on the ranked host the
serialized GPU stream is the critical path, so adding NTT stages to it extends
every tree's exclusive occupancy. The local `49.2%` GPU idle measured here does
not transfer to the M4 Pro, where the public map reports `89.6%` busy.

Caveat in the other direction: that A/B ran against an `8.0011 tx/s` frontier,
and the pipeline has changed enormously since (v11 streaming, buffer-set
pooling, ramp depth two). The evidence is strong but not current.

**2. The CPU/GPU concurrency this note proposed is already implemented.** The
promoted streamed path overlaps exactly as modelled:

> the backend absorbs each group of eight LDE columns while the CPU computes
> the next group, collapsing the serial FFT-then-hash commitment into
> `max(FFT, hash)`

So CPU FFT and GPU hashing already run concurrently in eight-column groups. The
`1.45x` split model above is not available headroom; it is largely the
mechanism already in place.

## Verdict

Close the FFT-to-GPU routing angle. The isolated head-to-head result stands
(GPU NTT is `0.795x`/`0.913x` of CPU LDE), but the routing decision it feeds was
already made against official evidence, and the concurrency it would have
enabled is already promoted.

The `41.5%` FFT figure remains the largest single CPU consumer and
`fill_lde_column_store` remains its origin. Any future attack on it must delete
or cheapen the CPU coset-FFT itself — `fft_classic_simd_single_layer_neon` and
`fft_classic_simd_two_layers_neon_w4` are `52,400` of the `70,145` FFT samples,
so `75%` of the prize sits in two NEON kernels — rather than move the work to a
GPU stream that is the ranked critical path.

## Related

- [[cpu-decomposition-note]] — the `41.5%` FFT figure that motivated this.
- [[merkle-production-gap-trace-note]] — the GPU idle figure this depends on.
- [[exclusive-d14-wire-ntt-note]] — the narrow prior rejection this generalizes.
