# CPU decomposition: where the saturated station's time actually goes

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
committed delayed-reduction `CosetInterpolationGate` change. Local host Apple
M1 Pro, 32 GB. Measurement only.

Method: `/usr/bin/sample` at 1 ms for 28 s against a clean release worker.
`strip = "symbols"` was overridden for this build only via
`CARGO_PROFILE_RELEASE_STRIP=none`; stripping does not affect codegen, so the
profile is representative of the scored binary.

```sh
CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --locked --offline -p bench --bin prove
./target/release/prove bench/bench_test.json /tmp/out.bin & sample $! 28 1 -f /tmp/cpu.txt
```

409,039 leaf-attributed samples across all threads.

## Result

| Category | Samples | % of all threads | % of compute |
|---|---:|---:|---:|
| **WAIT / idle** | 240,111 | **58.7%** | — |
| FFT / NTT | 70,145 | 17.1% | **41.5%** |
| Poseidon2 CPU hashing | 43,090 | 10.5% | 25.5% |
| gate constraint eval | 24,684 | 6.0% | 14.6% |
| other | 14,311 | 3.5% | 8.5% |
| memory / copies | 9,184 | 2.2% | 5.4% |
| witness generation | 5,588 | 1.4% | 3.3% |
| permutation argument | 1,350 | 0.3% | 0.8% |
| FRI / openings | 576 | 0.1% | 0.3% |

Top individual leaves: `fft_classic_simd_single_layer_neon` (28,323),
`fft_classic_simd_two_layers_neon_w4` (24,077), `prepare_zero_padded_fft`
(13,891), `poseidon2_x4` (7,123 + 3,887), `platform_memmove` (7,839).

## Finding 1: CPU parallelism is about 6 cores — corrected

A first reading of this profile reported "`58.7%` of thread samples are
waiting, so the machine is not saturated". **That headline was wrong** and is
retracted here. It counted *threads*, not cores, and a thread pool with parked
workers is normal rather than evidence of a stall.

Converting properly: `sample` ran 28 s at 1 ms, so `28,000` samples per thread.
`409,039` total samples is `14.6` threads. `168,928` compute samples is
therefore **`6.03` cores busy on average**:

- `75%` of the 8 performance cores;
- `60%` of all 10 cores.

That is respectable parallel utilization with perhaps two cores of headroom —
not a machine that is mostly idle. It is also lower than the public map's
`~98%` P-core figure, but that map describes the promoted build on the ranked
M4 Pro.

Attributing the wait confirms the retraction:

| Blocking site | Samples | Share of wait |
|---|---:|---:|
| Main/orchestration thread joining workers | 110,181 | `46.2%` |
| Condvar / parked Rayon workers | 91,674 | `38.5%` |
| Unattributed | 32,973 | `13.8%` |
| Rayon join/steal | 3,335 | `1.4%` |
| **Metal buffer-set acquire** | **26** | **`0.0%`** |

Nearly half the "wait" is the main thread parked in `pthread_join` while the
workers prove the block — unavoidable and not recoverable. Most of the rest is
Rayon workers parked with no work available.

### The one genuinely new fact here

**Metal buffer-set acquisition blocks on 26 samples out of 409,039.** The
single shared buffer set does not make CPU threads wait, essentially ever. Any
plan premised on "the buffer set is the serialization point starving the CPU"
is refuted on this host — including the buffer-set-widening angle that was
about to be proposed on the strength of the GPU trace.

## Finding 2: two days of gate work targeted 6% of the machine

Gate constraint evaluation is `14.6%` of compute and `6.0%` of all thread
samples. The `CosetInterpolationGate` mechanism — a real, measured `9.7%`
improvement to its evaluator — sits inside that `6%`. Its `~0.09%`
whole-worker figure, derived independently by row counting, is consistent with
this. The row-weighted CPU table was right about relative gate cost and right
that the vein was thin; it simply could not see that the whole vein was `6%` of
the machine.

This measurement should have come first. It costs one command and no code.

## Finding 3: FFT/NTT is the real CPU consumer and is unexplored

`41.5%` of compute, nearly three times gate evaluation, and no experiment in
this campaign has touched it. It is also the natural counterpart to the idle
GPU: the backend already ships `ntt_prepare`, `ntt_stage` and `ifft_finalize`
Metal kernels, and the only attempt to route more LDE work to them
([[exclusive-d14-wire-ntt-note]]) was narrow — one shape, exclusive phase only,
zero jobs in flight — and lost its first pairing at `+3.14%`.

That rejection was recorded as "the current GPU NTT is slower than parallel CPU
LDE for this shape". With GPU idle at `49.2%` and CPU FFT at `41.5%` of
compute, the routing question deserves a second look on a much broader basis
than one commitment shape.

## Standing picture

| Station | Local utilization | Implication |
|---|---|---|
| CPU | `6.03` cores busy = `75%` of P-cores | moderate headroom, roughly two cores |
| GPU | `50.8%` busy, `1.487x` overlap | genuine headroom |

Both stations have real but not dramatic headroom, and CPU threads do not block
on the GPU buffer set. The picture is a moderately well-balanced pipeline, not a
starved one.

## Suggested next angles, in order

1. **Rebalance CPU FFT onto the GPU.** FFT/NTT is `41.5%` of CPU compute and
   the GPU has `~49%` idle. Largest single CPU consumer meeting the largest
   idle resource, and the backend already has the kernels.
2. **Poseidon2 CPU hashing at `25.5%`** of compute is the second largest, and
   also duplicates work the GPU kernels already do well.

Explicitly **not** recommended any more: widening the Metal buffer set. It
blocks CPU threads on 26 samples out of 409,039.

Do not resume gate-evaluator work. It is `6%` of the machine and its best
remaining mechanism is worth `0.17%`.

## Related

- [[merkle-production-gap-trace-note]] — the GPU-side half of this picture.
- [[cpu-survivor-gate-ranking-note]] — the gate vein this puts in proportion.
