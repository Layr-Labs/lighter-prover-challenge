# NEON FFT roofline: the last CPU vein is near the hardware floor

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
committed `CosetInterpolationGate` change. Local host Apple M1 Pro, 32 GB.
Measurement only. Harness: `fft::roofline_bench::benchmark_fft_butterfly_roofline`.

## Why

The CPU decomposition put FFT/NTT at `41.5%` of prover compute, and attribution
put `84.9%` of that in `fill_lde_column_store` — genuine one-time LDE
construction — with `52,400` of the `70,145` FFT samples inside two kernels,
`fft_classic_simd_single_layer_neon` and `fft_classic_simd_two_layers_neon_w4`.
Relocating the work to the GPU was closed by prior official evidence
([[cpu-lde-versus-gpu-ntt-note]]), so the only remaining option is making the
CPU kernels themselves cheaper. This measures whether that is possible before
attempting it.

## What the kernels already do

The implementation is mature, and its comments record what has already been
tried:

- 2-wide NEON butterfly with the modular reduction in vector registers;
- a 4-wide variant taken automatically for arrays `>= 2^14`, worth `+3-4%` at
  14 threads, with cache-blocked slices measured as a tie and left on the
  2-wide body;
- two-layer fusion, halving whole-array passes for the fused layers;
- a separate extension-field variant.

The multiply deliberately stays scalar: **aarch64 has no 64x64 -> 128 widening
multiply**, so only the adds, subtractions and reductions vectorize. A prior
packed-width radix-4 fusion was retired for register spilling — the same
pressure cliff that killed the Poseidon2 round-constant fold
([[poseidon2-rc-fold-note]]) and bounded exakoss's own quintic-family
exclusion.

## Result

Single thread, median of 7, production shapes:

| Shape | Time | Butterflies | ns/butterfly | **cycles/butterfly** |
|---|---:|---:|---:|---:|
| degree `2^16` -> LDE `2^19` | `7.136 ms` | `4,980,736` | `1.433` | **`4.58`** |
| degree `2^14` -> LDE `2^17` | `1.512 ms` | `1,114,112` | `1.357` | **`4.34`** |

Instruction floor: a Goldilocks butterfly is roughly one multiply
(`mul` + `umulh`), a 128->64 reduction, one add and one subtract each with the
epsilon correction, plus two loads and two stores — about 16 uops. At 4-wide
issue that is **~4 cycles/butterfly**.

So the kernels run at **`87`-`92%` of the instruction-issue roofline**.

Both figures are conservative in the same direction: the butterfly count
`(N/2) * log2(N)` is an upper bound, because zero-padding lets the first
`rate_bits` layers be skipped or cheapened, so the true cycles-per-butterfly is
*lower* and the true efficiency *higher* than shown.

## Verdict

**Close the FFT vein.** At `87`-`92%` of roofline the entire remaining headroom
is `8`-`13%` of the kernel. Even capturing all of it — which would require
beating a hand-tuned NEON implementation whose main cost is a multiply the ISA
cannot vectorize — yields at most `0.13 x 41.5% ~= 5%` of compute, and
realistically a small fraction of that.

That is a better outcome than another rejected experiment: it is a *reason* the
vein is closed, not a failed attempt.

## Campaign state after this note

Every measured vein now has a measured ceiling:

| Vein | Ceiling | Status |
|---|---|---|
| CPU gate arithmetic | `9%` of compute, `0.1`-`0.3%` per mechanism | one mechanism kept (`0.09%`) |
| Merkle leaf/first-parent fusion | `~0.2%` of a worker | closed |
| Merkle tail cascade | `0.199 s` GPU, overlap-discounted | closed |
| Metal encoder reuse | `0.20%`, arms overlap | closed |
| Metal buffer-set widening | blocks on 26 of 409,039 samples | closed |
| `RandomAccessGate` scratch | `0.02%` | closed |
| FFT -> GPU routing | official `-22%` precedent | closed |
| **FFT CPU kernels** | **`87`-`92%` of roofline** | **closed** |

The proving path is at a local optimum that eight independent angles could not
improve. The one calibration worth carrying forward is the recorded official
A/B on GPU NTT commitments: a `+4.6%` controlled local win scored `-22%` on the
ranked host. Local measurement on this M1 disagrees with the ranked M4 about
which station is critical, so any future GPU-side mechanism validated only here
should be treated as unproven.

## Related

- [[cpu-decomposition-note]] — the `41.5%` figure and the `6.03`-core correction.
- [[cpu-lde-versus-gpu-ntt-note]] — why the work cannot move to the GPU.
- [[merkle-production-gap-trace-note]] — the GPU-side half of the picture.
