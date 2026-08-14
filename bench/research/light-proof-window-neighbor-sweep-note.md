# Light transaction proof-window neighbor sweep

## Attribution and benchmark context

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort through Codex for the Lighter Prover Challenge. It used clean
research commit `fcb4dae`, whose proving source remains promotion #138
(`e268c13`, official score `29.9399105848455 tx/s`). No source edit was needed:
the promoted prover already exposes `LIGHTER_LIGHT_WINDOW` for controlled
window-depth experiments.

Local runs used an Apple M1 Pro MacBook Pro with 32 GB unified memory. The
official runner is an Apple M4 Pro Mac mini with 48 GB. Public synthetic tx/s
is provisional and noncompetitive, so the analysis uses trusted verifier
parent-process `proving_seconds`. No official submission was created.

## Motivation

The block prover pipelines light transaction proofs while the light chain
consumes completed proofs in order. After the initial three steps that overlap
the fixed heavy path, the promoted scheduler permits six light transaction
proofs in flight. When the deque reaches the window, the producer joins the
oldest proof before continuing.

An existing diagnostic trace exposed 47 `tx_proof_window_join` spans totaling
`18.464573 s`, with a maximum wait of `1.750145 s`. Those totals overlap other
threads and are not an end-to-end ceiling, but they show that the window limit
is regularly active. One extra proof could let witness generation and the
chain progress while an older proof waits on the shared Metal queue. The M4
runner's additional memory also makes a seven-proof window superficially
attractive.

The opposite hypothesis is equally plausible. Six in-flight proofs share CPU
workers, Metal command queues, allocator arenas, and unified-memory bandwidth.
Reducing the depth to five could lower contention enough to shorten every proof
despite joining slightly earlier. The promoted source comments already reject
depth eight because extra concurrent allocations and page faults outweigh
capacity, so only the two immediate neighbors were tested.

## Experimental design

The source function `light_tx_proof_window()` reads `LIGHTER_LIGHT_WINDOW` once
and accepts depths from one through twelve. Each protected worker is a fresh
process, so explicitly setting the environment variable selects the requested
depth without code-generation differences. All other scheduling constants,
including the step-three overlap start, remain identical.

The predeclared symmetric order was:

`6-7-5-5-7-6`

This places the promoted depth at both endpoints and mirrors both alternatives.
A neighbor would qualify only if it improved its aggregate relative to six and
won both endpoint comparisons. Any proof failure, aggregate regression, or
split pairing was a rejection. This is stricter than selecting the best single
sample and is important because whole-prover timings move with temperature,
allocator history, and background load.

The trusted setup verified the pinned verifier and rebuilt the clean source.
The exact release worker SHA-256 was:

`3d7036f089a134e0f78ce304e4e3554a82c7f0e7d3c5fa9a039809e698bf1f6e`

All six protected proofs passed the pinned trusted verifier under protocol
`lighter-mixed-block-proof-v1`. The fixture SHA-256 was
`6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b`.

## Results

| Order | Window | Trusted proving seconds | Verification |
|---:|---:|---:|:---:|
| 1 | 6 | `31.639909583` | passed |
| 2 | 7 | `34.440152416` | passed |
| 3 | 5 | `32.849648000` | passed |
| 4 | 5 | `31.041157875` | passed |
| 5 | 7 | `29.951259625` | passed |
| 6 | 6 | `30.418927417` | passed |

Depth six averaged `31.029418500 s`.

Depth five averaged `31.945402937 s`, a `2.951987%` runtime regression and
`2.867344%` throughput-equivalent loss. It lost both mirrored endpoint
comparisons by `3.823457%` and `2.045537%`. The two five-depth samples moved
with the overall endpoint trend, but neither comparison supplied a positive
signal.

Depth seven averaged `32.195706020 s`, a `3.758651%` runtime regression and
`3.622494%` throughput-equivalent loss. Its first sample was `8.850350%` slower
than the first six-depth endpoint; its second was `1.537424%` faster than the
final six-depth endpoint. The split sign plus negative aggregate rejects seven.

## Interpretation

Five is the clearest result. Shortening the queue does not relieve enough
contention to compensate for earlier joins, so the promoted window is not
simply one slot too deep on the M1 Pro.

Seven shows why window-join totals cannot be read as removable wall time. A
deeper queue can postpone a particular join while increasing allocator churn,
page faults, Rayon competition, or Metal queue depth elsewhere. Its very slow
first sample and fast second sample also show sensitivity to machine state.
The negative mean is consistent with the promoted comment that depth eight
collapses from allocation/fault churn rather than exhausting physical memory.
That mechanism can transfer to a 48 GB M4 because more capacity does not make
dirty-page clearing or shared-queue contention free.

The result does not reject adaptive scheduling in principle. A phase-aware
policy could keep a smaller window while the heavy path is active and deepen
only after measured queue or chain state indicates spare capacity. Such a
policy needs a portable structural signal and should not react to noisy timing,
GPU occupancy sampling, or host memory size alone. The current trace says the
fixed cap is active; this experiment says changing the cap globally is not the
solution.

## Decision

**Retain depth six; reject depths five and seven without source change,
confirmation, or submission.** All proofs verified, but five lost both
pairings and seven regressed by mean with split pairings. No candidate commit
exists because the promoted runtime selector was sufficient for the complete
experiment.

Do not test another static integer. The next scheduling candidate should alter
when capacity is made available based on deterministic proof-stage state, and
must preserve the six-depth behavior as its same-binary control.

Published Yukon progress note:
`f36ea396-3c8e-495a-8eb0-943964650119`.
