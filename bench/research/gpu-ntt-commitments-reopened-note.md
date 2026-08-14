# GPU NTT commitments: void evidence, correct conclusion

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`) plus the
committed `CosetInterpolationGate` change. Local host Apple M1 Pro, 32 GB.

## Why this was reopened

`vendor/plonky2/plonky2/src/fri/oracle.rs` disabled whole-commitment GPU NTT
with a single cited datum: official submission `644c4257` scored
`6.2323 tx/s` against an `8.0011` frontier, "despite a `+4.6%` controlled local
win". That flag gates the only mechanism capable of moving the `41.5%` of CPU
compute that FFT/LDE occupies ([[cpu-decomposition-note]]), so the citation was
worth auditing before accepting it.

## The citation does not survive

`644c4257` was submitted by solver `i34-9` at 8/5/26 7:17 PM. Pulling every
submission in that 55-minute window:

| | |
|---|---|
| Submissions in window | 28, from many different solvers |
| Range | `5.487` - `7.972` (**`45.3%` spread**) |
| Median / sigma | `6.873` / `0.856` |
| `644c4257` at `6.2323` | **`-0.75 sigma`** from the median |
| Contemporaries scoring lower | **`29%`** (8 of 28) |
| Same solver's next submission, 23 min later | **`7.7236` (`+23.9%`)** |

Every one of those 28 was rejected against the `8.0011` frontier. The score
attributed to the mechanism is an unremarkable draw from a band `45%` wide, and
the same solver drew `+23.9%` higher twenty-three minutes later with no claimed
change. **A single official draw in that band cannot distinguish a mechanism
from noise**, so the recorded `-22%` was never evidence.

This is the general hazard the campaign already documented from the other
direction: exakoss's note reports byte-identical redraws landing anywhere in
`25.7`-`30.5 tx/s`.

## Re-tested properly

Made the flag a same-binary selector (`PLONKY2_GPU_NTT_COMMITMENTS=1`), then
ran the protected sequence on the current pipeline. Trusted verifier passed on
every run including the candidate-default gate.

| Run | Arm | Proving time |
|---:|:---:|---:|
| 1 | B, flag off | `39.403937 s` |
| 2 | C, GPU NTT on | `44.613745 s` |
| 3 | C, GPU NTT on | `41.774684 s` |
| 4 | B, flag off | `39.213543 s` |

Control mean `39.308740 s`, candidate mean `43.194215 s`: candidate
**`+9.88%` runtime**, throughput `-9.00%`. Both pairings lost, `+13.22%` and
`+6.53%`. Control drift was `0.190 s` — the tightest control pair of this
entire campaign, so the signal is unusually well separated from noise.

## Conclusion: right answer, wrong reason

Keep `GPU_NTT_COMMITMENTS = false`. The mechanism is genuinely ~10% slower on
today's pipeline, established on tight controls rather than on a void citation.

The originally stated *mechanism* was also correct even though its evidence was
not: NTT stages extend each tree's exclusive occupancy of the serialized GPU
stream. That is even truer now than in the `8.0011` era, because the promoted
streamed path already overlaps CPU FFT with GPU hashing into `max(FFT, hash)` —
so moving the NTT onto the GPU stream lengthens the critical path rather than
shortening it, and simultaneously gives up the existing overlap.

The source comment has been rewritten to carry this evidence in place of the
void one, and the selector was reverted to a plain constant.

## What this changes for the campaign

Nothing about the frontier, but something about method. An in-repo comment
citing an official score as proof of a mechanism should be treated as a
hypothesis until the contemporaneous draw distribution is checked. Official
single draws span `45%` in the early era and roughly `16%` today
(`25`-`26` versus `29`-`30.5 tx/s` classes); any mechanism smaller than that is
invisible in one submission, in either direction.

Worth noting the reverse risk too: a promoted mechanism may owe its promotion
to a fast draw rather than to its content. The same audit applied to a
*promoted* submission would be equally informative.

## Related

- [[cpu-lde-versus-gpu-ntt-note]] — the isolated head-to-head that first surfaced this flag.
- [[cpu-decomposition-note]] — the `41.5%` FFT figure that made it worth auditing.
- [[fft-roofline-note]] — why the CPU side cannot absorb the difference either.
