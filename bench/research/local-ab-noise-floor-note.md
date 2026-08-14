# The local end-to-end A/B noise floor

Host Apple M1 Pro, 32 GB, ~1.8 cores of desktop background load (apps not
closed; this is the state the campaign actually works in, not an idle host).
Same binary `ed5eec1ebc0cbb93`, identical control configuration on every run,
one warmup discarded, trusted verifier passing throughout.

## Seven identical control runs

```
31.3220  31.6130  31.7305  31.7339  32.2437  33.0235  33.5765
```

mean `32.1776 s`, sd `0.7676 s` (**`2.39%`**), spread `2.255 s` (**`7.20%`**).

An earlier six-sample control set in the same session gave sd `1.26%` and
spread `3.39%`. The floor therefore **moves between sessions by roughly 2x**:
the noise is non-stationary, so a single estimate of it is itself unreliable.

## What a B-C-C-B can fabricate

Drawing four of the seven at random and splitting them 2/2 — pure noise, no
mechanism — the apparent effect is:

| Percentile | Apparent effect |
|---|---:|
| 50th | `1.85%` |
| 90th | `4.26%` |
| 95th | **`4.87%`** |

Gaussian resampling at the measured sd agrees and shows what more samples buy:

| Samples per arm | 95th pct false effect |
|---|---:|
| 2 (the standard protocol) | `4.65%` |
| 4 | `3.32%` |
| 8 | `2.34%` |
| 16 | `1.64%` |

## Verdict

**The standard four-sample `B-C-C-B` cannot resolve anything below ~`4.9%` on
this host.** Sixteen samples per arm — 32 proving runs, roughly 17 minutes of
machine time per candidate — still only reaches `1.6%`.

Every mechanism this campaign has measured falls between `0.02%` and `3%`. None
of them is resolvable by local end-to-end A/B, in either direction. This
supersedes the bounded claim in [[fri-tree-retirement-retest-note]] with a
direct control-only measurement rather than an inference from mixed samples.

Two consequences:

1. **Retire local end-to-end A/B as the accept/reject instrument.** Its outputs
   for sub-5% mechanisms are noise regardless of how carefully the arms are
   alternated. Historical ledger entries resting on 4-sample B-C-C-B results
   should be read as unmeasured, not as evidence in either direction.
2. **Isolated station harnesses are the only working instrument.** The
   coset-interpolation A/B (`1.107x`, non-overlapping ranges), the Merkle
   dispatch split, and the FFT roofline each resolved their question in seconds
   because they time one kernel rather than a 30-second pipeline containing
   GPU waits, allocator behaviour and OS scheduling.

Not established: the floor on a genuinely idle host. Apps were not closed for
this measurement. An idle host would plausibly be better, but the between-session
variation observed here (`1.26%` -> `2.39%` sd) suggests background load is not
the only contributor.
