# Retest of the FRI tree-retirement stack, and the protocol's noise floor

Base: promoted frontier #145 (`7477de7`, `30.7380852237325 tx/s`).
Local host Apple M1 Pro, 32 GB. Same-binary release worker SHA-256 prefix
`ed5eec1ebc0cbb93`, built from `82635d8` re-applied onto #145 (revert of
`029af57`, auto-merged with no conflicts).

## Why

[[rejected-submission-audit-note]] showed the official rejection of `9345ccf`
was statistically void — a 53rd-percentile draw scored against a top-of-band
frontier draw. Its local evidence was the strongest in the ledger (`-2.94%`
runtime, both `B-C-C-B` pairings won, verified 5/5), so a `~3%` mechanism
looked like it had been discarded on a coin flip.

The restored commit carries independent selectors, allowing the two bundled
mechanisms to be measured apart in one binary for the first time:

```
PLONKY2_METAL_COLUMN_BEST_FIT=1     -> old best-fit  (exact-size reuse OFF)
PLONKY2_RETAIN_INITIAL_FRI_TREES=1  -> old retain    (tree retirement OFF)
```

## Conditions

A first attempt was discarded outright: the host was at load average `13.36`
and a four-arm screen drifted `58.92 -> 42.43 s` on the control arm alone, with
a monotonic decrease across four different arms that was warmup, not ranking.
Background load was then measured at `~2.0 cores` of desktop applications.

The reported runs were taken after a discarded warmup run, alternating arms
within each pass, with trusted verification passing on every run.

**The machine was NOT quiet, and an earlier revision of this note wrongly said
it was.** No applications were closed. Background load was `~1.9`-`2.0 cores`
of desktop software throughout (WindowServer, Claude helpers, Dock helper,
Firefox, Chrome, Telegram, ChatGPT), and the load average at the start of the
reported sequence was `4.44 / 5.13 / 7.39`. What changed relative to the
discarded screen was only that the self-inflicted load from back-to-back
prover runs had decayed, plus the warmup discard and arm alternation.

## Result

| Arm | Candidate | Control | Runtime delta | Pairings won |
|---|---:|---:|---:|:---:|
| exact-size reuse only | `30.8584` | `30.4207` | `+1.439%` | 1/2 |
| tree retirement only | `30.4207` | `30.3971` | `+0.077%` | 1/2 |
| both, as submitted | `30.7763` | `30.7808` | `-0.015%` | 1/2 |

**All three arms are neutral, and every one splits its pairings 1-1.** The
`-2.94%` local win does not reproduce on the #145 pipeline.

That is unsurprising in hindsight. The mechanism was measured on the
`#137`/`#138`-era tree, and promotion #132 ("Column-store buffer pool: stop
re-faulting 720 MiB of Metal pages every chain step") attacks the same
problem area. Whatever headroom exact-size reuse and tree retirement were
recovering has since been captured by the promoted pooling and streaming work.

## The larger finding: this protocol cannot see 3%

Six control samples were collected across the three A/Bs — same binary, same
configuration, same session:

```
29.9316  30.1013  30.6139  30.6929  30.9098  30.9477
```

Mean `30.5329 s`, standard deviation `0.3860 s` (**`1.26%`**), full spread
`1.016 s` (**`3.39%`**) — measured with `~2` cores of background contention
present against a prover that wants `~6` of the 8 P-cores.

Drawing four of those six at random and splitting them into two "arms" — pure
noise, no mechanism — produces an apparent effect whose 95th percentile is
**`2.57%`**. A four-sample `B-C-C-B` on this host can fabricate a `2.6%` result
from nothing, and can produce 2/2 pairings by chance a substantial fraction of
the time.

The original claim for this mechanism was `-2.94%` with 2/2 pairings. That sits
essentially at the fabrication ceiling. **The local evidence was never
conclusive either.** The audit was right that the official rejection was void,
but wrong to infer that a real `~3%` mechanism had been lost — both the
rejection and the original acceptance were noise.

### Consequence for the ledger, and its limit

Most `B-C-C-B` results in `experiment-results.tsv` are four-sample and report
effects between `0.1%` and `3%`. On a host in *this* state none of them was
resolvable, which offers an explanation for the recurring local-win-then-
official-failure pattern that needs no appeal to M1-versus-M4 architecture
differences.

That claim is bounded, though, and an earlier revision of this note overstated
it. The `3.39%` spread was measured under `~2` cores of contention. A genuinely
idle host may have a materially lower floor, in which case sub-3% mechanisms
would be measurable here and the ledger's results would deserve individual
re-examination rather than blanket dismissal. **Establishing the idle-host
noise floor is a prerequisite for any strong claim about the ledger**, and it
has not been done: it needs a control-only run of 8+ samples with the desktop
applications closed.

Minimum protocol change for any future claim below `3%`:

- discard a warmup run explicitly;
- verify background load before starting (this session's `2.0` idle cores of
  desktop apps is material against a prover wanting `~6`);
- collect at least 6-8 samples per arm, not 2;
- report the control-only spread alongside the candidate delta, so the reader
  can see the noise floor next to the claimed effect;
- treat any effect under the control spread as unmeasured, not as a result.

Isolated station-level harnesses remain far better: the coset-interpolation and
Merkle-dispatch measurements resolved `1.5%`-`10%` differences with
non-overlapping sample ranges in seconds, because they time one kernel instead
of a 30-second pipeline.

## Decision

Keep `82635d8` reverted; the research line already excludes it. The restoration
is preserved on branch `restore/fri-tree-retirement` with its selectors intact,
in case a future pipeline change makes the mechanism relevant again.

## Related

- [[rejected-submission-audit-note]] — the audit that motivated this retest.
- [[promotion-draw-audit-note]] — why official single draws cannot resolve this either.
