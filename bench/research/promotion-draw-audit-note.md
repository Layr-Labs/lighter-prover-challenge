# Promotion audit: functional promotions and redraws are statistically identical

Base: leaderboard state at 2026-08-13, frontier #145 (`a955674` / `7477de7`,
`30.7380852237325 tx/s`). 4,887 submissions parsed, 145 promoted.

## Why

The [[gpu-ntt-commitments-reopened-note]] audit showed that a *rejection* cited
as evidence was statistically void — one draw inside a `45%`-wide band. The same
question applies in reverse: does a *promotion* demonstrate that its mechanism
works, or only that it drew well?

## Method

For each promotion since 2026-08-11, take every other scored submission within
`+/-3 h` and keep the fast-class draws (`> 28 tx/s`), since the ranked host
serves two clearly separated classes (`25`-`26` and `29`-`30.7`). Then ask
where the promotion sits inside that fast-class band.

## Result

| Promotion | Solver | Kind | Score | Fast-class band | Position |
|---|---|---|---:|---|---|
| `4bfd557` #138 | FatihSolak | functional | `29.9399` | n=36, med `29.3072` | top `3%` |
| `5a25029` #139 | FatihSolak | functional | `30.3112` | n=35, med `29.3883` | top `3%` |
| `7b2d3a6` #140 | AlexLaevski | **redraw** | `30.4408` | n=38, med `29.5505` | top `8%` |
| `cdee956` #141 | AlexLaevski | **redraw** | `30.4759` | n=39, med `29.7829` | top `5%` |
| `c90e02f` #142 | AlexLaevski | **redraw** | `30.5343` | n=39, med `30.0097` | top `3%` |
| `9a57cdd` #143 | exakoss | functional | `30.6448` | n=35, med `30.0385` | top `3%` |
| `81fcf73` #144 | AlexLaevski | **redraw** | `30.6619` | n=33, med `30.1087` | top `0%` |
| `a955674` #145 | jungjipdo | — | `30.7381` | n=46, med `30.1505` | top `0%` |

**No signature separates functional promotions from marker-only redraws.** Every
promotion is the top `0`-`3%` of its contemporaneous fast-class band, and the
redraws sit *higher* positionally than the functional ones. A promotion records
a good draw; it does not, on its own, demonstrate a working mechanism.

## Where the mechanisms actually appear

In the drifting band median, not in the promoting draw:

```
29.3072 -> 29.3883 -> 29.5505 -> 29.7829 -> 30.0097 -> 30.0385 -> 30.1087 -> 30.1505
```

The whole field's fast-class floor rose `2.9%` over these eight promotions as
every solver synced to each new frontier. That drift is the real aggregate
effect of mechanism work. Individual promotions are draws off the top of a
rising floor.

## Consequences for strategy

At the last promotion the fast-class median was `30.1505` against a frontier of
`30.7381` — a promoting draw must land **`+1.95%` above the band median**, which
is historically the top `0`-`3%`. So roughly **one submission in ~33 promotes**,
for an executable sitting at the field median.

Our tree is the frontier plus the delayed-reduction `CosetInterpolationGate`
mechanism, worth `+0.09%`, which shifts our band median by `+0.027 tx/s`. That
is roughly `5%` of the gap that has to be crossed by luck. It does not
meaningfully change the odds.

Two honest readings follow, and they point the same way:

1. **Sub-percent mechanisms cannot be validated by submitting them.** The draw
   noise is `16%` between classes and the intra-class band is `~2%` wide;
   nothing this campaign produced (`0.02%`-`0.4%`) is observable in one
   submission, in either direction. Local isolated measurement is strictly
   better evidence than an official score for anything under ~2%.
2. **Promotion is a sampling problem, not only an optimization problem.** The
   solvers who promote most often (AlexLaevski: four of these eight) are the
   ones firing continuous redraws of an already-good executable.

## Caveats

The `+/-3 h` window and the `28 tx/s` class threshold are judgement calls; a
wider window mixes in field-wide drift, a narrower one loses sample size.
Results are stable under `+/-45 min` (checked) but the exact percentiles move by
a few points. The direction — that functional and redraw promotions are
indistinguishable — is not sensitive to either choice.

This does **not** say the promoted mechanisms are fake. #143's three mechanisms
are real, measurable code changes, and the rising band median is evidence that
such work compounds across the field. It says only that any *individual*
promotion is uninformative about its own mechanism.

## Related

- [[gpu-ntt-commitments-reopened-note]] — the same audit applied to a rejection.
- [[fft-roofline-note]] — the campaign state this strategy conclusion sits on top of.
