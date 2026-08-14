# Audit of our own official rejections: two verdicts are void

Base: leaderboard state 2026-08-13, frontier #145
(`a955674` / `7477de7`, `30.7380852237325 tx/s`). 4,887 submissions parsed.

Same method as [[gpu-ntt-commitments-reopened-note]] and
[[promotion-draw-audit-note]]: for each of our officially rejected submissions,
compare its score against contemporaneous fast-class draws (`> 28 tx/s`) within
`+/-3 h`, rather than against the frontier alone.

## The systematic error

A submission's official delta is computed against **the frontier**, which
[[promotion-draw-audit-note]] shows is always a top-`0`-`3%` draw. Our
submission gets *one* draw from the same band. So the recorded "delta" is
routinely a median draw measured against someone else's best-of-33 draw, and
carries no information about the mechanism unless the effect exceeds the band
width (`~2%` intra-class, `16%` between classes).

## Results

| Submission | Mechanism | Local evidence | Official | Position in fast band | Verdict |
|---|---|---|---:|---|---|
| `9345ccf` | exact-size Metal reuse + FRI initial-tree retirement | `-2.94%` runtime, **both pairings won**, verified 5/5 | `29.3561` | **53rd pct** (median `29.3072`) | **VOID** |
| `1d69b9c` | hybrid exact-bin Metal column-store pool | `-4.00%` median runtime, pairings split 2-2 | `29.4067` | **72nd pct** (median `29.1711`) | **VOID** |
| `7244133` | finished prover-state destructor elision | `-0.82%` mean, pairings 2-2 | `25.8191` | **slow-class draw**; 0 of 36 fast draws below it | **VOID** (immaterial) |
| `7b48073` | dense transaction scatter | — | `28.6955` | 40th pct | weakly negative |
| `f00ec67` | pre-resolved tx seed metadata | — | `26.6686` | 100% of window slow-class | uninformative |
| `c66e726` | earlier candidate | — | `18.6306` | 100% of window slow-class | uninformative |

### `9345ccf` is the important one

Its local evidence is the strongest in the entire ledger: `-2.94%` runtime
(`~+3.03%` throughput), **both** `B-C-C-B` pairings won, trusted verification
5/5. It was reverted in `029af57` solely because its official score
(`29.3561`) fell below the then-frontier (`29.9399`).

But `29.3561` is the **53rd percentile** of its own fast-class band, whose
median was `29.3072`. Our draw was an ordinary one; the frontier it lost to was
a top-of-band draw. The comparison measured draw luck, not the mechanism.

A `+3%` mechanism is roughly thirty times larger than anything this week's
campaign produced, and it was discarded on a coin flip.

### `1d69b9c`

Our draw sat at the **72nd percentile** — better than `72%` of fast-class
contemporaries — and was still recorded as a `-0.47 tx/s` regression. Its local
evidence is weaker (pairings split 2-2), so it is a lower-priority revisit, but
the official refutation is equally void.

### `7244133`

The draw was outright slow-class (`25.8191`, with zero of 36 fast contemporaries
below it), so the recorded `-4.06 tx/s` is a runner-class artifact. The roadmap
already suspected this ("landed in a broad low-service band and is inconclusive
about a `~0.26%` effect") but reverted anyway on the predeclared rule. Correct
process, void evidence — and immaterial either way at a `0.26%` ceiling.

## Recommended action

**Restore and retest `82635d8` (the `9345ccf` mechanism) on the #145 frontier.**
It is the single highest-value item on the board: strong local evidence, a void
rejection, and a magnitude that would actually move the band median rather than
the fourth decimal.

Caveats to hold while doing it:

- It was measured on the `#137`/`#138`-era pipeline. Column-store pooling,
  streaming, ramp depth two and the buffer-set policy have all changed since,
  and the mechanism interacts with exactly those. The local win may not
  reproduce.
- It bundles two changes — exact-size reuse below 256 MiB, and ownership
  retirement of the three per-proof initial FRI trees. The roadmap notes
  neither component received isolated credit. Test them separately; a
  two-mechanism candidate that loses tells you nothing about either half.
- Our M1 disagrees with the ranked M4 about which station is critical
  ([[gpu-ntt-commitments-reopened-note]]), so a local win remains provisional.

## Method note for the ledger

Every future official result should be recorded with its **contemporaneous
fast-class band position**, not only its delta against the frontier. The delta
against a top-`3%` frontier draw is uninformative for anything under `~2%`. The
`experiment-results.tsv` schema should carry a band-percentile column.

## Related

- [[promotion-draw-audit-note]] — the same audit applied to promotions.
- [[gpu-ntt-commitments-reopened-note]] — the same audit applied to an in-repo citation.
