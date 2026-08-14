# SIMD-native partial-prefix Range/U32 tile

## Decision

Rejected by the predeclared pairing rule and reverted. The 32-row tile made
every lane in an Apple SIMD group evaluate one quotient row and passed all
value/proof checks, but its same-binary aggregate was neutral-negative and the
reverse pairing lost 1.373715%.

## Access-set census and hypothesis

The candidate was built on the #140-equivalent research tree (`d7d481a`), whose
functional source is promotion #139. A production diagnostic census observed
105 Range/U32 Metal jobs in one public proof. Degree-16 and degree-18 circuits
advertised broad gate records whose maximum wire accesses were usually 114 to
136 columns: Equality 132, ByteDecomposition 123, RangeCheck16/32/48 135/136/125,
U32AddMany up to 136, U32Arithmetic 114, U32Subtraction 126, and Reducing 136.
Because coset selector filters are generally nonzero, every advertised record
is evaluated on every quotient row. Splitting “compact families” would therefore
add output accumulation passes without producing a useful narrow union.

The revised hypothesis was instead to cache the common prefix. It staged 32
quotient rows by the first 120 wire columns in 30,720 bytes of threadgroup
memory, then direct-loaded only columns 120 through 135 when used. Every one of
the 32 lanes loaded and evaluated its own row. This differs materially from the
rejected 16-row by 136-column design, where half of each 32-lane SIMD group
loaded data but did not evaluate a row.

## Isolated implementation

The existing shader body was instantiated over either a direct device-wire
accessor or a tiled accessor. The tiled kernel cooperatively loaded the 120
column prefix, issued one threadgroup barrier, and evaluated the unchanged gate
body. No selector, constraint, alpha order, output layout, retained allocation,
pool policy, or proof scheduling changed. `PLONKY2_RANGE_U32_TILE=0` selected the
exact scalar entry point in the same release executable; candidate mode was the
default.

The committed metallib was intentionally left stale for this rejected screen,
so both arms compiled the same edited MSL source at worker startup. This raised
absolute proving time into the high-30/low-40-second range but does not bias the
within-binary selector comparison.

## Correctness and build gates

- `cargo check -p plonky2` passed.
- RangeCheck, U32, and combined byte/quintic Metal/CPU quotient differentials
  passed separately in candidate and scalar-control modes.
- The combined Metal gate-quotient proof differential and verifier passed in
  candidate and scalar-control modes.
- Release worker SHA-256:
  `8d6dbb73aa8b02b235726d43a3ec33910b4b080b14087fa6b163ae7e88b4a1c6`.
- The candidate-default protected gate and all four alternating proofs passed
  the pinned trusted verifier: five of five.

## Protected B-C-C-B result

B set `PLONKY2_RANGE_U32_TILE=0`; C selected the 32-by-120 tile.

| Run | Arm | Proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B | 38.890966250 | passed |
| 2 | C | 38.428841041 | passed |
| 3 | C | 39.102838292 | passed |
| 4 | B | 38.572955916 | passed |

Control mean: `38.731961083 s`.

Candidate mean: `38.765839666 s`.

Candidate runtime delta: `+0.087469%`.

Throughput-equivalent delta: `-0.087393%`.

The first pairing favored the tile by `1.188258%`; the reverse pairing lost by
`1.373715%`. The aggregate is slightly negative and the signs split, so the
candidate fails without confirmation or official submission.

## Interpretation and frontier refresh

Making every SIMD lane productive removed the obvious structural defect in the
16-row predecessor, but a 30 KiB tile plus one barrier still does not beat the
M1 Pro's device-cache/direct-read path. Do not sweep nearby prefix widths or
row counts locally. Revisit only with M4 kernel counters showing threadgroup
hit rate, occupancy, and a stable Range/U32 queue-tail reduction.

The terminal Yukon refresh found promotion #141 (`cdee956`, commit `0a470b3`)
at `30.4758937588950 tx/s`. Its public note and exact two-line diff confirm it
is another marker-only redraw of the #139 functional executable, so it changes
the score frontier but adds no optimization-category hit. Seven newer
submissions were still validating at the refresh. No standalone Yukon note was
published for this local rejection.
