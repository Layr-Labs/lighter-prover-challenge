# Shape-preserving Metal column-store buffer reuse

## Context and attribution

This is a progress and candidate-submission note for the Lighter Prover
Challenge. The implementation and experiment used **GPT 5.6 Sol**, effort
**max**, through Codex. The local development machine is an Apple M1 Pro with
32 GB unified memory; the official runner is an Apple M4 Pro Mac mini with
48 GB. All local wall times below use the checked-in public synthetic fixture,
so they are screening evidence rather than official throughput scores.

The base is promoted frontier commit `59c0155`, promotion 137, with official
score `29.8785698468374 tx/s`. The working branch also contains research-log
commits only; the production delta described here is confined to
`vendor/plonky2/plonky2/src/hash/poseidon2/metal.rs`.

## Hypothesis

The promoted column-store pool uses smallest-fitting best-fit reuse. That is
normally a sensible allocator policy, but this workload has a small fixed set
of recurring commitment shapes. A 20 MiB request can borrow an idle 64 MiB
Metal shared buffer. If a 64 MiB commitment arrives before the borrower drops,
the pool no longer has that exact recurring shape and calls `new_buffer` again,
paying shared-page allocation and first-touch costs. The larger buffer was not
too small; it was temporarily assigned to the wrong future-use class.

The expected opportunity was deliberately narrow:

- retain exact-size bins for recurring stores below 256 MiB;
- retain best-fit for requests at or above 256 MiB, where reuse happens at
  startup/final edges or among the largest shapes and avoids one-off terminal
  allocations;
- preserve the existing 640 MiB per-buffer cap, 2.5 GiB total cap, nonblocking
  `try_lock`, lease lifetime, and fully-written-before-read invariant;
- provide `PLONKY2_METAL_COLUMN_BEST_FIT=1` as a same-binary local control.

This changes only which already-valid `MTLBuffer` is selected. It does not
change field values, Merkle leaves, digests, transcript order, proof layout,
thread count, or GPU command ordering.

## Counter-first diagnosis

Temporary code behind the existing `diagnostic_profile` feature recorded each
column-pool request, hit, miss, selected-buffer length, oversize delta, recycle,
and free-byte total. These counters were removed before the scored release
build and are not part of the candidate production diff.

One historical-best-fit public worker recorded:

| Metric | Best-fit control |
|---|---:|
| Requests | 281 |
| Hits | 255 |
| Misses | 26 |
| Bytes requested by misses | 10,716,446,720 |
| Oversized hits | 36 |
| Total excess mapped by oversized hits | 1,929,379,840 bytes |
| Recurring 544 MiB misses | 9 |

Thirty-one of the oversized loans repeatedly mapped the 20 MiB request onto a
64 MiB buffer, wasting 44 MiB of mapping each time and removing a 64 MiB buffer
from its future bin. Other early loans consumed 136 MiB for an 80 MiB request.
The trace still showed later 64, 80, and 544 MiB misses after warm-up, so this
was not merely the unavoidable first allocation of every shape.

## Course correction: why pure exact reuse was not kept

The first candidate required exact matches for every cacheable request. It
eliminated all oversized loans and reduced 544 MiB misses from nine to six, but
it also forced new 256 and 320 MiB allocations during the final proof. Under
best-fit those terminal requests can harmlessly consume idle 352 and 544 MiB
buffers because the recurring transaction/chain phase is already over. The
pure policy therefore increased total miss count from 26 to 27 and slightly
increased local system time. It was rejected before release A/B.

The revised policy preserves exact bins only below 256 MiB and allows best-fit
at or above that threshold. Its diagnostic worker recorded:

| Metric | Best-fit control | Hybrid candidate |
|---|---:|---:|
| Requests | 281 | 280 |
| Hits | 255 | 254 |
| Misses | 26 | 26 |
| Bytes requested by misses | 10,716,446,720 | 9,133,096,960 |
| Excess bytes in oversized hits | 1,929,379,840 | 335,544,320 |
| Recurring 544 MiB misses | 9 | 6 |
| Terminal 256/320 MiB misses | 0 | 0 |

The one-run request count can vary slightly with concurrent pool lock timing,
so the evidence is the size distribution, not the count alone. The hybrid
removed about 1.58 GB of newly allocated shared-buffer demand while retaining
the useful large terminal loans. Diagnostic wall time was intentionally not
used as the keep criterion because profiling and local thermal state add noise.

## Correctness and build gates

Commands were run with the pinned toolchain, offline dependencies, and native
Apple target features:

```text
RUSTFLAGS='-C target-cpu=native' CARGO_NET_OFFLINE=true \
  cargo check --locked --offline --workspace --all-targets

RUSTFLAGS='-C target-cpu=native' CARGO_NET_OFFLINE=true \
  cargo build --release --locked --offline -p bench --bin prove

./benchmark.sh
```

The workspace-wide check passed. The release build passed, producing worker
SHA-256
`20a1e68f5a22c7057d32395e795a91854bf88d23b3a60ac641c3f0d2cc52e4e9`.
The pinned trusted verifier accepted the exact default-candidate binary: one
proof verified out of one expected proof. Public synthetic proving time was
36.121446333 seconds (`13.84219212571252` provisional tx/s); this score is not
compared with the private-active leaderboard.

## Same-binary release A/B

The predeclared order was `B-C-C-B / C-B-B-C`. Both arms used the exact same
release binary and public fixture. `B` set
`PLONKY2_METAL_COLUMN_BEST_FIT=1`, restoring the promoted smallest-fitting
policy. `C` left the environment unset, selecting the hybrid default.

| Sample | Policy | Real seconds | User seconds | Sys seconds |
|---:|---|---:|---:|---:|
| 1 | B control | 35.65 | 170.92 | 17.45 |
| 2 | C candidate | 32.77 | 174.69 | 15.48 |
| 3 | C candidate | 30.73 | 170.78 | 14.71 |
| 4 | B control | 29.96 | 174.79 | 12.43 |
| 5 | C candidate | 34.83 | 184.48 | 17.38 |
| 6 | B control | 35.95 | 201.81 | 16.34 |
| 7 | B control | 34.77 | 198.95 | 15.46 |
| 8 | C candidate | 35.03 | 209.17 | 15.51 |

Control mean was `34.0825 s` and candidate mean was `33.3400 s`, a `2.1785%`
runtime reduction (`2.2271%` throughput-equivalent improvement). Control
median was `35.2100 s` and candidate median was `33.8000 s`, a `4.0045%`
runtime reduction (`4.1716%` throughput-equivalent improvement).

Pairwise signs split 2-2. This is an important caveat: local wall time has
large thermal and scheduling drift, and the experiment is not considered a
proven four-percent win. It is retained as a submission candidate because both
aggregate statistics are positive, the exact binary verifies, and independent
counters demonstrate deletion of about 1.58 GB of fresh shared allocations.
The M4 Pro has more unified memory and a different GPU, but the underlying page
fault and shape-cannibalization mechanism is architecture-stable.

## Production implementation

`ColumnStorePool::take_best_fit` now accepts `allow_oversize`. Exact matches are
always eligible. Oversized matches are eligible only when the historical
control is enabled or the request is at least 256 MiB. Selection remains
smallest-fitting among eligible buffers, and accounting still subtracts the
actual selected `Buffer::length()`.

The environment switch is intentionally narrow and defaults to the candidate.
It exists so the released executable can reproduce the frontier behavior
without a second build. The official runner does not set it.

## Risks and next steps

The main risk is pool-cap pressure: exact small bins may retain a small buffer
that best-fit could have avoided allocating. The hybrid threshold mitigates
this by preserving large flexibility, and the diagnostic trace showed lower,
not higher, newly allocated bytes. Another risk is workload specificity; the
threshold is justified by this fixed circuit suite's observed request sizes,
not presented as a general-purpose allocator rule.

Before any promotion claim, refresh the live frontier and inspect the official
result. If rejected, record the full submission ID, score, delta, and status in
`experiment-results.tsv`, `optimization-roadmap.md`, and
`promoted-options.md`; do not reinterpret the public smoke as ranked evidence.
If promoted, retain the same records and categorize the result under Memory,
buffers, copies and Allocators.
