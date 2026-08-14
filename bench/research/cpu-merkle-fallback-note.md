# CPU Merkle fallback: 15.5% of compute, and it corrects an earlier verdict

Host Apple M1 Pro. Source: the `sample` profile behind
[[cpu-decomposition-note]] (409,039 leaf samples), re-attributed by call graph.

## Where CPU Poseidon2 comes from

CPU-side Poseidon2 is `25.5%` of prover compute. Attributing its 45,350
samples to callers:

| Caller | Samples | Share |
|---|---:|---:|
| `hash_or_noop` | 26,253 | `57.9%` |
| merkle | 8,356 | `18.4%` |
| permutation | 4,552 | `10.0%` |
| fri | 4,070 | `9.0%` |
| generator | 1,501 | `3.3%` |

Both `hash_or_noop` callers resolve to `merkle_tree::fill_subtree_flat`, in two
monomorphizations (`circuit` 10,995, `prove` 6,586). That is **CPU Merkle tree
building** — `~15.5%` of prover compute — while the GPU is `49.2%` idle and
hashes at a measured `11.2 ns` per permutation.

## Why it runs on the CPU

`fill_subtree_flat` sits behind the comment `// CPU fallback` in
`merkle_tree.rs:815`. It is reached only when
`H::try_build_merkle_tree_column_store` returns `None` — that is, when the
Metal backend **declines** the tree.

## This corrects the buffer-set verdict

[[cpu-decomposition-note]] recorded that Metal buffer-set acquisition blocks on
26 samples out of 409,039, and concluded that buffer-set widening was refuted
because CPU threads essentially never wait on it.

That inference was wrong. **Threads do not block on the buffer set because the
code declines rather than waits**: when no set is available the GPU path
returns `None` and the caller silently builds the tree on the CPU. The absence
of blocking is therefore evidence of *fallback*, not of *sufficiency* — the two
look identical in a wait profile and are opposite in meaning.

So the cost of buffer-set scarcity is not visible as blocked threads. It is
visible as `~15.5%` of compute spent doing on the CPU what the GPU does at
`11.2 ns`/permutation.

## Why this is promising, and why it may still fail

Promising: the largest remaining CPU consumer after FFT, it is *duplicated
capability* rather than irreducible work, and the resource it needs is measured
idle.

Cautions, all earned this week:

- three Metal buffer sets "regressed badly" historically, attributed to memory
  pressure; this host has 32 GB against the ranked 48 GB;
- adding work to the serialized GPU stream has now failed twice on direct
  measurement (`GPU_NTT_COMMITMENTS` at `+9.88%`, exclusive D14 wire NTT at
  `+3.14%`), and the stated mechanism each time was exclusive-stream occupancy;
- the declined trees may be small ones where the measured `~104 us` dispatch
  floor exceeds the CPU cost, in which case declining is correct.

## Next measurement, not next mechanism

Instrument the decline: count `try_build_merkle_tree_column_store` calls that
return `None`, with the leaf count and column width of each. That says whether
the declines are large trees worth moving (a real opportunity) or small ones
below the dispatch floor (correct behaviour, and the angle closes). It needs
one counter and one traced run.

Do not add buffer sets before that count exists.
