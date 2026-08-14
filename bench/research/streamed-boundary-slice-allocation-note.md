# Streamed Metal boundary slice-allocation audit

## Attribution and decision

This experiment was designed and evaluated with **GPT 5.6 Sol** at maximum
reasoning effort through Codex. It started from research HEAD `9172492`; the
tracked proving source at that point was identical to promoted submission #138
(`e268c13`, official `29.9399105848455 tx/s`). The candidate was rejected and
fully reverted. No official submission was created.

The terminal decision is **do not replace the streamed Merkle builder's tiny
per-group slice-descriptor `Vec` with fixed stack storage**. The change removed
22 heap allocations and 2,752 descriptor bytes per public worker, but a locked,
same-binary `B-C-C-B` lost both pairings and regressed mean runtime by
`8.780076%`. More importantly, the ownership trace showed that promotion #138
has already removed the large payload copies that originally motivated this
search. The remaining descriptor allocation is not a proxy for the reported
wire-commit/Merkle work.

## Baseline and moving official frontier

The investigation was intentionally based on #138 because that was the
delegated source baseline. Its strongest relevant achievements are already in
the audited path:

- non-routed witness columns move into the wire IFFT rather than clone;
- routed columns use an out-of-place bit-reversal gather that directly creates
  the required coefficient vector, avoiding a preliminary clone;
- coefficient scaling writes directly into the final retained shared Metal
  column store;
- the Metal Merkle builder binds that same shared buffer as its leaf source,
  without a CPU-to-GPU staging upload or a row-major transpose;
- the exclusive final-proof path overlaps CPU production of eight-column LDE
  groups with Metal absorption of the previous group.

After the terminal local result, `yukon submissions --all` showed that the
official frontier had advanced to promoted submission `5a25029` at
`30.3111567697189 tx/s`. Its public note attributes the new gain primarily to
opening the early light-proof ramp to depth two, while preserving the inherited
Metal, FFT, witness, prewarm, readback, and fixed-window mechanisms. It also
reports a jointly saturated steady pipeline: about `89.6%` GPU busy and the
single Metal buffer set occupied about `95.9%` of the time. This experiment did
not sync or claim to test that later scheduler. The correct comparison is:

- source semantics tested here: #138 / `e268c13`;
- current official baseline after refresh: `5a25029`, `30.3111567697189 tx/s`;
- conclusion likely to transfer: removing a few host allocator calls around
  the final streamed groups is far below the payload and serialized-GPU costs
  identified by both traces.

## Motivation and candidate list

The supplied diagnostic totals contained roughly `35.253 s` of aggregate
`compute wires commitment` work and `53.604 s` of aggregate `build Merkle tree`
work. Those are overlapping work totals, not removable wall-clock ceilings,
but they justified a value-exact boundary audit. Before editing, four candidates
were inspected in order:

1. avoid an intermediate CPU allocation or copy between the wire IFFT values
   and the Metal column store;
2. move a uniquely owned coefficient or LDE buffer into the Merkle batch rather
   than clone or repack it;
3. fuse the leaf upload or column-layout conversion while preserving byte order;
4. delete a redundant staging allocation or transpose after proving its last
   reader and owner.

The first three were ruled out by source and protocol ownership, not by
speculation. The fourth exposed one real but very small allocation and became
the isolated candidate.

## Exact ownership and dataflow

`MatrixWitness` owns 136 column-major witness vectors. There are 80 routed
columns. The wire polynomial conversion has two cases:

- non-routed columns have no later witness reader, so `mem::take` transfers
  their allocation into the in-place IFFT;
- routed columns are still required after the wire cap has been observed, when
  transcript-derived beta/gamma challenges drive the permutation partial
  products. `ifft_borrowed` therefore must create a second buffer. It gathers
  directly into bit-reversed order and continues the IFFT there; it does not
  clone and then bit-reverse.

That routed allocation cannot be moved away without violating a simultaneous
ownership requirement. The wire commitment must exist before the challenger
can derive beta/gamma, but the original routed evaluations must remain alive
until those challenges are used. Delaying the IFFT would make the wire cap
depend on challenges that themselves depend on the cap. Moving the coefficient
buffer into Metal also cannot work: later opening construction reads the
coefficients while the Merkle leaves require a distinct, rate-eight LDE whose
in-place FFT overwrites its coefficient prefix.

`PolynomialBatch::from_coeffs` then allocates the final shared Metal column
store. `batch_multiply_into` reads each retained coefficient vector and writes
the scaled values directly into the destination prefix, followed by the
zero-padded FFT in that same Metal-backed column. There is no intermediate LDE
`Vec`. `MerkleTree::new_column_store` passes `ColumnStore::Shared` to the
Poseidon2 Metal backend, which binds the retained buffer directly. There is no
payload upload, clone, repack, or transpose on the successful Metal path.

For the final streamed path, the only surviving boundary allocation was:

```rust
let mut slices: Vec<&mut [F]> = (0..chunk)
    .map(|k| /* disjoint column slice */)
    .collect();
fill_group(group, &mut slices);
```

The vector contains only fat slice descriptors; it never contains or copies
field elements.

## Shape and byte accounting

The diagnostic worker contained 106 proofs:

| Degree | Proof calls | Wires | Routed wires | Rate |
|---:|---:|---:|---:|---:|
| `2^14` | 53 | 136 | 80 | 8 |
| `2^16` | 52 | 136 | 80 | 8 |
| `2^18` | 1 | 136 | 80 | 8 |

Across those calls, wire coefficient vectors cover `617,218,048` field
elements, or about `4.60 GiB` of aggregate allocation traffic. The routed
simultaneous buffers cover `363,069,440` elements, about `2.71 GiB`. The final
wire LDE stores cover eight times the coefficient elements:
`39,501,955,072` bytes, about `36.79 GiB`, written across the worker.

The same trace contained 324 `build Merkle tree` spans. Streaming was admitted
only for the final degree-18 proof's three wide commitments. Those commitments
contained 172 columns total and therefore 22 actual eight-column absorb groups
(21 full groups and one four-column group). On a 64-bit target each `&mut [F]`
is a 16-byte fat pointer, so the selected change removed only:

```text
172 descriptors * 16 bytes = 2,752 bytes
22 short-lived heap allocations
0 field-element payload bytes
```

This accounting was the key scope result. Even a perfect allocator saving here
cannot explain seconds of commitment and Merkle work.

## Isolated implementation

The experimental patch used `[MaybeUninit<&mut [F]>; 8]`, initialized exactly
the leading `chunk` slots, then exposed only that initialized prefix to the
existing fill closure. The candidate kept column order, group size, fill
closure, command-buffer submission order, Metal bindings, parent hashing, pool
policy, and retained memory unchanged.

`PLONKY2_STREAMED_STACK_SLICES=0` selected the exact #138 heap-vector body;
every other value selected stack descriptors. The selector was read once per
worker through a `LazyLock`, outside the group loop. Both arms therefore used
one release executable with SHA-256:

`a7a83b077d2ca435f68e47da64e7ed917ccfa7716b743262c44f0cf3c67686af`

## Correctness and build gates

The following checks were completed:

```text
cargo check --manifest-path vendor/plonky2/plonky2/Cargo.toml --features parallel
cargo test --manifest-path vendor/plonky2/plonky2/Cargo.toml --features parallel \
  streamed_merkle_keeps_digests_resident_and_matches_classic -- --nocapture
./setup.sh
./benchmark.sh
```

The first focused Metal test attempted inside the default sandbox could not
create a Metal context. Repeating the exact locked test outside the nested
Seatbelt sandbox passed. The candidate arm and the environment-selected heap
control each matched the classic shared-column tree's complete level-order
digest store and cap. Equality of every node fixes the Merkle root and every
authentication path. The four protected full proofs then passed the pinned
trusted verifier under `lighter-mixed-block-proof-v1`, supplying the full proof
differential.

The repository-wide formatter check was not used as a patch gate because the
checkout already contains broad unrelated rustfmt drift; it made no edits.
The candidate file passed `git diff --check`, and all source changes were
reverted after the performance decision.

## Controlled B-C-C-B result

Local hardware was an Apple M1 Pro with 32 GB unified memory. The official
runner is an Apple M4 Pro with 48 GB. The predeclared order was `B-C-C-B`, where
B set `PLONKY2_STREAMED_STACK_SLICES=0` and C used stack storage. All four runs
atomically held the shared heavy-command lock for the full sequence.

| Order | Arm | Trusted proving seconds | Verification |
|---:|:---:|---:|:---:|
| 1 | B, heap control | `36.380620708` | passed |
| 2 | C, stack candidate | `40.062462042` | passed |
| 3 | C, stack candidate | `33.563807417` | passed |
| 4 | B, heap control | `31.302977125` | passed |

Control mean: `33.841798917 s`.

Candidate mean: `36.813134729 s`.

Candidate runtime delta: `+8.780076%`.

Throughput-equivalent delta: `-8.071401%`.

The candidate lost the first mirrored pairing by `10.120337%` and the second by
`7.222413%`. The endpoint movement confirms meaningful host variance, but both
pair signs and the aggregate cross the rejection rule. There was no lost
pairing and no proof failure to quarantine.

## Interpretation and next work

This result should not be read as “heap allocation is faster than stack
storage” in general. The changed work is only tens of allocator operations,
whereas every sample contains tens of gigabytes of LDE traffic and a serialized
Metal queue. The correct conclusion is that the candidate is too small to
attribute, and the observed aggregate is decisively non-positive.

The valuable outcome is the negative ownership map:

- do not revisit routed-wire `ifft_borrowed` as a removable clone; the second
  buffer is protocol-required and the clone pass is already fused away;
- do not add a `bytesNoCopy` or coefficient-to-Metal ownership trick that loses
  coefficients needed by openings or increases retained memory;
- do not optimize `Vec<&mut [F]>` or Objective-C buffer-handle clones as if
  they were payload copies;
- do not resurrect exact-bin pooling plus initial-tree retirement; its combined
  official result was already negative on M4.

A future boundary experiment needs payload-scale evidence: decouple wire LDE
fill from absorb granularity, fuse an actual kernel pass, or remove a measured
field-element staging buffer in a fallback shape. The newly promoted frontier
note independently points to wires-commit fill/absorb granularity and the
range-U32 kernel's repeated LDE reads as larger targets. Those should be tested
one at a time on the current promoted source, with exact roots/proofs and an
official-host-aware admission argument.

## Final state

The experimental Metal source is reverted to #138 behavior. Only this research
record and the three standard ledgers remain. No submission was made because
the candidate lost both pairings and regressed its aggregate.
