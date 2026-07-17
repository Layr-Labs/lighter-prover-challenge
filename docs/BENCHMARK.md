# Prover challenge

The goal is to produce proofs for the frozen Lighter circuits faster. Higher
scores are better:

```text
score = baseline time / candidate time
```

## What contestants edit

Only `challenge/submission/` is editable. The repository's Cargo files,
circuits, `bench/src/bin/bench.rs`, fixtures, scripts, workflows, and the rest
of `challenge/` are trusted inputs.

The challenge is a standalone Cargo package, so adding it does not change the
original workspace. It pins circuit commit
`5bbb307dfb26276c48054f2c3ea9dcfe80d3678a`, which is the revision that matches
`bench/bench_test.json`.

## Local use

```bash
./setup.sh
./benchmark.sh --local-iterate  # first 32 transactions
./benchmark.sh --local-submit   # all 500 transactions
```

The immutable `challenge/src/bin/bench.rs` binary starts and times the separate
`prove` process. Contestant code is linked only into `prove`, which must write
its pre-execution proof and final recursive proof to disk. After `prove` exits,
`bench` constructs fresh frozen verifier circuits, verifies both proofs, and
compares their public outputs with the fixture. A failed verification produces
no score. The same trusted binary computes and writes the local score, so there
is no separate score command or log parser. Local results are not ranked.

## Ranked use

The ranked workflow runs on the same `m5-bench` hosts as MLX Fast. It uses the
host's audited `/opt/bench/bench-exec.sh` bridge to run the frozen baseline and
the submitted prover as the unprivileged `bench` user. The runner-owned steps
validate the commit, compile it, and seal the paired score; they never execute
the submitted prover directly.

The workflow:

1. Reject changes outside `challenge/submission/`, fetch locked dependencies
   from the trusted checkout, then build offline with the pinned Rust toolchain.
2. Run the pinned baseline and candidate on the same machine and case.
3. Use the trusted `bench` timer, never candidate-reported timings.
4. Require the verified proofs and expected public outputs before writing
   `score.json`.
5. Clean the workspace and invoke the host janitor after every run.

The M5 hosts must have `cargo`, `rustup`, `jq`, and the toolchain named by
`rust-toolchain` installed. No Lighter-specific host executable is required.

For now, the ranked workflow uses the checked-in 500-transaction
`bench/bench_test.json` fixture as its only case. Replace it with the private
fixture pool once the complete prover exports are available.

Use a rotating pool of complete prover fixtures and a separate final holdout
pool. Randomly changing JSON fields is invalid because state roots and Merkle
paths are coupled.

## Block data

`testdata/lighter-api/public/` contains normalized real blocks from Lighter's
Explorer API. `testdata/lighter-api/private/` contains additional local samples
and is ignored by Git.

Explorer blocks are provenance data, not prover fixtures: the public API does
not include the state snapshot and Merkle witnesses required by the circuit.
New ranked fixtures must be exported from Lighter's prover pipeline and checked
with the pinned baseline before use.
