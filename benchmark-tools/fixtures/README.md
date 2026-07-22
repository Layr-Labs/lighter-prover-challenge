# Trusted fixtures

`bench.json` is the repository's existing 500-transaction
`bench/bench_test.json` fixture, moved outside the editable surface. The trusted
CPU verifier validates it against the circuit revision pinned by
`benchmark-tools/harness/Cargo.toml` on every run.

Keep ranked fixtures outside the editable paths. A rotating private pool is
preferred for future benchmark versions.
