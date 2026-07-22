# Trusted fixtures

Ranked fixtures must be exported from the prover pipeline at the circuit
revision pinned by `benchmark-tools/harness/Cargo.toml`.

The historical `bench/bench_test.json` predates current `main`'s universal
cross-margin witness format and is not valid for this circuit. Before enabling
ranked runs, provide a complete current-main fixture here as
`bench-current-500.json` and validate it with the protected CPU verifier.

Keep ranked fixtures outside the editable paths. A rotating private pool is
preferred; the checked-in fixture is intended only for local correctness and
smoke testing.
