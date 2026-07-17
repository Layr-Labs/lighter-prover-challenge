# Test data

- `bench/bench_test.json` is the checked-in public prover fixture.
- `prover/public/bench_test_32.json` is its validated 32-transaction prefix.
- `lighter-api/public/` contains normalized real Lighter block responses.
- `lighter-api/private/` contains local samples and is ignored by Git.

The API responses are not prover fixtures: they do not contain the state and
Merkle witnesses required by the circuit. Complete private fixtures must come
from Lighter's prover pipeline and must pass the pinned baseline before use.
