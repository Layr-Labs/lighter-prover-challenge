#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
provenance="${root}/benchmark-tools/upstream-381fd52.snapshot"
readonly PINNED_COMMIT=381fd529eb61dfff9ad245d94fce214a0a64d927

# shellcheck source=/dev/null
source "${provenance}"

[[ "${UPSTREAM_COMMIT}" == "${PINNED_COMMIT}" ]] || {
  echo "unexpected upstream commit: ${UPSTREAM_COMMIT}" >&2
  exit 1
}

temporary_index="$(mktemp "${TMPDIR:-/tmp}/lighter-upstream-index.XXXXXX")"
rm -f "${temporary_index}"
cleanup() {
  rm -f "${temporary_index}"
}
trap cleanup EXIT

(
  cd "${root}"
  GIT_INDEX_FILE="${temporary_index}" git read-tree --empty
  GIT_INDEX_FILE="${temporary_index}" git add -f -- circuit
)
root_tree="$(GIT_INDEX_FILE="${temporary_index}" git -C "${root}" write-tree)"
actual_circuit_tree="$(git -C "${root}" rev-parse "${root_tree}:circuit")"
actual_circuit_file_count="$(
  git -C "${root}" ls-tree -r --name-only "${actual_circuit_tree}" | wc -l | tr -d '[:space:]'
)"

[[ "${actual_circuit_tree}" == "${UPSTREAM_CIRCUIT_TREE}" ]] || {
  echo "circuit tree mismatch: expected ${UPSTREAM_CIRCUIT_TREE}, got ${actual_circuit_tree}" >&2
  exit 1
}
[[ "${actual_circuit_file_count}" == "${UPSTREAM_CIRCUIT_FILE_COUNT}" ]] || {
  echo "circuit file count mismatch: expected ${UPSTREAM_CIRCUIT_FILE_COUNT}, got ${actual_circuit_file_count}" >&2
  exit 1
}

for entry in "${UPSTREAM_FILES[@]}"; do
  read -r expected_blob expected_sha file <<<"${entry}"
  actual_blob="$(git -C "${root}" hash-object "${root}/${file}")"
  actual_sha="$(shasum -a 256 "${root}/${file}")"
  actual_sha="${actual_sha%% *}"

  [[ "${actual_blob}" == "${expected_blob}" ]] || {
    echo "${file} Git blob mismatch: expected ${expected_blob}, got ${actual_blob}" >&2
    exit 1
  }
  [[ "${actual_sha}" == "${expected_sha}" ]] || {
    echo "${file} SHA-256 mismatch: expected ${expected_sha}, got ${actual_sha}" >&2
    exit 1
  }
done

python3 - "${root}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
source = (root / "bench/src/bin/bench.rs").read_text()
makefile = (root / "bench/Makefile").read_text()

defaults = {
    "tx_count": 500,
    "heavy_tx_per_proof": 4,
    "light_tx_per_proof": 10,
}
for field, value in defaults.items():
    pattern = rf"#\[arg\(long,\s*default_value_t\s*=\s*{value}\)\]\s*{field}:\s*usize"
    if re.search(pattern, source) is None:
        raise SystemExit(f"bench binary default mismatch for {field}: expected {value}")

if re.search(r"(?m)^TX_COUNT \?= 30$", makefile) is None:
    raise SystemExit("bench Makefile TX_COUNT default mismatch: expected 30")

execution_files = [
    root / "benchmark.sh",
    root / "setup.sh",
    *sorted((root / ".github/scripts").glob("*.sh")),
    *sorted((root / ".github/workflows").glob("*.yml")),
]
bare_make_run = re.compile(r"\bmake(?:\s+-C\s+\S*bench)?\s+run\b")
for path in execution_files:
    if not path.exists():
        continue
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if bare_make_run.search(line) and "TX_COUNT=" not in line:
            raise SystemExit(f"{path.relative_to(root)}:{number} relies on bare make run")
PY

echo "PASS: pinned upstream snapshot, defaults, and execution contract"
