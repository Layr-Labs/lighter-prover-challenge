#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
candidate_root="${LIGHTER_CANDIDATE_ROOT:-${root}}"
candidate_root="$(cd "${candidate_root}" && pwd -P)"
fixture="${LIGHTER_FIXTURE:-${root}/benchmark-tools/fixtures/bench.json}"
score="${LIGHTER_SCORE_PATH:-${root}/score.json}"
transactions="${LIGHTER_TRANSACTIONS:-500}"
mode="${LIGHTER_BENCHMARK_MODE:-official-throughput}"
worker="${candidate_root}/target/release/prove"
verifier="${root}/benchmark-tools/trusted/lighter-benchmark-verifier"

rm -f "${score}"
[[ -f "${fixture}" ]] || {
  echo "Missing protected benchmark fixture: ${fixture}" >&2
  exit 1
}
if [[ ! -x "${worker}" || ! -x "${verifier}" ]]; then
  LIGHTER_CANDIDATE_ROOT="${candidate_root}" "${root}/setup.sh"
fi
(
  cd "${root}/benchmark-tools/trusted"
  shasum -a 256 -c SHA256SUMS
)

scratch="$(mktemp -d "${TMPDIR:-/tmp}/lighter-benchmark.XXXXXX")"
scratch="$(cd "${scratch}" && pwd -P)"
sandbox_profile=""
cleanup() {
  rm -rf "${scratch}"
  [[ -z "${sandbox_profile}" ]] || rm -f "${sandbox_profile}"
}
trap cleanup EXIT

if [[ "$(uname -s)" == Darwin && -x /usr/bin/sandbox-exec ]]; then
  [[ "${scratch}" != *\"* && "${scratch}" != *\\* && "${scratch}" != *$'\n'* ]] || {
    echo "scratch path contains unsupported characters" >&2
    exit 1
  }
  sandbox_profile="$(mktemp -t lighter-benchmark.XXXXXX.sb)"
  printf '%s\n' \
    '(version 1)' \
    '(allow default)' \
    '(deny network*)' \
    '(deny process-fork)' \
    '(deny file-write*)' \
    "(allow file-write* (subpath \"${scratch}\"))" \
    > "${sandbox_profile}"
elif [[ "${LIGHTER_REQUIRE_SANDBOX:-0}" == 1 ]]; then
  echo "sandbox-exec is required for the ranked benchmark" >&2
  exit 1
else
  echo "WARNING: candidate worker is not sandboxed (local development only)" >&2
fi

candidate_sha="$(git -C "${candidate_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
args=("${worker}" "${fixture}" "${scratch}" "${score}"
  "${mode}" "${transactions}" "${candidate_sha}")
[[ -z "${sandbox_profile}" ]] || args+=("${sandbox_profile}")
"${verifier}" "${args[@]}"
