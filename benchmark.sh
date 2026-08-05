#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
candidate_root="${LIGHTER_CANDIDATE_ROOT:-${root}}"
candidate_root="$(cd "${candidate_root}" && pwd -P)"
fixture="${LIGHTER_FIXTURE:-${root}/bench/bench_test.json}"
score="${LIGHTER_SCORE_PATH:-${root}/score.json}"
transactions="${LIGHTER_TRANSACTIONS:-500}"
expected_public_fixture_sha256="${LIGHTER_PUBLIC_FIXTURE_SHA256:-6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b}"
expected_fixture_count="${LIGHTER_EXPECTED_FIXTURE_COUNT:-}"
mode="${LIGHTER_BENCHMARK_MODE:-public-synthetic-smoke}"
use_bench_bridge="${LIGHTER_USE_BENCH_BRIDGE:-0}"
worker="${candidate_root}/target/release/prove"
verifier="${root}/benchmark-tools/trusted/lighter-benchmark-verifier"
bridge="/opt/lighter-prover-challenge/bench-exec.sh"
bridge_work_root="/opt/lighter-prover-challenge/work"
bridge_wrapper="${root}/.github/scripts/prove-via-bench.sh"
bridge_worker="${candidate_root}/target/release/prove-bin"

rm -f "${score}"
case "${mode}" in
  public-synthetic-smoke)
    [[ -f "${fixture}" && ! -L "${fixture}" ]] || {
      echo "Missing protected public benchmark fixture" >&2
      exit 1
    }
    fixture="$(cd "$(dirname "${fixture}")" && pwd -P)/$(basename "${fixture}")"
    fixture_read_root="$(dirname "${fixture}")"
    ;;
  official-throughput)
    [[ "${expected_fixture_count}" =~ ^[1-9][0-9]*$ ]] || {
      echo "LIGHTER_EXPECTED_FIXTURE_COUNT must be a positive integer in private mode" >&2
      exit 1
    }
    [[ -d "${fixture}" && ! -L "${fixture}" ]] || {
      echo "Missing protected private benchmark fixture directory" >&2
      exit 1
    }
    fixture="$(cd "${fixture}" && pwd -P)"
    fixture_read_root="${fixture}"
    ;;
  *)
    echo "Unsupported benchmark mode" >&2
    exit 1
    ;;
esac
case "${use_bench_bridge}" in
  0|1) ;;
  *)
    echo "LIGHTER_USE_BENCH_BRIDGE must be 0 or 1" >&2
    exit 1
    ;;
esac

require_bridge_path() {
  local description="$1" path="$2"
  case "${path}" in
    "${bridge_work_root}"/*) ;;
    *)
      echo "${description} is outside the bench bridge work root: ${path}" >&2
      exit 1
      ;;
  esac
}

require_private_fixture_bridge_path() {
  case "$1" in
    "${bridge_work_root}"/*) ;;
    *)
      echo "Private fixture input is outside the bench bridge work root" >&2
      exit 1
      ;;
  esac
}

if [[ "${use_bench_bridge}" == 1 ]]; then
  [[ -f "${bridge}" && ! -L "${bridge}" && -x "${bridge}" ]] || {
    echo "Missing trusted root bench bridge: ${bridge}" >&2
    exit 1
  }
  [[ -f "${bridge_wrapper}" && ! -L "${bridge_wrapper}" ]] || {
    echo "Missing trusted bench bridge wrapper: ${bridge_wrapper}" >&2
    exit 1
  }
  [[ -f "${worker}" && ! -L "${worker}" && -x "${worker}" ]] || {
    echo "Installed bench bridge wrapper is missing or unsafe: ${worker}" >&2
    exit 1
  }
  [[ -f "${bridge_worker}" && ! -L "${bridge_worker}" && -x "${bridge_worker}" ]] || {
    echo "Installed candidate worker is missing or unsafe: ${bridge_worker}" >&2
    exit 1
  }
  /usr/bin/cmp -s "${bridge_wrapper}" "${worker}" || {
    echo "Installed bench bridge wrapper does not match the trusted source" >&2
    exit 1
  }
  [[ "${fixture}" == /* && ! -L "${fixture}" ]] || {
    echo "Bench bridge fixture input must be absolute and non-symlink" >&2
    exit 1
  }
  case "${mode}" in
    public-synthetic-smoke) [[ -f "${fixture}" ]] ;;
    official-throughput) [[ -d "${fixture}" ]] ;;
  esac || {
    echo "Bench bridge fixture input has the wrong type" >&2
    exit 1
  }
  require_bridge_path "candidate root" "${candidate_root}"
  require_bridge_path "candidate worker" "${bridge_worker}"
  case "${mode}" in
    public-synthetic-smoke) require_bridge_path "fixture" "${fixture}" ;;
    official-throughput) require_private_fixture_bridge_path "${fixture}" ;;
  esac
  require_bridge_path "TMPDIR" "${TMPDIR:?}"
elif [[ ! -x "${worker}" || ! -x "${verifier}" ]]; then
  LIGHTER_CANDIDATE_ROOT="${candidate_root}" "${root}/setup.sh"
fi
[[ -x "${worker}" && -x "${verifier}" ]] || {
  echo "Missing candidate worker or trusted verifier" >&2
  exit 1
}
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

if [[ "${use_bench_bridge}" == 1 ]]; then
  require_bridge_path "proof scratch" "${scratch}"
  require_bridge_path "proof output" "${scratch}/proof.bin"
fi

if [[ "${use_bench_bridge}" == 0 ]]; then
  if [[ "$(uname -s)" == Darwin && -x /usr/bin/sandbox-exec ]]; then
    sandbox_profile="$(mktemp -t lighter-benchmark.XXXXXX.sb)"
    "${root}/.github/scripts/write-benchmark-sandbox-profile.sh" \
      "${scratch}" "${sandbox_profile}" "${worker}" \
      "${candidate_root}" "${root}" "${fixture_read_root}"
  elif [[ "${LIGHTER_REQUIRE_SANDBOX:-0}" == 1 ]]; then
    echo "sandbox-exec is required for the ranked benchmark" >&2
    exit 1
  else
    echo "WARNING: candidate worker is not sandboxed (local development only)" >&2
  fi
fi

candidate_sha="${LIGHTER_CANDIDATE_SHA:-$(git -C "${candidate_root}" rev-parse HEAD 2>/dev/null || echo unknown)}"
args=("${worker}" "${fixture}" "${scratch}" "${score}"
  "${mode}" "${transactions}" "${candidate_sha}")
[[ "${use_bench_bridge}" == 0 && -n "${sandbox_profile}" ]] &&
  args+=("${sandbox_profile}")
if [[ "${mode}" == official-throughput ]]; then
  export LIGHTER_EXPECTED_FIXTURE_COUNT="${expected_fixture_count}"
fi
LIGHTER_EXPECTED_FIXTURE_SHA256="${expected_public_fixture_sha256}" \
  "${verifier}" "${args[@]}"
