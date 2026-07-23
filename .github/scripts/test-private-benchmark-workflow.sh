#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
workflow="${root}/.github/workflows/benchmark.yml"
ci_workflow="${root}/.github/workflows/ci.yml"
benchmark_script="${root}/benchmark.sh"
profile_writer="${root}/.github/scripts/write-benchmark-sandbox-profile.sh"
bridge_wrapper="${root}/.github/scripts/prove-via-bench.sh"

require() {
  if ! rg -Fq -- "$1" "${workflow}"; then
    echo "benchmark workflow is missing: $1" >&2
    exit 1
  fi
}

require_in() {
  local section="$1" expected="$2"
  if ! printf '%s\n' "${section}" | rg -Fq -- "${expected}"; then
    echo "workflow section is missing: ${expected}" >&2
    exit 1
  fi
}

reject_in() {
  local section="$1" forbidden="$2"
  if printf '%s\n' "${section}" | rg -Fq -- "${forbidden}"; then
    echo "workflow section unexpectedly contains: ${forbidden}" >&2
    exit 1
  fi
}

require_before() {
  local first="$1" second="$2" first_line second_line
  first_line="$(rg -n -F -- "${first}" "${workflow}" | awk -F: 'NR == 1 { print $1 }')"
  second_line="$(rg -n -F -- "${second}" "${workflow}" | awk -F: 'NR == 1 { print $1 }')"
  if [[ -z "${first_line}" || -z "${second_line}" || "${first_line}" -ge "${second_line}" ]]; then
    echo "benchmark workflow ordering is wrong: ${first} must precede ${second}" >&2
    exit 1
  fi
}

build_job="$(awk '/^  build:/{capture=1} /^  benchmark:/{capture=0} capture' "${workflow}")"
benchmark_job="$(awk '/^  benchmark:/{capture=1} capture' "${workflow}")"

require_in "${build_job}" "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m3]"
require_in "${build_job}" 'ref: ${{ github.sha }}'
require_in "${build_job}" "set-safe-directory: false"
require_in "${build_job}" 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}'
require_in "${build_job}" '[[ "${GITHUB_REF_TYPE}" == branch ]]'
require_in "${build_job}" '[[ "${GITHUB_REF_NAME}" == "${DEFAULT_BRANCH}" ||'
require_in "${build_job}" '"${GITHUB_REF_NAME}" == submissions/* ||'
require_in "${build_job}" '"${GITHUB_REF_NAME}" == yukon/baseline/* ]]'
require_in "${build_job}" 'bridge="/opt/lighter-prover-challenge/bench-exec.sh"'
require_in "${build_job}" '[[ -f "${bridge}" && ! -L "${bridge}" && -x "${bridge}" ]]'
require_in "${build_job}" '/usr/bin/sudo -n -l "${bridge}"'
require_in "${build_job}" 'work_root="/opt/lighter-prover-challenge/work"'
require_in "${build_job}" 'case "${RUNNER_TEMP:?}" in'
require_in "${build_job}" 'case "${GITHUB_WORKSPACE:?}" in'
require_in "${build_job}" 'archive="${RUNNER_TEMP}/lighter-candidate-source.tar"'
require_in "${build_job}" '/usr/bin/env -i \'
require_in "${build_job}" '/usr/bin/git archive --format=tar HEAD'
require_in "${build_job}" '/bin/chmod 0644 "${archive}"'
require_in "${build_job}" 'sudo -n /opt/lighter-prover-challenge/bench-exec.sh build \'
require_in "${build_job}" '"${archive}" "${worker}"'
require_in "${build_job}" '[[ -f "${worker}" && ! -L "${worker}" ]]'
require_in "${build_job}" "shasum -a 256 prove > prove.sha256"
require_in "${build_job}" "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1"
require_in "${build_job}" 'name: lighter-prover-candidate-${{ github.run_id }}-${{ github.run_attempt }}'
reject_in "${build_job}" "./setup.sh"
reject_in "${build_job}" "cargo build"
reject_in "${build_job}" "environment:"
reject_in "${build_job}" "secrets.R2_"
reject_in "${build_job}" "GITHUB_OUTPUT"
reject_in "${build_job}" "candidate_sha:"

require_in "${benchmark_job}" "needs: build"
require_in "${benchmark_job}" "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m3]"
require_in "${benchmark_job}" "environment: benchmark-private-data"
require_in "${benchmark_job}" 'PATH: /usr/bin:/bin:/usr/sbin:/sbin'
require_in "${benchmark_job}" "GIT_CONFIG_GLOBAL: /dev/null"
require_in "${benchmark_job}" 'GIT_CONFIG_NOSYSTEM: "1"'
require_in "${benchmark_job}" "AWS_CONFIG_FILE: /dev/null"
require_in "${benchmark_job}" "AWS_SHARED_CREDENTIALS_FILE: /dev/null"
require_in "${benchmark_job}" 'AWS_EC2_METADATA_DISABLED: "true"'
require_in "${benchmark_job}" "R2_AWS_CLI: /usr/local/bin/aws"
require_in "${benchmark_job}" 'TMPDIR: ${{ runner.temp }}'
require_in "${benchmark_job}" 'LIGHTER_USE_BENCH_BRIDGE: "1"'
require_in "${benchmark_job}" 'ref: ${{ github.event.repository.default_branch }}'
require_in "${benchmark_job}" "set-safe-directory: false"
require_in "${benchmark_job}" 'bridge="/opt/lighter-prover-challenge/bench-exec.sh"'
require_in "${benchmark_job}" '/usr/bin/sudo -n -l "${bridge}"'
require_in "${benchmark_job}" 'work_root="/opt/lighter-prover-challenge/work"'
require_in "${benchmark_job}" 'case "${RUNNER_TEMP:?}" in'
require_in "${benchmark_job}" "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1"
require_in "${benchmark_job}" 'name: lighter-prover-candidate-${{ github.run_id }}-${{ github.run_attempt }}'
require_in "${benchmark_job}" 'LIGHTER_CANDIDATE_SHA: ${{ github.sha }}'
require_in "${benchmark_job}" '[[ -x "${R2_AWS_CLI}" ]]'
require_in "${benchmark_job}" '/usr/bin/stat -Lf "%u" "${path_entry}"'
require_in "${benchmark_job}" '(( (8#${path_mode} & 8#022) == 0 ))'
require_in "${benchmark_job}" "/usr/bin/shasum -a 256 -c SHA256SUMS"
require_in "${benchmark_job}" "/usr/bin/codesign --verify --strict"
require_in "${benchmark_job}" "/usr/bin/find"
require_in "${benchmark_job}" '[[ "${artifact_entry_count}" == 2 ]]'
require_in "${benchmark_job}" '[[ -f "${artifact_dir}/prove" && ! -L "${artifact_dir}/prove" ]]'
require_in "${benchmark_job}" '[[ -f "${artifact_dir}/prove.sha256" && ! -L "${artifact_dir}/prove.sha256" ]]'
require_in "${benchmark_job}" 'checksum_pattern='
require_in "${benchmark_job}" "/usr/bin/shasum -a 256 -c prove.sha256"
require_in "${benchmark_job}" '"${candidate_root}/target/release/prove-bin"'
require_in "${benchmark_job}" '".github/scripts/prove-via-bench.sh"'
require_in "${benchmark_job}" '"${candidate_root}/target/release/prove"'
reject_in "${benchmark_job}" "run: ./setup.sh"

require "environment: benchmark-private-data"
require 'R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}'
require 'R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}'
require 'R2_BUCKET_ENDPOINT: ${{ secrets.R2_BUCKET_ENDPOINT }}'
require 'LIGHTER_PRIVATE_FIXTURE_SHA256: ${{ vars.LIGHTER_PRIVATE_FIXTURE_SHA256 }}'
require ".github/scripts/download-r2-object.sh"
require "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT"
require 'LIGHTER_FIXTURE="${private_fixture}" ./benchmark.sh'
require "trap cleanup EXIT"
require 'name: lighter-prover-score-${{ github.run_id }}-${{ github.run_attempt }}'
require 'LIGHTER_CANDIDATE_SHA256='
require '/usr/bin/shasum -a 256 "${LIGHTER_CANDIDATE_ROOT}/target/release/prove-bin"'
require_before ".github/scripts/download-r2-object.sh" \
  "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT"
require_before "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT" \
  'LIGHTER_FIXTURE="${private_fixture}" ./benchmark.sh'
require_before 'LIGHTER_CANDIDATE_SHA256=' \
  'LIGHTER_FIXTURE="${private_fixture}" ./benchmark.sh'
require_before '[[ "${GITHUB_REF_NAME}" == "${DEFAULT_BRANCH}" ||' \
  "Archive candidate source"
require_before "Archive candidate source" \
  "sudo -n /opt/lighter-prover-challenge/bench-exec.sh build"
if rg -Fq -- "if: always()" "${workflow}"; then
  echo "score upload must run only after benchmark success" >&2
  exit 1
fi

if ! rg -Fq -- \
  'candidate_sha="${LIGHTER_CANDIDATE_SHA:-$(git -C "${candidate_root}" rev-parse HEAD 2>/dev/null || echo unknown)}"' \
  "${benchmark_script}"; then
  echo "benchmark.sh is missing the LIGHTER_CANDIDATE_SHA override contract" >&2
  exit 1
fi
for expected in \
  'use_bench_bridge="${LIGHTER_USE_BENCH_BRIDGE:-0}"' \
  'bridge_worker="${candidate_root}/target/release/prove-bin"' \
  '.github/scripts/prove-via-bench.sh' \
  '/opt/lighter-prover-challenge/bench-exec.sh' \
  'if [[ "${use_bench_bridge}" == 0 ]]; then' \
  '[[ "${use_bench_bridge}" == 0 && -n "${sandbox_profile}" ]]'; do
  if ! rg -Fq -- "${expected}" "${benchmark_script}"; then
    echo "benchmark.sh is missing bridge-mode contract: ${expected}" >&2
    exit 1
  fi
done
[[ -f "${bridge_wrapper}" ]] || {
  echo "missing trusted bench bridge wrapper" >&2
  exit 1
}

for denied_service in \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.system.mDNSResponder"))' \
  '(deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder.dnsproxy"))'; do
  if ! rg -Fq -- "${denied_service}" "${profile_writer}"; then
    echo "sandbox profile is missing DNS Mach lookup denial: ${denied_service}" >&2
    exit 1
  fi
done
if ! rg -Fq -- ".github/scripts/write-benchmark-sandbox-profile.sh" \
  "${benchmark_script}"; then
  echo "benchmark.sh does not use the shared sandbox profile writer" >&2
  exit 1
fi

if ! rg -Fq -- '[[ "${R2_AWS_CLI}" == /* ]]' \
  "${root}/.github/scripts/download-r2-object.sh"; then
  echo "downloader does not require an absolute R2_AWS_CLI override" >&2
  exit 1
fi

[[ -f "${ci_workflow}" ]] || {
  echo "missing workflow_dispatch-only CI workflow" >&2
  exit 1
}
if ! rg -Fq -- "workflow_dispatch:" "${ci_workflow}"; then
  echo "CI workflow is missing workflow_dispatch" >&2
  exit 1
fi
for forbidden in "pull_request:" "push:" "environment:" "secrets."; do
  if rg -Fq -- "${forbidden}" "${ci_workflow}"; then
    echo "CI workflow unexpectedly contains: ${forbidden}" >&2
    exit 1
  fi
done
for required in \
  "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m3]" \
  ".github/scripts/test-private-benchmark-workflow.sh" \
  ".github/scripts/test-prove-via-bench.sh" \
  ".github/scripts/test-benchmark-sandbox.sh" \
  ".github/scripts/test-trusted-verifier.sh" \
  "bash -n"; do
  if ! rg -Fq -- "${required}" "${ci_workflow}"; then
    echo "CI workflow is missing: ${required}" >&2
    exit 1
  fi
done
