#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
workflow="${root}/.github/workflows/benchmark.yml"
ci_workflow="${root}/.github/workflows/ci.yml"
benchmark_script="${root}/benchmark.sh"
profile_writer="${root}/.github/scripts/write-benchmark-sandbox-profile.sh"
bridge_wrapper="${root}/.github/scripts/prove-via-bench.sh"
bundle_extractor="${root}/.github/scripts/extract-private-fixture-bundle.sh"
manifest="${root}/benchmark.json"

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

count_matches_in() {
  local section="$1" pattern="$2" mode="${3:-regex}" matches
  if [[ "${mode}" == fixed ]]; then
    matches="$(printf '%s\n' "${section}" | rg -F -- "${pattern}" || true)"
  else
    matches="$(printf '%s\n' "${section}" | rg -- "${pattern}" || true)"
  fi
  if [[ -z "${matches}" ]]; then
    printf '0\n'
  else
    printf '%s\n' "${matches}" | wc -l | tr -d '[:space:]'
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

require_in "${build_job}" "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m4]"
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
# Nothing in the archive may configure Cargo for the build root.
require_in "${build_job}" '/usr/bin/tar -tf "${archive}" |'
require_in "${build_job}" '/usr/bin/awk -F/'
require_in "${build_job}" '[[ -z "${root_cargo}" ]] || {'
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
require_in "${benchmark_job}" "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m4]"
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
require_in "${benchmark_job}" "LIGHTER_BENCHMARK_MODE: \${{ vars.LIGHTER_BENCHMARK_MODE || 'official-throughput' }}"
require_in "${benchmark_job}" "LIGHTER_EXPECTED_FIXTURE_COUNT: \${{ vars.LIGHTER_EXPECTED_FIXTURE_COUNT || '5' }}"
require_in "${benchmark_job}" "LIGHTER_TRANSACTIONS: \${{ vars.LIGHTER_TRANSACTIONS || '2500' }}"
require_in "${benchmark_job}" "LIGHTER_PRIVATE_FIXTURE_BUNDLE_R2_PATH: fixtures/ranked-v1.tar.gz"
reject_in "${benchmark_job}" "fixtures/ranked-v1.json"
reject_in "${benchmark_job}" 'LIGHTER_PRIVATE_FIXTURE_SHA256:'
reject_in "${benchmark_job}" 'LIGHTER_PUBLIC_FIXTURE_SHA256:'
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
require 'LIGHTER_PRIVATE_FIXTURE_BUNDLE_SHA256: ${{ secrets.LIGHTER_PRIVATE_FIXTURE_BUNDLE_SHA256 }}'
require ".github/scripts/download-r2-object.sh"
require '.github/scripts/extract-private-fixture-bundle.sh \'
require '"${private_bundle}" "${private_fixtures}" \'
require '"${LIGHTER_EXPECTED_FIXTURE_COUNT}" >/dev/null 2>&1'
require '/bin/rm -f "${private_bundle}"'
require "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT \\"
require "  LIGHTER_PRIVATE_FIXTURE_BUNDLE_SHA256"
require 'LIGHTER_FIXTURE="${private_fixtures}" ./benchmark.sh'
require "trap cleanup EXIT"
require 'trap interrupted HUP INT TERM'
require '/bin/rm -rf "${private_root}" "${LIGHTER_CANDIDATE_ROOT}"'
require 'name: lighter-prover-score-${{ github.run_id }}-${{ github.run_attempt }}'
require 'LIGHTER_CANDIDATE_SHA256='
require '/usr/bin/shasum -a 256 "${LIGHTER_CANDIDATE_ROOT}/target/release/prove-bin"'
require_before ".github/scripts/download-r2-object.sh" \
  ".github/scripts/extract-private-fixture-bundle.sh \\"
require_before ".github/scripts/extract-private-fixture-bundle.sh \\" \
  '/bin/rm -f "${private_bundle}"'
require_before '/bin/rm -f "${private_bundle}"' \
  "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT \\"
require_before "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT \\" \
  '/usr/bin/shasum -a 256 "${LIGHTER_CANDIDATE_ROOT}/target/release/prove-bin"'
require_before "unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET_ENDPOINT \\" \
  'LIGHTER_FIXTURE="${private_fixtures}" ./benchmark.sh'
require_before 'LIGHTER_CANDIDATE_SHA256=' \
  'LIGHTER_FIXTURE="${private_fixtures}" ./benchmark.sh'
require_before '[[ "${GITHUB_REF_NAME}" == "${DEFAULT_BRANCH}" ||' \
  "Archive candidate source"
require_before "Archive candidate source" \
  "Reject in-tree Cargo configuration"
require_before "Reject in-tree Cargo configuration" \
  "sudo -n /opt/lighter-prover-challenge/bench-exec.sh build"
if rg -Fq -- "if: always()" "${workflow}"; then
  echo "score upload must run only after benchmark success" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${benchmark_job}" |
  rg -c 'actions/upload-artifact@')" != 1 ]]; then
  echo "benchmark job must upload only the aggregate score artifact" >&2
  exit 1
fi
require_in "${benchmark_job}" "path: score.json"
reject_in "${benchmark_job}" 'path: ${private_'

# Exercise the workflow's own Cargo-configuration filter rather than a copy of
# it: everything the archive root could name .cargo has to be caught, including
# the case variants a case-insensitive volume would still hand to Cargo, and
# nothing below the root may be.
cargo_filter_line="$(rg -m1 -N -- '/usr/bin/awk -F/' "${workflow}")"
cargo_filter="${cargo_filter_line#*\'}"
cargo_filter="${cargo_filter%\'*}"
[[ -n "${cargo_filter}" ]] || {
  echo "could not extract the Cargo configuration filter from the workflow" >&2
  exit 1
}
rejected="$(
  printf '%s\n' \
    '.cargo/config.toml' \
    '.cargo/config' \
    '.CARGO/Config.TOML' \
    '.cargo/' \
    '.cargo' \
    './.cargo/config.toml' \
    'bench/.cargo/config.toml' \
    'vendor/plonky2/.cargo/config.toml' \
    '.cargo-ok' \
    'Cargo.toml' \
    'circuit/src/lib.rs' |
    awk -F/ "${cargo_filter}" | tr '\n' ' '
)"
expected_rejected='.cargo/config.toml .cargo/config .CARGO/Config.TOML .cargo/ .cargo ./.cargo/config.toml '
if [[ "${rejected}" != "${expected_rejected}" ]]; then
  echo "workflow Cargo configuration filter rejected [${rejected}]," \
    "expected [${expected_rejected}]" >&2
  exit 1
fi

if ! rg -Fq -- \
  'candidate_sha="${LIGHTER_CANDIDATE_SHA:-$(git -C "${candidate_root}" rev-parse HEAD 2>/dev/null || echo unknown)}"' \
  "${benchmark_script}"; then
  echo "benchmark.sh is missing the LIGHTER_CANDIDATE_SHA override contract" >&2
  exit 1
fi
for expected in \
  'fixture="${LIGHTER_FIXTURE:-${root}/bench/bench_test.json}"' \
  'expected_public_fixture_sha256="${LIGHTER_PUBLIC_FIXTURE_SHA256:-6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b}"' \
  'LIGHTER_EXPECTED_FIXTURE_SHA256="${expected_public_fixture_sha256}"' \
  'mode="${LIGHTER_BENCHMARK_MODE:-public-synthetic-smoke}"' \
  'use_bench_bridge="${LIGHTER_USE_BENCH_BRIDGE:-0}"' \
  'expected_fixture_count="${LIGHTER_EXPECTED_FIXTURE_COUNT:-}"' \
  'public-synthetic-smoke)' \
  'official-throughput)' \
  'fixture_read_root="$(dirname "${fixture}")"' \
  'fixture_read_root="${fixture}"' \
  'LIGHTER_EXPECTED_FIXTURE_COUNT="${expected_fixture_count}"' \
  'bridge_worker="${candidate_root}/target/release/prove-bin"' \
  '.github/scripts/prove-via-bench.sh' \
  '/opt/lighter-prover-challenge/bench-exec.sh' \
  'require_bridge_path "proof output" "${scratch}/proof.bin"' \
  'require_private_fixture_bridge_path "${fixture}"' \
  'Private fixture input is outside the bench bridge work root' \
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
[[ -f "${bundle_extractor}" && ! -L "${bundle_extractor}" &&
   -x "${bundle_extractor}" ]] || {
  echo "missing private fixture bundle extractor" >&2
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
if rg -Fq -- '(allow default)' "${profile_writer}"; then
  echo "sandbox profile writer is a default-allow blocklist" >&2
  exit 1
fi
if ! rg -Fq -- "'(deny default)'" "${profile_writer}"; then
  echo "sandbox profile writer does not emit (deny default)" >&2
  exit 1
fi
if ! rg -Fq -- ".github/scripts/write-benchmark-sandbox-profile.sh" \
  "${benchmark_script}"; then
  echo "benchmark.sh does not use the shared sandbox profile writer" >&2
  exit 1
fi

# The build-time threat model is stated against a fixed editable set: files
# outside it are trusted because upstream submission validation restricts a
# submission to it. Adding .cargo, rust-toolchain, or setup.sh here would
# invalidate that reasoning, so a change to the set has to be deliberate.
manifest_editable="$(
  sed -n '/"editablePaths"[[:space:]]*:[[:space:]]*\[/,/\]/p' "${manifest}" |
    sed -n 's/^[[:space:]]*"\([^"]*\)",\{0,1\}[[:space:]]*$/\1/p' |
    LC_ALL=C sort | tr '\n' ' '
)"
if [[ "${manifest_editable}" != "Cargo.lock Cargo.toml bench circuit vendor " ]]; then
  echo "benchmark.json editablePaths changed: ${manifest_editable}" >&2
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
ci_job="$(
  awk '
    /^  test:/ { capture=1 }
    capture && !/^  test:/ &&
      /^  [[:alnum:]_-]+:/ { exit }
    capture { print }
  ' "${ci_workflow}"
)"
ci_job_env="$(
  printf '%s\n' "${ci_job}" |
    awk '
      /^    env:/ { capture=1 }
      capture && !/^    env:/ && /^    [^ ]/ { exit }
      capture { print }
    '
)"
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
require_in "${ci_job_env}" 'RUST_TOOLCHAIN_BIN: /opt/lighter-prover-challenge/rustup/toolchains/nightly-2025-12-06-aarch64-apple-darwin/bin'
require_in "${ci_job_env}" 'RUSTC: /opt/lighter-prover-challenge/rustup/toolchains/nightly-2025-12-06-aarch64-apple-darwin/bin/rustc'
require_in "${ci_job_env}" 'PATH: /opt/lighter-prover-challenge/rustup/toolchains/nightly-2025-12-06-aarch64-apple-darwin/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
require_in "${ci_job_env}" 'RUSTFMT: /opt/lighter-prover-challenge/rustup/toolchains/nightly-2025-12-06-aarch64-apple-darwin/bin/rustfmt'
require_in "${ci_job}" 'toolchain="$(< rust-toolchain)"'
require_in "${ci_job}" '[[ "${toolchain}" == "nightly-2025-12-06" ]] || {'
require_in "${ci_job}" 'for binary in cargo rustc rustfmt cargo-fmt; do'
require_in "${ci_job}" 'path="${RUST_TOOLCHAIN_BIN}/${binary}"'
require_in "${ci_job}" '[[ -f "${path}" && ! -L "${path}" && -x "${path}" ]] || {'
require_in "${ci_job}" '"${RUST_TOOLCHAIN_BIN}/cargo" test --locked --manifest-path benchmark-tools/harness/Cargo.toml'
require_in "${ci_job}" '"${RUST_TOOLCHAIN_BIN}/cargo" build --release --locked --manifest-path Cargo.toml -p bench --bin prove'
direct_cargo_fmt='"${RUST_TOOLCHAIN_BIN}/cargo-fmt" fmt'
if [[ "$(count_matches_in "${ci_job}" "${direct_cargo_fmt}" fixed)" != 2 ]]; then
  echo "CI workflow must run exactly two direct pinned cargo-fmt commands" >&2
  exit 1
fi
cargo_fmt_command_pattern='cargo-fmt"?[[:space:]]+fmt([[:space:]]|$)'
if [[ "$(count_matches_in "${ci_job}" "${cargo_fmt_command_pattern}")" != 2 ]]; then
  echo "CI workflow must invoke cargo-fmt exactly twice" >&2
  exit 1
fi
cargo_command_pattern='cargo"?[[:space:]]+(fmt|test|build)([[:space:]]|$)'
if [[ "$(count_matches_in "${ci_job}" "${cargo_command_pattern}")" != 2 ]]; then
  echo "CI workflow must invoke Cargo exactly twice for pinned test and build" >&2
  exit 1
fi
cargo_fmt_via_cargo_pattern='cargo"?[[:space:]]+fmt([[:space:]]|$)'
if [[ "$(count_matches_in "${ci_job}" "${cargo_fmt_via_cargo_pattern}")" != 0 ]]; then
  echo "CI workflow unexpectedly invokes cargo fmt" >&2
  exit 1
fi
validation_end_line="$(
  printf '%s\n' "${ci_job}" |
    awk '
      /^          for binary in cargo rustc rustfmt cargo-fmt; do$/ {
        validation=1
      }
      validation && /^          done$/ { print NR; exit }
    '
)"
if [[ -z "${validation_end_line}" ]]; then
  echo "CI workflow is missing the complete four-tool validation block" >&2
  exit 1
fi
for invocation in \
  "${direct_cargo_fmt}" \
  '"${RUST_TOOLCHAIN_BIN}/cargo" test --locked --manifest-path benchmark-tools/harness/Cargo.toml' \
  '"${RUST_TOOLCHAIN_BIN}/cargo" build --release --locked --manifest-path Cargo.toml -p bench --bin prove'; do
  invocation_line="$(
    printf '%s\n' "${ci_job}" |
      awk -v invocation="${invocation}" '
        index($0, invocation) { print NR; exit }
      '
  )"
  if [[ -z "${invocation_line}" ||
        "${validation_end_line}" -ge "${invocation_line}" ]]; then
    echo "CI workflow must validate all four pinned tools before: ${invocation}" >&2
    exit 1
  fi
done
if rg -q -- '(^|[[:space:]])rustup[[:space:]]' "${ci_workflow}"; then
  echo "CI workflow unexpectedly invokes rustup" >&2
  exit 1
fi
if rg -Fq -- ".cargo/bin" "${ci_workflow}"; then
  echo "CI workflow unexpectedly contains: .cargo/bin" >&2
  exit 1
fi
if rg -q -- "cargo[\"']?[[:space:]]+[\"']?\\+" "${ci_workflow}"; then
  echo "CI workflow unexpectedly contains a Cargo +toolchain directive" >&2
  exit 1
fi
for required in \
  "runs-on: [self-hosted, macOS, ARM64, lighter-prover-challenge-m4]" \
  ".github/scripts/test-private-benchmark-workflow.sh" \
  ".github/scripts/test-private-fixture-bundle.sh" \
  ".github/scripts/test-prove-via-bench.sh" \
  ".github/scripts/test-benchmark-sandbox.sh" \
  ".github/scripts/test-trusted-verifier.sh" \
  "bash -n"; do
  if ! rg -Fq -- "${required}" "${ci_workflow}"; then
    echo "CI workflow is missing: ${required}" >&2
    exit 1
  fi
done
