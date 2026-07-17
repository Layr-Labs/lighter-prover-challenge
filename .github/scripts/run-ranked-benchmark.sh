#!/usr/bin/env bash
set -euo pipefail

candidate_sha="${1:?usage: run-ranked-benchmark.sh CANDIDATE_SHA}"
root="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
workspace="${LIGHTER_JOB_WS:?LIGHTER_JOB_WS is required}"
bridge="${LIGHTER_BENCH_EXEC:?LIGHTER_BENCH_EXEC is required}"
result="${root}/score.json"

[[ "${candidate_sha}" =~ ^[0-9a-f]{40}$ ]]

cd "${root}"
baseline_sha="$(git rev-parse HEAD)"
git fetch --no-tags origin "${candidate_sha}"
git cat-file -e "${candidate_sha}^{commit}"
git merge-base --is-ancestor "${baseline_sha}" "${candidate_sha}"

invalid="$(git diff --name-only --no-renames "${baseline_sha}" "${candidate_sha}" \
  | grep -v '^challenge/submission/' || true)"
if [[ -n "${invalid}" ]]; then
  echo "Only challenge/submission/ may change:" >&2
  echo "${invalid}" >&2
  exit 1
fi

rm -rf "${workspace}"
if [[ -e "${workspace}" || -L "${workspace}" ]]; then
  echo "stale benchmark workspace could not be removed: ${workspace}" >&2
  exit 1
fi
cp -c -R "${root}" "${workspace}" 2>/dev/null || cp -R "${root}" "${workspace}"
mkdir -p "${workspace}/tmp"

cd "${workspace}"
toolchain="$(tr -d '[:space:]' < rust-toolchain)"
rustup run "${toolchain}" rustc --version >/dev/null
RUSTUP_TOOLCHAIN="${toolchain}" cargo fetch --locked --manifest-path challenge/Cargo.toml
RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
CARGO_NET_OFFLINE=true \
  cargo build --release --locked --manifest-path challenge/Cargo.toml --bins

/bin/chmod -R +a \
  "user:bench allow list,search,readattr,readextattr,read,execute,add_file,add_subdirectory,delete_child,write,append,writeattr,writeextattr,file_inherit,directory_inherit" \
  "${workspace}"
/bin/chmod -R +a \
  "user:bench deny write,append,writeattr,writeextattr,delete,delete_child,add_file,add_subdirectory,chown,file_inherit,directory_inherit" \
  "${workspace}/.github" \
  "${workspace}/bench" \
  "${workspace}/challenge/src" \
  "${workspace}/challenge/submission" \
  "${workspace}/challenge/target" \
  "${workspace}/circuit" \
  "${workspace}/testdata"
/bin/chmod +a \
  "user:bench deny write,append,writeattr,writeextattr,delete,chown" \
  "${workspace}/benchmark.sh" \
  "${workspace}/benchmark.json" \
  "${workspace}/setup.sh" \
  "${workspace}/Cargo.toml" \
  "${workspace}/Cargo.lock" \
  "${workspace}/challenge/Cargo.toml" \
  "${workspace}/challenge/Cargo.lock"

run_prover() {
  local name="$1"
  local output="${RUNNER_TEMP}/lighter-prover-${name}.json"
  # The runner, not the untrusted bench uid, must own captured results.
  # shellcheck disable=SC2024
  sudo -n "${bridge}" "${workspace}" \
    /usr/bin/env \
      TMPDIR="${workspace}/tmp" \
      LIGHTER_SCORE_PATH="score.${name}.json" \
      /bin/bash "${workspace}/benchmark.sh" --local-submit \
    > "${output}"
  jq -s -e '
    if length == 1
      and .[0].passed == true
      and .[0].metrics.transactions == 500
      and .[0].metrics.timing_authority == "trusted bench parent"
      and .[0].metrics.proving_seconds > 0
    then .[0]
    else error("invalid benchmark output")
    end
  ' "${output}"
}

baseline_json="$(run_prover baseline)"

rm -rf challenge/submission
git archive "${candidate_sha}" challenge/submission | tar -x
RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
CARGO_NET_OFFLINE=true \
  cargo build --release --locked --manifest-path challenge/Cargo.toml --bins
/bin/chmod -R +a \
  "user:bench deny write,append,writeattr,writeextattr,delete,delete_child,add_file,add_subdirectory,chown,file_inherit,directory_inherit" \
  challenge/submission

candidate_json="$(run_prover candidate)"

baseline_seconds="$(jq -r '.metrics.proving_seconds' <<< "${baseline_json}")"
candidate_seconds="$(jq -r '.metrics.proving_seconds' <<< "${candidate_json}")"
circuit_digest="$(
  git -C "${root}" ls-tree -r "${baseline_sha}" -- \
    challenge/src/api.rs challenge/src/bin/bench.rs challenge/Cargo.toml challenge/Cargo.lock circuit \
    | shasum -a 256 | awk '{print $1}'
)"

jq -n \
  --argjson baseline_seconds "${baseline_seconds}" \
  --argjson candidate_seconds "${candidate_seconds}" \
  --arg baseline_sha "${baseline_sha}" \
  --arg candidate_sha "${candidate_sha}" \
  --arg circuit_digest "${circuit_digest}" \
  '{
    score: ($baseline_seconds / $candidate_seconds),
    passed: true,
    metrics: {
      runtime: "official-paired",
      candidate_seconds: $candidate_seconds,
      baseline_seconds: $baseline_seconds,
      speedup: ($baseline_seconds / $candidate_seconds),
      verified_proofs: 2,
      expected_proofs: 2,
      transactions: 500,
      candidate_sha: $candidate_sha,
      baseline_sha: $baseline_sha,
      circuit_digest: $circuit_digest,
      case_id: "bench-test-500"
    }
  }' > "${result}"

jq -e '
  .passed == true
  and .score > 0
  and .score == .metrics.speedup
  and .metrics.runtime == "official-paired"
  and .metrics.verified_proofs == .metrics.expected_proofs
' "${result}" >/dev/null
shasum -a 256 "${result}" > "${result}.sha256"
