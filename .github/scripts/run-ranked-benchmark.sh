#!/usr/bin/env bash
set -euo pipefail

root="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
cd "${root}"
root="$(pwd -P)"
candidate_root="$(cd "${root}/.lighter-candidate" && pwd -P)"
result="${root}/score.json"
baseline_sha="$(git rev-parse HEAD)"
candidate_sha="$(git -C "${candidate_root}" rev-parse HEAD)"
[[ "${candidate_sha}" =~ ^[0-9a-f]{40}$ ]]
git -C "${candidate_root}" cat-file -e "${baseline_sha}^{commit}"
base_sha="$(git -C "${candidate_root}" merge-base "${baseline_sha}" "${candidate_sha}")"
if [[ "${base_sha}" != "${baseline_sha}" ]]; then
  echo "Candidate must be based on current master ${baseline_sha}; merge-base is ${base_sha}" >&2
  exit 1
fi

invalid="$(git -C "${candidate_root}" diff --name-only --no-renames "${baseline_sha}" "${candidate_sha}" \
  | grep -v '^challenge/submission/' || true)"
if [[ -n "${invalid}" ]]; then
  echo "Only challenge/submission/ may change:" >&2
  echo "${invalid}" >&2
  exit 1
fi

toolchain="$(tr -d '[:space:]' < rust-toolchain)"
rustup toolchain install "${toolchain}" --profile minimal --no-self-update
RUSTUP_TOOLCHAIN="${toolchain}" cargo fetch --locked --manifest-path challenge/Cargo.toml

# Swap in the candidate submission before the single build. The baseline is no
# longer built or proved: the ranked score is the candidate's own proving
# throughput (transactions per second), not a baseline/candidate ratio.
rm -rf challenge/submission
git -C "${candidate_root}" archive "${candidate_sha}" challenge/submission | tar -x -C "${root}"
rm -rf "${candidate_root}"

RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
CARGO_NET_OFFLINE=true \
  cargo build --release --locked --manifest-path challenge/Cargo.toml --bins

install_sandbox() {
  mv -f challenge/target/release/prove challenge/target/release/prove-bin
  cp .github/scripts/sandbox-prove.sh challenge/target/release/prove
  cp .github/scripts/prover.sb challenge/target/release/prover.sb
  chmod +x challenge/target/release/prove
}

circuit_digest="$(
  git ls-tree -r "${baseline_sha}" -- \
    challenge/src/api.rs challenge/src/bin/bench.rs challenge/Cargo.toml challenge/Cargo.lock circuit \
    | shasum -a 256 | awk '{print $1}'
)"

run_prover() {
  local name="$1"
  local output
  output="$(LIGHTER_SCORE_PATH="score.${name}.json" ./benchmark.sh --local-submit)"
  jq -s -e '
    if length == 1
      and .[0].passed == true
      and .[0].metrics.transactions == 500
      and .[0].metrics.timing_authority == "trusted bench parent"
      and .[0].metrics.proving_seconds > 0
    then .[0]
    else error("invalid benchmark output")
    end
  ' <<< "${output}"
}

install_sandbox
candidate_json="$(run_prover candidate)"

candidate_seconds="$(jq -r '.metrics.proving_seconds' <<< "${candidate_json}")"
transactions="$(jq -r '.metrics.transactions' <<< "${candidate_json}")"

jq -n \
  --argjson candidate_seconds "${candidate_seconds}" \
  --argjson transactions "${transactions}" \
  --arg baseline_sha "${baseline_sha}" \
  --arg candidate_sha "${candidate_sha}" \
  --arg circuit_digest "${circuit_digest}" \
  '{
    score: ($transactions / $candidate_seconds),
    passed: true,
    metrics: {
      runtime: "official-throughput",
      candidate_seconds: $candidate_seconds,
      transactions_per_second: ($transactions / $candidate_seconds),
      verified_proofs: 1,
      expected_proofs: 1,
      transactions: $transactions,
      candidate_sha: $candidate_sha,
      baseline_sha: $baseline_sha,
      circuit_digest: $circuit_digest,
      case_id: "bench-test-500"
    }
  }' > "${result}"

jq -e '
  .passed == true
  and .score > 0
  and .score == .metrics.transactions_per_second
  and .metrics.runtime == "official-throughput"
  and .metrics.verified_proofs == .metrics.expected_proofs
' "${result}" >/dev/null
(cd "${root}" && shasum -a 256 score.json > score.json.sha256)
