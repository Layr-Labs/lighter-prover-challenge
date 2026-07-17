#!/usr/bin/env bash
set -euo pipefail

candidate_sha="${1:?usage: run-ranked-benchmark.sh CANDIDATE_SHA}"
root="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
result="${root}/score.json"

[[ "${candidate_sha}" =~ ^[0-9a-f]{40}$ ]]

cd "${root}"
baseline_sha="$(git rev-parse HEAD)"
git cat-file -e "${candidate_sha}^{commit}"
git merge-base --is-ancestor "${baseline_sha}" "${candidate_sha}"

invalid="$(git diff --name-only --no-renames "${baseline_sha}" "${candidate_sha}" \
  | grep -v '^challenge/submission/' || true)"
if [[ -n "${invalid}" ]]; then
  echo "Only challenge/submission/ may change:" >&2
  echo "${invalid}" >&2
  exit 1
fi

toolchain="$(tr -d '[:space:]' < rust-toolchain)"
rustup toolchain install "${toolchain}" --profile minimal --no-self-update
RUSTUP_TOOLCHAIN="${toolchain}" cargo fetch --locked --manifest-path challenge/Cargo.toml
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

remove_sandbox() {
  mv -f challenge/target/release/prove-bin challenge/target/release/prove
  rm -f challenge/target/release/prover.sb
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
baseline_json="$(run_prover baseline)"
remove_sandbox

rm -rf challenge/submission
git archive "${candidate_sha}" challenge/submission | tar -x
RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
CARGO_NET_OFFLINE=true \
  cargo build --release --locked --manifest-path challenge/Cargo.toml --bins

install_sandbox
candidate_json="$(run_prover candidate)"

baseline_seconds="$(jq -r '.metrics.proving_seconds' <<< "${baseline_json}")"
candidate_seconds="$(jq -r '.metrics.proving_seconds' <<< "${candidate_json}")"

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
