#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
candidate_root="${LIGHTER_CANDIDATE_ROOT:-${root}}"
candidate_root="$(cd "${candidate_root}" && pwd -P)"
trusted="${root}/benchmark-tools/trusted/lighter-benchmark-verifier"

die() { echo "setup.sh: $*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin ]] || die "macOS is required"
[[ "$(uname -m)" == arm64 ]] || die "Apple Silicon is required"
for tool in cargo codesign rustup shasum; do
  command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"
done
if [[ ! -x /usr/bin/sandbox-exec ]]; then
  [[ "${LIGHTER_REQUIRE_SANDBOX:-0}" == 1 ]] && die "sandbox-exec is required"
  echo "setup.sh: warning: sandbox-exec is missing; ranked runs will fail" >&2
fi

[[ -x "${trusted}" ]] || die "trusted verifier is missing"
(
  cd "${root}/benchmark-tools/trusted"
  shasum -a 256 -c SHA256SUMS
)
codesign --verify --strict "${trusted}" || die "trusted verifier signature is invalid"

toolchain="$(tr -d '[:space:]' < "${root}/rust-toolchain")"
if ! rustup toolchain list 2>/dev/null \
  | awk '{print $1}' \
  | grep -Eq "^${toolchain}(-|$)"; then
  rustup toolchain install "${toolchain}" --profile minimal --no-self-update
fi

RUSTUP_TOOLCHAIN="${toolchain}" \
  cargo fetch --locked --manifest-path "${candidate_root}/Cargo.toml"
RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
CARGO_NET_OFFLINE=true \
  cargo build --release --locked --offline \
    --manifest-path "${candidate_root}/Cargo.toml" -p bench --bin prove

echo "candidate worker: ${candidate_root}/target/release/prove"
echo "trusted verifier: ${trusted}"
