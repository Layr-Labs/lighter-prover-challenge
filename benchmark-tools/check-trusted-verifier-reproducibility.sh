#!/usr/bin/env bash
# Independently rebuild the published trusted verifier from REVIEWED_COMMIT and
# compare the full SHA-256 of the result against benchmark-tools/trusted/SHA256SUMS.
#
# Why this exists: the verifier is ad-hoc code-signed, so `codesign --verify`
# proves tamper-evidence and not signer authenticity. The SHA-256 pin is the only
# authenticity control, and this script is the only way to check that pin against
# something other than the commit that introduced it. See
# benchmark-tools/trusted/README.md.
#
# This script is read-only with respect to the repository. It never writes into
# the worktree, never creates a git worktree, and never touches the published
# binary or SHA256SUMS. It exits non-zero unless the rebuild matches the pin
# byte for byte.
#
# KNOWN RESULT: this check currently FAILS, because the dependency graph pulls in
# const-random, which bakes fresh compile-time entropy into every build. The
# published pin is therefore not reproducible by anyone. Read the "Reproducibility"
# section of benchmark-tools/trusted/README.md before interpreting a failure here.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: check-trusted-verifier-reproducibility.sh [--single] [--keep]

  --single  perform one rebuild instead of two (faster, but cannot tell a wrong
            pin apart from a build that is simply not deterministic)
  --keep    keep the scratch build directory and print its path
USAGE
}

builds=2
keep=0
while (( $# > 0 )); do
  case "$1" in
    --single) builds=1 ;;
    --keep) keep=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
publisher="${root}/benchmark-tools/build-trusted-verifier.sh"
trusted_dir="${root}/benchmark-tools/trusted"
published="${trusted_dir}/lighter-benchmark-verifier"

die() { echo "check-trusted-verifier-reproducibility.sh: $*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin ]] || die "the published verifier is a macOS arm64 binary; rebuild on macOS"
[[ "$(uname -m)" == arm64 ]] || die "Apple Silicon is required to rebuild the published arm64 binary"
for tool in cargo codesign git rustup shasum; do
  command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"
done
[[ -f "${publisher}" ]] || die "missing ${publisher}"
[[ -f "${published}" ]] || die "missing ${published}"
[[ -f "${trusted_dir}/SHA256SUMS" ]] || die "missing ${trusted_dir}/SHA256SUMS"

# Take the commit, toolchain, and build flags from the publication script itself
# so this check cannot drift away from how the binary was actually produced.
reviewed_commit="$(
  awk -F'=' '$1 == "readonly REVIEWED_COMMIT" { print $2 }' "${publisher}"
)"
toolchain="$(
  awk -F'=' '$1 == "readonly TOOLCHAIN" { print $2 }' "${publisher}"
)"
[[ "${reviewed_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || die "could not read a 40-hex REVIEWED_COMMIT from ${publisher}"
[[ -n "${toolchain}" ]] || die "could not read TOOLCHAIN from ${publisher}"

git -C "${root}" cat-file -e "${reviewed_commit}^{commit}" 2>/dev/null \
  || die "REVIEWED_COMMIT ${reviewed_commit} is not present; fetch the full history first"

expected_sha256="$(awk '$2 == "lighter-benchmark-verifier" { print $1 }' "${trusted_dir}/SHA256SUMS")"
[[ "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "could not read the pinned SHA-256 from SHA256SUMS"
published_sha256="$(shasum -a 256 "${published}" | awk '{print $1}')"
[[ "${published_sha256}" == "${expected_sha256}" ]] \
  || die "the checked-in verifier does not match its own SHA256SUMS pin; stop and investigate"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/lighter-verifier-repro.XXXXXX")"
scratch="$(cd "${scratch}" && pwd -P)"
cleanup() {
  if (( keep == 1 )); then
    echo "scratch build directory kept at ${scratch}"
  else
    rm -rf "${scratch}"
  fi
}
trap cleanup EXIT

echo "reviewed commit: ${reviewed_commit}"
echo "toolchain:       ${toolchain}"
echo "pinned SHA-256:  ${expected_sha256}"
echo

rustup toolchain install "${toolchain}" --profile minimal --no-self-update

# Materialize the reviewed source read-only. `git archive` is used instead of
# `git worktree add` so this check never mutates the caller's repository.
rebuild() {
  local index="$1"
  local checkout="${scratch}/build-${index}/.trusted-benchmark"
  local target="${scratch}/build-${index}/target/trusted-author-build"
  local output_dir="${scratch}/out-${index}"
  local output="${output_dir}/lighter-benchmark-verifier"

  mkdir -p "${checkout}" "${output_dir}"
  git -C "${root}" archive "${reviewed_commit}" | tar -x -C "${checkout}"

  (
    cd "${checkout}"
    CARGO_INCREMENTAL=0 \
    MACOSX_DEPLOYMENT_TARGET=11.0 \
    RUSTFLAGS="-C target-cpu=apple-m1" \
      cargo "+${toolchain}" build --locked --release \
        --manifest-path benchmark-tools/harness/Cargo.toml \
        --target-dir "${target}"
  ) >&2

  # Publish exactly the way build-trusted-verifier.sh does. The basename matters:
  # codesign derives the ad-hoc code directory identifier from it.
  cp "${target}/release/lighter-benchmark-harness" "${output}"
  chmod 755 "${output}"
  codesign --force --sign - "${output}" >&2
  shasum -a 256 "${output}" | awk '{print $1}'
}

declare -a rebuilt=()
for ((index = 1; index <= builds; index++)); do
  echo "--- rebuild ${index} of ${builds}" >&2
  rebuilt+=("$(rebuild "${index}")")
done

echo
for ((index = 0; index < builds; index++)); do
  printf 'rebuild %d SHA-256: %s\n' "$((index + 1))" "${rebuilt[index]}"
done
printf 'pinned    SHA-256: %s\n' "${expected_sha256}"
echo

if [[ "${rebuilt[0]}" == "${expected_sha256}" ]]; then
  echo "REPRODUCED: the rebuild from ${reviewed_commit} matches the pinned binary byte for byte."
  exit 0
fi

echo "NOT REPRODUCED: the rebuild does not match the pinned binary." >&2
if (( builds > 1 )) && [[ "${rebuilt[0]}" != "${rebuilt[1]}" ]]; then
  cat >&2 <<'EOF'

Two rebuilds of the same commit, on this machine, with the same toolchain and
flags, also disagree with each other. The build itself is not deterministic, so
this tells you nothing about whether the pinned binary corresponds to the
reviewed source. See the "Reproducibility" section of
benchmark-tools/trusted/README.md for the known cause (compile-time entropy from
const-random) and for what a future publication has to change.
EOF
elif (( builds > 1 )); then
  cat >&2 <<'EOF'

The two rebuilds agree with each other but not with the pinned binary. The build
is reproducible here, so the pinned binary was produced from something other
than this source, this toolchain, or this environment. Investigate before
trusting the pin.
EOF
fi
exit 1
