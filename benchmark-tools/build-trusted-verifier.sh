#!/usr/bin/env bash
# Author-only reproducibility tool. Ranked runs never execute this script.
# It materializes the reviewed source in the same detached-worktree pattern as
# the Flock challenge, then publishes the verifier into the caller's worktree.
set -euo pipefail

readonly REVIEWED_COMMIT=REPLACE_AFTER_SOURCE_COMMIT
readonly TOOLCHAIN=nightly-2025-12-06

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
checkout="${root}/.trusted-benchmark"
target="${root}/target/trusted-author-build"
output="${root}/benchmark-tools/trusted"

if [[ "${REVIEWED_COMMIT}" == REPLACE_AFTER_SOURCE_COMMIT ]]; then
  echo "set REVIEWED_COMMIT after the protected source diff is reviewed and committed" >&2
  exit 1
fi

if [[ ! -d "${checkout}/.git" && ! -f "${checkout}/.git" ]]; then
  git -C "${root}" worktree add --detach "${checkout}" "${REVIEWED_COMMIT}"
else
  [[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] || {
    echo "trusted checkout is not clean" >&2
    exit 1
  }
  git -C "${checkout}" checkout --detach "${REVIEWED_COMMIT}"
fi
[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${REVIEWED_COMMIT}" ]] || {
  echo "trusted checkout is not ${REVIEWED_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] || {
  echo "trusted checkout is not clean" >&2
  exit 1
}

rustup toolchain install "${TOOLCHAIN}" --profile minimal --no-self-update
(
  cd "${checkout}"
  CARGO_INCREMENTAL=0 \
  MACOSX_DEPLOYMENT_TARGET=11.0 \
  RUSTFLAGS="-C target-cpu=apple-m1" \
    cargo "+${TOOLCHAIN}" build --locked --release \
      --manifest-path benchmark-tools/harness/Cargo.toml \
      --target-dir "${target}"
)

mkdir -p "${output}"
cp "${target}/release/lighter-benchmark-harness" \
  "${output}/lighter-benchmark-verifier"
chmod 755 "${output}/lighter-benchmark-verifier"
codesign --force --sign - "${output}/lighter-benchmark-verifier"
(
  cd "${output}"
  shasum -a 256 lighter-benchmark-verifier > SHA256SUMS
)

echo "wrote ${output}/lighter-benchmark-verifier"
cat "${output}/SHA256SUMS"
