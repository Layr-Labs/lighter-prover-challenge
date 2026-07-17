#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${root}"

for tool in cargo rustup; do
  command -v "${tool}" >/dev/null || {
    echo "missing required tool: ${tool}" >&2
    exit 1
  }
done

toolchain="$(tr -d '[:space:]' < rust-toolchain)"
rustup toolchain install "${toolchain}" --profile minimal

RUSTUP_TOOLCHAIN="${toolchain}" \
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
  cargo build --release --locked --manifest-path challenge/Cargo.toml --bins
