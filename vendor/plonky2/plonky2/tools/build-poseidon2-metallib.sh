#!/usr/bin/env bash
# Copyright (c) Elliot Technologies, Inc.
# SPDX-License-Identifier: BUSL-1.1
#
# Regenerates BOTH precompiled blobs and BOTH digests in build.rs, in one shot,
# so no pair can be updated independently:
#
#   poseidon2.metal  --xcrun metal-->  poseidon2.metallib   (AIR; skips the
#                                       Metal front end at startup)
#   poseidon2.metallib --GPU-------->  poseidon2.binarchive (lowered pipelines;
#                                       skips the Metal back end at startup)
#
# The archive step needs a real GPU, so it runs the prover workspace's
# `gen-gpu-archive` binary on this machine. The archive it produces is valid for
# this GPU family and driver; if the machine that later runs the prover rejects
# it, Metal simply lowers the pipelines as it did before — a miss costs the
# speedup and nothing else.
#
# DEVELOPER TOOL ONLY. It is never invoked by a build: the ranked build
# environment is not guaranteed to have the Metal toolchain, so build.rs
# verifies the checked-in blob rather than producing it. Run this after ANY
# edit to poseidon2.metal, then rebuild — `cargo build` prints a warning for as
# long as the blob is stale, and the prover silently falls back to compiling
# the shader from source (correct, just slower to start).

set -euo pipefail

crate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
shader="${crate}/src/hash/poseidon2/poseidon2.metal"
metallib="${crate}/src/hash/poseidon2/poseidon2.metallib"
archive="${crate}/src/hash/poseidon2/poseidon2.binarchive"
build_rs="${crate}/build.rs"
workspace="$(cd "${crate}/../../.." && pwd -P)"

die() { echo "build-poseidon2-metallib.sh: $*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin ]] || die "macOS is required"
command -v xcrun >/dev/null 2>&1 || die "xcrun is required"
xcrun --sdk macosx --find metal >/dev/null 2>&1 \
  || die "the Metal toolchain is not installed (xcrun --sdk macosx --find metal failed).
Install it from Xcode > Settings > Components, or run: xcodebuild -downloadComponent MetalToolchain"
[[ -f "${shader}" ]] || die "missing ${shader}"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# No -ffast-math / -O overrides: the runtime fallback uses a default-constructed
# MTLCompileOptions, and these kernels are pure 64-bit integer arithmetic
# (Goldilocks limbs), so float-math flags cannot change a single value either
# way. Keeping both paths on compiler defaults is what makes them equivalent.
xcrun -sdk macosx metal -c "${shader}" -o "${tmp}/poseidon2.air"
xcrun -sdk macosx metallib "${tmp}/poseidon2.air" -o "${tmp}/poseidon2.metallib"
mv "${tmp}/poseidon2.metallib" "${metallib}"

# The archive is lowered from the metallib that was just written, by the same
# `ARCHIVED_KERNELS` list the prover loads, so it can never cover a different
# kernel set than the one the prover builds pipelines for.
(
  cd "${workspace}"
  cargo run --release --quiet --bin gen-gpu-archive -- "${archive}"
)

fnv() {
  python3 - "$1" <<'PY'
import sys

# Must stay bit-identical to `digest()` in build.rs.
OFFSET = 0x6c62272e07bb014262b821756295c58d
PRIME = 0x0000000001000000000000000000013b
MASK = (1 << 128) - 1

data = open(sys.argv[1], 'rb').read()
h = OFFSET
for b in data:
    h = ((h ^ b) * PRIME) & MASK
h = ((h ^ len(data)) * PRIME) & MASK
print(f'0x{h:032x}')
PY
}

shader_digest="$(fnv "${shader}")"
metallib_digest="$(fnv "${metallib}")"

python3 - "${build_rs}" "${shader_digest}" "${metallib_digest}" <<'PY'
import re
import sys

path, shader_digest, metallib_digest = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
for name, value in (
    ('RECORDED_SHADER_DIGEST', shader_digest),
    ('RECORDED_METALLIB_DIGEST', metallib_digest),
):
    text, n = re.subn(
        rf'(const {name}: u128 = )0x[0-9a-fA-F_]+(;)',
        lambda m: m.group(1) + value + m.group(2),
        text,
    )
    if n != 1:
        raise SystemExit(f'expected exactly one {name} assignment, found {n}')
open(path, 'w').write(text)
PY

echo "regenerated ${metallib} ($(wc -c <"${metallib}" | tr -d ' ') bytes)"
echo "regenerated ${archive} ($(wc -c <"${archive}" | tr -d ' ') bytes)"
echo "recorded shader digest   ${shader_digest} in ${build_rs}"
echo "recorded metallib digest ${metallib_digest} in ${build_rs}"
