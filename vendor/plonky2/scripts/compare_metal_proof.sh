#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/tmp/metal-proof-compare}"
CPU_PROOF="$OUT_DIR/proof_cpu.bin"
METAL_PROOF="$OUT_DIR/proof_metal.bin"

mkdir -p "$OUT_DIR"
rm -f "$CPU_PROOF" "$METAL_PROOF"

cd "$ROOT_DIR"

echo "Generating CPU proof..."
RUSTC_BOOTSTRAP=1 cargo run -p plonky2 --example fibonacci_proof_bin -- "$CPU_PROOF"

echo "Generating Metal proof..."
RUSTC_BOOTSTRAP=1 cargo run -p plonky2 --features metal --example fibonacci_proof_bin -- "$METAL_PROOF"

echo "CPU  : $(shasum -a 256 "$CPU_PROOF")"
echo "Metal: $(shasum -a 256 "$METAL_PROOF")"

if cmp -s "$CPU_PROOF" "$METAL_PROOF"; then
    echo "proofs are identical"
else
    echo "proofs differ" >&2
    exit 1
fi
