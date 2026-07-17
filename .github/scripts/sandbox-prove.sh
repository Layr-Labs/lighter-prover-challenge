#!/bin/bash
set -euo pipefail

bin_dir="$(cd "$(dirname "$0")" && pwd -P)"
scratch="$(pwd -P)"
exec /usr/bin/sandbox-exec \
  -D "SCRATCH=${scratch}" \
  -f "${bin_dir}/prover.sb" \
  "${bin_dir}/prove-bin" "$@"
