#!/bin/bash
set -euo pipefail

[[ "$#" -eq 2 ]] || {
  echo "usage: prove FIXTURE PROOF" >&2
  exit 2
}

script_dir="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && /bin/pwd -P)"
worker="${script_dir}/prove-bin"
[[ -f "${worker}" && ! -L "${worker}" && -x "${worker}" ]] || {
  echo "prove: adjacent prove-bin is missing or unsafe" >&2
  exit 1
}

exec /usr/bin/sudo -n \
  /opt/lighter-prover-challenge/bench-exec.sh \
  prove "${worker}" "$1" "$2" \
  </dev/null >/dev/null 2>/dev/null
