#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
wrapper="${root}/.github/scripts/prove-via-bench.sh"

fail() {
  echo "prove-via-bench contract: $*" >&2
  exit 1
}

[[ -f "${wrapper}" && ! -L "${wrapper}" ]] || fail "trusted wrapper is missing or unsafe"

for expected in \
  '[[ "$#" -eq 2 ]]' \
  'script_dir="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && /bin/pwd -P)"' \
  'worker="${script_dir}/prove-bin"' \
  '[[ -f "${worker}" && ! -L "${worker}" && -x "${worker}" ]]' \
  'exec /usr/bin/sudo -n \' \
  '/opt/lighter-prover-challenge/bench-exec.sh \' \
  'prove "${worker}" "$1" "$2" \' \
  '</dev/null >/dev/null 2>/dev/null'; do
  rg -Fq -- "${expected}" "${wrapper}" ||
    fail "missing static contract: ${expected}"
done

for forbidden in \
  '"$@"' \
  'sudo -E' \
  'R2_ACCESS_KEY_ID' \
  'R2_SECRET_ACCESS_KEY' \
  'R2_BUCKET_ENDPOINT' \
  'GITHUB_TOKEN' \
  'GH_TOKEN'; do
  if rg -Fq -- "${forbidden}" "${wrapper}"; then
    fail "wrapper forwards forbidden state: ${forbidden}"
  fi
done

for args in 'fixture-only' 'fixture proof extra'; do
  status=0
  # shellcheck disable=SC2086
  /bin/bash "${wrapper}" ${args} >/dev/null 2>&1 || status=$?
  [[ "${status}" == 2 ]] ||
    fail "wrong argument count returned ${status}, expected 2"
done

echo "PASS: prove-via-bench uses adjacent prove-bin and the fixed root bridge"
