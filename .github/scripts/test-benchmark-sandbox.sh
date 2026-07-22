#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
writer="${root}/.github/scripts/write-benchmark-sandbox-profile.sh"
scratch="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-test.XXXXXX")"
outside="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-outside.XXXXXX")"
profile="$(/usr/bin/mktemp -t lighter-sandbox-test.XXXXXX.sb)"
trap '/bin/rm -rf "${scratch}" "${outside}" "${profile}"' EXIT

[[ "$(/usr/bin/uname -s)" == Darwin ]]
[[ -x /usr/bin/sandbox-exec ]]
"${writer}" "${scratch}" "${profile}"

/usr/bin/sandbox-exec -f "${profile}" /bin/sh -c \
  'printf permitted > "$1/scratch-write"' sandbox-test "${scratch}"
[[ -f "${scratch}/scratch-write" ]]

if /usr/bin/sandbox-exec -f "${profile}" /bin/sh -c \
  'printf forbidden > "$1/outside-write"' sandbox-test "${outside}" \
  >/dev/null 2>&1; then
  echo "sandbox allowed a non-scratch write" >&2
  exit 1
fi

if /usr/bin/sandbox-exec -f "${profile}" /bin/sh -c \
  '/bin/sleep 0 & wait' >/dev/null 2>&1; then
  echo "sandbox allowed process fork" >&2
  exit 1
fi

if /usr/bin/sandbox-exec -f "${profile}" \
  /usr/bin/curl -sS --max-time 3 https://1.1.1.1/ >/dev/null 2>&1; then
  echo "sandbox allowed network access" >&2
  exit 1
fi

if /usr/bin/sandbox-exec -f "${profile}" \
  /usr/bin/dscacheutil -q host -a name example.com 2>/dev/null |
    /usr/bin/grep -q ip_address; then
  echo "sandbox allowed resolver IPC" >&2
  exit 1
fi
