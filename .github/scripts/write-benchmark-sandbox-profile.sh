#!/bin/bash
set -euo pipefail

[[ "$#" -eq 2 ]] || {
  echo "usage: write-benchmark-sandbox-profile.sh SCRATCH PROFILE" >&2
  exit 2
}

scratch="$1"
profile="$2"
scratch="$(cd "${scratch}" && pwd -P)"
[[ "${scratch}" != *\"* && "${scratch}" != *\\* && "${scratch}" != *$'\n'* ]] || {
  echo "sandbox profile: scratch path contains unsupported characters" >&2
  exit 1
}

umask 077
printf '%s\n' \
  '(version 1)' \
  '(allow default)' \
  '(deny network*)' \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.system.mDNSResponder"))' \
  '(deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder.dnsproxy"))' \
  '(deny process-fork)' \
  '(deny file-write*)' \
  "(allow file-write* (subpath \"${scratch}\"))" \
  > "${profile}"
