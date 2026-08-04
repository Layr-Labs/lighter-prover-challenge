#!/bin/bash
set -euo pipefail

[[ "$#" -ge 3 ]] || {
  echo "usage: write-benchmark-sandbox-profile.sh SCRATCH PROFILE WORKER [READ_ROOT...]" >&2
  exit 2
}

scratch="$1"
profile="$2"
worker="$3"
shift 3

reject_unsupported() {
  [[ "$1" != *\"* && "$1" != *\\* && "$1" != *$'\n'* ]] || {
    echo "sandbox profile: path contains unsupported characters" >&2
    exit 1
  }
}

resolve_dir() {
  local path
  path="$(cd "$1" && pwd -P)"
  reject_unsupported "${path}"
  printf '%s' "${path}"
}

resolve_file() {
  local path
  path="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")"
  reject_unsupported "${path}"
  printf '%s' "${path}"
}

scratch="$(resolve_dir "${scratch}")"
worker="$(resolve_file "${worker}")"
[[ -f "${worker}" && ! -L "${worker}" && -x "${worker}" ]] || {
  echo "sandbox profile: worker must be a regular executable non-symlink file" >&2
  exit 1
}

read_roots=()
add_read_root() {
  local candidate="$1" existing
  for existing in ${read_roots[@]+"${read_roots[@]}"}; do
    [[ "${existing}" != "${candidate}" ]] || return 0
  done
  read_roots+=("${candidate}")
}

add_read_root "${scratch}"
add_read_root "$(dirname "${worker}")"
for read_root in "$@"; do
  add_read_root "$(resolve_dir "${read_root}")"
done

umask 077
{
  # Default-deny allowlist. Every operation the candidate worker is permitted to
  # perform is enumerated below; anything absent is denied by (deny default).
  printf '%s\n' '(version 1)' '(deny default)'

  # Redundant under (deny default), but kept as explicit, testable statements of
  # the load-bearing denials so a future broad allow cannot silently undo them.
  printf '%s\n' \
    '(deny network*)' \
    '(deny mach-lookup (global-name "com.apple.mDNSResponder"))' \
    '(deny mach-lookup (global-name "com.apple.system.mDNSResponder"))' \
    '(deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))' \
    '(deny mach-lookup (global-name "com.apple.mDNSResponder.dnsproxy"))' \
    '(deny process-fork)' \
    '(deny process-exec*)' \
    '(deny file-write*)'

  # Denials the prover triggers thousands of times per run, silenced so the
  # kernel stops generating a backtraced violation report for each one. A
  # 500-transaction ranked run produced ~2300 reports, 99% of them these:
  # dirhelper and opendirectoryd back confstr(_CS_DARWIN_USER_*_DIR) and the
  # passwd/group lookups that Metal and the Rust runtime retry on every miss.
  # These stay denied; only the reporting is suppressed, and every *other*
  # denial still reports so an unexpected one remains diagnosable via
  # `log stream --predicate 'eventMessage contains "deny("'`.
  printf '%s\n' \
    '(deny mach-lookup (global-name "com.apple.bsd.dirhelper") (with no-report))' \
    '(deny mach-lookup (global-name "com.apple.system.opendirectoryd.membership") (with no-report))' \
    '(deny mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo") (with no-report))' \
    '(deny file-read* (literal "/private/etc/passwd") (with no-report))'

  # dyld opens the volume root to locate the shared cache before main() runs.
  printf '%s\n' '(allow file-read* (literal "/"))'
  # stat(2)-level metadata on any path. Path resolution needs it on every
  # component, including the /var, /tmp and /etc symlinks, and getcwd() needs it
  # on every ancestor of the working directory. File *contents* stay allowlisted
  # below, so this grants no way to read data.
  printf '%s\n' '(allow file-read-metadata)'
  # Read-only system code and data: the dyld shared cache and frameworks, the
  # Metal GPU driver bundle (/System/Library/Extensions/AGXMetal*.bundle), the
  # IOGPU private framework, and the ICU locale tables.
  printf '%s\n' \
    '(allow file-read* (subpath "/System"))' \
    '(allow file-read* (subpath "/usr/lib"))' \
    '(allow file-read* (subpath "/usr/share"))'
  # Some Mac models ship the GPU driver bundle here instead of /System.
  printf '%s\n' '(allow file-read* (subpath "/Library/GPUBundles"))'
  # Metal's per-user shader/pipeline cache, scoped to the com.apple.metal
  # directories inside the per-user Darwin cache dir. Read-only: writing it stays
  # denied, so a cache miss costs only a cold pipeline compile.
  printf '%s\n' \
    '(allow file-read* (regex #"^/private/var/folders/[^/]+/[^/]+/C/com\.apple\.metal"))'
  # Character devices used by the Rust runtime and by discarded worker output.
  printf '%s\n' \
    '(allow file-read* (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))' \
    '(allow file-write* (literal "/dev/null"))'
  # Host capability and topology queries (page size, core count, GPU family)
  # made by the Rust runtime, rayon, and Metal. Read-only; sysctl-write stays
  # denied, and without it the Rust runtime cannot even install its stack guard.
  printf '%s\n' '(allow sysctl-read)'
  # Metal compiles the prover's embedded metallibs through this service.
  printf '%s\n' '(allow mach-lookup (global-name "com.apple.MTLCompilerService"))'
  # The GPU compute channel itself: the Apple GPU device user client and the
  # IOSurface client Metal allocates its buffers through.
  printf '%s\n' \
    '(allow iokit-open (iokit-user-client-class "AGXDeviceUserClient"))' \
    '(allow iokit-open (iokit-user-client-class "IOGPUDeviceUserClient"))' \
    '(allow iokit-open (iokit-user-client-class "IOSurfaceRootUserClient"))' \
    '(allow iokit-get-properties)'
  # Only the concrete worker binary may be executed. process-fork stays denied,
  # so this cannot be used to spawn a second process.
  printf '(allow process-exec (literal "%s"))\n' "${worker}"
  # The worker's own inputs: its binary directory, the fixture, and the scratch
  # directory. Writes are confined to scratch.
  for read_root in "${read_roots[@]}"; do
    printf '(allow file-read* (subpath "%s"))\n' "${read_root}"
  done
  printf '(allow file-write* (subpath "%s"))\n' "${scratch}"
} > "${profile}"
