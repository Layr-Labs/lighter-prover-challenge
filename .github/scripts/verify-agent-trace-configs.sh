#!/bin/bash
# Verifies the checked-in coding-agent trace hook files against the exact content the challenge
# CLI's trace installer generates. All eight managed files — the four agent configs, the trace
# wrapper, the OpenCode plugin, and the OMP and Pi extensions — are compared byte for byte against a
# locally regenerated or digest-pinned copy, so any edit that is not a bump of the inert
# trace-config version marker fails here.
#
#   verify-agent-trace-configs.sh [<repo-root>]      check the working tree (default: this repo)
#   verify-agent-trace-configs.sh --emit <dir>       write the expected files into <dir>
#
# The templates below are the source of truth, so --emit is how the managed files are refreshed:
# emit into a scratch directory, diff it against the repository, and commit the result. The OpenCode
# plugin and the two Pi-family extensions are verbatim CLI source rather than templates, so --emit
# does not produce them; copy those files from the CLI release being adopted and update
# PLUGIN_SHA256 / EXTENSION_SHA256.
#
# Refreshing after an intentional CLI change: run the check, read the diff it prints, confirm the
# new content against the CLI's trace installer, then update the templates here in the same
# reviewed commit. Reviewing that template diff is the point — a bare hash bump is not a review.
set -euo pipefail

readonly CLI=yukon
readonly WRAPPER=".${CLI}/hooks/${CLI}-trace.sh"
readonly PLUGIN=".opencode/plugin/${CLI}-trace.ts"
readonly OMP_EXTENSION=".omp/extensions/${CLI}-trace.ts"
readonly PI_EXTENSION=".pi/extensions/${CLI}-trace.ts"

# The CLI's TRACE_CONFIG_VERSION, used only by --emit. A checkout may carry any version marker;
# see the comment on the wrapper check below for why that digit is not a security boundary.
readonly TRACE_CONFIG_VERSION=15

# sha256 of .opencode/plugin/<cli>-trace.ts, which is opaque TypeScript emitted verbatim by the
# CLI and so is pinned by digest rather than regenerated.
readonly PLUGIN_SHA256=a364b32825558f5862e7c25f6f878da78a2079a8471cbe55629509b909726780

# sha256 of the OMP and Pi extensions, likewise verbatim CLI source. One digest covers both files:
# the CLI emits a single Pi-family extension source for both adapters, which decides at runtime
# which of the two it is running under, so the bytes are identical and only the path differs. Each
# path is still verified separately, so a missing or tampered file is caught whichever one it is.
# Confirmed byte-identical to the copies checked in to Layr-Labs/flock-challenge.
readonly EXTENSION_SHA256=1eb56e819f1810b0ac9ead2dba9fae3aae6dae7849703f343ad0735d53a32d21

mode=verify
emit_dir=""
if [[ "${1:-}" == --emit ]]; then
  [[ -n "${2:-}" ]] || { echo "usage: $0 --emit <dir>" >&2; exit 2; }
  mode=emit
  emit_dir="$2"
  shift 2
fi

fail() {
  echo "agent trace config check failed: $*" >&2
  exit 1
}

# Writes every managed file this script can generate into "$1". The OpenCode plugin is excluded:
# it is CLI source, not a template.
generate_managed_files() {
  local dir="$1" version="$2"
  mkdir -p "${dir}/.${CLI}/hooks" "${dir}/.cursor" "${dir}/.claude" "${dir}/.codex"

  sed -e "s/@CLI@/${CLI}/g" -e "s/@VERSION@/${version}/g" > "${dir}/${WRAPPER}" <<'WRAPPER'
#!/usr/bin/env sh
# @CLI@-trace-config-v@VERSION@
# Thin transport only: the CLI owns identity, checkpoints, source handling, retry, and upload.
agent="${1:-unknown}"
kind="${2:-capture}"
detail="${3:-}"
repo="$(CDPATH= cd -- "$(dirname -- "$0")/../.." 2>/dev/null && pwd)" || repo="$PWD"
payload="$(cat 2>/dev/null || true)"
if [ "$kind" = "session" ]; then
  cd "$repo" 2>/dev/null || exit 1
  printf '%s' "$payload" | @CLI@ trace session "$agent"
  exit $?
fi
( cd "$repo" 2>/dev/null && printf '%s' "$payload" | @CLI@ trace hook "$agent" >/dev/null 2>&1 & ) >/dev/null 2>&1 || true
# Codex consumes command-hook stdout. UserPromptSubmit must remain empty because plain text is
# injected into the prompt; the other capture hooks accept this explicit continue response.
[ "$agent" = "codex" ] && [ "$detail" != "UserPromptSubmit" ] && printf '{"continue":true}\n'
exit 0
WRAPPER
  chmod 0644 "${dir}/${WRAPPER}"

  jq -n \
    --arg w "sh \"${WRAPPER}\" cursor" \
    --arg cli "${CLI}" \
    '{
      version: 1,
      hooks: {
        sessionStart: [{command: ($w + " session")}, {command: ($w + " capture")}],
        beforeShellExecution: [{command: ($w + " session"), matcher: $cli}],
        beforeSubmitPrompt: [{command: ($w + " capture")}],
        stop: [{command: ($w + " capture")}],
        sessionEnd: [{command: ($w + " capture")}],
        subagentStop: [{command: ($w + " capture")}]
      }
    }' > "${dir}/.cursor/hooks.json"

  jq -n \
    --arg w "sh \"\$CLAUDE_PROJECT_DIR/${WRAPPER}\" claude" \
    '{
      hooks: {
        SessionStart: [{hooks: [
          {type: "command", command: ($w + " session")},
          {type: "command", command: ($w + " capture")}
        ]}],
        SubagentStart: [{hooks: [{type: "command", command: ($w + " session")}]}],
        UserPromptSubmit: [{hooks: [{type: "command", command: ($w + " capture")}]}],
        Stop: [{hooks: [{type: "command", command: ($w + " capture")}]}],
        StopFailure: [{hooks: [{type: "command", command: ($w + " capture")}]}],
        SessionEnd: [{hooks: [{type: "command", command: ($w + " capture")}]}],
        SubagentStop: [{hooks: [{type: "command", command: ($w + " capture")}]}]
      }
    }' > "${dir}/.claude/settings.local.json"

  jq -n --arg p "./${PLUGIN}" \
    '{"$schema": "https://opencode.ai/config.json", plugin: [$p]}' > "${dir}/opencode.json"

  # Only the managed block is CLI-generated; the installer preserves everything outside it, so the
  # preamble is repository-owned text pinned here as a literal.
  {
    printf '# Yukon trace hooks are managed by `%s trace init`.\n\n' "${CLI}"
    sed -e "s/@CLI@/${CLI}/g" <<'CODEX'
# >>> @CLI@ trace hook (managed) >>>
[[hooks.SessionStart]]
[[hooks.SessionStart.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex session"
[[hooks.SessionStart.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex capture SessionStart"
[[hooks.SubagentStart]]
[[hooks.SubagentStart.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex session"
[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex capture UserPromptSubmit"
[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex capture Stop"
[[hooks.SubagentStop]]
[[hooks.SubagentStop.hooks]]
type = "command"
command = "sh \"$(git rev-parse --show-toplevel)/.@CLI@/hooks/@CLI@-trace.sh\" codex capture SubagentStop"
# <<< @CLI@ trace hook (managed) <<<
CODEX
  } > "${dir}/.codex/config.toml"
}

if [[ "${mode}" == emit ]]; then
  mkdir -p "${emit_dir}"
  generate_managed_files "${emit_dir}" "${TRACE_CONFIG_VERSION}"
  echo "wrote expected agent trace files to ${emit_dir} (cli=${CLI}, trace-config v${TRACE_CONFIG_VERSION})"
  echo "note: ${PLUGIN}, ${OMP_EXTENSION} and ${PI_EXTENSION} are CLI source and are not"
  echo "      generated; copy them from the CLI release"
  exit 0
fi

root="${1:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${root}"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/lighter-agent-configs.XXXXXX")"
cleanup() { rm -rf "${tmp}"; }
trap cleanup EXIT

expect_regular_file() {
  [[ -f "$1" && ! -L "$1" ]] || fail "$1 is missing or is not a regular file"
}

compare_expected() {
  local path="$1"
  if ! cmp -s "${tmp}/expected/${path}" "${path}"; then
    echo "--- expected ${path}" >&2
    echo "+++ actual   ${path}" >&2
    diff -u "${tmp}/expected/${path}" "${path}" >&2 || true
    fail "${path} does not match the content the CLI trace installer generates"
  fi
}

expect_tree() {
  local dir="$1"
  shift
  printf '%s\n' "$@" | LC_ALL=C sort > "${tmp}/expected-tree"
  find "${dir}" -mindepth 1 | LC_ALL=C sort > "${tmp}/actual-tree"
  if ! cmp -s "${tmp}/expected-tree" "${tmp}/actual-tree"; then
    diff -u "${tmp}/expected-tree" "${tmp}/actual-tree" >&2 || true
    fail "${dir}/ contains unexpected or missing entries"
  fi
}

# 1. Inventory. The managed files must all be present as regular files, and no unmanaged agent
#    surface may sit beside them: without this an attacker could leave every pinned file untouched
#    and add a second, unpinned hook file. .omp/ and .pi/ are managed here because `yukon trace init`
#    writes them for those adapters and because the CLI's routine refresh only touches adapters that
#    already have a config present. They are pinned by digest below, so they are validated rather
#    than tolerated.
for path in \
  "${WRAPPER}" \
  "${PLUGIN}" \
  "${OMP_EXTENSION}" \
  "${PI_EXTENSION}" \
  .claude/settings.local.json \
  .cursor/hooks.json \
  .codex/config.toml \
  opencode.json
do
  expect_regular_file "${path}"
done

expect_tree ".${CLI}" ".${CLI}/hooks" "${WRAPPER}"
expect_tree .opencode .opencode/plugin "${PLUGIN}"
expect_tree .omp .omp/extensions "${OMP_EXTENSION}"
expect_tree .pi .pi/extensions "${PI_EXTENSION}"
expect_tree .claude .claude/settings.local.json
expect_tree .cursor .cursor/hooks.json
expect_tree .codex .codex/config.toml

# 2. Regenerate and compare. Only the digits of the wrapper's version marker are free: that marker
#    is a comment the CLI reads to decide whether to reinstall, every executable line around it is
#    fixed, and leaving it free means a trace-config bump that changes nothing else stays green.
version="$(sed -n "2s/^# ${CLI}-trace-config-v\([0-9][0-9]*\)\$/\1/p" "${WRAPPER}")"
[[ -n "${version}" ]] || fail "${WRAPPER} line 2 is not a '# ${CLI}-trace-config-v<N>' marker"

generate_managed_files "${tmp}/expected" "${version}"
for path in \
  "${WRAPPER}" \
  .cursor/hooks.json \
  .claude/settings.local.json \
  opencode.json \
  .codex/config.toml
do
  compare_expected "${path}"
done

# 3. The three TypeScript files the CLI emits verbatim — the OpenCode plugin and the OMP and Pi
#    extensions — are pinned by digest rather than regenerated, plus an explicit check that the
#    wrapper each one spawns is the one this checkout ships. The digest subsumes that check today; it
#    is kept so that a careless digest refresh still cannot bless a file pointed somewhere else.
#    Note that these files legitimately contain Yukon session-id environment variable names, so
#    this remains a digest comparison rather than a content-pattern check.
check_cli_source() {
  local path="$1" expected="$2" actual
  grep -Fqx "const WRAPPER = \"${WRAPPER}\";" "${path}" \
    || fail "${path} does not spawn ${WRAPPER}"
  actual="$(shasum -a 256 "${path}" | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] \
    || fail "${path} sha256 is ${actual}, expected ${expected}"
}

check_cli_source "${PLUGIN}" "${PLUGIN_SHA256}"
check_cli_source "${OMP_EXTENSION}" "${EXTENSION_SHA256}"
check_cli_source "${PI_EXTENSION}" "${EXTENSION_SHA256}"

echo "agent trace config check passed (cli=${CLI}, trace-config v${version})"
