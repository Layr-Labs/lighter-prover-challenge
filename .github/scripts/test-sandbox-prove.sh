#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd -P)"
launcher="${script_dir}/sandbox-prove.sh"
probe="${script_dir}/prove-bin"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/lighter-prover-sandbox.XXXXXX")"

cleanup() {
  rm -f "${probe}"
  rm -rf "${scratch}"
}
trap cleanup EXIT

cat > "${probe}" <<'EOF'
#!/bin/bash
set -euo pipefail

(
  /bin/sleep 1
  /usr/bin/touch child-ran
) &
EOF
chmod +x "${probe}"

set +e
(
  cd "${scratch}"
  "${launcher}" ignored
) 2>"${scratch}/probe.stderr"
set -e

/bin/sleep 2
if [[ -e "${scratch}/child-ran" ]]; then
  echo "sandbox allowed a child process after prove-bin exited" >&2
  exit 1
fi
