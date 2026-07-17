#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
bench_source="${root}/challenge/target/release/bench"
fixture="${root}/testdata/prover/public/bench_test_32.json"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/lighter-prover-timeout.XXXXXX")"
bin="${tmp}/bin"
scratch="${tmp}/scratch"

cleanup() {
  rm -rf "${tmp}"
}
trap cleanup EXIT

mkdir -p "${bin}" "${scratch}"
cp "${bench_source}" "${bin}/bench"
cat > "${bin}/prove" <<'EOF'
#!/bin/bash
exec /bin/sleep 2
EOF
chmod +x "${bin}/prove"
cp "${fixture}" "${scratch}/bench_test.json"

set +e
output="$(
  cd "${scratch}"
  LIGHTER_PROVE_TIMEOUT_SECONDS=1 \
    "${bin}/bench" local-iterate 32 timeout-test "${scratch}/score.json" 2>&1
)"
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
  echo "sleeping prover unexpectedly passed" >&2
  exit 1
fi
if [[ "${output}" != *"prover timed out after 1s"* ]]; then
  echo "expected the prover timeout, got:" >&2
  printf '%s\n' "${output}" >&2
  exit 1
fi
