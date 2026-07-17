#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${root}"

case "${1:---local-iterate}" in
  --local-iterate)
    mode=local-iterate
    transactions=32
    fixture=testdata/prover/public/bench_test_32.json
    ;;
  --local-submit)
    mode=local-submit
    transactions=500
    fixture=bench/bench_test.json
    ;;
  *) echo "usage: ./benchmark.sh [--local-iterate|--local-submit]" >&2; exit 2 ;;
esac

bench="${root}/challenge/target/release/bench"
prove="${root}/challenge/target/release/prove"
if [[ ! -x "${bench}" || ! -x "${prove}" ]]; then
  ./setup.sh
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/lighter-prover.XXXXXX")"
trap 'rm -rf "${scratch}"' EXIT

cp "${fixture}" "${scratch}/bench_test.json"

output="${LIGHTER_SCORE_PATH:-score.${mode}.json}"
[[ "${output}" = /* ]] || output="${root}/${output}"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
(cd "${scratch}" && RUST_LOG="info,circuit=error" \
  "${bench}" "${mode}" "${transactions}" "${commit}" "${output}")

if [[ "${output}" != "${root}/score.json" ]]; then
  cp "${output}" "${root}/score.json"
fi
