#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: download-r2-object.sh OBJECT_PATH OUTPUT_PATH EXPECTED_SHA256" >&2
  exit 2
fi

object_path="$1"
output_path="$2"
expected_sha256="$3"

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY is required}"
: "${R2_BUCKET_ENDPOINT:?R2_BUCKET_ENDPOINT is required}"
: "${R2_BUCKET:?R2_BUCKET is required}"

[[ -n "${object_path}" && "${object_path}" != /* ]] || {
  echo "download-r2-object: invalid object path" >&2
  exit 2
}
case "/${object_path}/" in
  *"/../"*|*"/./"*)
    echo "download-r2-object: object path contains a dot segment" >&2
    exit 2
    ;;
esac
[[ "${object_path}" != *$'\n'* && "${object_path}" != *$'\r'* ]] || {
  echo "download-r2-object: object path contains a control character" >&2
  exit 2
}
[[ "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "download-r2-object: EXPECTED_SHA256 must be 64 lowercase hex characters" >&2
  exit 2
}

endpoint="${R2_BUCKET_ENDPOINT%/}"
[[ "${endpoint}" == https://* ]] || {
  echo "download-r2-object: R2_BUCKET_ENDPOINT must use https" >&2
  exit 2
}
if [[ -n "${R2_AWS_CLI:-}" ]]; then
  [[ "${R2_AWS_CLI}" == /* ]] || {
    echo "download-r2-object: R2_AWS_CLI must be an absolute path" >&2
    exit 2
  }
  [[ -x "${R2_AWS_CLI}" && ! -d "${R2_AWS_CLI}" ]] || {
    echo "download-r2-object: R2_AWS_CLI is not executable: ${R2_AWS_CLI}" >&2
    exit 1
  }
  aws_cli="${R2_AWS_CLI}"
else
  aws_cli="$(command -v aws || true)"
  [[ -n "${aws_cli}" && -x "${aws_cli}" && ! -d "${aws_cli}" ]] || {
    echo "download-r2-object: aws CLI is required" >&2
    exit 1
  }
fi
[[ ! -d "${output_path}" ]] || {
  echo "download-r2-object: OUTPUT_PATH must not be a directory" >&2
  exit 2
}

umask 077
output_dir="$(/usr/bin/dirname "${output_path}")"
/bin/mkdir -p "${output_dir}"
tmp_path="$(/usr/bin/mktemp "${output_path}.tmp.XXXXXX")"
trap '/bin/rm -f "${tmp_path}"' EXIT

AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
AWS_DEFAULT_REGION=auto \
AWS_EC2_METADATA_DISABLED=true \
  "${aws_cli}" --endpoint-url "${endpoint}" \
    s3 cp "s3://${R2_BUCKET}/${object_path}" "${tmp_path}" \
    --only-show-errors --no-progress

[[ -f "${tmp_path}" && ! -L "${tmp_path}" ]] || {
  echo "download-r2-object: download did not produce a regular file" >&2
  exit 1
}
actual_sha256="$(/usr/bin/shasum -a 256 "${tmp_path}" | /usr/bin/awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "download-r2-object: fixture SHA-256 mismatch" >&2
  exit 1
fi

/bin/chmod 600 "${tmp_path}"
/bin/mv -f "${tmp_path}" "${output_path}"
trap - EXIT
