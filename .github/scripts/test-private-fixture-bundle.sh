#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
extractor="${root}/.github/scripts/extract-private-fixture-bundle.sh"
scratch="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-bundle-test.XXXXXX")"
trap '/bin/rm -rf "${scratch}"' EXIT

fail() {
  echo "private fixture bundle test: $*" >&2
  exit 1
}

[[ -f "${extractor}" && ! -L "${extractor}" && -x "${extractor}" ]] ||
  fail "extractor is missing or unsafe"

make_archive() {
  local source="$1" archive="$2"
  shift 2
  COPYFILE_DISABLE=1 /usr/bin/tar -czf "${archive}" -C "${source}" "$@"
  /bin/chmod 0600 "${archive}"
}

make_named_archive() {
  local archive="$1" name="$2" payload="$3"
  /usr/bin/perl -MArchive::Tar -e '
    my ($archive, $name, $payload) = @ARGV;
    my $tar = Archive::Tar->new;
    $tar->add_data($name, $payload);
    $tar->write($archive, Archive::Tar::COMPRESS_GZIP());
  ' "${archive}" "${name}" "${payload}"
  /bin/chmod 0600 "${archive}"
}

make_duplicate_archive() {
  local archive="$1"
  /usr/bin/perl -MArchive::Tar -e '
    my ($archive) = @ARGV;
    my $tar = Archive::Tar->new;
    $tar->add_data("fixture.json", "{\"synthetic\":1}\n");
    $tar->add_data("fixture.json", "{\"synthetic\":2}\n");
    $tar->write($archive, Archive::Tar::COMPRESS_GZIP());
  ' "${archive}"
  /bin/chmod 0600 "${archive}"
}

assert_private_modes() {
  local directory="$1" directory_mode file file_mode
  directory_mode="$(/usr/bin/stat -f '%Lp' "${directory}")"
  [[ "${directory_mode}" == 700 ]] ||
    fail "destination mode is ${directory_mode}, expected 700"
  for file in "${directory}"/*.json; do
    [[ -f "${file}" && ! -L "${file}" ]] ||
      fail "extracted entry is not a regular non-symlink file"
    file_mode="$(/usr/bin/stat -f '%Lp' "${file}")"
    [[ "${file_mode}" == 600 ]] ||
      fail "extracted file mode is ${file_mode}, expected 600"
  done
}

assert_failure() {
  local archive="$1" output="$2" expected_count="$3" label="$4"
  local stdout="${scratch}/${label}.stdout" stderr="${scratch}/${label}.stderr"
  /bin/rm -rf "${output}"
  if "${extractor}" "${archive}" "${output}" "${expected_count}" \
    >"${stdout}" 2>"${stderr}"; then
    fail "${label} archive was accepted"
  fi
  [[ ! -e "${output}" && ! -L "${output}" ]] ||
    fail "${label} left a partial destination"
  if /usr/bin/grep -Eq '\.json|synthetic-private-marker|[0-9a-f]{64}' \
    "${stdout}" "${stderr}"; then
    fail "${label} output disclosed fixture metadata"
  fi
}

valid_source="${scratch}/valid-source"
/bin/mkdir "${valid_source}"
printf '%s\n' '{"case":"synthetic-a"}' > "${valid_source}/fixture-a.json"
printf '%s\n' '{"case":"synthetic-b"}' > "${valid_source}/fixture-b.json"
printf '%s\n' '{"case":"synthetic-c"}' > "${valid_source}/fixture-c.json"
valid_archive="${scratch}/valid.tar.gz"
make_archive "${valid_source}" "${valid_archive}" \
  fixture-a.json fixture-b.json fixture-c.json

valid_output="${scratch}/valid-output"
valid_stdout="${scratch}/valid.stdout"
valid_stderr="${scratch}/valid.stderr"
"${extractor}" "${valid_archive}" "${valid_output}" 3 \
  >"${valid_stdout}" 2>"${valid_stderr}"
[[ ! -s "${valid_stdout}" && ! -s "${valid_stderr}" ]] ||
  fail "successful extraction produced output"
[[ "$(/usr/bin/find "${valid_output}" -mindepth 1 -maxdepth 1 -type f |
  /usr/bin/wc -l | /usr/bin/tr -d '[:space:]')" == 3 ]] ||
  fail "valid dynamic-count archive did not extract exactly three files"
assert_private_modes "${valid_output}"

assert_failure "${valid_archive}" "${scratch}/missing-output" 4 "missing-count"
assert_failure "${valid_archive}" "${scratch}/extra-output" 2 "extra-count"
assert_failure "${valid_archive}" "${scratch}/zero-output" 0 "zero-count"

non_json_source="${scratch}/non-json-source"
/bin/mkdir "${non_json_source}"
printf '%s\n' 'synthetic-private-marker' > "${non_json_source}/fixture.txt"
non_json_archive="${scratch}/non-json.tar.gz"
make_archive "${non_json_source}" "${non_json_archive}" fixture.txt
assert_failure "${non_json_archive}" "${scratch}/non-json-output" 1 "non-json"

nested_archive="${scratch}/nested.tar.gz"
make_named_archive "${nested_archive}" "nested/fixture.json" \
  '{"case":"synthetic-private-marker"}'
assert_failure "${nested_archive}" "${scratch}/nested-output" 1 "nested"

traversal_archive="${scratch}/traversal.tar.gz"
make_named_archive "${traversal_archive}" "../fixture.json" \
  '{"case":"synthetic-private-marker"}'
assert_failure "${traversal_archive}" "${scratch}/traversal-output" 1 "traversal"

absolute_archive="${scratch}/absolute.tar.gz"
make_named_archive "${absolute_archive}" "/fixture.json" \
  '{"case":"synthetic-private-marker"}'
assert_failure "${absolute_archive}" "${scratch}/absolute-output" 1 "absolute"

control_archive="${scratch}/control.tar.gz"
make_named_archive "${control_archive}" $'fixture\nbreak.json' \
  '{"case":"synthetic-private-marker"}'
assert_failure "${control_archive}" "${scratch}/control-output" 1 "control"

duplicate_archive="${scratch}/duplicate.tar.gz"
make_duplicate_archive "${duplicate_archive}"
assert_failure "${duplicate_archive}" "${scratch}/duplicate-output" 2 "duplicate"

link_source="${scratch}/link-source"
/bin/mkdir "${link_source}"
printf '%s\n' '{"case":"synthetic-private-marker"}' > "${link_source}/fixture-a.json"
/bin/ln -s fixture-a.json "${link_source}/fixture-b.json"
symlink_archive="${scratch}/symlink.tar.gz"
make_archive "${link_source}" "${symlink_archive}" fixture-a.json fixture-b.json
assert_failure "${symlink_archive}" "${scratch}/symlink-output" 2 "symlink"

/bin/rm "${link_source}/fixture-b.json"
/bin/ln "${link_source}/fixture-a.json" "${link_source}/fixture-b.json"
hardlink_archive="${scratch}/hardlink.tar.gz"
make_archive "${link_source}" "${hardlink_archive}" fixture-a.json fixture-b.json
assert_failure "${hardlink_archive}" "${scratch}/hardlink-output" 2 "hardlink"

unsafe_mode_archive="${scratch}/unsafe-mode.tar.gz"
/bin/cp "${valid_archive}" "${unsafe_mode_archive}"
/bin/chmod 0644 "${unsafe_mode_archive}"
assert_failure "${unsafe_mode_archive}" "${scratch}/unsafe-mode-output" 3 "unsafe-mode"

bundle_symlink="${scratch}/bundle-link.tar.gz"
/bin/ln -s "${valid_archive}" "${bundle_symlink}"
assert_failure "${bundle_symlink}" "${scratch}/bundle-link-output" 3 "bundle-link"

preexisting="${scratch}/preexisting-output"
/bin/mkdir "${preexisting}"
preexisting_stdout="${scratch}/preexisting.stdout"
preexisting_stderr="${scratch}/preexisting.stderr"
if "${extractor}" "${valid_archive}" "${preexisting}" 3 \
  >"${preexisting_stdout}" 2>"${preexisting_stderr}"; then
  fail "pre-existing output was accepted"
fi
[[ -d "${preexisting}" ]] || fail "pre-existing output was removed"
[[ -z "$(/usr/bin/find "${preexisting}" -mindepth 1 -print -quit)" ]] ||
  fail "pre-existing output was modified"

echo "PASS: private fixture bundle extraction rejects unsafe synthetic archives"
