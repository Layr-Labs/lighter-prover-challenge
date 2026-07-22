#!/usr/bin/env bash
set -euo pipefail

fail() {
  local message="$1"
  local log="${2:-}"

  echo "FAIL: ${message}" >&2
  if [[ -n "${log}" && -f "${log}" ]]; then
    echo "--- benchmark output ---" >&2
    while IFS= read -r line; do
      printf '%s\n' "${line}" >&2
    done < "${log}"
    echo "--- end benchmark output ---" >&2
  fi
  exit 1
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
benchmark="${root}/benchmark.sh"
fixture="${root}/benchmark-tools/fixtures/bench.json"
trusted_dir="${root}/benchmark-tools/trusted"
verifier="${trusted_dir}/lighter-benchmark-verifier"

[[ "$(uname -s)" == Darwin ]] || fail "this integration test requires macOS"
[[ "$(uname -m)" == arm64 ]] || fail "this integration test requires the published arm64 architecture"
[[ -x /usr/bin/clang ]] || fail "/usr/bin/clang is required to build the temporary test worker"
[[ -x /usr/bin/lipo ]] || fail "/usr/bin/lipo is required to inspect the published verifier"
[[ -x /usr/bin/sandbox-exec ]] || fail "/usr/bin/sandbox-exec is required to test Seatbelt"
[[ -x "${benchmark}" ]] || fail "benchmark.sh is not executable"
[[ -f "${fixture}" ]] || fail "protected fixture is missing"
[[ -x "${verifier}" ]] || fail "published trusted verifier is not executable"
[[ "$(/usr/bin/lipo -archs "${verifier}")" == arm64 ]] \
  || fail "published trusted verifier is not a single-architecture arm64 binary"

(
  cd "${trusted_dir}"
  shasum -a 256 -c SHA256SUMS
)
codesign --verify --strict --verbose=2 "${verifier}"

fixture_sha256_before="$(shasum -a 256 "${fixture}" | awk '{print $1}')"
verifier_sha256_before="$(shasum -a 256 "${verifier}" | awk '{print $1}')"

test_root="$(mktemp -d "${TMPDIR:-/tmp}/lighter-verifier-integration.XXXXXX")"
worker_source="${test_root}/test-worker.c"
worker_binary="${test_root}/test-worker"
active_benchmark_pid=""
active_benchmark_pgid=""
active_group_file=""
active_ack_file=""
active_benchmark_reaped=0
launch_in_progress=0
deferred_signal_status=0
test_shell_pgid="$(/bin/ps -o pgid= -p "$$" | tr -d '[:space:]')"

process_group_alive() {
  local pgid="$1"

  [[ "${pgid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_numeric_pid_file() {
  local path="$1"
  local owner_pid="$2"
  local attempts="$3"
  local value=""

  for ((attempt = 0; attempt < attempts; attempt++)); do
    if [[ -f "${path}" ]]; then
      value="$(<"${path}")"
      if [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${value}"
        return 0
      fi
    fi
    kill -0 "${owner_pid}" 2>/dev/null || return 1
    sleep 0.02
  done
  return 1
}

wait_for_process_group_exit() {
  local pgid="$1"

  for ((attempt = 0; attempt < 50; attempt++)); do
    reap_active_benchmark_if_waitable || true
    process_group_alive "${pgid}" || return 0
    sleep 0.02
  done
  return 1
}

process_is_waitable() {
  local pid="$1"
  local state=""

  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  state="$(/bin/ps -o stat= -p "${pid}" 2>/dev/null \
    | tr -d '[:space:]' || true)"
  [[ -z "${state}" || "${state}" == Z* ]]
}

reap_active_benchmark_if_waitable() {
  (( active_benchmark_reaped == 0 )) || return 0
  process_is_waitable "${active_benchmark_pid}" || return 1
  wait "${active_benchmark_pid}" 2>/dev/null || true
  active_benchmark_reaped=1
}

wait_for_process_exit() {
  local pid="$1"
  local attempts="${2:-50}"

  for ((attempt = 0; attempt < attempts; attempt++)); do
    process_is_waitable "${pid}" && return 0
    sleep 0.02
  done
  return 1
}

resolve_active_process_group() {
  local value=""

  if [[ "${active_benchmark_pgid}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if [[ -n "${active_group_file}" && -f "${active_group_file}" ]]; then
    value="$(<"${active_group_file}")"
    if [[ "${value}" =~ ^[0-9]+$ && "${value}" == "${active_benchmark_pid}" ]]; then
      active_benchmark_pgid="${value}"
      return 0
    fi
  fi
  value="$(/bin/ps -o pgid= -p "${active_benchmark_pid}" 2>/dev/null \
    | tr -d '[:space:]' || true)"
  if [[ "${value}" =~ ^[0-9]+$ && "${value}" == "${active_benchmark_pid}" ]]; then
    active_benchmark_pgid="${value}"
  fi
}

terminate_active_benchmark() {
  local failed=0

  [[ "${active_benchmark_pid}" =~ ^[0-9]+$ ]] || return 0
  resolve_active_process_group

  if [[ "${active_benchmark_pgid}" =~ ^[0-9]+$ ]]; then
    if [[ "${active_benchmark_pgid}" == "${test_shell_pgid}" ]]; then
      echo "FAIL: refusing to signal the test shell's process group" >&2
      return 1
    fi
    if process_group_alive "${active_benchmark_pgid}"; then
      kill -TERM -- "-${active_benchmark_pgid}" 2>/dev/null || true
      if ! wait_for_process_group_exit "${active_benchmark_pgid}"; then
        kill -KILL -- "-${active_benchmark_pgid}" 2>/dev/null || true
        if ! wait_for_process_group_exit "${active_benchmark_pgid}"; then
          echo "FAIL: process group ${active_benchmark_pgid} survived SIGKILL" >&2
          failed=1
        fi
      fi
    fi
  elif kill -0 "${active_benchmark_pid}" 2>/dev/null; then
    # The launcher calls setpgid before it can create descendants. If cleanup
    # catches it before then, terminating the launcher PID is sufficient.
    kill -TERM "${active_benchmark_pid}" 2>/dev/null || true
    if ! wait_for_process_exit "${active_benchmark_pid}"; then
      kill -KILL "${active_benchmark_pid}" 2>/dev/null || true
      if ! wait_for_process_exit "${active_benchmark_pid}"; then
        echo "FAIL: benchmark launcher ${active_benchmark_pid} survived SIGKILL" >&2
        failed=1
      fi
    fi
  fi

  reap_active_benchmark_if_waitable || true
  if (( active_benchmark_reaped == 0 )); then
    echo "FAIL: benchmark launcher ${active_benchmark_pid} was not reaped" >&2
    failed=1
  fi
  if (( failed == 0 )); then
    rm -f "${active_group_file}" "${active_ack_file}"
    active_benchmark_pid=""
    active_benchmark_pgid=""
    active_group_file=""
    active_ack_file=""
    active_benchmark_reaped=0
  fi
  return "${failed}"
}

verify_protected_state() {
  local fixture_sha256_after=""
  local verifier_sha256_after=""
  local failed=0

  if ! fixture_sha256_after="$(shasum -a 256 "${fixture}" | awk '{print $1}')"; then
    echo "FAIL: could not hash the protected fixture during cleanup" >&2
    failed=1
  elif [[ "${fixture_sha256_after}" != "${fixture_sha256_before}" ]]; then
    echo "FAIL: protected fixture changed during the integration test" >&2
    failed=1
  fi
  if ! verifier_sha256_after="$(shasum -a 256 "${verifier}" | awk '{print $1}')"; then
    echo "FAIL: could not hash the published verifier during cleanup" >&2
    failed=1
  elif [[ "${verifier_sha256_after}" != "${verifier_sha256_before}" ]]; then
    echo "FAIL: published trusted verifier changed during the integration test" >&2
    failed=1
  fi
  if ! (
    cd "${trusted_dir}"
    shasum -a 256 -c SHA256SUMS >/dev/null
  ); then
    echo "FAIL: published verifier checksum validation failed during cleanup" >&2
    failed=1
  fi
  if ! codesign --verify --strict "${verifier}" >/dev/null 2>&1; then
    echo "FAIL: published verifier signature validation failed during cleanup" >&2
    failed=1
  fi
  return "${failed}"
}

cleanup() {
  local original_status="$1"
  local cleanup_failed=0
  local final_status="${original_status}"

  trap - EXIT HUP INT TERM
  terminate_active_benchmark || cleanup_failed=1
  verify_protected_state || cleanup_failed=1
  rm -rf "${test_root}" || cleanup_failed=1
  if (( final_status == 0 && cleanup_failed != 0 )); then
    final_status=1
  fi
  exit "${final_status}"
}

handle_signal() {
  local status="$1"

  if (( launch_in_progress != 0 )); then
    deferred_signal_status="${status}"
    return 0
  fi
  exit "${status}"
}

launch_isolated_process() {
  local group_file="$1"
  local ack_file="$2"
  shift 2

  rm -f "${group_file}" "${ack_file}" "${ack_file}.tmp"
  active_group_file="${group_file}"
  active_ack_file="${ack_file}"
  launch_in_progress=1
  "${worker_binary}" --process-group "${group_file}" "${ack_file}" "$@" &
  active_benchmark_pid=$!
  launch_in_progress=0
  active_benchmark_reaped=0
  if (( deferred_signal_status != 0 )); then
    local status="${deferred_signal_status}"
    deferred_signal_status=0
    exit "${status}"
  fi
}

acknowledge_process_group() {
  local pgid="$1"
  local temporary_ack="${active_ack_file}.tmp"

  printf '%s' "${pgid}" > "${temporary_ack}"
  mv -f "${temporary_ack}" "${active_ack_file}"
}

start_isolated_process() {
  local group_file="$1"
  local ack_file="$2"
  local observed_pgid=""
  shift 2

  launch_isolated_process "${group_file}" "${ack_file}" "$@"
  if ! observed_pgid="$(
    wait_for_numeric_pid_file "${group_file}" "${active_benchmark_pid}" 100
  )"; then
    return 1
  fi
  [[ "${observed_pgid}" == "${active_benchmark_pid}" ]] || return 1
  active_benchmark_pgid="${observed_pgid}"
  acknowledge_process_group "${observed_pgid}"
}

trap 'cleanup "$?"' EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

cat > "${worker_source}" <<'EOF'
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MAX_PROOF_BYTES (256ULL * 1024ULL * 1024ULL)

static int write_bytes(const char *path, const void *bytes, size_t length) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        return 10;
    }
    const unsigned char *cursor = bytes;
    while (length > 0) {
        ssize_t written = write(fd, cursor, length);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            close(fd);
            return 11;
        }
        cursor += (size_t)written;
        length -= (size_t)written;
    }
    return close(fd) == 0 ? 0 : 12;
}

static int write_pid_file(const char *path) {
    char temporary_path[PATH_MAX];
    char pid[32];
    int path_length = snprintf(temporary_path, sizeof(temporary_path), "%s.tmp", path);
    int pid_length = snprintf(pid, sizeof(pid), "%ld", (long)getpid());
    if (path_length < 0 || (size_t)path_length >= sizeof(temporary_path)
        || pid_length < 0 || (size_t)pid_length >= sizeof(pid)) {
        return 20;
    }
    int result = write_bytes(temporary_path, pid, (size_t)pid_length);
    if (result != 0) {
        return 21;
    }
    if (rename(temporary_path, path) != 0) {
        unlink(temporary_path);
        return 22;
    }
    return 0;
}

static int wait_for_acknowledgement(const char *path) {
    char expected[32];
    int expected_length = snprintf(expected, sizeof(expected), "%ld", (long)getpid());
    if (expected_length < 0 || (size_t)expected_length >= sizeof(expected)) {
        return 84;
    }

    for (int attempt = 0; attempt < 500; attempt++) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            char contents[32];
            char extra;
            ssize_t length = read(fd, contents, sizeof(contents));
            ssize_t extra_length = length >= 0 ? read(fd, &extra, 1) : -1;
            int saved_errno = errno;
            close(fd);
            errno = saved_errno;
            if (length == expected_length && extra_length == 0
                && memcmp(contents, expected, (size_t)expected_length) == 0) {
                return 0;
            }
            return 85;
        }
        if (errno != ENOENT) {
            return 86;
        }
        struct timespec delay = { .tv_sec = 0, .tv_nsec = 10 * 1000 * 1000 };
        while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
        }
    }
    return 87;
}

static int timeout_worker(const char *proof_path) {
    char pid_path[PATH_MAX];
    int length = snprintf(pid_path, sizeof(pid_path), "%s.pid", proof_path);
    if (length < 0 || (size_t)length >= sizeof(pid_path)) {
        return 23;
    }
    if (write_pid_file(pid_path) != 0) {
        return 24;
    }
    for (;;) {
        pause();
    }
}

static int oversized_worker(const char *proof_path) {
    int fd = open(proof_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        return 30;
    }
    if (ftruncate(fd, (off_t)(MAX_PROOF_BYTES + 1)) != 0) {
        close(fd);
        return 31;
    }
    return close(fd) == 0 ? 0 : 32;
}

static int seatbelt_worker(const char *program_path) {
    static const char suffix[] = "/target/release/prove";
    static const char marker_suffix[] = "/outside-marker";
    char marker_path[PATH_MAX];
    size_t program_length = strlen(program_path);
    size_t suffix_length = sizeof(suffix) - 1;

    if (program_length <= suffix_length
        || strcmp(program_path + program_length - suffix_length, suffix) != 0) {
        return 70;
    }
    size_t root_length = program_length - suffix_length;
    if (root_length + sizeof(marker_suffix) > sizeof(marker_path)) {
        return 71;
    }
    memcpy(marker_path, program_path, root_length);
    memcpy(marker_path + root_length, marker_suffix, sizeof(marker_suffix));

    int fd = open(marker_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        return errno == EACCES || errno == EPERM ? 73 : 74;
    }
    static const char marker[] = "outside write succeeded\n";
    int result = write_bytes(marker_path, marker, sizeof(marker) - 1);
    close(fd);
    return result == 0 ? 0 : 75;
}

static int process_group_launcher(int argc, char **argv) {
    if (argc < 5) {
        return 80;
    }
    if (setpgid(0, 0) != 0 && getpgrp() != getpid()) {
        return 81;
    }
    if (write_pid_file(argv[2]) != 0) {
        return 82;
    }
    if (wait_for_acknowledgement(argv[3]) != 0) {
        return 83;
    }
    execv(argv[4], &argv[4]);
    return 88;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--process-group") == 0) {
        return process_group_launcher(argc, argv);
    }
    if (argc != 3) {
        return 64;
    }
    if (strstr(argv[0], "/timeout/") != NULL) {
        return timeout_worker(argv[2]);
    }
    if (strstr(argv[0], "/malformed/") != NULL) {
        static const char invalid_proof[] = "not-a-valid-proof";
        return write_bytes(argv[2], invalid_proof, sizeof(invalid_proof) - 1);
    }
    if (strstr(argv[0], "/oversized/") != NULL) {
        return oversized_worker(argv[2]);
    }
    if (strstr(argv[0], "/seatbelt/") != NULL) {
        return seatbelt_worker(argv[0]);
    }
    return 65;
}
EOF

/usr/bin/clang \
  -std=c11 \
  -O2 \
  -Wall \
  -Wextra \
  -Werror \
  "${worker_source}" \
  -o "${worker_binary}"

create_candidate() {
  local name="$1"
  local candidate="${test_root}/candidates/${name}"

  mkdir -p "${candidate}/target/release"
  cp "${worker_binary}" "${candidate}/target/release/prove"
  chmod 755 "${candidate}/target/release/prove"
  printf '%s\n' "${candidate}"
}

pre_ack_group_file="${test_root}/pre-ack-cleanup-group.pid"
pre_ack_file="${test_root}/pre-ack-cleanup.ack"
pre_ack_exec_marker="${test_root}/pre-ack-exec-marker"
launch_isolated_process \
  "${pre_ack_group_file}" \
  "${pre_ack_file}" \
  /bin/sh \
  -c \
  'printf executed > "${1}"; /bin/sleep 60' \
  pre-ack-cleanup \
  "${pre_ack_exec_marker}"
if ! pre_ack_pgid="$(
  wait_for_numeric_pid_file "${pre_ack_group_file}" "${active_benchmark_pid}" 100
)"; then
  fail "cleanup self-test: launcher did not publish its pre-acknowledgement PGID"
fi
[[ "${pre_ack_pgid}" == "${active_benchmark_pid}" ]] \
  || fail "cleanup self-test: pre-acknowledgement PGID did not match the launcher"
active_benchmark_pgid="${pre_ack_pgid}"
[[ ! -e "${pre_ack_file}" ]] \
  || fail "cleanup self-test: acknowledgement appeared before parent release"
terminate_active_benchmark \
  || fail "cleanup self-test: could not cancel the launcher before acknowledgement"
process_group_alive "${pre_ack_pgid}" \
  && fail "cleanup self-test: pre-acknowledgement process group remained alive"
[[ ! -e "${pre_ack_exec_marker}" ]] \
  || fail "cleanup self-test: launcher executed descendants before acknowledgement"

cleanup_self_test_group_file="${test_root}/cleanup-self-test-group.pid"
cleanup_self_test_ack_file="${test_root}/cleanup-self-test.ack"
cleanup_self_test_child_file="${test_root}/cleanup-self-test-child.pid"
if ! start_isolated_process \
  "${cleanup_self_test_group_file}" \
  "${cleanup_self_test_ack_file}" \
  /bin/sh \
  -c \
  '/bin/sleep 60 & child=$!; printf "%s" "${child}" > "${1}"; wait "${child}"' \
  cleanup-self-test \
  "${cleanup_self_test_child_file}"
then
  fail "cleanup self-test: could not start an isolated process group"
fi
cleanup_self_test_pgid="${active_benchmark_pgid}"
if ! cleanup_self_test_child_pid="$(
  wait_for_numeric_pid_file \
    "${cleanup_self_test_child_file}" \
    "${active_benchmark_pid}" \
    100
)"; then
  fail "cleanup self-test: did not observe a complete numeric child PID"
fi
terminate_active_benchmark \
  || fail "cleanup self-test: could not terminate and reap the isolated process tree"
process_group_alive "${cleanup_self_test_pgid}" \
  && fail "cleanup self-test: isolated process group remained alive"
kill -0 "${cleanup_self_test_child_pid}" 2>/dev/null \
  && fail "cleanup self-test: descendant ${cleanup_self_test_child_pid} remained alive"
if [[ "${LIGHTER_CLEANUP_SELF_TEST_ONLY:-0}" == 1 ]]; then
  echo "PASS: bounded process-group cleanup self-test"
  exit 0
fi

assert_failure() {
  local case_name="$1"
  local candidate="$2"
  local expected_output="$3"
  local score="${test_root}/${case_name}-score.json"
  local log="${test_root}/${case_name}.log"
  local status

  printf '{"stale":true}\n' > "${score}"
  if TMPDIR="${test_root}" \
    LIGHTER_CANDIDATE_ROOT="${candidate}" \
    LIGHTER_SCORE_PATH="${score}" \
    LIGHTER_PROVE_TIMEOUT_SECONDS=10 \
    LIGHTER_REQUIRE_SANDBOX=1 \
    "${benchmark}" > "${log}" 2>&1
  then
    status=0
  else
    status=$?
  fi

  (( status != 0 )) || fail "${case_name}: benchmark unexpectedly succeeded" "${log}"
  [[ ! -e "${score}" ]] || fail "${case_name}: benchmark left a score file" "${log}"
  [[ "$(<"${log}")" == *"${expected_output}"* ]] \
    || fail "${case_name}: expected output not found: ${expected_output}" "${log}"
}

timeout_candidate="$(create_candidate timeout)"
malformed_candidate="$(create_candidate malformed)"
oversized_candidate="$(create_candidate oversized)"
seatbelt_candidate="$(create_candidate seatbelt)"

timeout_score="${test_root}/timeout-score.json"
timeout_log="${test_root}/timeout.log"
timeout_group_file="${test_root}/timeout-group.pid"
timeout_ack_file="${test_root}/timeout.ack"
printf '{"stale":true}\n' > "${timeout_score}"
if ! start_isolated_process \
  "${timeout_group_file}" \
  "${timeout_ack_file}" \
  /usr/bin/env \
  "TMPDIR=${test_root}" \
  "LIGHTER_CANDIDATE_ROOT=${timeout_candidate}" \
  "LIGHTER_SCORE_PATH=${timeout_score}" \
  LIGHTER_PROVE_TIMEOUT_SECONDS=1 \
  LIGHTER_REQUIRE_SANDBOX=1 \
  "${benchmark}" \
  > "${timeout_log}" 2>&1
then
  fail "timeout: could not start benchmark in an isolated process group" "${timeout_log}"
fi
benchmark_pid="${active_benchmark_pid}"
benchmark_pgid="${active_benchmark_pgid}"

shopt -s nullglob
worker_pid=""
for ((attempt = 0; attempt < 300; attempt++)); do
  pid_files=("${test_root}"/lighter-benchmark.*/proof.bin.pid)
  if (( ${#pid_files[@]} > 0 )); then
    candidate_worker_pid="$(<"${pid_files[0]}")"
    if [[ "${candidate_worker_pid}" =~ ^[0-9]+$ ]]; then
      worker_pid="${candidate_worker_pid}"
      break
    fi
  fi
  kill -0 "${benchmark_pid}" 2>/dev/null || break
  sleep 0.02
done

if ! wait_for_process_exit "${benchmark_pid}" 300; then
  terminate_active_benchmark \
    || fail "timeout: bounded cleanup could not terminate the benchmark process tree" "${timeout_log}"
  fail "timeout: benchmark did not exit within the bounded wait" "${timeout_log}"
fi
if wait "${benchmark_pid}"; then
  timeout_status=0
else
  timeout_status=$?
fi
active_benchmark_reaped=1
if process_group_alive "${benchmark_pgid}"; then
  fail "timeout: benchmark descendants remained after the verifier exited" "${timeout_log}"
fi
active_benchmark_pid=""
active_benchmark_pgid=""
active_group_file=""
active_ack_file=""
active_benchmark_reaped=0
(( timeout_status != 0 )) || fail "timeout: benchmark unexpectedly succeeded" "${timeout_log}"
[[ "${worker_pid}" =~ ^[0-9]+$ ]] || fail "timeout: did not observe the worker PID" "${timeout_log}"
[[ ! -e "${timeout_score}" ]] || fail "timeout: benchmark left a score file" "${timeout_log}"
[[ "$(<"${timeout_log}")" == *"candidate worker timed out"* ]] \
  || fail "timeout: verifier did not report its trusted timeout" "${timeout_log}"
if kill -0 "${worker_pid}" 2>/dev/null; then
  fail "timeout: worker ${worker_pid} was not killed and reaped" "${timeout_log}"
fi
if [[ "${LIGHTER_TIMEOUT_TEST_ONLY:-0}" == 1 ]]; then
  echo "PASS: timeout worker was killed and all isolated descendants were reaped"
  exit 0
fi

assert_failure \
  malformed \
  "${malformed_candidate}" \
  "UnexpectedEof"

assert_failure \
  oversized \
  "${oversized_candidate}" \
  "proof output is empty or exceeds the trusted size limit"

seatbelt_worker="${seatbelt_candidate}/target/release/prove"
outside_marker="${seatbelt_candidate}/outside-marker"
direct_proof="${test_root}/seatbelt-negative-control-proof.bin"
"${seatbelt_worker}" "${fixture}" "${direct_proof}" \
  || fail "seatbelt: unsandboxed negative control worker failed"
[[ -f "${outside_marker}" ]] \
  || fail "seatbelt: unsandboxed negative control did not create the outside marker"
rm -f "${outside_marker}" "${direct_proof}"

assert_failure \
  seatbelt \
  "${seatbelt_candidate}" \
  "candidate worker failed with exit status: 73"
[[ ! -e "${outside_marker}" ]] \
  || fail "seatbelt: sandboxed worker created the outside marker"

echo "PASS: timeout, malformed proof, oversized proof, and Seatbelt write denial"
