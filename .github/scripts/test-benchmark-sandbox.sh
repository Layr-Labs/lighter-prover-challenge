#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -P)"
writer="${root}/.github/scripts/write-benchmark-sandbox-profile.sh"
scratch="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-test.XXXXXX")"
outside="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-outside.XXXXXX")"
readable="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-readable.XXXXXX")"
tools="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/lighter-sandbox-tools.XXXXXX")"
profile="$(/usr/bin/mktemp -t lighter-sandbox-test.XXXXXX.sb)"
tool_profile="$(/usr/bin/mktemp -t lighter-sandbox-tool.XXXXXX.sb)"
trap '/bin/rm -rf "${scratch}" "${outside}" "${readable}" "${tools}" \
  "${profile}" "${tool_profile}"' EXIT

fail() {
  echo "$1" >&2
  exit 1
}

render() {
  local worker="$1" destination="$2"
  shift 2
  /bin/rm -f "${destination}"
  "${writer}" "${scratch}" "${destination}" "${worker}" "$@"
}

[[ "$(/usr/bin/uname -s)" == Darwin ]]
[[ -x /usr/bin/sandbox-exec ]]
render /bin/bash "${profile}" "${readable}"

# The profile must be a default-deny allowlist: no (allow default) anywhere, and
# (deny default) as the first rule after the version header.
! /usr/bin/grep -Fq '(allow default)' "${profile}" ||
  fail "sandbox profile still contains (allow default)"
[[ "$(/usr/bin/sed -n '2p' "${profile}")" == '(deny default)' ]] ||
  fail "sandbox profile does not start with (deny default)"

# Every denial the blocklist profile used to spell out must still be spelled out,
# so a future edit cannot quietly drop one behind (deny default).
for required in \
  '(deny network*)' \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.system.mDNSResponder"))' \
  '(deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))' \
  '(deny mach-lookup (global-name "com.apple.mDNSResponder.dnsproxy"))' \
  '(deny process-fork)' \
  '(deny process-exec*)' \
  '(deny file-write*)'; do
  /usr/bin/grep -Fq -- "${required}" "${profile}" ||
    fail "sandbox profile is missing an explicit denial: ${required}"
done

# High-frequency benign denials are silenced, not relaxed. Each must still be a
# deny rule, and only these may carry (with no-report) — a stray no-report on
# anything else would blind the log that diagnoses an over-tight profile.
for silenced in \
  '(deny mach-lookup (global-name "com.apple.bsd.dirhelper") (with no-report))' \
  '(deny mach-lookup (global-name "com.apple.system.opendirectoryd.membership") (with no-report))' \
  '(deny mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo") (with no-report))' \
  '(deny file-read* (literal "/private/etc/passwd") (with no-report))'; do
  /usr/bin/grep -Fq -- "${silenced}" "${profile}" ||
    fail "sandbox profile is missing a silenced denial: ${silenced}"
done
[[ "$(/usr/bin/grep -c 'with no-report' "${profile}")" -eq 4 ]] ||
  fail "sandbox profile carries unexpected (with no-report) rules"
! /usr/bin/grep -E 'with no-report' "${profile}" | /usr/bin/grep -Fq '(allow' ||
  fail "(with no-report) must never appear on an allow rule"

# Silencing must not have granted the read: /etc/passwd stays unreadable.
if /usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  '/bin/cat /private/etc/passwd' >/dev/null 2>&1; then
  fail "silenced denial became an allow: /private/etc/passwd is readable"
fi

# The worker may write its scratch directory.
/usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  'printf permitted > "$1/scratch-write"' sandbox-test "${scratch}"
[[ -f "${scratch}/scratch-write" ]]

if /usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  'printf forbidden > "$1/outside-write"' sandbox-test "${outside}" \
  >/dev/null 2>&1; then
  fail "sandbox allowed a non-scratch write"
fi

if /usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  '/bin/sleep 0 & wait' >/dev/null 2>&1; then
  fail "sandbox allowed process fork"
fi

# Default-deny coverage: only the rendered worker may be executed, so the shell
# cannot start any other program even though it runs fine itself.
/usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c 'exit 0' ||
  fail "sandbox denied the rendered worker itself"
if /usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  '/bin/echo unexpected' >/dev/null 2>&1; then
  fail "sandbox allowed exec of a program that is not the worker"
fi

# Default-deny coverage: reads are an allowlist too. A path handed to the writer
# stays readable; an unlisted path does not.
printf 'listed\n' > "${readable}/input"
printf 'unlisted\n' > "${outside}/input"
/usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  'read -r line < "$1/input"' sandbox-test "${readable}" ||
  fail "sandbox denied a read root it was given"
if /usr/bin/sandbox-exec -f "${profile}" /bin/bash --noprofile --norc -c \
  'read -r line < "$1/input"' sandbox-test "${outside}" >/dev/null 2>&1; then
  fail "sandbox allowed a read outside its read roots"
fi

# Network egress and the DNS resolver stay unreachable. Each tool is rendered as
# its own profile's worker so the failure is the denied operation and not a
# denied exec.
render /usr/bin/curl "${tool_profile}"
if /usr/bin/sandbox-exec -f "${tool_profile}" \
  /usr/bin/curl -sS --max-time 3 https://1.1.1.1/ >/dev/null 2>&1; then
  fail "sandbox allowed network access"
fi

render /usr/bin/dscacheutil "${tool_profile}"
if /usr/bin/sandbox-exec -f "${tool_profile}" \
  /usr/bin/dscacheutil -q host -a name example.com 2>/dev/null |
    /usr/bin/grep -q ip_address; then
  fail "sandbox allowed resolver IPC"
fi

# The remaining checks need a compiler. They assert on the kernel's answer
# (EPERM from the sandbox) and on real GPU access, which is what a too-tight
# allowlist would break.
if [[ ! -x /usr/bin/clang ]]; then
  echo "SKIP: /usr/bin/clang is unavailable; skipped errno and Metal checks" >&2
  exit 0
fi

/bin/cat > "${tools}/prober.c" <<'PROBER'
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* Exit 40 means the operation was refused with EPERM, which is how Seatbelt
   reports a denial. Exit 0 means it was permitted. */
int main(int argc, char **argv) {
    if (argc != 2) return 64;
    if (strcmp(argv[1], "connect") == 0) {
        struct sockaddr_in address;
        int refusal;
        int handle = socket(AF_INET, SOCK_STREAM, 0);
        if (handle < 0) {
            refusal = errno;
            fprintf(stderr, "socket errno=%d\n", refusal);
            return refusal == EPERM ? 40 : 41;
        }
        memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_port = htons(443);
        address.sin_addr.s_addr = htonl(0x01010101);
        if (connect(handle, (struct sockaddr *)&address, sizeof(address)) == 0) {
            return 0;
        }
        refusal = errno;
        fprintf(stderr, "connect errno=%d\n", refusal);
        return refusal == EPERM ? 40 : 41;
    }
    if (strcmp(argv[1], "fork") == 0) {
        pid_t child = fork();
        int refusal = errno;
        if (child == 0) _exit(0);
        if (child < 0) {
            fprintf(stderr, "fork errno=%d\n", refusal);
            return refusal == EPERM ? 40 : 41;
        }
        return 0;
    }
    return 64;
}
PROBER
/usr/bin/clang -O1 "${tools}/prober.c" -o "${tools}/prober"
render "${tools}/prober" "${tool_profile}"
for operation in connect fork; do
  status=0
  /usr/bin/sandbox-exec -f "${tool_profile}" "${tools}/prober" "${operation}" \
    >/dev/null 2>&1 || status=$?
  [[ "${status}" -eq 40 ]] ||
    fail "sandbox did not refuse ${operation} with EPERM (prober exit ${status})"
done

/bin/cat > "${tools}/gpu-probe.m" <<'GPUPROBE'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Compiles a Metal kernel and runs it, the way the plonky2 prover does, so a
   too-tight allowlist that loses GPU access fails this test loudly instead of
   silently costing throughput. */
int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "kernel void twice(device uint *values [[buffer(0)]],\n"
                            "                  uint index [[thread_position_in_grid]]) {\n"
                            "    values[index] = values[index] * 2;\n"
                            "}\n";
        const NSUInteger count = 256;
        id<MTLLibrary> library = nil;
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLBuffer> buffer = nil;
        id<MTLCommandBuffer> commands = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        uint32_t *words = NULL;
        if (device == nil) return 10;
        library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) return 11;
        pipeline = [device newComputePipelineStateWithFunction:
                              [library newFunctionWithName:@"twice"]
                                                        error:&error];
        if (pipeline == nil) return 12;
        buffer = [device newBufferWithLength:count * sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];
        if (buffer == nil) return 13;
        words = buffer.contents;
        for (NSUInteger index = 0; index < count; index++) {
            words[index] = (uint32_t)index;
        }
        commands = [[device newCommandQueue] commandBuffer];
        encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) return 14;
        for (NSUInteger index = 0; index < count; index++) {
            if (words[index] != (uint32_t)index * 2) return 15;
        }
        return 0;
    }
}
GPUPROBE
if ! /usr/bin/clang -fobjc-arc -O1 "${tools}/gpu-probe.m" \
  -framework Foundation -framework Metal -o "${tools}/gpu-probe" 2>/dev/null; then
  echo "SKIP: Metal SDK is unavailable; skipped the GPU allowlist check" >&2
  exit 0
fi
if ! "${tools}/gpu-probe" >/dev/null 2>&1; then
  echo "SKIP: no usable GPU outside the sandbox; skipped the GPU allowlist check" >&2
  exit 0
fi
render "${tools}/gpu-probe" "${tool_profile}"
status=0
/usr/bin/sandbox-exec -f "${tool_profile}" "${tools}/gpu-probe" >/dev/null 2>&1 ||
  status=$?
[[ "${status}" -eq 0 ]] ||
  fail "sandbox profile broke Metal GPU compute (gpu-probe exit ${status})"
