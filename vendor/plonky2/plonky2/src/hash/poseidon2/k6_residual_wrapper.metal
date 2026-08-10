// Executable entry point for lighter_k6::quotient. This source stays small so
// a worker linking the build-host dynamic library only pays a thin MSL front
// end, not the full residual-gate implementation.

#include <metal_stdlib>
using namespace metal;

constant ulong COSET_16_DOMAIN[16] = {
    0x1UL, 0x1000UL, 0x1000000UL, 0x1000000000UL,
    0x1000000000000UL, 0x1000000000000000UL, 0xffffffff00UL,
    0xffffffff00000UL, 0xffffffff00000000UL, 0xfffffffefffff001UL,
    0xfffffffeff000001UL, 0xffffffef00000001UL, 0xfffeffff00000001UL,
    0xefffffff00000001UL, 0xfffffeff00000101UL, 0xffefffff00100001UL,
};

constant ulong COSET_16_WEIGHTS[16] = {
    0xefffffff10000001UL, 0x100UL, 0x100000UL, 0x100000000UL,
    0x100000000000UL, 0x100000000000000UL, 0xffffffff0UL,
    0xffffffff0000UL, 0xffffffff0000000UL, 0xfffffffeffffff01UL,
    0xfffffffefff00001UL, 0xfffffffe00000001UL, 0xffffefff00000001UL,
    0xfeffffff00000001UL, 0xffffffef00000011UL, 0xfffeffff00010001UL,
};

namespace lighter_k6 {
void quotient(
    const device ulong* wires,
    const device ulong* constants,
    device ulong* output,
    constant ulong* alpha_powers,
    constant uint* metadata,
    constant uint& lde_rows,
    constant uint& quotient_rows,
    constant uint& step,
    constant uint& alpha_stride,
    constant uint& k6_count,
    constant ulong* public_inputs_hash,
    constant ulong* coset_domain,
    constant ulong* coset_weights,
    uint gid);
} // namespace lighter_k6

kernel void k6_residual_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* metadata [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& alpha_stride [[buffer(8)]],
    constant uint& k6_count [[buffer(9)]],
    constant ulong* public_inputs_hash [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    lighter_k6::quotient(
        wires,
        constants,
        output,
        alpha_powers,
        metadata,
        lde_rows,
        quotient_rows,
        step,
        alpha_stride,
        k6_count,
        public_inputs_hash,
        COSET_16_DOMAIN,
        COSET_16_WEIGHTS,
        gid);
}
