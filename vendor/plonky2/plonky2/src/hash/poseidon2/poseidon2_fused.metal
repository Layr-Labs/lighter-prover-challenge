#include "poseidon2.metal"

// Fuses two adjacent radix-2 DIT stages. Four values are loaded once, both
// stages execute in registers, and the four results are written once. A pair
// of ordinary `ntt_stage` dispatches performs the same butterflies with twice
// the device-memory traffic and an extra encoder boundary.
kernel void ntt_stage2(
    device ulong* values [[buffer(0)]],
    const device ulong* roots0 [[buffer(1)]],
    const device ulong* roots1 [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& log_quarter_m [[buffer(4)]],
    constant uint& canonicalize [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint quarter_butterflies = n >> 2;
    if (t >= quarter_butterflies) {
        return;
    }

    ulong colbase = (ulong)gid.y * n;
    uint quarter_m = 1u << log_quarter_m;
    uint j = t & (quarter_m - 1u);
    uint base = ((t >> log_quarter_m) << (log_quarter_m + 2u));
    uint i0 = base + j;
    uint i1 = i0 + quarter_m;
    uint i2 = i1 + quarter_m;
    uint i3 = i2 + quarter_m;

    ulong a = values[colbase + i0];
    ulong b = values[colbase + i1];
    ulong c = values[colbase + i2];
    ulong d = values[colbase + i3];

    ulong wb = gl_mul(roots0[j], b);
    ulong wd = gl_mul(roots0[j], d);
    ulong ab_add = gl_add(a, wb);
    ulong ab_sub = gl_sub(a, wb);
    ulong cd_add = gl_add(c, wd);
    ulong cd_sub = gl_sub(c, wd);

    ulong w_cd_add = gl_mul(roots1[j], cd_add);
    ulong w_cd_sub = gl_mul(roots1[j + quarter_m], cd_sub);
    ulong out0 = gl_add(ab_add, w_cd_add);
    ulong out2 = gl_sub(ab_add, w_cd_add);
    ulong out1 = gl_add(ab_sub, w_cd_sub);
    ulong out3 = gl_sub(ab_sub, w_cd_sub);
    if (canonicalize != 0u) {
        out0 = gl_canonicalize(out0);
        out1 = gl_canonicalize(out1);
        out2 = gl_canonicalize(out2);
        out3 = gl_canonicalize(out3);
    }

    values[colbase + i0] = out0;
    values[colbase + i1] = out1;
    values[colbase + i2] = out2;
    values[colbase + i3] = out3;
}
