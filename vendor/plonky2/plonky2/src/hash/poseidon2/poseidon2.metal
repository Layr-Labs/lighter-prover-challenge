#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

inline ulong gl_add(ulong a, ulong b) {
    ulong sum = a + b;
    ulong carry = sum < a;
    sum += carry * GOLDILOCKS_EPSILON;
    ulong carry2 = (carry != 0UL) && (sum < GOLDILOCKS_EPSILON);
    return sum + carry2 * GOLDILOCKS_EPSILON;
}

inline ulong gl_mul(ulong a, ulong b) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong high_high = high >> 32;
    ulong high_low = high & GOLDILOCKS_EPSILON;

    // Since 2^64 = 2^32 - 1 and 2^32(2^32 - 1) = -1 modulo p,
    // low + high * 2^64 reduces to low - high_high + high_low * (2^32 - 1).
    ulong reduced = low - high_high;
    if (reduced > low) {
        reduced -= GOLDILOCKS_EPSILON;
    }
    ulong addend = high_low * GOLDILOCKS_EPSILON;
    ulong result = reduced + addend;
    return result + (result < reduced) * GOLDILOCKS_EPSILON;
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

inline ulong gl_sub(ulong a, ulong b) {
    ulong diff = a - b;
    ulong under = diff > a;
    diff -= under * GOLDILOCKS_EPSILON;
    ulong under2 = (under != 0UL) && (diff > (~0UL - GOLDILOCKS_EPSILON));
    return diff - under2 * GOLDILOCKS_EPSILON;
}

inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value4 = gl_mul(value2, value2);
    ulong value3 = gl_mul(value, value2);
    return gl_mul(value3, value4);
}

// ---------------------------------------------------------------------------
// Lazy (deferred) reduction.
//
// The only contract on the permutation is congruence modulo
// p = 2^64 - 2^32 + 1: every kernel canonicalizes exactly once when writing
// final digests, and all Goldilocks operations accept arbitrary 64-bit
// representatives. The linear layers therefore accumulate exact integers in
// 96 bits -- a ulong low word plus a uint counting 2^64 overflows -- and
// reduce once per output element (mirroring the CPU reference, which runs
// external_linear_layer_u128 in u128 with one final reduction), instead of
// paying a double-corrected gl_add per addition.
//
// Identities, with EPSILON = 2^32 - 1:
//   2^64 ≡ EPSILON (mod p)     and     2^96 ≡ -1 (mod p).
// Hence lo + 2^64*hi ≡ lo + EPSILON*hi (mod p). For any hi < 2^32 the fold-in
// t = hi*EPSILON satisfies t <= (2^32-1)^2 = 2^64 - 2^33 + 1, and if
// lo + t wraps, the wrapped value r < t, so r + EPSILON < 2^64: one overflow
// correction is always enough. Every hi tracked below stays below 32.

struct u96 {
    ulong lo;
    uint hi;
};

inline u96 u96_of(ulong x) {
    return u96{x, 0u};
}

// Exact 96-bit addition: a single u64 + u64 addition wraps at most once, and
// the hi words (all < 32 here) can never overflow a uint.
inline u96 u96_add(u96 a, u96 b) {
    ulong lo = a.lo + b.lo;
    return u96{lo, a.hi + b.hi + (uint)(lo < b.lo)};
}

inline u96 u96_add64(u96 a, ulong b) {
    ulong lo = a.lo + b;
    return u96{lo, a.hi + (uint)(lo < b)};
}

// lo + 2^64*hi ≡ lo + EPSILON*hi (mod p), single-correction (see above).
inline ulong u96_reduce(u96 a) {
    ulong t = (ulong)a.hi * GOLDILOCKS_EPSILON;
    ulong r = a.lo + t;
    return r + (ulong)(r < t) * GOLDILOCKS_EPSILON;
}

// (a*b + c) mod p for arbitrary 64-bit representatives. With
// a*b = low + 2^64*high and high = hh*2^32 + hl:
//   2^64*high = hh*2^96 + hl*2^64 ≡ hl*EPSILON - hh (mod p),
// so a*b + c ≡ c + low + hl*EPSILON + (p - hh) (mod p). All four terms are
// nonnegative and < 2^64 (hl*EPSILON <= (2^32-1)^2, 0 < p - hh <= p), so the
// exact sum fits u96 with hi <= 3 and one reduction finishes the job. This
// fuses the multiply's final correction with the following addition, which
// the eager gl_add(c, gl_mul(a, b)) form pays for separately.
inline ulong gl_mul_add(ulong a, ulong b, ulong c) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    u96 acc = u96_add64(u96_of(c), low);
    acc = u96_add64(acc, (high & GOLDILOCKS_EPSILON) * GOLDILOCKS_EPSILON);
    acc = u96_add64(acc, GOLDILOCKS_PRIME - (high >> 32));
    return u96_reduce(acc);
}

// (a*b + c + rc) mod p; one more term keeps hi <= 4.
inline ulong gl_mul_add_rc(ulong a, ulong b, ulong c, ulong rc) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    u96 acc = u96_add64(u96_of(c), low);
    acc = u96_add64(acc, rc);
    acc = u96_add64(acc, (high & GOLDILOCKS_EPSILON) * GOLDILOCKS_EPSILON);
    acc = u96_add64(acc, GOLDILOCKS_PRIME - (high >> 32));
    return u96_reduce(acc);
}

// M_4 block of the external layer, kept as an exact integer:
//   y0 = 2x0 + 3x1 +  x2 +  x3
//   y1 =  x0 + 2x1 + 3x2 +  x3
//   y2 =  x0 +  x1 + 2x2 + 3x3
//   y3 = 3x0 +  x1 +  x2 + 2x3
// Row weight 7, so y_i <= 7*(2^64 - 1) and hi <= 6.
inline void mat4_lazy(thread const ulong* x, thread u96* y) {
    u96 t01 = u96_add64(u96_of(x[0]), x[1]);
    u96 t23 = u96_add64(u96_of(x[2]), x[3]);
    u96 total = u96_add(t01, t23);
    y[0] = u96_add64(u96_add(total, t01), x[1]);
    y[1] = u96_add64(u96_add64(u96_add64(total, x[1]), x[2]), x[2]);
    y[2] = u96_add64(u96_add(total, t23), x[3]);
    y[3] = u96_add64(u96_add64(u96_add64(total, x[3]), x[0]), x[0]);
}

// Shared unreduced external layer: after the three M_4 blocks (hi <= 6), the
// circulant column sums reach 21*(2^64 - 1) (hi <= 20) and each output
// y_i + sums[i % 4] reaches 28*(2^64 - 1) (hi <= 27) -- comfortably inside
// u96, exactly like the CPU's u128 accumulation, which peaks below 2^69.
inline void external_layer_lazy(thread const ulong state[12], thread u96 acc[12]) {
    mat4_lazy(state, acc);
    mat4_lazy(state + 4, acc + 4);
    mat4_lazy(state + 8, acc + 8);

    u96 sums[4];
    for (uint k = 0; k < 4; ++k) {
        sums[k] = u96_add(u96_add(acc[k], acc[k + 4]), acc[k + 8]);
    }
    for (uint i = 0; i < 12; ++i) {
        acc[i] = u96_add(acc[i], sums[i & 3]);
    }
}

inline void external_linear_layer(thread ulong state[12]) {
    u96 acc[12];
    external_layer_lazy(state, acc);
    for (uint i = 0; i < 12; ++i) {
        state[i] = u96_reduce(acc[i]);
    }
}

// External layer with the next round's constants folded into the accumulator
// before its single reduction (hi <= 28 with the extra canonical constant):
// algebraically identical to external_linear_layer followed by per-element
// gl_add of the constants, at a fraction of the cost.
inline void external_linear_layer_rc(thread ulong state[12], constant ulong* rc) {
    u96 acc[12];
    external_layer_lazy(state, acc);
    for (uint i = 0; i < 12; ++i) {
        state[i] = u96_reduce(u96_add64(acc[i], rc[i]));
    }
}

// External layer folding a single constant into element 0 (the upcoming
// partial round's constant add touches only state[0]).
inline void external_linear_layer_rc0(thread ulong state[12], ulong rc0) {
    u96 acc[12];
    external_layer_lazy(state, acc);
    state[0] = u96_reduce(u96_add64(acc[0], rc0));
    for (uint i = 1; i < 12; ++i) {
        state[i] = u96_reduce(acc[i]);
    }
}

// Carry-counted 12-element sum (hi <= 11), reduced once.
inline ulong sum_state(thread const ulong state[12]) {
    u96 sum = u96_of(state[0]);
    for (uint i = 1; i < 12; ++i) {
        sum = u96_add64(sum, state[i]);
    }
    return u96_reduce(sum);
}

// Internal layer with the next partial round's constant folded into element
// 0's fused mul-add.
inline void internal_linear_layer_rc0(thread ulong state[12], constant ulong* diagonal, ulong rc0) {
    ulong sum = sum_state(state);
    state[0] = gl_mul_add_rc(state[0], diagonal[0], sum, rc0);
    for (uint i = 1; i < 12; ++i) {
        state[i] = gl_mul_add(state[i], diagonal[i], sum);
    }
}

// Internal layer folding a full external-constant row into every output (used
// by the last partial round, which the second block of full rounds follows).
inline void internal_linear_layer_rc12(thread ulong state[12], constant ulong* diagonal, constant ulong* rc) {
    ulong sum = sum_state(state);
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_mul_add_rc(state[i], diagonal[i], sum, rc[i]);
    }
}

inline void sbox_layer(thread ulong state[12]) {
    for (uint i = 0; i < 12; ++i) {
        state[i] = pow7(state[i]);
    }
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
//
// The round schedule is the canonical one with every add-round-constant step
// folded into the linear layer that precedes it, so the executed operation
// sequence expands to exactly the reference order:
//   L A0 S L A1 S L A2 S L A3 S L | a0 s I a1 s I ... a21 s I | A4 S L ... A7 S L
// (L = external layer, A = 12-constant add, S = full sbox, a = state[0]
// constant add, s = state[0] sbox, I = internal layer).
inline void poseidon2(thread ulong state[12], constant ulong* parameters) {
    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;

    external_linear_layer_rc(state, external_constants);

    for (uint round = 0; round < 3; ++round) {
        sbox_layer(state);
        external_linear_layer_rc(state, external_constants + (round + 1) * 12);
    }
    sbox_layer(state);
    external_linear_layer_rc0(state, internal_constants[0]);

    for (uint round = 0; round < 21; ++round) {
        state[0] = pow7(state[0]);
        internal_linear_layer_rc0(state, diagonal, internal_constants[round + 1]);
    }
    state[0] = pow7(state[0]);
    internal_linear_layer_rc12(state, diagonal, external_constants + 4 * 12);

    for (uint round = 4; round < 7; ++round) {
        sbox_layer(state);
        external_linear_layer_rc(state, external_constants + (round + 1) * 12);
    }
    sbox_layer(state);
    external_linear_layer(state);
}

// Two independent permutations with stage-interleaved instruction streams:
// corresponding stages of the two states sit adjacent in the (fully unrolled)
// instruction sequence, so the scheduler can overlap one state's dependency
// chains -- the serial pow7 of the partial rounds in particular -- with the
// other's.
inline void poseidon2_x2(thread ulong state[2][12], constant ulong* parameters) {
    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;

    for (uint n = 0; n < 2; ++n) {
        external_linear_layer_rc(state[n], external_constants);
    }

    for (uint round = 0; round < 3; ++round) {
        for (uint n = 0; n < 2; ++n) {
            sbox_layer(state[n]);
        }
        for (uint n = 0; n < 2; ++n) {
            external_linear_layer_rc(state[n], external_constants + (round + 1) * 12);
        }
    }
    for (uint n = 0; n < 2; ++n) {
        sbox_layer(state[n]);
    }
    for (uint n = 0; n < 2; ++n) {
        external_linear_layer_rc0(state[n], internal_constants[0]);
    }

    for (uint round = 0; round < 21; ++round) {
        for (uint n = 0; n < 2; ++n) {
            state[n][0] = pow7(state[n][0]);
        }
        for (uint n = 0; n < 2; ++n) {
            internal_linear_layer_rc0(state[n], diagonal, internal_constants[round + 1]);
        }
    }
    for (uint n = 0; n < 2; ++n) {
        state[n][0] = pow7(state[n][0]);
    }
    for (uint n = 0; n < 2; ++n) {
        internal_linear_layer_rc12(state[n], diagonal, external_constants + 4 * 12);
    }

    for (uint round = 4; round < 7; ++round) {
        for (uint n = 0; n < 2; ++n) {
            sbox_layer(state[n]);
        }
        for (uint n = 0; n < 2; ++n) {
            external_linear_layer_rc(state[n], external_constants + (round + 1) * 12);
        }
    }
    for (uint n = 0; n < 2; ++n) {
        sbox_layer(state[n]);
    }
    for (uint n = 0; n < 2; ++n) {
        external_linear_layer(state[n]);
    }
}

kernel void poseidon2_hash_leaves(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    const device ulong* input = leaves + (ulong)gid * leaf_width;
    device ulong* output = hashes + (ulong)gid * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(input[i]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            // Absorbed raw: the permutation is mod-p arithmetic on 64-bit
            // representatives (the CPU sponge also absorbs noncanonical
            // values verbatim), and only final digest writes canonicalize.
            state[i] = input[offset + i];
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

// Writes the state of the zero-padded, coset-shifted coefficient array as it
// exists after `reverse_index_bits_in_place` plus the first `rate_bits`
// replication rounds of `fft_classic`: position i of column `col` holds
// shift^k * coeffs[col][k] with k = reverse_bits(i >> rate_bits, log_degree).
kernel void ntt_prepare(
    const device ulong* coeffs [[buffer(0)]],
    const device ulong* shift_pows [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant uint& degree [[buffer(3)]],
    constant uint& lde_size [[buffer(4)]],
    constant uint& log_degree [[buffer(5)]],
    constant uint& rate_bits [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= lde_size) {
        return;
    }
    uint col = gid.y;
    uint k = log_degree == 0
        ? 0
        : (reverse_bits(i >> rate_bits) >> (32 - log_degree));
    ulong value = gl_mul(shift_pows[k], coeffs[(ulong)col * degree + k]);
    out[(ulong)col * lde_size + i] = value;
}

// One radix-2 decimation-in-time butterfly stage over every column, matching
// fft_classic: (u, v) := (u + w*v, u - w*v) with w = roots[j]. The final stage
// canonicalizes so downstream consumers see canonical representations.
kernel void ntt_stage(
    device ulong* values [[buffer(0)]],
    const device ulong* roots [[buffer(1)]],
    constant uint& lde_size [[buffer(2)]],
    constant uint& log_half_m [[buffer(3)]],
    constant uint& canonicalize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint half_butterflies = lde_size >> 1;
    if (t >= half_butterflies) {
        return;
    }
    ulong colbase = (ulong)gid.y * lde_size;
    uint half_m = 1u << log_half_m;
    uint j = t & (half_m - 1u);
    uint base = ((t >> log_half_m) << (log_half_m + 1u));
    uint u_index = base + j;
    uint v_index = u_index + half_m;

    ulong u = values[colbase + u_index];
    ulong v = values[colbase + v_index];
    ulong w = roots[j];
    ulong wv = gl_mul(w, v);
    ulong out_u = gl_add(u, wv);
    ulong out_v = gl_sub(u, wv);
    if (canonicalize != 0u) {
        out_u = gl_canonicalize(out_u);
        out_v = gl_canonicalize(out_v);
    }
    values[colbase + u_index] = out_u;
    values[colbase + v_index] = out_v;
}

// Converts a forward-FFT output into IFFT coefficients, matching plonky2's
// `ifft`: coeffs[i] = fft_out[(n - i) mod n] * n^{-1}, canonicalized so the
// CPU-side readback and the downstream LDE prepare see canonical values.
kernel void ifft_finalize(
    const device ulong* fft_out [[buffer(0)]],
    device ulong* coeffs [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& n_inv [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= n) {
        return;
    }
    ulong colbase = (ulong)gid.y * n;
    uint src = (n - i) & (n - 1u);
    coeffs[colbase + i] = gl_canonicalize(gl_mul(fft_out[colbase + src], n_inv));
}

kernel void poseidon2_hash_leaves_colmajor(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    // Input is column-major: column j occupies leaves[j * leaf_count..(j + 1) *
    // leaf_count] in natural row order, so adjacent threads read adjacent
    // addresses. The digest of natural row gid belongs to tree leaf
    // reverse_bits(gid), so outputs scatter by bit reversal.
    uint out_row = log_leaf_count == 0
        ? gid
        : (reverse_bits(gid) >> (32 - log_leaf_count));
    device ulong* output = hashes + (ulong)out_row * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(leaves[(ulong)i * leaf_count + gid]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            // Raw absorb; see poseidon2_hash_leaves.
            state[i] = leaves[(ulong)(offset + i) * leaf_count + gid];
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

kernel void poseidon2_hash_parents(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& parent_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= parent_count) {
        return;
    }

    const device ulong* input = children + (ulong)gid * 8;
    ulong state[12] = { 0 };
    for (uint i = 0; i < 8; ++i) {
        state[i] = input[i];
    }
    poseidon2(state, parameters);

    device ulong* output = parents + (ulong)gid * 4;
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

// Two parent nodes per thread, permuted with interleaved instruction streams
// (poseidon2_x2) for instruction-level parallelism. Selected by the host-side
// PARENT_NODES_PER_THREAD toggle; the one-node kernel above stays available
// as the low-register-pressure fallback. An odd tail duplicates the last
// node's input into the second lane and simply skips its write.
kernel void poseidon2_hash_parents_x2(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& parent_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint first = gid * 2;
    if (first >= parent_count) {
        return;
    }
    bool has_second = first + 1 < parent_count;
    uint second = has_second ? first + 1 : first;

    ulong state[2][12] = { { 0 }, { 0 } };
    const device ulong* input0 = children + (ulong)first * 8;
    const device ulong* input1 = children + (ulong)second * 8;
    for (uint i = 0; i < 8; ++i) {
        state[0][i] = input0[i];
        state[1][i] = input1[i];
    }
    poseidon2_x2(state, parameters);

    device ulong* output0 = parents + (ulong)first * 4;
    for (uint i = 0; i < 4; ++i) {
        output0[i] = gl_canonicalize(state[0][i]);
    }
    if (has_second) {
        device ulong* output1 = parents + (ulong)second * 4;
        for (uint i = 0; i < 4; ++i) {
            output1[i] = gl_canonicalize(state[1][i]);
        }
    }
}
