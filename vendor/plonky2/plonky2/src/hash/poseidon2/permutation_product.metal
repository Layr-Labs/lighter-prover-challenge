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

inline ulong gl_sub(ulong a, ulong b) {
    ulong diff = a - b;
    ulong under = diff > a;
    diff -= under * GOLDILOCKS_EPSILON;
    ulong under2 = (under != 0UL) && (diff > (~0UL - GOLDILOCKS_EPSILON));
    return diff - under2 * GOLDILOCKS_EPSILON;
}

inline ulong gl_mul(ulong a, ulong b) {
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong high_high = high >> 32;
    ulong high_low = high & GOLDILOCKS_EPSILON;
    ulong reduced = low - high_high;
    if (reduced > low) {
        reduced -= GOLDILOCKS_EPSILON;
    }
    ulong addend = high_low * GOLDILOCKS_EPSILON;
    ulong result = reduced + addend;
    return result + (result < reduced) * GOLDILOCKS_EPSILON;
}

inline ulong gl_mul_add(ulong a, ulong b, ulong addend) {
    return gl_add(gl_mul(a, b), addend);
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

inline ulong gl_pow_u32(ulong base, uint exponent) {
    ulong result = 1;
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result = gl_mul(result, base);
        }
        base = gl_mul(base, base);
        exponent >>= 1u;
    }
    return result;
}

// Evaluates the permutation argument's partial-product checks at one natural
// quotient-domain point. The two L_0 boundary terms stay on the CPU; alpha
// powers begin after those two rows and preserve the ordinary challenge-major
// chunk order exactly.
kernel void permutation_product_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants_sigmas [[buffer(1)]],
    const device ulong* zs_partial_products [[buffer(2)]],
    device ulong* output [[buffer(3)]],
    constant ulong* betas [[buffer(4)]],
    constant ulong* gammas [[buffer(5)]],
    constant ulong* beta_k_is [[buffer(6)]],
    constant ulong* alpha_powers [[buffer(7)]],
    constant uint& lde_rows [[buffer(8)]],
    constant uint& quotient_rows [[buffer(9)]],
    constant uint& step [[buffer(10)]],
    constant uint& next_step [[buffer(11)]],
    constant uint& sigma_start [[buffer(12)]],
    constant uint& zs_start [[buffer(13)]],
    constant uint& partial_products_start [[buffer(14)]],
    constant uint& num_routed_wires [[buffer(15)]],
    constant uint& chunk_size [[buffer(16)]],
    constant uint& num_partial_products [[buffer(17)]],
    constant uint& alpha_stride [[buffer(18)]],
    constant ulong& root [[buffer(19)]],
    constant ulong& coset_shift [[buffer(20)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    uint next_gid = (gid + next_step) & (quotient_rows - 1u);
    uint next_source_row = next_gid * step;
    ulong x = gl_mul(coset_shift, gl_pow_u32(root, gid));
    uint num_chunks = num_partial_products + 1u;
    ulong accumulators[2] = { 0, 0 };

    for (uint challenge = 0; challenge < 2u; ++challenge) {
        for (uint chunk = 0; chunk < num_chunks; ++chunk) {
            uint wire_start = chunk * chunk_size;
            uint wire_end = min(wire_start + chunk_size, num_routed_wires);
            ulong wire = wires[(ulong)wire_start * lde_rows + source_row];
            ulong numerator = gl_add(
                gl_add(
                    wire,
                    gl_mul(beta_k_is[challenge * num_routed_wires + wire_start], x)),
                gammas[challenge]);
            ulong sigma = constants_sigmas[
                (ulong)(sigma_start + wire_start) * lde_rows + source_row];
            ulong denominator = gl_add(
                gl_add(wire, gl_mul(betas[challenge], sigma)),
                gammas[challenge]);

            for (uint wire_index = wire_start + 1u;
                 wire_index < wire_end;
                 ++wire_index) {
                wire = wires[(ulong)wire_index * lde_rows + source_row];
                ulong numerator_factor = gl_add(
                    gl_add(
                        wire,
                        gl_mul(
                            beta_k_is[challenge * num_routed_wires + wire_index],
                            x)),
                    gammas[challenge]);
                sigma = constants_sigmas[
                    (ulong)(sigma_start + wire_index) * lde_rows + source_row];
                ulong denominator_factor = gl_add(
                    gl_add(wire, gl_mul(betas[challenge], sigma)),
                    gammas[challenge]);
                numerator = gl_mul(numerator, numerator_factor);
                denominator = gl_mul(denominator, denominator_factor);
            }

            ulong previous;
            if (chunk == 0u) {
                previous = zs_partial_products[
                    (ulong)(zs_start + challenge) * lde_rows + source_row];
            } else {
                uint column = partial_products_start
                    + challenge * num_partial_products + chunk - 1u;
                previous = zs_partial_products[(ulong)column * lde_rows + source_row];
            }
            ulong next;
            if (chunk == num_partial_products) {
                next = zs_partial_products[
                    (ulong)(zs_start + challenge) * lde_rows + next_source_row];
            } else {
                uint column = partial_products_start
                    + challenge * num_partial_products + chunk;
                next = zs_partial_products[(ulong)column * lde_rows + source_row];
            }

            ulong term = gl_sub(gl_mul(previous, numerator), gl_mul(next, denominator));
            uint term_index = challenge * num_chunks + chunk;
            accumulators[0] = gl_mul_add(
                term,
                alpha_powers[term_index],
                accumulators[0]);
            accumulators[1] = gl_mul_add(
                term,
                alpha_powers[alpha_stride + term_index],
                accumulators[1]);
        }
    }

    output[(ulong)gid * 2] = gl_canonicalize(accumulators[0]);
    output[(ulong)gid * 2 + 1] = gl_canonicalize(accumulators[1]);
}
