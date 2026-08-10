// K6 implementation shared by two link modes. The ordinary runtime source
// appends k6_residual_wrapper.metal and lets the compiler inline this function.
// A build-host probe compiles this file as a Metal dynamic library, serializes
// it, and links the same tiny wrapper against it in the worker. Keeping both
// modes on one implementation prevents their arithmetic from drifting.

inline void gl_ext2_mul(
    ulong a0,
    ulong a1,
    ulong b0,
    ulong b1,
    thread ulong& output0,
    thread ulong& output1) {
    output0 = gl_mul_add(7, gl_mul(a1, b1), gl_mul(a0, b0));
    output1 = gl_mul_add(a1, b0, gl_mul(a0, b1));
}

inline void coset16_interpolation_step(
    ulong point0,
    ulong point1,
    ulong value0,
    ulong value1,
    uint domain_index,
    constant ulong* coset_domain,
    constant ulong* coset_weights,
    thread ulong& eval0,
    thread ulong& eval1,
    thread ulong& product0,
    thread ulong& product1) {
    ulong term0 = gl_sub(point0, coset_domain[domain_index]);
    ulong term1 = point1;
    ulong eval_term0;
    ulong eval_term1;
    ulong weighted_product0;
    ulong weighted_product1;
    ulong next_product0;
    ulong next_product1;
    gl_ext2_mul(eval0, eval1, term0, term1, eval_term0, eval_term1);
    gl_ext2_mul(
        gl_mul(value0, coset_weights[domain_index]),
        gl_mul(value1, coset_weights[domain_index]),
        product0,
        product1,
        weighted_product0,
        weighted_product1);
    gl_ext2_mul(product0, product1, term0, term1, next_product0, next_product1);
    eval0 = gl_add(eval_term0, weighted_product0);
    eval1 = gl_add(eval_term1, weighted_product1);
    product0 = next_product0;
    product1 = next_product1;
}

inline void k6_emit(
    ulong constraint,
    constant ulong* alpha_powers,
    uint alpha_stride,
    thread ulong accumulators[2],
    uint constraint_index) {
    accumulators[0] =
        gl_mul_add(constraint, alpha_powers[constraint_index], accumulators[0]);
    accumulators[1] = gl_mul_add(
        constraint, alpha_powers[alpha_stride + constraint_index], accumulators[1]);
}

// Each record is ten uints:
//   selector column, gate index, selector group start/end, include UNUSED,
//   kind (13..18), operation count, raw constant-column base, reserved, reserved.
#if defined(LIGHTER_K6_DYNAMIC_LIBRARY)
#define LIGHTER_K6_LINKAGE __attribute__((visibility("default")))
#else
#define LIGHTER_K6_LINKAGE inline
#endif

namespace lighter_k6 {
LIGHTER_K6_LINKAGE void quotient(
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
    uint gid) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong total[2] = { 0, 0 };
    for (uint k6_index = 0; k6_index < k6_count; ++k6_index) {
        constant uint* spec = metadata + k6_index * 10u;
        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint kind = spec[5];
        uint num_ops = spec[6];
        uint constant_base = spec[7];

        ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        ulong gate_accumulators[2] = { 0, 0 };
        uint constraint_index = 0;
        if (kind == 13u) {
            // ArithmeticGate: four wires per operation and two gate constants.
            ulong const_0 = constants[(ulong)constant_base * lde_rows + source_row];
            ulong const_1 =
                constants[((ulong)constant_base + 1u) * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 4u;
                ulong multiplicand_0 =
                    wires[(wire_base + 0u) * lde_rows + source_row];
                ulong multiplicand_1 =
                    wires[(wire_base + 1u) * lde_rows + source_row];
                ulong addend = wires[(wire_base + 2u) * lde_rows + source_row];
                ulong output_value =
                    wires[(wire_base + 3u) * lde_rows + source_row];
                ulong computed = gl_mul_add(
                    const_1,
                    addend,
                    gl_mul(const_0, gl_mul(multiplicand_0, multiplicand_1)));
                k6_emit(
                    gl_sub(output_value, computed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 14u) {
            // ArithmeticExtensionGate<2>: a, b, z, out are extension pairs.
            ulong const_0 = constants[(ulong)constant_base * lde_rows + source_row];
            ulong const_1 =
                constants[((ulong)constant_base + 1u) * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 8u;
                ulong a0 = wires[(wire_base + 0u) * lde_rows + source_row];
                ulong a1 = wires[(wire_base + 1u) * lde_rows + source_row];
                ulong b0 = wires[(wire_base + 2u) * lde_rows + source_row];
                ulong b1 = wires[(wire_base + 3u) * lde_rows + source_row];
                ulong z0 = wires[(wire_base + 4u) * lde_rows + source_row];
                ulong z1 = wires[(wire_base + 5u) * lde_rows + source_row];
                ulong out0 = wires[(wire_base + 6u) * lde_rows + source_row];
                ulong out1 = wires[(wire_base + 7u) * lde_rows + source_row];
                ulong product_0 =
                    gl_mul_add(7, gl_mul(a1, b1), gl_mul(a0, b0));
                ulong product_1 = gl_mul_add(a1, b0, gl_mul(a0, b1));
                ulong computed_0 =
                    gl_mul_add(const_1, z0, gl_mul(const_0, product_0));
                ulong computed_1 =
                    gl_mul_add(const_1, z1, gl_mul(const_0, product_1));
                k6_emit(
                    gl_sub(out0, computed_0),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                k6_emit(
                    gl_sub(out1, computed_1),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 15u) {
            // MulExtensionGate<2>: a, b and out are extension pairs.
            ulong const_0 = constants[(ulong)constant_base * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 6u;
                ulong a0 = wires[(wire_base + 0u) * lde_rows + source_row];
                ulong a1 = wires[(wire_base + 1u) * lde_rows + source_row];
                ulong b0 = wires[(wire_base + 2u) * lde_rows + source_row];
                ulong b1 = wires[(wire_base + 3u) * lde_rows + source_row];
                ulong out0 = wires[(wire_base + 4u) * lde_rows + source_row];
                ulong out1 = wires[(wire_base + 5u) * lde_rows + source_row];
                ulong product_0 =
                    gl_mul_add(7, gl_mul(a1, b1), gl_mul(a0, b0));
                ulong product_1 = gl_mul_add(a1, b0, gl_mul(a0, b1));
                k6_emit(
                    gl_sub(out0, gl_mul(const_0, product_0)),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                k6_emit(
                    gl_sub(out1, gl_mul(const_0, product_1)),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 16u) {
            // CosetInterpolationGate<GoldilocksField, 2>, subgroup_bits=4,
            // degree=6. Layout: shift; 16 extension values; evaluation point
            // and value; two eval checkpoints; two product checkpoints;
            // shifted evaluation point.
            ulong shift = wires[source_row];
            ulong point0 = wires[(ulong)33u * lde_rows + source_row];
            ulong point1 = wires[(ulong)34u * lde_rows + source_row];
            ulong shifted0 = wires[(ulong)45u * lde_rows + source_row];
            ulong shifted1 = wires[(ulong)46u * lde_rows + source_row];
            k6_emit(
                gl_sub(point0, gl_mul(shifted0, shift)),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);
            k6_emit(
                gl_sub(point1, gl_mul(shifted1, shift)),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);

            ulong eval0 = 0;
            ulong eval1 = 0;
            ulong product0 = 1;
            ulong product1 = 0;
            for (uint i = 0; i < 6u; ++i) {
                ulong value_base = 1u + 2u * i;
                coset16_interpolation_step(
                    shifted0,
                    shifted1,
                    wires[value_base * lde_rows + source_row],
                    wires[(value_base + 1u) * lde_rows + source_row],
                    i,
                    coset_domain,
                    coset_weights,
                    eval0,
                    eval1,
                    product0,
                    product1);
            }
            for (uint checkpoint = 0; checkpoint < 2u; ++checkpoint) {
                ulong expected_eval0 =
                    wires[((ulong)37u + 2u * checkpoint) * lde_rows + source_row];
                ulong expected_eval1 =
                    wires[((ulong)38u + 2u * checkpoint) * lde_rows + source_row];
                ulong expected_product0 =
                    wires[((ulong)41u + 2u * checkpoint) * lde_rows + source_row];
                ulong expected_product1 =
                    wires[((ulong)42u + 2u * checkpoint) * lde_rows + source_row];
                k6_emit(
                    gl_sub(expected_eval0, eval0),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                k6_emit(
                    gl_sub(expected_eval1, eval1),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                k6_emit(
                    gl_sub(expected_product0, product0),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                k6_emit(
                    gl_sub(expected_product1, product1),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                eval0 = expected_eval0;
                eval1 = expected_eval1;
                product0 = expected_product0;
                product1 = expected_product1;

                uint start = checkpoint == 0u ? 6u : 11u;
                uint end = checkpoint == 0u ? 11u : 16u;
                for (uint i = start; i < end; ++i) {
                    ulong value_base = 1u + 2u * i;
                    coset16_interpolation_step(
                        shifted0,
                        shifted1,
                        wires[value_base * lde_rows + source_row],
                        wires[(value_base + 1u) * lde_rows + source_row],
                        i,
                        coset_domain,
                        coset_weights,
                        eval0,
                        eval1,
                        product0,
                        product1);
                }
            }
            k6_emit(
                gl_sub(wires[(ulong)35u * lde_rows + source_row], eval0),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);
            k6_emit(
                gl_sub(wires[(ulong)36u * lde_rows + source_row], eval1),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);
        } else if (kind == 17u) {
            for (uint i = 0; i < 2u; ++i) {
                k6_emit(
                    gl_sub(
                        constants[((ulong)constant_base + i) * lde_rows + source_row],
                        wires[(ulong)i * lde_rows + source_row]),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 18u) {
            for (uint i = 0; i < 4u; ++i) {
                k6_emit(
                    gl_sub(
                        wires[(ulong)i * lde_rows + source_row],
                        public_inputs_hash[i]),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else {
            // A malformed host record is unsatisfiable instead of omitted.
            k6_emit(
                1,
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);
        }

        total[0] = gl_mul_add(filter, gate_accumulators[0], total[0]);
        total[1] = gl_mul_add(filter, gate_accumulators[1], total[1]);
    }

    // The prebuilt Range/U32 command precedes this tail command on the same
    // queue and initializes both words. Fold K6 into that retained vector so
    // the host performs one wait, one denominator pass and one recycle.
    output[(ulong)gid * 2u] =
        gl_canonicalize(gl_add(output[(ulong)gid * 2u], total[0]));
    output[(ulong)gid * 2u + 1u] = gl_canonicalize(
        gl_add(output[(ulong)gid * 2u + 1u], total[1]));
}
} // namespace lighter_k6

#undef LIGHTER_K6_LINKAGE
