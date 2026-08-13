// Generated exact747 split-roster R2 kernels.
// Arithmetic bodies are mechanically extracted from exact747's generic
// range_check_gate_quotient, including deferred alpha_acc_t accumulation and
// the strict quintic transitions. Generic production fallback remains required.

kernel void range_check_gate_quotient_static_chain_d14_r2(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* _metadata [[buffer(4)]],
    constant uint& _lde_rows [[buffer(5)]],
    constant uint& _quotient_rows [[buffer(6)]],
    constant uint& _step [[buffer(7)]],
    constant uint& _alpha_stride [[buffer(8)]],
    constant uint& _range_count [[buffer(9)]],
    constant uint& _u32_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint STATIC_LDE_ROWS = 131072u;
    constexpr uint STATIC_ALPHA_STRIDE = 123u;
    if (gid >= STATIC_LDE_ROWS) {
        return;
    }
    uint source_row = gid;
    ulong total[2] = { 0, 0 };
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 3u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 10u;

        constexpr uint num_ops = 26u;

        constexpr uint num_addends = 4u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // AdditionGate: three routed words per operation (x, y, output).
            // The addend-count slot carries the first of its two raw constant
            // columns, immediately after the selector prefix.
            uint constant_base = num_addends;
            ulong const_0 = constants[(ulong)constant_base * STATIC_LDE_ROWS + source_row];
            ulong const_1 = constants[((ulong)constant_base + 1u) * STATIC_LDE_ROWS + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 3u;
                ulong addend_0 = wires[(wire_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong addend_1 = wires[(wire_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong output_value = wires[(wire_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong computed = gl_add(
                    gl_mul(addend_0, const_0),
                    gl_mul(addend_1, const_1));
                range_check_gate_emit(
                    gl_sub(output_value, computed),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 4u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 11u;

        constexpr uint num_ops = 63u;

        constexpr uint num_addends = 2u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // BaseSumGate: wire 0 is the sum and the next `num_ops` wires are
            // little-endian limbs. The addend-count slot carries base 2 or 4.
            //
            // The Horner step's `gl_mul(computed, base)` looks like free money
            // -- both bases are powers of two, so doubling or `gl_quadruple`
            // would replace a 128-bit product with one or two field adds, 63
            // of them per row on the widest family. Measured, it is not:
            // specializing the base outside the loop costs more in code
            // duplication than the arithmetic saves. Recomposition-only,
            // against the deferred-accumulator kernel: d18 160.9 -> 161.6 ms,
            // d16-heavy 60.40 -> 60.53 ms, d14 4.135 -> 4.112 ms; splitting the
            // range-constraint loop as well costs another 1.5 ms on d18. Both
            // arms bit-exact, so this is a scheduling/footprint effect, not an
            // arithmetic one. Keep the multiply.
            ulong base = num_addends;
            ulong computed = 0;
            for (uint remaining = num_ops; remaining > 0u; --remaining) {
                uint limb = remaining - 1u;
                computed = gl_add(
                    gl_mul(computed, base),
                    wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row]);
            }
            range_check_gate_emit(
                gl_sub(computed, wires[source_row]),
                alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                constraint_index++);
            for (uint limb = 0; limb < num_ops; ++limb) {
                ulong x = wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (base == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 5u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 8u;

        constexpr uint num_ops = 22u;

        constexpr uint num_addends = 4u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // EqualityGate: three routed words per operation (x, y, equal)
            // followed by three unrouted temporaries (diff, invdiff, prod).
            // The addend slot carries the constants column holding the gate's
            // first constant, its "one" value.
            uint constant_column = num_addends;
            ulong const_0 = constants[(ulong)constant_column * STATIC_LDE_ROWS + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 3u;
                ulong x = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong y = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong equal = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong temporary_base = (ulong)num_ops * 3u + (ulong)op * 3u;
                ulong difference = wires[(temporary_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong inverse = wires[(temporary_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong product = wires[(temporary_base + 2u) * STATIC_LDE_ROWS + source_row];

                range_check_gate_emit(
                    gl_sub(gl_sub(x, y), difference),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(difference, inverse), product),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(product, difference), difference),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(const_0, product), equal),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 6u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 9u;

        constexpr uint num_ops = 33u;

        constexpr uint num_addends = 1u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // ReducingGate / ReducingExtensionGate at D == 2. Wires 0..2 are
            // the output, 2..4 alpha, 4..6 the incoming accumulator, then one
            // (base) or two (extension) wires per coefficient, then one
            // two-wire accumulator per step except the last, which aliases the
            // output. Each step emits the two components of
            // `acc * alpha + coeff - next_acc` in that order.
            //
            // The quadratic extension is F[x]/(x^2 - 7) for Goldilocks:
            //   (a0 + a1 x)(b0 + b1 x) = (a0 b0 + 7 a1 b1) + (a0 b1 + a1 b0) x.
            uint extension_coeffs = num_addends;
            uint coeff_wires = extension_coeffs != 0u ? 2u : 1u;
            uint coeff_start = 6u;
            uint acc_start = coeff_start + num_ops * coeff_wires;
            ulong alpha_0 = wires[(ulong)2u * STATIC_LDE_ROWS + source_row];
            ulong alpha_1 = wires[(ulong)3u * STATIC_LDE_ROWS + source_row];
            ulong acc_0 = wires[(ulong)4u * STATIC_LDE_ROWS + source_row];
            ulong acc_1 = wires[(ulong)5u * STATIC_LDE_ROWS + source_row];
            for (uint i = 0; i < num_ops; ++i) {
                uint next_start = (i + 1u == num_ops) ? 0u : acc_start + 2u * i;
                ulong next_0 = wires[(ulong)next_start * STATIC_LDE_ROWS + source_row];
                ulong next_1 = wires[((ulong)next_start + 1u) * STATIC_LDE_ROWS + source_row];

                uint coeff_wire = coeff_start + i * coeff_wires;
                ulong coeff_0 = wires[(ulong)coeff_wire * STATIC_LDE_ROWS + source_row];
                ulong coeff_1 = extension_coeffs != 0u
                    ? wires[((ulong)coeff_wire + 1u) * STATIC_LDE_ROWS + source_row]
                    : 0;

                ulong product_0 = gl_add(
                    gl_mul(acc_0, alpha_0),
                    gl_mul(7, gl_mul(acc_1, alpha_1)));
                ulong product_1 = gl_add(
                    gl_mul(acc_0, alpha_1),
                    gl_mul(acc_1, alpha_0));
                range_check_gate_emit(
                    gl_sub(gl_add(product_0, coeff_0), next_0),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(product_1, coeff_1), next_1),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);

                acc_0 = next_0;
                acc_1 = next_1;
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 1u;
        constexpr uint gate_index = 7u;
        constexpr uint group_start = 7u;
        constexpr uint group_end = 12u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 9u;

        constexpr uint num_ops = 44u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // ReducingGate / ReducingExtensionGate at D == 2. Wires 0..2 are
            // the output, 2..4 alpha, 4..6 the incoming accumulator, then one
            // (base) or two (extension) wires per coefficient, then one
            // two-wire accumulator per step except the last, which aliases the
            // output. Each step emits the two components of
            // `acc * alpha + coeff - next_acc` in that order.
            //
            // The quadratic extension is F[x]/(x^2 - 7) for Goldilocks:
            //   (a0 + a1 x)(b0 + b1 x) = (a0 b0 + 7 a1 b1) + (a0 b1 + a1 b0) x.
            uint extension_coeffs = num_addends;
            uint coeff_wires = extension_coeffs != 0u ? 2u : 1u;
            uint coeff_start = 6u;
            uint acc_start = coeff_start + num_ops * coeff_wires;
            ulong alpha_0 = wires[(ulong)2u * STATIC_LDE_ROWS + source_row];
            ulong alpha_1 = wires[(ulong)3u * STATIC_LDE_ROWS + source_row];
            ulong acc_0 = wires[(ulong)4u * STATIC_LDE_ROWS + source_row];
            ulong acc_1 = wires[(ulong)5u * STATIC_LDE_ROWS + source_row];
            for (uint i = 0; i < num_ops; ++i) {
                uint next_start = (i + 1u == num_ops) ? 0u : acc_start + 2u * i;
                ulong next_0 = wires[(ulong)next_start * STATIC_LDE_ROWS + source_row];
                ulong next_1 = wires[((ulong)next_start + 1u) * STATIC_LDE_ROWS + source_row];

                uint coeff_wire = coeff_start + i * coeff_wires;
                ulong coeff_0 = wires[(ulong)coeff_wire * STATIC_LDE_ROWS + source_row];
                ulong coeff_1 = extension_coeffs != 0u
                    ? wires[((ulong)coeff_wire + 1u) * STATIC_LDE_ROWS + source_row]
                    : 0;

                ulong product_0 = gl_add(
                    gl_mul(acc_0, alpha_0),
                    gl_mul(7, gl_mul(acc_1, alpha_1)));
                ulong product_1 = gl_add(
                    gl_mul(acc_0, alpha_1),
                    gl_mul(acc_1, alpha_0));
                range_check_gate_emit(
                    gl_sub(gl_add(product_0, coeff_0), next_0),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(product_1, coeff_1), next_1),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);

                acc_0 = next_0;
                acc_1 = next_1;
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 1u;
        constexpr uint gate_index = 8u;
        constexpr uint group_start = 7u;
        constexpr uint group_end = 12u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 12u;

        constexpr uint num_ops = 20u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // SelectionGate: four routed wires per operation followed by one
            // temporary wire per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong b = wires[((ulong)(4u * op)) * STATIC_LDE_ROWS + source_row];
                ulong x = wires[((ulong)(4u * op + 1u)) * STATIC_LDE_ROWS + source_row];
                ulong y = wires[((ulong)(4u * op + 2u)) * STATIC_LDE_ROWS + source_row];
                ulong result = wires[((ulong)(4u * op + 3u)) * STATIC_LDE_ROWS + source_row];
                ulong temp = wires[
                    ((ulong)(4u * num_ops + op)) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, y), y), temp),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, x), temp), result),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 12u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 15u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 3u;

        constexpr uint num_ops = 3u;

        constexpr uint num_addends = 8u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // ByteDecompositionGate: per operation, one routed sum wire and
            // `num_addends` routed byte wires (the metadata word carries the
            // byte count for this kind), then four base-4 aux limbs per
            // byte. Constraint order matches the CPU gate exactly: aux range
            // products in ascending wire order, one base-4 recomposition per
            // byte, then the base-256 byte-to-sum recomposition.
            uint num_limbs = num_addends;
            uint routed_per_op = 1u + num_limbs;
            uint aux_per_op = 4u * num_limbs;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong aux_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * aux_per_op;
                for (uint j = 0; j < aux_per_op; ++j) {
                    ulong x = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }
                for (uint byte_index = 0; byte_index < num_limbs; ++byte_index) {
                    ulong chunk = aux_base + (ulong)byte_index * 4u;
                    ulong recomposed = wires[(chunk + 3u) * STATIC_LDE_ROWS + source_row];
                    for (uint remaining = 3u; remaining > 0u; --remaining) {
                        uint k = remaining - 1u;
                        recomposed = gl_add(
                            gl_quadruple(recomposed),
                            wires[(chunk + k) * STATIC_LDE_ROWS + source_row]);
                    }
                    ulong byte_value =
                        wires[(routed_base + 1u + byte_index) * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit(
                        gl_sub(recomposed, byte_value),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }
                ulong recomposed_sum =
                    wires[(routed_base + num_limbs) * STATIC_LDE_ROWS + source_row];
                for (uint remaining = num_limbs - 1u; remaining > 0u; --remaining) {
                    uint k = remaining - 1u;
                    recomposed_sum = gl_add(
                        gl_mul(recomposed_sum, 256),
                        wires[(routed_base + 1u + k) * STATIC_LDE_ROWS + source_row]);
                }
                ulong expected_sum = wires[routed_base * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(recomposed_sum, expected_sum),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 13u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 15u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 6u;

        constexpr uint num_ops = 4u;

        constexpr uint num_addends = 4u;

        constexpr uint result_limbs = 2u;

        constexpr uint num_carry_limbs = 4u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            uint bits = num_addends;
            uint num_extra_constants = result_limbs;
            uint constant_base = num_carry_limbs;
            uint num_copies = num_ops;
            ulong vec_size = 1UL << bits;
            ulong routed_per_copy = vec_size + 2u;
            ulong extra_wire_base = routed_per_copy * num_copies;
            ulong bit_base = extra_wire_base + num_extra_constants;

            for (uint copy = 0; copy < num_copies; ++copy) {
                ulong copy_base = routed_per_copy * copy;

                // RandomAccessGate emits boolean constraints for b_0 upward.
                for (uint i = 0; i < bits; ++i) {
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit(
                        gl_mul(b, gl_sub(b, 1)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }

                // Reconstruct the little-endian index in the CPU's exact
                // reverse-bit `acc.double() + b` order.
                ulong reconstructed_index = 0;
                for (uint remaining = bits; remaining > 0u; --remaining) {
                    uint i = remaining - 1u;
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    reconstructed_index = gl_add(
                        gl_add(reconstructed_index, reconstructed_index), b);
                }
                ulong access_index = wires[copy_base * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(reconstructed_index, access_index),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                // Fold each eight-item block in ascending pair order, then fold
                // block results with the remaining bits in the same order.
                ulong block_results[8];
                uint block_count = (uint)(vec_size / 8u);
                ulong list_base = copy_base + 2u;
                ulong copy_bit_base = bit_base + (ulong)copy * bits;
                for (uint block = 0; block < block_count; ++block) {
                    block_results[block] = random_access_select_8(
                        wires, STATIC_LDE_ROWS, source_row, list_base, copy_bit_base, block);
                }
                uint level_size = block_count;
                for (uint i = 3u; i < bits; ++i) {
                    ulong b = wires[(copy_bit_base + i) * STATIC_LDE_ROWS + source_row];
                    for (uint k = 0; k < level_size / 2u; ++k) {
                        ulong x = block_results[2u * k];
                        ulong y = block_results[2u * k + 1u];
                        block_results[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
                    }
                    level_size /= 2u;
                }
                ulong claimed_element = wires[(copy_base + 1u) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(block_results[0], claimed_element),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }

            // Raw local constants follow all gate and lookup selectors.
            for (uint i = 0; i < num_extra_constants; ++i) {
                ulong local_constant = constants[
                    ((ulong)constant_base + i) * STATIC_LDE_ROWS + source_row];
                ulong extra_wire = wires[
                    (extra_wire_base + i) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(local_constant, extra_wire),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }

    output[(ulong)gid * 2] = gl_canonicalize(total[0]);
    output[(ulong)gid * 2 + 1] = gl_canonicalize(total[1]);
}

kernel void range_check_gate_quotient_static_tx_d16_r2(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* _metadata [[buffer(4)]],
    constant uint& _lde_rows [[buffer(5)]],
    constant uint& _quotient_rows [[buffer(6)]],
    constant uint& _step [[buffer(7)]],
    constant uint& _alpha_stride [[buffer(8)]],
    constant uint& _range_count [[buffer(9)]],
    constant uint& _u32_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint STATIC_LDE_ROWS = 524288u;
    constexpr uint STATIC_ALPHA_STRIDE = 136u;
    if (gid >= STATIC_LDE_ROWS) {
        return;
    }
    uint source_row = gid;
    ulong total[2] = { 0, 0 };
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 14u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 17u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint num_ops = 15u;

        constexpr uint num_aux = 8u;

        constexpr uint final_limb_range = 4u;

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
for (uint op = 0; op < num_ops; ++op) {
            ulong input = wires[(ulong)op * STATIC_LDE_ROWS + source_row];
            ulong aux_base = (ulong)num_ops + (ulong)num_aux * op;
            ulong computed = wires[(aux_base + num_aux - 1u) * STATIC_LDE_ROWS + source_row];
            for (uint remaining = num_aux - 1u; remaining > 0u; --remaining) {
                uint j = remaining - 1u;
                ulong limb = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                computed = gl_add(gl_quadruple(computed), limb);
            }
            range_check_gate_emit(
                gl_sub(computed, input),
                alpha_powers,
                STATIC_ALPHA_STRIDE,
                gate_accumulators,
                constraint_index++);

            for (uint j = 0; j < num_aux; ++j) {
                ulong x = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (j + 1u == num_aux && final_limb_range == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    // x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3),
                    // exactly the production CPU specialization.
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        }

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    
    }
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 15u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 17u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint num_ops = 5u;

        constexpr uint num_aux = 24u;

        constexpr uint final_limb_range = 4u;

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
for (uint op = 0; op < num_ops; ++op) {
            ulong input = wires[(ulong)op * STATIC_LDE_ROWS + source_row];
            ulong aux_base = (ulong)num_ops + (ulong)num_aux * op;
            ulong computed = wires[(aux_base + num_aux - 1u) * STATIC_LDE_ROWS + source_row];
            for (uint remaining = num_aux - 1u; remaining > 0u; --remaining) {
                uint j = remaining - 1u;
                ulong limb = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                computed = gl_add(gl_quadruple(computed), limb);
            }
            range_check_gate_emit(
                gl_sub(computed, input),
                alpha_powers,
                STATIC_ALPHA_STRIDE,
                gate_accumulators,
                constraint_index++);

            for (uint j = 0; j < num_aux; ++j) {
                ulong x = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (j + 1u == num_aux && final_limb_range == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    // x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3),
                    // exactly the production CPU specialization.
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        }

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    
    }
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 16u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 17u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint num_ops = 8u;

        constexpr uint num_aux = 16u;

        constexpr uint final_limb_range = 4u;

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
for (uint op = 0; op < num_ops; ++op) {
            ulong input = wires[(ulong)op * STATIC_LDE_ROWS + source_row];
            ulong aux_base = (ulong)num_ops + (ulong)num_aux * op;
            ulong computed = wires[(aux_base + num_aux - 1u) * STATIC_LDE_ROWS + source_row];
            for (uint remaining = num_aux - 1u; remaining > 0u; --remaining) {
                uint j = remaining - 1u;
                ulong limb = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                computed = gl_add(gl_quadruple(computed), limb);
            }
            range_check_gate_emit(
                gl_sub(computed, input),
                alpha_powers,
                STATIC_ALPHA_STRIDE,
                gate_accumulators,
                constraint_index++);

            for (uint j = 0; j < num_aux; ++j) {
                ulong x = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (j + 1u == num_aux && final_limb_range == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    // x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3),
                    // exactly the production CPU specialization.
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        }

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 2u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 10u;

        constexpr uint num_ops = 26u;

        constexpr uint num_addends = 6u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // AdditionGate: three routed words per operation (x, y, output).
            // The addend-count slot carries the first of its two raw constant
            // columns, immediately after the selector prefix.
            uint constant_base = num_addends;
            ulong const_0 = constants[(ulong)constant_base * STATIC_LDE_ROWS + source_row];
            ulong const_1 = constants[((ulong)constant_base + 1u) * STATIC_LDE_ROWS + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 3u;
                ulong addend_0 = wires[(wire_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong addend_1 = wires[(wire_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong output_value = wires[(wire_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong computed = gl_add(
                    gl_mul(addend_0, const_0),
                    gl_mul(addend_1, const_1));
                range_check_gate_emit(
                    gl_sub(output_value, computed),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 3u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 11u;

        constexpr uint num_ops = 63u;

        constexpr uint num_addends = 2u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // BaseSumGate: wire 0 is the sum and the next `num_ops` wires are
            // little-endian limbs. The addend-count slot carries base 2 or 4.
            //
            // The Horner step's `gl_mul(computed, base)` looks like free money
            // -- both bases are powers of two, so doubling or `gl_quadruple`
            // would replace a 128-bit product with one or two field adds, 63
            // of them per row on the widest family. Measured, it is not:
            // specializing the base outside the loop costs more in code
            // duplication than the arithmetic saves. Recomposition-only,
            // against the deferred-accumulator kernel: d18 160.9 -> 161.6 ms,
            // d16-heavy 60.40 -> 60.53 ms, d14 4.135 -> 4.112 ms; splitting the
            // range-constraint loop as well costs another 1.5 ms on d18. Both
            // arms bit-exact, so this is a scheduling/footprint effect, not an
            // arithmetic one. Keep the multiply.
            ulong base = num_addends;
            ulong computed = 0;
            for (uint remaining = num_ops; remaining > 0u; --remaining) {
                uint limb = remaining - 1u;
                computed = gl_add(
                    gl_mul(computed, base),
                    wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row]);
            }
            range_check_gate_emit(
                gl_sub(computed, wires[source_row]),
                alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                constraint_index++);
            for (uint limb = 0; limb < num_ops; ++limb) {
                ulong x = wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (base == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 4u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 8u;

        constexpr uint num_ops = 22u;

        constexpr uint num_addends = 6u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // EqualityGate: three routed words per operation (x, y, equal)
            // followed by three unrouted temporaries (diff, invdiff, prod).
            // The addend slot carries the constants column holding the gate's
            // first constant, its "one" value.
            uint constant_column = num_addends;
            ulong const_0 = constants[(ulong)constant_column * STATIC_LDE_ROWS + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 3u;
                ulong x = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong y = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong equal = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong temporary_base = (ulong)num_ops * 3u + (ulong)op * 3u;
                ulong difference = wires[(temporary_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong inverse = wires[(temporary_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong product = wires[(temporary_base + 2u) * STATIC_LDE_ROWS + source_row];

                range_check_gate_emit(
                    gl_sub(gl_sub(x, y), difference),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(difference, inverse), product),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(product, difference), difference),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(const_0, product), equal),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 5u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 4u;

        constexpr uint num_ops = 5u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            ulong strict_accumulators[2] = { 0, 0 };
            // QuinticMultiplicationGate: fifteen routed words per operation
            // (five limbs each for a, b and the claimed product c). The five
            // constraints are the schoolbook product limbs reduced by
            // u^5 = 3, minus the claimed output limbs, in ascending limb
            // order exactly like the CPU accumulator.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 15u;
                ulong a[5];
                ulong b[5];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * STATIC_LDE_ROWS + source_row];
                    b[j] = wires[(routed_base + 5u + j) * STATIC_LDE_ROWS + source_row];
                }
                ulong d[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                for (uint j = 0; j < 5u; ++j) {
                    for (uint k = 0; k < 5u; ++k) {
                        d[j + k] = gl_add(d[j + k], gl_mul(a[j], b[k]));
                    }
                }
                for (uint k = 0; k < 5u; ++k) {
                    ulong term = k < 4u
                        ? gl_add(d[k], gl_mul(3, d[k + 5u]))
                        : d[k];
                    ulong c = wires[(routed_base + 10u + k) * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit_strict(
                        gl_sub(term, c),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        strict_accumulators,
                        constraint_index++);
                }
            }
            gate_accumulators[0] = alpha_acc_of(strict_accumulators[0]);
            gate_accumulators[1] = alpha_acc_of(strict_accumulators[1]);
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 0u;
        constexpr uint gate_index = 6u;
        constexpr uint group_start = 0u;
        constexpr uint group_end = 7u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 5u;

        constexpr uint num_ops = 6u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            ulong strict_accumulators[2] = { 0, 0 };
            // QuinticSquaringGate: ten routed words per operation (input
            // limbs a then output limbs c) plus ten temporary wires. Each
            // constraint checks one accumulation step of the squaring
            // against its temporary or output, in the exact CPU emission
            // order.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 10u;
                ulong temp_base = (ulong)num_ops * 10u + (ulong)op * 10u;
                ulong a[5];
                ulong c[5];
                ulong extra[10];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * STATIC_LDE_ROWS + source_row];
                    c[j] = wires[(routed_base + 5u + j) * STATIC_LDE_ROWS + source_row];
                }
                for (uint j = 0; j < 10u; ++j) {
                    extra[j] = wires[(temp_base + j) * STATIC_LDE_ROWS + source_row];
                }

                // c[0]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[0], a[0]), extra[0]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[1]), a[4]), extra[0]), extra[1]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[3]), extra[1]), c[0]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);

                // c[1]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(gl_mul(3, a[3]), a[3]), extra[2]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[1]), extra[2]), extra[3]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[4]), extra[3]), c[1]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);

                // c[2]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[1], a[1]), extra[4]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[2]), extra[4]), extra[5]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[3]), a[4]), extra[5]), c[2]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);

                // c[3]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(gl_mul(3, a[4]), a[4]), extra[6]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[3]), extra[6]), extra[7]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[2]), extra[7]), c[3]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);

                // c[4]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[2], a[2]), extra[8]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[4]), extra[8]), extra[9]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[3]), extra[9]), c[4]),
                    alpha_powers, STATIC_ALPHA_STRIDE, strict_accumulators,
                    constraint_index++);
            }
            gate_accumulators[0] = alpha_acc_of(strict_accumulators[0]);
            gate_accumulators[1] = alpha_acc_of(strict_accumulators[1]);
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 1u;
        constexpr uint gate_index = 7u;
        constexpr uint group_start = 7u;
        constexpr uint group_end = 12u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 12u;

        constexpr uint num_ops = 20u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // SelectionGate: four routed wires per operation followed by one
            // temporary wire per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong b = wires[((ulong)(4u * op)) * STATIC_LDE_ROWS + source_row];
                ulong x = wires[((ulong)(4u * op + 1u)) * STATIC_LDE_ROWS + source_row];
                ulong y = wires[((ulong)(4u * op + 2u)) * STATIC_LDE_ROWS + source_row];
                ulong result = wires[((ulong)(4u * op + 3u)) * STATIC_LDE_ROWS + source_row];
                ulong temp = wires[
                    ((ulong)(4u * num_ops + op)) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, y), y), temp),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, x), temp), result),
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 1u;
        constexpr uint gate_index = 10u;
        constexpr uint group_start = 7u;
        constexpr uint group_end = 12u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 11u;

        constexpr uint num_ops = 16u;

        constexpr uint num_addends = 4u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // BaseSumGate: wire 0 is the sum and the next `num_ops` wires are
            // little-endian limbs. The addend-count slot carries base 2 or 4.
            //
            // The Horner step's `gl_mul(computed, base)` looks like free money
            // -- both bases are powers of two, so doubling or `gl_quadruple`
            // would replace a 128-bit product with one or two field adds, 63
            // of them per row on the widest family. Measured, it is not:
            // specializing the base outside the loop costs more in code
            // duplication than the arithmetic saves. Recomposition-only,
            // against the deferred-accumulator kernel: d18 160.9 -> 161.6 ms,
            // d16-heavy 60.40 -> 60.53 ms, d14 4.135 -> 4.112 ms; splitting the
            // range-constraint loop as well costs another 1.5 ms on d18. Both
            // arms bit-exact, so this is a scheduling/footprint effect, not an
            // arithmetic one. Keep the multiply.
            ulong base = num_addends;
            ulong computed = 0;
            for (uint remaining = num_ops; remaining > 0u; --remaining) {
                uint limb = remaining - 1u;
                computed = gl_add(
                    gl_mul(computed, base),
                    wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row]);
            }
            range_check_gate_emit(
                gl_sub(computed, wires[source_row]),
                alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                constraint_index++);
            for (uint limb = 0; limb < num_ops; ++limb) {
                ulong x = wires[((ulong)1u + limb) * STATIC_LDE_ROWS + source_row];
                ulong constraint;
                if (base == 2u) {
                    constraint = gl_mul(x, gl_sub(x, 1));
                } else {
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    constraint = gl_mul(y, gl_add(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers, STATIC_ALPHA_STRIDE, gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 1u;
        constexpr uint gate_index = 11u;
        constexpr uint group_start = 7u;
        constexpr uint group_end = 12u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 3u;

        constexpr uint num_ops = 3u;

        constexpr uint num_addends = 8u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // ByteDecompositionGate: per operation, one routed sum wire and
            // `num_addends` routed byte wires (the metadata word carries the
            // byte count for this kind), then four base-4 aux limbs per
            // byte. Constraint order matches the CPU gate exactly: aux range
            // products in ascending wire order, one base-4 recomposition per
            // byte, then the base-256 byte-to-sum recomposition.
            uint num_limbs = num_addends;
            uint routed_per_op = 1u + num_limbs;
            uint aux_per_op = 4u * num_limbs;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong aux_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * aux_per_op;
                for (uint j = 0; j < aux_per_op; ++j) {
                    ulong x = wires[(aux_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }
                for (uint byte_index = 0; byte_index < num_limbs; ++byte_index) {
                    ulong chunk = aux_base + (ulong)byte_index * 4u;
                    ulong recomposed = wires[(chunk + 3u) * STATIC_LDE_ROWS + source_row];
                    for (uint remaining = 3u; remaining > 0u; --remaining) {
                        uint k = remaining - 1u;
                        recomposed = gl_add(
                            gl_quadruple(recomposed),
                            wires[(chunk + k) * STATIC_LDE_ROWS + source_row]);
                    }
                    ulong byte_value =
                        wires[(routed_base + 1u + byte_index) * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit(
                        gl_sub(recomposed, byte_value),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }
                ulong recomposed_sum =
                    wires[(routed_base + num_limbs) * STATIC_LDE_ROWS + source_row];
                for (uint remaining = num_limbs - 1u; remaining > 0u; --remaining) {
                    uint k = remaining - 1u;
                    recomposed_sum = gl_add(
                        gl_mul(recomposed_sum, 256),
                        wires[(routed_base + 1u + k) * STATIC_LDE_ROWS + source_row]);
                }
                ulong expected_sum = wires[routed_base * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(recomposed_sum, expected_sum),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 2u;
        constexpr uint gate_index = 13u;
        constexpr uint group_start = 12u;
        constexpr uint group_end = 17u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 6u;

        constexpr uint num_ops = 8u;

        constexpr uint num_addends = 3u;

        constexpr uint result_limbs = 0u;

        constexpr uint num_carry_limbs = 6u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            uint bits = num_addends;
            uint num_extra_constants = result_limbs;
            uint constant_base = num_carry_limbs;
            uint num_copies = num_ops;
            ulong vec_size = 1UL << bits;
            ulong routed_per_copy = vec_size + 2u;
            ulong extra_wire_base = routed_per_copy * num_copies;
            ulong bit_base = extra_wire_base + num_extra_constants;

            for (uint copy = 0; copy < num_copies; ++copy) {
                ulong copy_base = routed_per_copy * copy;

                // RandomAccessGate emits boolean constraints for b_0 upward.
                for (uint i = 0; i < bits; ++i) {
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit(
                        gl_mul(b, gl_sub(b, 1)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }

                // Reconstruct the little-endian index in the CPU's exact
                // reverse-bit `acc.double() + b` order.
                ulong reconstructed_index = 0;
                for (uint remaining = bits; remaining > 0u; --remaining) {
                    uint i = remaining - 1u;
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    reconstructed_index = gl_add(
                        gl_add(reconstructed_index, reconstructed_index), b);
                }
                ulong access_index = wires[copy_base * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(reconstructed_index, access_index),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                // Fold each eight-item block in ascending pair order, then fold
                // block results with the remaining bits in the same order.
                ulong block_results[8];
                uint block_count = (uint)(vec_size / 8u);
                ulong list_base = copy_base + 2u;
                ulong copy_bit_base = bit_base + (ulong)copy * bits;
                for (uint block = 0; block < block_count; ++block) {
                    block_results[block] = random_access_select_8(
                        wires, STATIC_LDE_ROWS, source_row, list_base, copy_bit_base, block);
                }
                uint level_size = block_count;
                for (uint i = 3u; i < bits; ++i) {
                    ulong b = wires[(copy_bit_base + i) * STATIC_LDE_ROWS + source_row];
                    for (uint k = 0; k < level_size / 2u; ++k) {
                        ulong x = block_results[2u * k];
                        ulong y = block_results[2u * k + 1u];
                        block_results[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
                    }
                    level_size /= 2u;
                }
                ulong claimed_element = wires[(copy_base + 1u) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(block_results[0], claimed_element),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }

            // Raw local constants follow all gate and lookup selectors.
            for (uint i = 0; i < num_extra_constants; ++i) {
                ulong local_constant = constants[
                    ((ulong)constant_base + i) * STATIC_LDE_ROWS + source_row];
                ulong extra_wire = wires[
                    (extra_wire_base + i) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(local_constant, extra_wire),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 3u;
        constexpr uint gate_index = 17u;
        constexpr uint group_start = 17u;
        constexpr uint group_end = 22u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 1u;

        constexpr uint num_ops = 10u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 8u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // U16/U32/U48 SubtractionGate: five routed words followed by
            // `result_limbs` base-4 result limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 5u;
                ulong input_x = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong input_y = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong input_borrow = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong output_result = wires[(routed_base + 3u) * STATIC_LDE_ROWS + source_row];
                ulong output_borrow = wires[(routed_base + 4u) * STATIC_LDE_ROWS + source_row];
                ulong result_initial = gl_sub(gl_sub(input_x, input_y), input_borrow);
                ulong borrowed = gl_add(
                    result_initial,
                    gl_mul(word_base, output_borrow));
                range_check_gate_emit(
                    gl_sub(output_result, borrowed),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 5u + (ulong)op * result_limbs;
                ulong recomposed = 0;
                for (uint remaining = result_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                    recomposed = gl_add(gl_quadruple(recomposed), x);
                }
                range_check_gate_emit(
                    gl_sub(recomposed, output_result),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_mul(output_borrow, gl_sub(1, output_borrow)),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 3u;
        constexpr uint gate_index = 18u;
        constexpr uint group_start = 17u;
        constexpr uint group_end = 22u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 2u;

        constexpr uint num_ops = 5u;

        constexpr uint num_addends = 6u;

        constexpr uint result_limbs = 16u;

        constexpr uint num_carry_limbs = 2u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // U16/U32 AddManyGate: num_addends inputs, carry/result/output-carry,
            // then `result_limbs` result and `num_carry_limbs` carry base-4
            // limbs per operation.
            uint routed_per_op = num_addends + 3u;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong computed = wires[(routed_base + num_addends) * STATIC_LDE_ROWS + source_row];
                for (uint j = 0; j < num_addends; ++j) {
                    computed = gl_add(
                        computed,
                        wires[(routed_base + j) * STATIC_LDE_ROWS + source_row]);
                }
                ulong output_result =
                    wires[(routed_base + num_addends + 1u) * STATIC_LDE_ROWS + source_row];
                ulong output_carry =
                    wires[(routed_base + num_addends + 2u) * STATIC_LDE_ROWS + source_row];
                ulong combined = gl_add(gl_mul(output_carry, word_base), output_result);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                uint total_limbs = result_limbs + num_carry_limbs;
                ulong limb_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * total_limbs;
                ulong combined_result = 0;
                ulong combined_carry = 0;
                for (uint remaining = total_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                    if (j < result_limbs) {
                        combined_result = gl_add(gl_quadruple(combined_result), x);
                    } else {
                        combined_carry = gl_add(gl_quadruple(combined_carry), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_result, output_result),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_carry, output_carry),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 3u;
        constexpr uint gate_index = 19u;
        constexpr uint group_start = 17u;
        constexpr uint group_end = 22u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 0u;

        constexpr uint num_ops = 3u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 16u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // U32ArithmeticGate: six routed words followed by 32 base-4
            // output limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 6u;
                ulong multiplicand_0 = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong multiplicand_1 = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong addend = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong output_low = wires[(routed_base + 3u) * STATIC_LDE_ROWS + source_row];
                ulong output_high = wires[(routed_base + 4u) * STATIC_LDE_ROWS + source_row];
                ulong inverse = wires[(routed_base + 5u) * STATIC_LDE_ROWS + source_row];

                ulong high_diff = gl_sub(0xffffffffUL, output_high);
                ulong high_not_max = gl_sub(gl_mul(inverse, high_diff), 1);
                range_check_gate_emit(
                    gl_mul(high_not_max, output_low),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                ulong computed = gl_add(gl_mul(multiplicand_0, multiplicand_1), addend);
                ulong combined = gl_add(gl_mul(output_high, 4294967296UL), output_low);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 6u + (ulong)op * 32u;
                ulong combined_low = 0;
                ulong combined_high = 0;
                for (uint remaining = 32u; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                    if (j < 16u) {
                        combined_low = gl_add(gl_quadruple(combined_low), x);
                    } else {
                        combined_high = gl_add(gl_quadruple(combined_high), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_low, output_low),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_high, output_high),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 3u;
        constexpr uint gate_index = 20u;
        constexpr uint group_start = 17u;
        constexpr uint group_end = 22u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 1u;

        constexpr uint num_ops = 6u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 16u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // U16/U32/U48 SubtractionGate: five routed words followed by
            // `result_limbs` base-4 result limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 5u;
                ulong input_x = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong input_y = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong input_borrow = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong output_result = wires[(routed_base + 3u) * STATIC_LDE_ROWS + source_row];
                ulong output_borrow = wires[(routed_base + 4u) * STATIC_LDE_ROWS + source_row];
                ulong result_initial = gl_sub(gl_sub(input_x, input_y), input_borrow);
                ulong borrowed = gl_add(
                    result_initial,
                    gl_mul(word_base, output_borrow));
                range_check_gate_emit(
                    gl_sub(output_result, borrowed),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 5u + (ulong)op * result_limbs;
                ulong recomposed = 0;
                for (uint remaining = result_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                    recomposed = gl_add(gl_quadruple(recomposed), x);
                }
                range_check_gate_emit(
                    gl_sub(recomposed, output_result),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_mul(output_borrow, gl_sub(1, output_borrow)),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 3u;
        constexpr uint gate_index = 21u;
        constexpr uint group_start = 17u;
        constexpr uint group_end = 22u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 1u;

        constexpr uint num_ops = 4u;

        constexpr uint num_addends = 0u;

        constexpr uint result_limbs = 24u;

        constexpr uint num_carry_limbs = 0u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            // U16/U32/U48 SubtractionGate: five routed words followed by
            // `result_limbs` base-4 result limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 5u;
                ulong input_x = wires[(routed_base + 0u) * STATIC_LDE_ROWS + source_row];
                ulong input_y = wires[(routed_base + 1u) * STATIC_LDE_ROWS + source_row];
                ulong input_borrow = wires[(routed_base + 2u) * STATIC_LDE_ROWS + source_row];
                ulong output_result = wires[(routed_base + 3u) * STATIC_LDE_ROWS + source_row];
                ulong output_borrow = wires[(routed_base + 4u) * STATIC_LDE_ROWS + source_row];
                ulong result_initial = gl_sub(gl_sub(input_x, input_y), input_borrow);
                ulong borrowed = gl_add(
                    result_initial,
                    gl_mul(word_base, output_borrow));
                range_check_gate_emit(
                    gl_sub(output_result, borrowed),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 5u + (ulong)op * result_limbs;
                ulong recomposed = 0;
                for (uint remaining = result_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * STATIC_LDE_ROWS + source_row];
                    ulong y = gl_mul(x, gl_sub(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add(y, 2)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                    recomposed = gl_add(gl_quadruple(recomposed), x);
                }
                range_check_gate_emit(
                    gl_sub(recomposed, output_result),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_mul(output_borrow, gl_sub(1, output_borrow)),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }
    {
        constexpr uint selector_column = 4u;
        constexpr uint gate_index = 22u;
        constexpr uint group_start = 22u;
        constexpr uint group_end = 24u;
        constexpr uint include_unused_selector = 1u;

        constexpr uint kind = 6u;

        constexpr uint num_ops = 4u;

        constexpr uint num_addends = 4u;

        constexpr uint result_limbs = 2u;

        constexpr uint num_carry_limbs = 6u;

        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * STATIC_LDE_ROWS + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
            uint bits = num_addends;
            uint num_extra_constants = result_limbs;
            uint constant_base = num_carry_limbs;
            uint num_copies = num_ops;
            ulong vec_size = 1UL << bits;
            ulong routed_per_copy = vec_size + 2u;
            ulong extra_wire_base = routed_per_copy * num_copies;
            ulong bit_base = extra_wire_base + num_extra_constants;

            for (uint copy = 0; copy < num_copies; ++copy) {
                ulong copy_base = routed_per_copy * copy;

                // RandomAccessGate emits boolean constraints for b_0 upward.
                for (uint i = 0; i < bits; ++i) {
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    range_check_gate_emit(
                        gl_mul(b, gl_sub(b, 1)),
                        alpha_powers,
                        STATIC_ALPHA_STRIDE,
                        gate_accumulators,
                        constraint_index++);
                }

                // Reconstruct the little-endian index in the CPU's exact
                // reverse-bit `acc.double() + b` order.
                ulong reconstructed_index = 0;
                for (uint remaining = bits; remaining > 0u; --remaining) {
                    uint i = remaining - 1u;
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * STATIC_LDE_ROWS + source_row];
                    reconstructed_index = gl_add(
                        gl_add(reconstructed_index, reconstructed_index), b);
                }
                ulong access_index = wires[copy_base * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(reconstructed_index, access_index),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);

                // Fold each eight-item block in ascending pair order, then fold
                // block results with the remaining bits in the same order.
                ulong block_results[8];
                uint block_count = (uint)(vec_size / 8u);
                ulong list_base = copy_base + 2u;
                ulong copy_bit_base = bit_base + (ulong)copy * bits;
                for (uint block = 0; block < block_count; ++block) {
                    block_results[block] = random_access_select_8(
                        wires, STATIC_LDE_ROWS, source_row, list_base, copy_bit_base, block);
                }
                uint level_size = block_count;
                for (uint i = 3u; i < bits; ++i) {
                    ulong b = wires[(copy_bit_base + i) * STATIC_LDE_ROWS + source_row];
                    for (uint k = 0; k < level_size / 2u; ++k) {
                        ulong x = block_results[2u * k];
                        ulong y = block_results[2u * k + 1u];
                        block_results[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
                    }
                    level_size /= 2u;
                }
                ulong claimed_element = wires[(copy_base + 1u) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(block_results[0], claimed_element),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }

            // Raw local constants follow all gate and lookup selectors.
            for (uint i = 0; i < num_extra_constants; ++i) {
                ulong local_constant = constants[
                    ((ulong)constant_base + i) * STATIC_LDE_ROWS + source_row];
                ulong extra_wire = wires[
                    (extra_wire_base + i) * STATIC_LDE_ROWS + source_row];
                range_check_gate_emit(
                    gl_sub(local_constant, extra_wire),
                    alpha_powers,
                    STATIC_ALPHA_STRIDE,
                    gate_accumulators,
                    constraint_index++);
            }
        

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }

    output[(ulong)gid * 2] = gl_canonicalize(total[0]);
    output[(ulong)gid * 2 + 1] = gl_canonicalize(total[1]);
}
