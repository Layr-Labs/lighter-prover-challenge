/*
 * GPU kernel for quotient polynomial evaluation — FUSED variant.
 * One thread per LDE evaluation point.
 * Each thread: gate constraints + permutation + alpha reduction + Z_H division.
 *
 * KEY OPTIMIZATION: No work_mem buffer. Only one gate has a non-zero selector
 * filter at each evaluation point, so we fuse gate evaluation with alpha
 * reduction using register-resident accumulators. This eliminates the 1.3GB
 * device memory buffer that caused terrible coalescing and 49% GPU slowdown.
 *
 * Register budget: ~24 Fp values = 192 bytes (well within GPU registers).
 */
#include <metal_stdlib>
using namespace metal;

#include "goldilocks.metal"
#include "goldilocks_ext2.metal"
#include "quotient_gates.metal"

using namespace GoldilocksField;

#define MAX_CHALLENGES 4

struct QuotientUniformsDevice {
    uint lde_size;
    uint degree_bits;
    uint quotient_degree_bits;
    uint num_wires;
    uint num_constants;
    uint num_routed_wires;
    uint num_challenges;
    uint num_partial_products;
    uint num_gate_constraints;
    uint max_degree;
    uint num_gates;
    uint num_selectors;
    uint num_lookup_selectors;
    uint zs_stride;
    uint wires_stride;
    uint cs_stride;
};

kernel void compute_quotient_poly(
    device const ulong* flat_wires       [[buffer(0)]],
    device const ulong* flat_cs          [[buffer(1)]],
    device const ulong* flat_zs          [[buffer(2)]],
    device const ulong* flat_zs_next     [[buffer(3)]],
    device const GateDescriptor* gates   [[buffer(4)]],
    device const ulong* k_is             [[buffer(5)]],
    device const ulong* betas            [[buffer(6)]],
    device const ulong* gammas           [[buffer(7)]],
    device const ulong* alphas           [[buffer(8)]],
    device const ulong* z_h_inverses     [[buffer(9)]],
    device const ulong* l_0_values       [[buffer(10)]],
    device const ulong* points           [[buffer(11)]],
    device const ulong* pub_inputs_hash  [[buffer(12)]],
    device const ulong* coset_interp_data [[buffer(13)]],
    device const QuotientUniformsDevice* u_ptr [[buffer(14)]],
    device ulong* output                 [[buffer(15)]],
    device ulong* debug_out              [[buffer(16)]],
    device const uint* thread_offset_ptr [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    QuotientUniformsDevice u = *u_ptr;
    uint thread_offset = *thread_offset_ptr;
    uint real_gid = gid + thread_offset;
    if (real_gid >= u.lde_size) return;

    // Wire/constant read helpers
    uint w_off = real_gid * u.wires_stride;
    uint cs_off = real_gid * u.cs_stride;
    uint zs_off = real_gid * u.zs_stride;
    uint zs_next_off = real_gid * u.zs_stride;
    uint const_prefix = u.num_selectors + u.num_lookup_selectors;

    #define W(i) Fp(flat_wires[w_off + (i)])
    #define CS(i) Fp(flat_cs[cs_off + const_prefix + (i)])

    uint num_ch = u.num_challenges;

    //=================================================================
    // Register-resident accumulators for fused alpha reduction.
    // cumul[ch] accumulates: sum( vt[t] * alpha[ch]^t )
    // alpha_pow[ch] tracks current power of alpha for each challenge.
    //=================================================================
    Fp cumul[MAX_CHALLENGES];
    Fp alpha_pow[MAX_CHALLENGES];
    Fp alpha_val[MAX_CHALLENGES];
    for (uint ch = 0; ch < num_ch; ch++) {
        cumul[ch] = Fp(0);
        alpha_pow[ch] = Fp(1);
        alpha_val[ch] = Fp(alphas[ch]);
    }

    // Inline macro: accumulate one vanishing term into all challenge accumulators
    #define ACCUM_VT(term) do { \
        Fp _t = (term); \
        for (uint _ch = 0; _ch < num_ch; _ch++) { \
            cumul[_ch] = cumul[_ch] + _t * alpha_pow[_ch]; \
            alpha_pow[_ch] = alpha_pow[_ch] * alpha_val[_ch]; \
        } \
    } while(0)

    // Skip alpha power without contributing a value (for zero-valued terms)
    #define SKIP_VT() do { \
        for (uint _ch = 0; _ch < num_ch; _ch++) { \
            alpha_pow[_ch] = alpha_pow[_ch] * alpha_val[_ch]; \
        } \
    } while(0)

    //=================================================================
    // Phase 1: L_0(x) * (Z(x) - 1) — one term per challenge
    //=================================================================
    Fp l_0_x = Fp(l_0_values[real_gid]);
    Fp x = Fp(points[real_gid]);

    for (uint ch_i = 0; ch_i < num_ch; ch_i++) {
        Fp z_x = Fp(flat_zs[zs_off + ch_i]);
        ACCUM_VT(l_0_x * fp_sub(z_x, Fp(1)));
    }

    //=================================================================
    // Phase 2: Partial product checks
    //=================================================================
    uint num_prods = u.num_partial_products;
    for (uint ch_i = 0; ch_i < num_ch; ch_i++) {
        Fp z_x = Fp(flat_zs[zs_off + ch_i]);
        Fp z_gx = Fp(flat_zs_next[zs_next_off + ch_i]);
        Fp beta = Fp(betas[ch_i]);
        Fp gamma = Fp(gammas[ch_i]);

        uint num_routed = u.num_routed_wires;
        uint chunk_size = u.max_degree;
        uint num_chunks = (num_routed + chunk_size - 1) / chunk_size;

        for (uint chunk = 0; chunk < num_chunks; chunk++) {
            uint start = chunk * chunk_size;
            uint end = min(start + chunk_size, num_routed);

            Fp num_prod = Fp(1);
            Fp den_prod = Fp(1);
            for (uint j = start; j < end; j++) {
                Fp wire_val = Fp(flat_wires[w_off + j]);
                Fp s_id = Fp(k_is[j]) * x;
                Fp s_sigma = Fp(flat_cs[cs_off + u.num_constants + j]);
                num_prod = num_prod * (wire_val + beta * s_id + gamma);
                den_prod = den_prod * (wire_val + beta * s_sigma + gamma);
            }

            Fp prev_acc, next_acc;
            if (chunk == 0) {
                prev_acc = z_x;
            } else {
                prev_acc = Fp(flat_zs[zs_off + num_ch + ch_i * num_prods + (chunk - 1)]);
            }
            if (chunk == num_chunks - 1) {
                next_acc = z_gx;
            } else {
                next_acc = Fp(flat_zs[zs_off + num_ch + ch_i * num_prods + chunk]);
            }

            ACCUM_VT(fp_sub(prev_acc * num_prod, next_acc * den_prod));
        }
    }

    //=================================================================
    // Phase 3: Gate constraints with correct alpha reduction.
    //
    // IMPORTANT: On the LDE coset, multiple gates can have non-zero
    // filters simultaneously. The CPU sums all filtered constraints
    // at the SAME position j before alpha-reducing:
    //   GC[j] = sum_g filter_g * eval_g(j)
    //   result += sum_j alpha^j * GC[j]
    //
    // This equals: result += sum_g filter_g * reduce_with_powers(eval_g, alpha)
    //
    // So each gate gets INDEPENDENT alpha powers starting from alpha^0,
    // and their contributions are summed into gc_sum[ch].
    // Finally: cumul[ch] += alpha_base[ch] * gc_sum[ch].
    //=================================================================
    Fp alpha_base[MAX_CHALLENGES];
    Fp gc_sum[MAX_CHALLENGES];
    for (uint ch = 0; ch < num_ch; ch++) {
        alpha_base[ch] = alpha_pow[ch];
        gc_sum[ch] = Fp(0);
    }

    uint dbg_off = 0; // debug write offset for thread 0
    // DEBUG: limit to first N gates for bisection (set to u.num_gates for all)
    uint max_gates_debug = u.num_gates;
    for (uint g = 0; g < max_gates_debug; g++) {
        GateDescriptor gate = gates[g];
        if (gate.gate_type == GATE_NOOP) continue;

        Fp selector_val = Fp(flat_cs[cs_off + gate.selector_index]);
        bool many_selectors = u.num_selectors > 1;
        Fp filter = compute_filter(gate.row, gate.group_start, gate.group_end,
                                    selector_val, many_selectors);

        // Skip gates with zero filter (optimization, not correctness-critical).
        if ((ulong)filter == 0) continue;

        // Save gc_sum before this gate for per-gate debug
        Fp gc_sum_before[MAX_CHALLENGES];
        for (uint ch = 0; ch < num_ch; ch++) {
            gc_sum_before[ch] = gc_sum[ch];
        }

        // Per-gate alpha powers: reset to alpha^0 for each gate.
        // This ensures each gate's constraints get the same alpha powers
        // as on the CPU (position j gets alpha^j regardless of gate).
        Fp ga_pow[MAX_CHALLENGES];
        for (uint ch = 0; ch < num_ch; ch++) {
            ga_pow[ch] = Fp(1);
        }

        // Macro: accumulate one gate constraint into gc_sum with per-gate alpha powers.
        #define ACCUM_GC(term) do { \
            Fp _t = (term); \
            for (uint _ch = 0; _ch < num_ch; _ch++) { \
                gc_sum[_ch] = gc_sum[_ch] + _t * ga_pow[_ch]; \
                ga_pow[_ch] = ga_pow[_ch] * alpha_val[_ch]; \
            } \
        } while(0)

        switch (gate.gate_type) {
            case GATE_ARITHMETIC: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                Fp c1 = CS(1);
                for (uint i = 0; i < num_ops; i++) {
                    Fp m0 = W(4*i);
                    Fp m1 = W(4*i + 1);
                    Fp addend = W(4*i + 2);
                    Fp out = W(4*i + 3);
                    ACCUM_GC(filter * fp_sub(out, m0 * m1 * c0 + addend * c1));

                }
                break;
            }
            case GATE_ARITHMETIC_EXTENSION: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                Fp c1 = CS(1);
                for (uint i = 0; i < num_ops; i++) {
                    uint base = 4 * D * i;
                    FpE m0 = FpE(W(base), W(base + 1));
                    FpE m1 = FpE(W(base + D), W(base + D + 1));
                    FpE addend = FpE(W(base + 2*D), W(base + 2*D + 1));
                    FpE out = FpE(W(base + 3*D), W(base + 3*D + 1));
                    FpE diff = out - ((m0 * m1).scalar_mul(c0) + addend.scalar_mul(c1));
                    ACCUM_GC(filter * diff.c0);

                    ACCUM_GC(filter * diff.c1);

                }
                break;
            }
            case GATE_MUL_EXTENSION: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                for (uint i = 0; i < num_ops; i++) {
                    uint base = 3 * D * i;
                    FpE m0 = FpE(W(base), W(base + 1));
                    FpE m1 = FpE(W(base + D), W(base + D + 1));
                    FpE out = FpE(W(base + 2*D), W(base + 2*D + 1));
                    FpE diff = out - (m0 * m1).scalar_mul(c0);
                    ACCUM_GC(filter * diff.c0);

                    ACCUM_GC(filter * diff.c1);

                }
                break;
            }
            case GATE_CONSTANT: {
                uint num_consts = gate.param0;
                for (uint i = 0; i < num_consts; i++) {
                    ACCUM_GC(filter * fp_sub(CS(i), W(i)));

                }
                break;
            }
            case GATE_PUBLIC_INPUT: {
                for (uint i = 0; i < 4; i++) {
                    ACCUM_GC(filter * fp_sub(W(i), Fp(pub_inputs_hash[i])));

                }
                break;
            }
            case GATE_BASE_SUM: {
                uint num_limbs = gate.param0;
                uint base_B = gate.param1;
                Fp sum_wire = W(0);
                Fp acc = Fp(0);
                Fp base_fp = Fp(base_B);
                for (int i = num_limbs - 1; i >= 0; i--) {
                    acc = acc * base_fp + W(1 + i);
                }
                ACCUM_GC(filter * fp_sub(acc, sum_wire));

                for (uint i = 0; i < num_limbs; i++) {
                    Fp limb = W(1 + i);
                    Fp prod = Fp(1);
                    for (uint j = 0; j < base_B; j++) {
                        prod = prod * fp_sub(limb, Fp(j));
                    }
                    ACCUM_GC(filter * prod);

                }
                break;
            }
            case GATE_REDUCING: {
                uint num_coeffs = gate.param0;
                FpE out = FpE(W(0), W(1));
                FpE alpha_r = FpE(W(D), W(D + 1));
                FpE old_acc = FpE(W(2*D), W(2*D + 1));
                uint start_coeffs = 3 * D;
                uint start_accs = start_coeffs + num_coeffs;
                FpE racc = old_acc;
                for (uint i = 0; i < num_coeffs; i++) {
                    Fp coeff = W(start_coeffs + i);
                    FpE acc_new = racc * alpha_r + FpE(coeff);
                    FpE target;
                    if (i < num_coeffs - 1) {
                        uint ai = start_accs + D * i;
                        target = FpE(W(ai), W(ai + 1));
                    } else {
                        target = out;
                    }
                    FpE diff = acc_new - target;
                    ACCUM_GC(filter * diff.c0);

                    ACCUM_GC(filter * diff.c1);

                    racc = target;
                }
                break;
            }
            case GATE_REDUCING_EXTENSION: {
                uint num_coeffs = gate.param0;
                FpE out = FpE(W(0), W(1));
                FpE alpha_r = FpE(W(D), W(D + 1));
                FpE old_acc = FpE(W(2*D), W(2*D + 1));
                uint start_coeffs = 3 * D;
                uint start_accs = start_coeffs + D * num_coeffs;
                FpE racc = old_acc;
                for (uint i = 0; i < num_coeffs; i++) {
                    uint cs = start_coeffs + D * i;
                    FpE coeff = FpE(W(cs), W(cs + 1));
                    FpE acc_new = racc * alpha_r + coeff;
                    FpE target;
                    if (i < num_coeffs - 1) {
                        uint ai = start_accs + D * i;
                        target = FpE(W(ai), W(ai + 1));
                    } else {
                        target = out;
                    }
                    FpE diff = acc_new - target;
                    ACCUM_GC(filter * diff.c0);

                    ACCUM_GC(filter * diff.c1);

                    racc = target;
                }
                break;
            }
            case GATE_RANDOM_ACCESS: {
                uint bits = gate.param0;
                uint num_copies = gate.param1;
                uint num_extra_constants = gate.param2;
                uint num_rw = gate.param3;
                uint vec_size = 1u << bits;
                for (uint copy = 0; copy < num_copies; copy++) {
                    uint base_routed = (2 + vec_size) * copy;
                    Fp access_index = W(base_routed);
                    Fp claimed_element = W(base_routed + 1);
                    uint bit_base = num_rw + copy * bits;
                    // Boolean constraints (forward order)
                    for (uint i = 0; i < bits; i++) {
                        Fp b = W(bit_base + i);
                        ACCUM_GC(filter * (b * fp_sub(b, Fp(1))));
    
                    }
                    // Reconstruction: REVERSE order
                    Fp reconstructed = Fp(0);
                    for (int i = bits - 1; i >= 0; i--) {
                        reconstructed = reconstructed + reconstructed + W(bit_base + (uint)i);
                    }
                    ACCUM_GC(filter * fp_sub(reconstructed, access_index));

                    // Binary fold: FORWARD order
                    Fp fold[64];
                    for (uint j = 0; j < vec_size && j < 64; j++) {
                        fold[j] = W(base_routed + 2 + j);
                    }
                    uint cur_size = vec_size;
                    for (uint i = 0; i < bits; i++) {
                        Fp b = W(bit_base + i);
                        uint new_size = cur_size / 2;
                        for (uint j = 0; j < new_size; j++) {
                            fold[j] = fold[2*j] + b * fp_sub(fold[2*j + 1], fold[2*j]);
                        }
                        cur_size = new_size;
                    }
                    ACCUM_GC(filter * fp_sub(fold[0], claimed_element));

                }
                // Extra constant constraints
                for (uint i = 0; i < num_extra_constants; i++) {
                    Fp constant_val = CS(i);
                    uint wire_idx = (2 + vec_size) * num_copies + i;
                    ACCUM_GC(filter * fp_sub(constant_val, W(wire_idx)));

                }
                break;
            }
            case GATE_COSET_INTERPOLATION: {
                uint subgroup_bits = gate.param0;
                uint degree_param = gate.param1;
                uint num_points = 1u << subgroup_bits;
                uint num_intermediates = (num_points - 2) / (degree_param - 1);
                uint data_off = gate.param2;

                uint val_start = 1;
                uint eval_pt_start = val_start + num_points * D;
                uint eval_val_start = eval_pt_start + D;
                uint inter_start = eval_val_start + D;

                FpE eval_point = FpE(W(eval_pt_start), W(eval_pt_start + 1));
                uint shifted_pt_start = inter_start + D * 2 * num_intermediates;
                FpE shifted_pt = FpE(W(shifted_pt_start), W(shifted_pt_start + 1));
                Fp shift = W(0);

                FpE diff0 = eval_point - shifted_pt.scalar_mul(shift);
                ACCUM_GC(filter * diff0.c0);;
                ACCUM_GC(filter * diff0.c1);;

                FpE computed_eval = FpE::zero();
                FpE computed_prod = FpE::one();
                uint chunk_end = min(degree_param, num_points);
                for (uint j = 0; j < chunk_end; j++) {
                    Fp domain_pt = Fp(coset_interp_data[data_off + j]);
                    Fp weight = Fp(coset_interp_data[data_off + num_points + j]);
                    FpE val = FpE(W(val_start + j*D), W(val_start + j*D + 1));
                    FpE diff_x = shifted_pt - FpE(domain_pt);
                    computed_eval = computed_eval * diff_x + val.scalar_mul(weight) * computed_prod;
                    computed_prod = computed_prod * diff_x;
                }
                for (uint inter = 0; inter < num_intermediates; inter++) {
                    uint ie_start = inter_start + D * inter;
                    uint ip_start = inter_start + D * (num_intermediates + inter);
                    FpE inter_eval = FpE(W(ie_start), W(ie_start + 1));
                    FpE inter_prod = FpE(W(ip_start), W(ip_start + 1));
                    FpE de = inter_eval - computed_eval;
                    FpE dp = inter_prod - computed_prod;
                    ACCUM_GC(filter * de.c0);;
                    ACCUM_GC(filter * de.c1);;
                    ACCUM_GC(filter * dp.c0);;
                    ACCUM_GC(filter * dp.c1);;
                    uint start_index = 1 + (degree_param - 1) * (inter + 1);
                    uint end_index = min(start_index + degree_param - 1, num_points);
                    computed_eval = inter_eval;
                    computed_prod = inter_prod;
                    for (uint j = start_index; j < end_index; j++) {
                        Fp domain_pt = Fp(coset_interp_data[data_off + j]);
                        Fp weight = Fp(coset_interp_data[data_off + num_points + j]);
                        FpE val = FpE(W(val_start + j*D), W(val_start + j*D + 1));
                        FpE diff_x = shifted_pt - FpE(domain_pt);
                        computed_eval = computed_eval * diff_x + val.scalar_mul(weight) * computed_prod;
                        computed_prod = computed_prod * diff_x;
                    }
                }
                FpE eval_value = FpE(W(eval_val_start), W(eval_val_start + 1));
                FpE final_diff = eval_value - computed_eval;
                ACCUM_GC(filter * final_diff.c0);;
                ACCUM_GC(filter * final_diff.c1);;
                break;
            }
            case GATE_POSEIDON2: {
                // Small thread-local state (96 bytes) — within register budget
                Fp state[12];

                Fp swap = W(P2_WIRE_SWAP);
                ACCUM_GC(filter * (swap * fp_sub(swap, Fp(1))));


                for (uint i = 0; i < 4; i++) {
                    Fp input_lhs = W(i);
                    Fp input_rhs = W(i + 4);
                    Fp delta_i = W(P2_START_DELTA + i);
                    ACCUM_GC(filter * fp_sub(swap * fp_sub(input_rhs, input_lhs), delta_i));

                }

                for (uint i = 0; i < 4; i++) {
                    Fp delta_i = W(P2_START_DELTA + i);
                    state[i] = W(i) + delta_i;
                    state[i + 4] = fp_sub(W(i + 4), delta_i);
                }
                for (uint i = 8; i < 12; i++) state[i] = W(i);

                p2_external_linear_layer(state);

                for (uint r = 0; r < P2_ROUNDS_F_HALF; r++) {
                    for (uint i = 0; i < 12; i++) {
                        state[i] = state[i] + Fp(P2_EXTERNAL_RC[r][i]);
                    }
                    if (r != 0) {
                        for (uint i = 0; i < 12; i++) {
                            Fp sbox_in = W(P2_START_ROUND_F_BEGIN + 12*(r-1) + i);
                            ACCUM_GC(filter * fp_sub(state[i], sbox_in));
        
                            state[i] = sbox_in;
                        }
                    }
                    p2_sbox(state);
                    p2_external_linear_layer(state);
                }

                for (uint r = 0; r < P2_ROUNDS_P; r++) {
                    state[0] = state[0] + Fp(P2_INTERNAL_RC[r]);
                    Fp sbox_in = W(P2_START_PARTIAL + r);
                    ACCUM_GC(filter * fp_sub(state[0], sbox_in));

                    state[0] = p2_sbox_p(sbox_in);
                    p2_internal_linear_layer(state);
                }

                for (uint r = P2_ROUNDS_F_HALF; r < P2_ROUNDS_F; r++) {
                    for (uint i = 0; i < 12; i++) {
                        state[i] = state[i] + Fp(P2_EXTERNAL_RC[r][i]);
                    }
                    for (uint i = 0; i < 12; i++) {
                        Fp sbox_in = W(P2_START_ROUND_F_END + 12*(r - P2_ROUNDS_F_HALF) + i);
                        ACCUM_GC(filter * fp_sub(state[i], sbox_in));
    
                        state[i] = sbox_in;
                    }
                    p2_sbox(state);
                    p2_external_linear_layer(state);
                }

                for (uint i = 0; i < 12; i++) {
                    ACCUM_GC(filter * fp_sub(state[i], W(12 + i)));

                }
                break;
            }
            case GATE_ADDITION: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                Fp c1 = CS(1);
                for (uint i = 0; i < num_ops; i++) {
                    Fp a0 = W(3*i);
                    Fp a1 = W(3*i + 1);
                    Fp out = W(3*i + 2);
                    ACCUM_GC(filter * fp_sub(out, a0 * c0 + a1 * c1));

                }
                break;
            }
            case GATE_SELECTION: {
                uint num_ops = gate.param0;
                for (uint i = 0; i < num_ops; i++) {
                    Fp b = W(4*i);
                    Fp x_sel = W(4*i + 1);
                    Fp y = W(4*i + 2);
                    Fp result = W(4*i + 3);
                    Fp temp = W(4*num_ops + i);
                    ACCUM_GC(filter * fp_sub(b * y, y + temp));

                    ACCUM_GC(filter * fp_sub(b * x_sel, temp + result));

                }
                break;
            }
            case GATE_EQUALITY: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                for (uint i = 0; i < num_ops; i++) {
                    Fp x_eq = W(3*i);
                    Fp y = W(3*i + 1);
                    Fp equal = W(3*i + 2);
                    Fp diff = W(3*num_ops + 3*i);
                    Fp invdiff = W(3*num_ops + 3*i + 1);
                    Fp prod = W(3*num_ops + 3*i + 2);
                    ACCUM_GC(filter * fp_sub(fp_sub(x_eq, y), diff));

                    ACCUM_GC(filter * fp_sub(diff * invdiff, prod));

                    ACCUM_GC(filter * fp_sub(prod * diff, diff));

                    ACCUM_GC(filter * fp_sub(fp_sub(c0, prod), equal));

                }
                break;
            }
            case GATE_EXPONENTIATION: {
                uint num_power_bits = gate.param0;
                Fp base_e = W(0);
                for (uint i = 0; i < num_power_bits; i++) {
                    Fp prev;
                    if (i == 0) {
                        prev = Fp(1);
                    } else {
                        Fp prev_inter = W(2 + num_power_bits + (i - 1));
                        prev = prev_inter * prev_inter;
                    }
                    Fp bit = W(1 + (num_power_bits - 1 - i));
                    Fp inter = W(2 + num_power_bits + i);
                    Fp factor = bit * fp_sub(base_e, Fp(1)) + Fp(1);
                    ACCUM_GC(filter * fp_sub(prev * factor, inter));

                }
                Fp final_inter = W(2 + num_power_bits + (num_power_bits - 1));
                Fp out = W(1 + num_power_bits);
                ACCUM_GC(filter * fp_sub(out, final_inter));

                break;
            }
            case GATE_MULTIPLICATION: {
                uint num_ops = gate.param0;
                Fp c0 = CS(0);
                for (uint i = 0; i < num_ops; i++) {
                    Fp m0 = W(3*i);
                    Fp m1 = W(3*i + 1);
                    Fp out = W(3*i + 2);
                    ACCUM_GC(filter * fp_sub(out, m0 * m1 * c0));

                }
                break;
            }
            case GATE_BYTE_DECOMPOSITION: {
                uint num_limbs = gate.param0;
                uint num_ops = gate.param1;
                for (uint op = 0; op < num_ops; op++) {
                    uint sum_wire = op * (1 + num_limbs);
                    uint limbs_start = 1 + op * (1 + num_limbs);
                    uint aux_start = (1 + num_limbs) * num_ops + op * 4 * num_limbs;
                    // Range check aux limbs
                    for (uint j = 0; j < 4 * num_limbs; j++) {
                        Fp limb = W(aux_start + j);
                        Fp r = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * r);
    
                    }
                    // Byte = reduce_with_powers(aux_chunk, base=4)
                    for (uint j = 0; j < num_limbs; j++) {
                        Fp sum = Fp(0);
                        Fp base_pow = Fp(1);
                        for (uint k = 0; k < 4; k++) {
                            sum = sum + W(aux_start + j*4 + k) * base_pow;
                            base_pow = base_pow * Fp(4);
                        }
                        ACCUM_GC(filter * fp_sub(sum, W(limbs_start + j)));
    
                    }
                    // Sum = reduce_with_powers(bytes, base=256)
                    Fp byte_sum = Fp(0);
                    Fp byte_pow = Fp(1);
                    for (uint j = 0; j < num_limbs; j++) {
                        byte_sum = byte_sum + W(limbs_start + j) * byte_pow;
                        byte_pow = byte_pow * Fp(256);
                    }
                    ACCUM_GC(filter * fp_sub(byte_sum, W(sum_wire)));

                }
                break;
            }
            case GATE_U48_SUBTRACTION:
            case GATE_U32_SUBTRACTION: {
                // Parametric subtraction: param0=num_ops, param1=bit_width (48 or 32)
                uint num_ops = gate.param0;
                uint bit_width = gate.param1;
                uint num_limbs = bit_width / 2; // 24 for U48, 16 for U32
                Fp base_val = Fp((ulong)1 << bit_width);

                for (uint i = 0; i < num_ops; i++) {
                    Fp input_x = W(5*i);
                    Fp input_y = W(5*i + 1);
                    Fp input_borrow = W(5*i + 2);
                    Fp output_result = W(5*i + 3);
                    Fp output_borrow = W(5*i + 4);

                    // 1. Correctness: result = (x - y - borrow) + base * out_borrow
                    Fp result_initial = fp_sub(fp_sub(input_x, input_y), input_borrow);
                    ACCUM_GC(filter * fp_sub(output_result, result_initial + base_val * output_borrow));


                    // 2. Limb range checks (REVERSE order: num_limbs-1 down to 0)
                    uint limb_base_w = 5 * num_ops;
                    Fp combined = Fp(0);
                    for (int j = (int)num_limbs - 1; j >= 0; j--) {
                        Fp limb = W(limb_base_w + num_limbs * i + (uint)j);
                        Fp prod = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * prod);
    
                        combined = combined * Fp(4) + limb;
                    }

                    // 3. Reconstruction
                    ACCUM_GC(filter * fp_sub(combined, output_result));


                    // 4. Borrow bit check
                    ACCUM_GC(filter * (output_borrow * fp_sub(Fp(1), output_borrow)));

                }
                break;
            }
            case GATE_U16_ADD_MANY: {
                // param0=num_addends, param1=num_ops
                uint num_addends = gate.param0;
                uint num_ops = gate.param1;
                uint num_result_limbs = 8;  // 16 bits / 2-bit limbs
                uint num_carry_limbs = 2;   // 4 bits / 2-bit limbs
                uint num_limbs = num_result_limbs + num_carry_limbs; // 10
                uint op_stride = num_addends + 3;
                Fp base_16 = Fp((ulong)1 << 16);

                for (uint i = 0; i < num_ops; i++) {
                    // Read wires
                    Fp carry_in = W(op_stride * i + num_addends);
                    Fp output_result = W(op_stride * i + num_addends + 1);
                    Fp output_carry = W(op_stride * i + num_addends + 2);

                    // Sum all addends + carry_in
                    Fp computed = carry_in;
                    for (uint j = 0; j < num_addends; j++) {
                        computed = computed + W(op_stride * i + j);
                    }
                    Fp combined_output = output_carry * base_16 + output_result;

                    // 1. Addition constraint
                    ACCUM_GC(filter * fp_sub(combined_output, computed));


                    // 2. Limb range checks (REVERSE order: num_limbs-1 down to 0)
                    uint limb_base_w = op_stride * num_ops;
                    Fp combined_result = Fp(0);
                    Fp combined_carry = Fp(0);
                    for (int j = (int)num_limbs - 1; j >= 0; j--) {
                        Fp limb = W(limb_base_w + num_limbs * i + (uint)j);
                        Fp prod = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * prod);
    
                        if ((uint)j < num_result_limbs) {
                            combined_result = combined_result * Fp(4) + limb;
                        } else {
                            combined_carry = combined_carry * Fp(4) + limb;
                        }
                    }

                    // 3. Result reconstruction
                    ACCUM_GC(filter * fp_sub(combined_result, output_result));


                    // 4. Carry reconstruction
                    ACCUM_GC(filter * fp_sub(combined_carry, output_carry));

                }
                break;
            }
            case GATE_RANGE_CHECK: {
                // param0=num_ops, param1=bit_size
                uint num_ops = gate.param0;
                uint bit_size = gate.param1;
                uint aux_limbs = (bit_size + 1) / 2; // ceil(bit_size / 2)

                for (uint i = 0; i < num_ops; i++) {
                    Fp input_val = W(i);

                    // Compute decomposition: reduce_with_powers(aux_limbs, base=4)
                    Fp computed_sum = Fp(0);
                    Fp base_pow = Fp(1);
                    for (uint j = 0; j < aux_limbs; j++) {
                        Fp limb = W(num_ops + aux_limbs * i + j);
                        computed_sum = computed_sum + limb * base_pow;
                        base_pow = base_pow * Fp(4);
                    }

                    // 1. Decomposition constraint
                    ACCUM_GC(filter * fp_sub(computed_sum, input_val));


                    // 2. Range check non-last limbs (FORWARD order)
                    for (uint j = 0; j < aux_limbs - 1; j++) {
                        Fp limb = W(num_ops + aux_limbs * i + j);
                        Fp prod = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * prod);
    
                    }

                    // 3. Last limb (restricted if bit_size is odd)
                    Fp last_limb = W(num_ops + aux_limbs * i + aux_limbs - 1);
                    if (bit_size % 2 == 1) {
                        ACCUM_GC(filter * (last_limb * fp_sub(last_limb, Fp(1))));
                    } else {
                        Fp prod = last_limb * fp_sub(last_limb, Fp(1)) * fp_sub(last_limb, Fp(2)) * fp_sub(last_limb, Fp(3));
                        ACCUM_GC(filter * prod);
                    }

                }
                break;
            }
            case GATE_U32_ADD_MANY: {
                // param0=num_addends, param1=num_ops
                // Same as U16AddMany but with 32-bit values:
                //   num_result_limbs=16 (32/2), num_carry_limbs=2, total=18
                uint num_addends_32 = gate.param0;
                uint num_ops_32 = gate.param1;
                uint num_result_limbs_32 = 16;
                uint num_carry_limbs_32 = 2;
                uint num_limbs_32 = num_result_limbs_32 + num_carry_limbs_32; // 18
                uint op_stride_32 = num_addends_32 + 3;
                Fp base_32 = Fp((ulong)1 << 32);

                for (uint i = 0; i < num_ops_32; i++) {
                    Fp carry_in = W(op_stride_32 * i + num_addends_32);
                    Fp output_result = W(op_stride_32 * i + num_addends_32 + 1);
                    Fp output_carry = W(op_stride_32 * i + num_addends_32 + 2);

                    Fp computed = carry_in;
                    for (uint j = 0; j < num_addends_32; j++) {
                        computed = computed + W(op_stride_32 * i + j);
                    }
                    Fp combined_output = output_carry * base_32 + output_result;

                    // 1. Addition constraint
                    ACCUM_GC(filter * fp_sub(combined_output, computed));

                    // 2. Limb range checks (REVERSE order)
                    uint limb_base_w = op_stride_32 * num_ops_32;
                    Fp combined_result = Fp(0);
                    Fp combined_carry = Fp(0);
                    for (int j = (int)num_limbs_32 - 1; j >= 0; j--) {
                        Fp limb = W(limb_base_w + num_limbs_32 * i + (uint)j);
                        Fp prod = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * prod);

                        if ((uint)j < num_result_limbs_32) {
                            combined_result = combined_result * Fp(4) + limb;
                        } else {
                            combined_carry = combined_carry * Fp(4) + limb;
                        }
                    }

                    // 3. Result reconstruction
                    ACCUM_GC(filter * fp_sub(combined_result, output_result));

                    // 4. Carry reconstruction
                    ACCUM_GC(filter * fp_sub(combined_carry, output_carry));
                }
                break;
            }
            case GATE_U32_ARITHMETIC: {
                // U32ArithmeticGate: mul-add with 64-bit output split into two 32-bit halves
                // param0 = num_ops
                // Wire layout per op i:
                //   6*i+0: multiplicand_0, 6*i+1: multiplicand_1, 6*i+2: addend
                //   6*i+3: output_low, 6*i+4: output_high, 6*i+5: inverse
                //   6*num_ops + 32*i + j: limb j (j=0..31, 2-bit each)
                // Constraints per op: 36 = 1 canonicity + 1 output + 32 limb range + 2 reconstruction
                uint num_arith_ops = gate.param0;
                uint num_arith_limbs = 32;  // 64 bits / 2-bit limbs
                uint midpoint = 16;  // low half = limbs 0..15, high half = limbs 16..31
                Fp base_arith = Fp((ulong)1 << 32);
                Fp u32_max = Fp((ulong)0xFFFFFFFF);

                for (uint i = 0; i < num_arith_ops; i++) {
                    Fp m0 = W(6*i);
                    Fp m1 = W(6*i + 1);
                    Fp addend = W(6*i + 2);
                    Fp output_low = W(6*i + 3);
                    Fp output_high = W(6*i + 4);
                    Fp inv = W(6*i + 5);

                    // 1. Canonicity: (inv * (u32_max - output_high) - 1) * output_low = 0
                    Fp diff_h = fp_sub(u32_max, output_high);
                    Fp hi_not_max = fp_sub(inv * diff_h, Fp(1));
                    ACCUM_GC(filter * (hi_not_max * output_low));

                    // 2. Output check: output_high * 2^32 + output_low == m0*m1 + addend
                    Fp combined = output_high * base_arith + output_low;
                    Fp computed = m0 * m1 + addend;
                    ACCUM_GC(filter * fp_sub(combined, computed));

                    // 3. Limb range checks (REVERSE order) + reconstruction
                    uint limb_base_arith = 6 * num_arith_ops;
                    Fp combined_low = Fp(0);
                    Fp combined_high = Fp(0);
                    for (int j = (int)num_arith_limbs - 1; j >= 0; j--) {
                        Fp limb = W(limb_base_arith + num_arith_limbs * i + (uint)j);
                        Fp prod = limb * fp_sub(limb, Fp(1)) * fp_sub(limb, Fp(2)) * fp_sub(limb, Fp(3));
                        ACCUM_GC(filter * prod);

                        if ((uint)j < midpoint) {
                            combined_low = combined_low * Fp(4) + limb;
                        } else {
                            combined_high = combined_high * Fp(4) + limb;
                        }
                    }

                    // 4. Low reconstruction
                    ACCUM_GC(filter * fp_sub(combined_low, output_low));

                    // 5. High reconstruction
                    ACCUM_GC(filter * fp_sub(combined_high, output_high));
                }
                break;
            }
            case GATE_QUINTIC_SQUARING: {
                // QuinticSquaringGate: 15 constraints per op
                // param0 = num_ops
                // Wire layout per op i:
                //   a[j] = W(10*i + j)      j=0..4 (input)
                //   c[j] = W(10*i + 5 + j)  j=0..4 (output)
                //   extra[j] = W(10*num_ops + 10*i + j)  j=0..9 (temps)
                uint num_sq_ops = gate.param0;
                Fp const_2_sq = Fp(2);
                Fp const_3_sq = Fp(3);
                Fp const_6_sq = Fp(6);

                for (uint i = 0; i < num_sq_ops; i++) {
                    uint ab = 10 * i;        // a/c base
                    uint eb = 10 * num_sq_ops + 10 * i; // extra base
                    Fp a0 = W(ab); Fp a1 = W(ab+1); Fp a2 = W(ab+2);
                    Fp a3 = W(ab+3); Fp a4 = W(ab+4);
                    Fp c0 = W(ab+5); Fp c1 = W(ab+6); Fp c2 = W(ab+7);
                    Fp c3 = W(ab+8); Fp c4 = W(ab+9);
                    Fp e0 = W(eb); Fp e1 = W(eb+1); Fp e2 = W(eb+2);
                    Fp e3 = W(eb+3); Fp e4 = W(eb+4); Fp e5 = W(eb+5);
                    Fp e6 = W(eb+6); Fp e7 = W(eb+7); Fp e8 = W(eb+8);
                    Fp e9 = W(eb+9);

                    // c[0]: 3 constraints
                    ACCUM_GC(filter * fp_sub(a0 * a0, e0));
                    ACCUM_GC(filter * fp_sub(const_6_sq * a1 * a4 + e0, e1));
                    ACCUM_GC(filter * fp_sub(const_6_sq * a2 * a3 + e1, c0));
                    // c[1]: 3 constraints
                    ACCUM_GC(filter * fp_sub(const_3_sq * a3 * a3, e2));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a0 * a1 + e2, e3));
                    ACCUM_GC(filter * fp_sub(const_6_sq * a2 * a4 + e3, c1));
                    // c[2]: 3 constraints
                    ACCUM_GC(filter * fp_sub(a1 * a1, e4));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a0 * a2 + e4, e5));
                    ACCUM_GC(filter * fp_sub(const_6_sq * a3 * a4 + e5, c2));
                    // c[3]: 3 constraints
                    ACCUM_GC(filter * fp_sub(const_3_sq * a4 * a4, e6));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a0 * a3 + e6, e7));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a1 * a2 + e7, c3));
                    // c[4]: 3 constraints
                    ACCUM_GC(filter * fp_sub(a2 * a2, e8));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a0 * a4 + e8, e9));
                    ACCUM_GC(filter * fp_sub(const_2_sq * a1 * a3 + e9, c4));
                }
                break;
            }
            case GATE_QUINTIC_MULTIPLICATION: {
                // QuinticMultiplicationGate: 5 constraints per op
                // param0 = num_ops
                uint num_qm_ops = gate.param0;
                Fp const_3 = Fp(3);

                for (uint i = 0; i < num_qm_ops; i++) {
                    uint base_w = 15 * i;
                    Fp a[5], b[5], c[5];
                    for (uint j = 0; j < 5; j++) {
                        a[j] = W(base_w + j);
                        b[j] = W(base_w + 5 + j);
                        c[j] = W(base_w + 10 + j);
                    }

                    // Schoolbook multiplication
                    Fp d[9] = {Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0)};
                    for (uint j = 0; j < 5; j++) {
                        for (uint k = 0; k < 5; k++) {
                            d[j + k] = d[j + k] + a[j] * b[k];
                        }
                    }

                    // Reduction: u^5 = 3
                    for (uint k = 0; k < 5; k++) {
                        Fp term = d[k];
                        if (k + 5 <= 8) {
                            term = term + const_3 * d[k + 5];
                        }
                        ACCUM_GC(filter * fp_sub(term, c[k]));
                    }
                }

                break;
            }
            default:
                break;
        }

        #undef ACCUM_GC

        // Debug: write per-gate contribution (delta gc_sum) for thread 0
        if (real_gid == 0 && dbg_off + 2 + num_ch < 256) {
            debug_out[dbg_off++] = gate.gate_type;
            debug_out[dbg_off++] = (ulong)filter;
            for (uint ch = 0; ch < num_ch; ch++) {
                // Per-gate contribution = current gc_sum - saved gc_sum before this gate
                debug_out[dbg_off++] = (ulong)fp_sub(gc_sum[ch], gc_sum_before[ch]);
            }
        }
    }

    // Combine gate constraint sum into cumul:
    //   cumul[ch] += alpha_base[ch] * gc_sum[ch]
    // Then advance alpha_pow past all gate constraint positions.
    for (uint ch = 0; ch < num_ch; ch++) {
        cumul[ch] = cumul[ch] + alpha_base[ch] * gc_sum[ch];
    }

    //=================================================================
    // Output: Z_H division and store
    //=================================================================
    uint rate = 1u << u.quotient_degree_bits;
    Fp z_h_inv = Fp(z_h_inverses[real_gid % rate]);
    for (uint ch = 0; ch < num_ch; ch++) {
        output[real_gid * num_ch + ch] = (ulong)(cumul[ch] * z_h_inv);
    }

    #undef ACCUM_VT
    #undef SKIP_VT
    #undef W
    #undef CS
}
