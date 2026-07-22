/*
 * Gate constraint evaluations for quotient polynomial GPU kernel.
 * Each function evaluates one gate type's unfiltered constraints.
 * The caller applies the selector filter and accumulates into the constraint buffer.
 *
 * Wire and constant indices refer to the "stripped" constants
 * (after removing num_selectors + num_lookup_selectors prefix).
 */
#ifndef quotient_gates_metal
#define quotient_gates_metal

#include "goldilocks.metal"
#include "goldilocks_ext2.metal"

using namespace GoldilocksField;

// Gate type enum
enum GateType : uint {
    GATE_NOOP = 0,
    GATE_ARITHMETIC = 1,
    GATE_ARITHMETIC_EXTENSION = 2,
    GATE_MUL_EXTENSION = 3,
    GATE_CONSTANT = 4,
    GATE_PUBLIC_INPUT = 5,
    GATE_BASE_SUM = 6,
    GATE_REDUCING = 7,
    GATE_REDUCING_EXTENSION = 8,
    GATE_RANDOM_ACCESS = 9,
    GATE_COSET_INTERPOLATION = 10,
    GATE_POSEIDON2 = 11,
    GATE_ADDITION = 12,
    GATE_SELECTION = 13,
    GATE_EQUALITY = 14,
    GATE_EXPONENTIATION = 15,
    GATE_MULTIPLICATION = 16,
    GATE_BYTE_DECOMPOSITION = 17,
    GATE_U48_SUBTRACTION = 18,
    GATE_U32_SUBTRACTION = 19,
    GATE_U16_ADD_MANY = 20,
    GATE_RANGE_CHECK = 21,
    GATE_U16_SUBTRACTION = 22,
    GATE_QUINTIC_MULTIPLICATION = 23,
    GATE_QUINTIC_SQUARING = 24,
    GATE_U32_ADD_MANY = 25,
    GATE_U32_ARITHMETIC = 26,
};

// Gate descriptor passed from CPU
struct GateDescriptor {
    uint gate_type;
    uint row;
    uint selector_index;
    uint group_start;
    uint group_end;
    uint num_constraints;
    uint constraint_offset; // starting index in gate_constraints array
    uint param0;  // num_ops / num_limbs / subgroup_bits / num_power_bits / num_coeffs / num_consts / bits
    uint param1;  // degree / base / num_copies / etc.
    uint param2;  // num_extra_constants / etc.
    uint param3;  // reserved
};

// Maximum gate constraints per gate instance (Poseidon2 is ~103, total circuit ≤ 123).
// Keep small to reduce per-thread memory and avoid GPU resource exhaustion on large dispatches.
#define MAX_GATE_CONSTRAINTS 128
// Poseidon2 dimensions
#define P2_WIDTH 12
#define P2_ROUNDS_F 8
#define P2_ROUNDS_F_HALF 4
#define P2_ROUNDS_P 22
// Extension degree
#define D 2
// UNUSED_SELECTOR value used in filter when many_selectors=true
#define UNUSED_SELECTOR_VAL 4294967295UL

//===================================================================
// Selector filter computation
//===================================================================
inline Fp compute_filter(uint row, uint group_start, uint group_end,
                          Fp selector_val, bool many_selectors) {
    Fp filter = Fp(1);
    for (uint i = group_start; i < group_end; i++) {
        if (i != row) {
            filter = filter * fp_sub(Fp(i), selector_val);
        }
    }
    if (many_selectors) {
        filter = filter * fp_sub(Fp(UNUSED_SELECTOR_VAL), selector_val);
    }
    return filter;
}

//===================================================================
// Gate: Noop — no constraints
//===================================================================
// Nothing to emit.

//===================================================================
// Gate: Arithmetic (arithmetic_base.rs)
// Constraint: output - (mult0 * mult1 * c0 + addend * c1)
// param0 = num_ops
//===================================================================
inline void eval_arithmetic_gate(
    thread const Fp* wires,
    thread const Fp* consts,  // stripped constants
    uint num_ops,
    thread Fp* constraints,
    uint stride  // constraint output stride (1 if writing contiguously)
) {
    Fp c0 = consts[0];
    Fp c1 = consts[1];
    for (uint i = 0; i < num_ops; i++) {
        Fp m0 = wires[4*i];
        Fp m1 = wires[4*i + 1];
        Fp addend = wires[4*i + 2];
        Fp output = wires[4*i + 3];
        Fp computed = m0 * m1 * c0 + addend * c1;
        constraints[i] = fp_sub(output, computed);
    }
}

//===================================================================
// Gate: ArithmeticExtension (arithmetic_extension.rs)
// Constraint: output - (c0 * mult0 * mult1 + c1 * addend) in FpE
// param0 = num_ops, outputs D constraints per op
//===================================================================
inline void eval_arithmetic_extension_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_ops,
    thread Fp* constraints
) {
    Fp c0 = consts[0];
    Fp c1 = consts[1];
    for (uint i = 0; i < num_ops; i++) {
        uint base = 4 * D * i;
        FpE m0 = FpE(wires[base], wires[base + 1]);
        FpE m1 = FpE(wires[base + D], wires[base + D + 1]);
        FpE addend = FpE(wires[base + 2*D], wires[base + 2*D + 1]);
        FpE output = FpE(wires[base + 3*D], wires[base + 3*D + 1]);
        FpE computed = m0 * m1;
        computed = computed.scalar_mul(c0) + addend.scalar_mul(c1);
        FpE diff = output - computed;
        constraints[i * D] = diff.c0;
        constraints[i * D + 1] = diff.c1;
    }
}

//===================================================================
// Gate: MulExtension (multiplication_extension.rs)
// Constraint: output - c0 * mult0 * mult1 in FpE
// param0 = num_ops
//===================================================================
inline void eval_mul_extension_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_ops,
    thread Fp* constraints
) {
    Fp c0 = consts[0];
    for (uint i = 0; i < num_ops; i++) {
        uint base = 3 * D * i;
        FpE m0 = FpE(wires[base], wires[base + 1]);
        FpE m1 = FpE(wires[base + D], wires[base + D + 1]);
        FpE output = FpE(wires[base + 2*D], wires[base + 2*D + 1]);
        FpE computed = (m0 * m1).scalar_mul(c0);
        FpE diff = output - computed;
        constraints[i * D] = diff.c0;
        constraints[i * D + 1] = diff.c1;
    }
}

//===================================================================
// Gate: Constant (constant.rs)
// Constraint: constants[i] - wires[i]
// param0 = num_consts
//===================================================================
inline void eval_constant_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_consts,
    thread Fp* constraints
) {
    for (uint i = 0; i < num_consts; i++) {
        constraints[i] = fp_sub(consts[i], wires[i]);
    }
}

//===================================================================
// Gate: PublicInput (public_input.rs)
// Constraint: wires[i] - public_inputs_hash[i]
// param0 = 4 (always)
//===================================================================
inline void eval_public_input_gate(
    thread const Fp* wires,
    thread const Fp* pub_hash,  // 4 elements
    thread Fp* constraints
) {
    for (uint i = 0; i < 4; i++) {
        constraints[i] = fp_sub(wires[i], pub_hash[i]);
    }
}

//===================================================================
// Gate: BaseSumGate (base_sum.rs)
// Constraint 0: sum - reduce_with_powers(limbs, B)
// Constraint 1..num_limbs: prod(limb - j for j in 0..B)
// param0 = num_limbs, param1 = base B
//===================================================================
inline void eval_base_sum_gate(
    thread const Fp* wires,
    uint num_limbs,
    uint base_B,
    thread Fp* constraints
) {
    // Wire 0 = sum, wires 1..1+num_limbs = limbs
    Fp sum_wire = wires[0];

    // Compute reduce_with_powers: sum = limb[n-1]*B^(n-1) + ... + limb[0]
    // Horner: acc = 0; for i in rev(0..num_limbs): acc = acc * B + limb[i]
    Fp acc = Fp(0);
    Fp base_fp = Fp(base_B);
    for (int i = num_limbs - 1; i >= 0; i--) {
        acc = acc * base_fp + wires[1 + i];
    }
    constraints[0] = fp_sub(acc, sum_wire);

    // Range check each limb: prod(limb - j) for j = 0..B-1 should be 0
    for (uint i = 0; i < num_limbs; i++) {
        Fp limb = wires[1 + i];
        Fp prod = Fp(1);
        for (uint j = 0; j < base_B; j++) {
            prod = prod * fp_sub(limb, Fp(j));
        }
        constraints[1 + i] = prod;
    }
}

//===================================================================
// Gate: ReducingGate (reducing.rs)
// Horner's method in FpE: sum(alpha^i * coeff_i)
// Wires: output(0..D), alpha(D..2D), old_acc(2D..3D), coeffs(3D..3D+num_coeffs), accs(...)
// param0 = num_coeffs
//===================================================================
inline void eval_reducing_gate(
    thread const Fp* wires,
    uint num_coeffs,
    thread Fp* constraints
) {
    FpE output = FpE(wires[0], wires[1]);
    FpE alpha = FpE(wires[D], wires[D + 1]);
    FpE old_acc = FpE(wires[2*D], wires[2*D + 1]);

    uint start_coeffs = 3 * D;
    uint start_accs = start_coeffs + num_coeffs;

    FpE acc = old_acc;
    for (uint i = 0; i < num_coeffs; i++) {
        // coeff is a base field element, embedded into FpE
        Fp coeff = wires[start_coeffs + i];
        // acc_new = acc * alpha + coeff
        FpE acc_new = acc * alpha + FpE(coeff);

        // Get the target accumulator
        FpE target;
        if (i < num_coeffs - 1) {
            uint acc_start = start_accs + D * i;
            target = FpE(wires[acc_start], wires[acc_start + 1]);
        } else {
            target = output;
        }

        FpE diff = acc_new - target;
        constraints[i * D] = diff.c0;
        constraints[i * D + 1] = diff.c1;
        acc = target;
    }
}

//===================================================================
// Gate: ReducingExtensionGate (reducing_extension.rs)
// Horner's method with FpE coefficients: sum(alpha^i * coeff_i)
// param0 = num_coeffs
//===================================================================
inline void eval_reducing_extension_gate(
    thread const Fp* wires,
    uint num_coeffs,
    thread Fp* constraints
) {
    FpE output = FpE(wires[0], wires[1]);
    FpE alpha = FpE(wires[D], wires[D + 1]);
    FpE old_acc = FpE(wires[2*D], wires[2*D + 1]);

    uint start_coeffs = 3 * D;
    uint start_accs = start_coeffs + D * num_coeffs;

    FpE acc = old_acc;
    for (uint i = 0; i < num_coeffs; i++) {
        uint coeff_start = start_coeffs + D * i;
        FpE coeff = FpE(wires[coeff_start], wires[coeff_start + 1]);
        FpE acc_new = acc * alpha + coeff;

        FpE target;
        if (i < num_coeffs - 1) {
            uint acc_start_idx = start_accs + D * i;
            target = FpE(wires[acc_start_idx], wires[acc_start_idx + 1]);
        } else {
            target = output;
        }

        FpE diff = acc_new - target;
        constraints[i * D] = diff.c0;
        constraints[i * D + 1] = diff.c1;
        acc = target;
    }
}

//===================================================================
// Gate: RandomAccess (random_access.rs)
// param0 = bits, param1 = num_copies, param2 = num_extra_constants
// param3 = num_routed_wires (for bit wire offset)
//===================================================================
inline void eval_random_access_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint bits,
    uint num_copies,
    uint num_extra_constants,
    uint num_routed_wires,
    thread Fp* constraints
) {
    uint vec_size = 1u << bits;
    uint c_idx = 0;

    for (uint copy = 0; copy < num_copies; copy++) {
        uint base_routed = (2 + vec_size) * copy;
        Fp access_index = wires[base_routed];
        Fp claimed_element = wires[base_routed + 1];

        // Bit decomposition constraints
        Fp reconstructed = Fp(0);
        for (uint i = 0; i < bits; i++) {
            uint bit_wire = num_routed_wires + copy * bits + i;
            Fp b = wires[bit_wire];
            // b * (b - 1) = 0
            constraints[c_idx++] = b * fp_sub(b, Fp(1));
            // reconstruct: MSB first
            reconstructed = reconstructed + reconstructed + b;
        }
        // reconstructed - access_index = 0
        constraints[c_idx++] = fp_sub(reconstructed, access_index);

        // Binary fold: walk from list to single element
        // We fold the vec_size list items using the bits
        // Start with all list items, fold by each bit from MSB to LSB
        // temp[j] = list[2j] + bit * (list[2j+1] - list[2j])
        // Actually: fold uses bits from MSB to LSB
        // Let's use a simplified approach: iterate bits, halving the list
        // Since vec_size can be up to 2^bits, we need temp storage
        // For GPU, use a local array
        Fp fold[64]; // max vec_size (typically ≤ 64)
        for (uint j = 0; j < vec_size; j++) {
            fold[j] = wires[base_routed + 2 + j];
        }
        uint cur_size = vec_size;
        for (int i = bits - 1; i >= 0; i--) {
            uint bit_wire = num_routed_wires + copy * bits + (uint)i;
            Fp b = wires[bit_wire];
            uint new_size = cur_size / 2;
            for (uint j = 0; j < new_size; j++) {
                // fold[j] = fold[2j] + b * (fold[2j+1] - fold[2j])
                fold[j] = fold[2*j] + b * fp_sub(fold[2*j + 1], fold[2*j]);
            }
            cur_size = new_size;
        }
        // fold[0] should equal claimed_element
        constraints[c_idx++] = fp_sub(fold[0], claimed_element);
    }

    // Extra constant constraints
    for (uint i = 0; i < num_extra_constants; i++) {
        Fp constant_val = consts[i];
        uint wire_idx = (2 + vec_size) * num_copies + i;
        constraints[c_idx++] = fp_sub(constant_val, wires[wire_idx]);
    }
}

//===================================================================
// Gate: Addition (addition_base.rs)
// Constraint: output - (addend0 * c0 + addend1 * c1)
// param0 = num_ops
//===================================================================
inline void eval_addition_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_ops,
    thread Fp* constraints
) {
    Fp c0 = consts[0];
    Fp c1 = consts[1];
    for (uint i = 0; i < num_ops; i++) {
        Fp a0 = wires[3*i];
        Fp a1 = wires[3*i + 1];
        Fp output = wires[3*i + 2];
        Fp computed = a0 * c0 + a1 * c1;
        constraints[i] = fp_sub(output, computed);
    }
}

//===================================================================
// Gate: Selection (select_base.rs)
// Constraint 1: b*y - y - temp = 0  →  temp = y*(b-1)
// Constraint 2: b*x - temp - result = 0
// param0 = num_ops
//===================================================================
inline void eval_selection_gate(
    thread const Fp* wires,
    uint num_ops,
    thread Fp* constraints
) {
    for (uint i = 0; i < num_ops; i++) {
        Fp b = wires[4*i];          // selector
        Fp x = wires[4*i + 1];     // element_0
        Fp y = wires[4*i + 2];     // element_1
        Fp result = wires[4*i + 3]; // output
        Fp temp = wires[4*num_ops + i]; // temporary (unrouted)

        // Constraint 1: b*y - y - temp = 0  →  temp = y*(b-1)
        constraints[2*i] = fp_sub(b * y, y + temp);
        // Constraint 2: b*x - temp - result = 0
        constraints[2*i + 1] = fp_sub(b * x, temp + result);
    }
}

//===================================================================
// Gate: Equality (equality_base.rs)
// param0 = num_ops
//===================================================================
inline void eval_equality_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_ops,
    thread Fp* constraints
) {
    Fp c0 = consts[0]; // = 1
    for (uint i = 0; i < num_ops; i++) {
        Fp x = wires[3*i];
        Fp y = wires[3*i + 1];
        Fp equal = wires[3*i + 2]; // output
        Fp diff = wires[3*num_ops + 3*i];
        Fp invdiff = wires[3*num_ops + 3*i + 1];
        Fp prod = wires[3*num_ops + 3*i + 2];

        constraints[4*i]     = fp_sub(fp_sub(x, y), diff);      // diff = x - y
        constraints[4*i + 1] = fp_sub(diff * invdiff, prod);     // prod = diff * invdiff
        constraints[4*i + 2] = fp_sub(prod * diff, diff);        // prod * diff = diff (i.e., if diff≠0 then prod=1)
        constraints[4*i + 3] = fp_sub(fp_sub(c0, prod), equal);  // equal = 1 - prod
    }
}

//===================================================================
// Gate: Multiplication (multiplication_base.rs)
// Constraint: output - mult0 * mult1 * c0
// param0 = num_ops
//===================================================================
inline void eval_multiplication_gate(
    thread const Fp* wires,
    thread const Fp* consts,
    uint num_ops,
    thread Fp* constraints
) {
    Fp c0 = consts[0];
    for (uint i = 0; i < num_ops; i++) {
        Fp m0 = wires[3*i];
        Fp m1 = wires[3*i + 1];
        Fp output = wires[3*i + 2];
        constraints[i] = fp_sub(output, m0 * m1 * c0);
    }
}

//===================================================================
// Gate: Exponentiation (exponentiation.rs)
// Binary exponentiation: base^exponent
// param0 = num_power_bits
//===================================================================
inline void eval_exponentiation_gate(
    thread const Fp* wires,
    uint num_power_bits,
    thread Fp* constraints
) {
    Fp base = wires[0];
    Fp output = wires[1 + num_power_bits];
    // wire_power_bit(i) = 1 + i (little-endian)
    // wire_intermediate_value(i) = 2 + num_power_bits + i

    uint c_idx = 0;
    for (uint i = 0; i < num_power_bits; i++) {
        Fp prev;
        if (i == 0) {
            prev = Fp(1);
        } else {
            Fp prev_inter = wires[2 + num_power_bits + (i - 1)];
            prev = prev_inter * prev_inter; // squaring
        }
        // Bit in big-endian order: bit index = num_power_bits - 1 - i
        Fp bit = wires[1 + (num_power_bits - 1 - i)];
        Fp inter = wires[2 + num_power_bits + i];

        // intermediate = prev * (bit * base + (1 - bit))
        //              = prev * (bit * (base - 1) + 1)
        Fp factor = bit * fp_sub(base, Fp(1)) + Fp(1);
        Fp expected = prev * factor;
        constraints[c_idx++] = fp_sub(inter, expected);
    }
    // Final constraint: output = intermediate[num_power_bits - 1]
    Fp final_inter = wires[2 + num_power_bits + (num_power_bits - 1)];
    constraints[c_idx++] = fp_sub(output, final_inter);
}

//===================================================================
// Gate: Poseidon2 (poseidon2.rs)
// Full Poseidon2 permutation constraint checking
// No params needed (fixed width=12, rounds=8+22)
//===================================================================

// Poseidon2 round constants in constant address space
constant unsigned long P2_EXTERNAL_RC[8][12] = {
    {15492826721047263190UL, 11728330187201910315UL, 8836021247773420868UL, 16777404051263952451UL,
     5510875212538051896UL, 6173089941271892285UL, 2927757366422211339UL, 10340958981325008808UL,
     8541987352684552425UL, 9739599543776434497UL, 15073950188101532019UL, 12084856431752384512UL},
    {4584713381960671270UL, 8807052963476652830UL, 54136601502601741UL, 4872702333905478703UL,
     5551030319979516287UL, 12889366755535460989UL, 16329242193178844328UL, 412018088475211848UL,
     10505784623379650541UL, 9758812378619434837UL, 7421979329386275117UL, 375240370024755551UL},
    {3331431125640721931UL, 15684937309956309981UL, 578521833432107983UL, 14379242000670861838UL,
     17922409828154900976UL, 8153494278429192257UL, 15904673920630731971UL, 11217863998460634216UL,
     3301540195510742136UL, 9937973023749922003UL, 3059102938155026419UL, 1895288289490976132UL},
    {5580912693628927540UL, 10064804080494788323UL, 9582481583369602410UL, 10186259561546797986UL,
     247426333829703916UL, 13193193905461376067UL, 6386232593701758044UL, 17954717245501896472UL,
     1531720443376282699UL, 2455761864255501970UL, 11234429217864304495UL, 4746959618548874102UL},
    {13571697342473846203UL, 17477857865056504753UL, 15963032953523553760UL, 16033593225279635898UL,
     14252634232868282405UL, 8219748254835277737UL, 7459165569491914711UL, 15855939513193752003UL,
     16788866461340278896UL, 7102224659693946577UL, 3024718005636976471UL, 13695468978618890430UL},
    {8214202050877825436UL, 2670727992739346204UL, 16259532062589659211UL, 11869922396257088411UL,
     3179482916972760137UL, 13525476046633427808UL, 3217337278042947412UL, 14494689598654046340UL,
     15837379330312175383UL, 8029037639801151344UL, 2153456285263517937UL, 8301106462311849241UL},
    {13294194396455217955UL, 17394768489610594315UL, 12847609130464867455UL, 14015739446356528640UL,
     5879251655839607853UL, 9747000124977436185UL, 8950393546890284269UL, 10765765936405694368UL,
     14695323910334139959UL, 16366254691123000864UL, 15292774414889043182UL, 10910394433429313384UL},
    {17253424460214596184UL, 3442854447664030446UL, 3005570425335613727UL, 10859158614900201063UL,
     9763230642109343539UL, 6647722546511515039UL, 909012944955815706UL, 18101204076790399111UL,
     11588128829349125809UL, 15863878496612806566UL, 5201119062417750399UL, 176665553780565743UL}
};

constant unsigned long P2_INTERNAL_RC[22] = {
    11921381764981422944UL, 10318423381711320787UL, 8291411502347000766UL, 229948027109387563UL,
    9152521390190983261UL, 7129306032690285515UL, 15395989607365232011UL, 8641397269074305925UL,
    17256848792241043600UL, 6046475228902245682UL, 12041608676381094092UL, 12785542378683951657UL,
    14546032085337914034UL, 3304199118235116851UL, 16499627707072547655UL, 10386478025625759321UL,
    13475579315436919170UL, 16042710511297532028UL, 1411266850385657080UL, 9024840976168649958UL,
    14047056970978379368UL, 838728605080212101UL
};

constant unsigned long P2_DIAG[12] = {
    0xc3b6c08e23ba9300UL, 0xd84b5de94a324fb6UL, 0x0d0c371c5b35b84fUL, 0x7964f570e7188037UL,
    0x5daf18bbd996604bUL, 0x6743bc47b9595257UL, 0x5528b9362c59bb70UL, 0xac45e25b7127b68bUL,
    0xa2077d7dfbb606b5UL, 0xf3faac6faee378aeUL, 0x0c6388b51545e883UL, 0xd27dbb6944917b60UL
};

// M4 circulant matrix for external linear layer: circ(2,3,1,1)
// Row 0: 2*x[0] + 3*x[1] + 1*x[2] + 1*x[3]
// Row 1: 1*x[0] + 2*x[1] + 3*x[2] + 1*x[3]
// Row 2: 1*x[0] + 1*x[1] + 2*x[2] + 3*x[3]
// Row 3: 3*x[0] + 1*x[1] + 1*x[2] + 2*x[3]
inline void p2_m4(thread Fp* x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp x0 = x[0];
    Fp x2 = x[2];
    x[0] = t0123 + t01 + x[1];     // 2*x0 + 3*x1 + x2 + x3
    x[1] = t0123 + x[1] + x2 + x2; // x0 + 2*x1 + 3*x2 + x3
    x[2] = t0123 + t23 + x[3];     // x0 + x1 + 2*x2 + 3*x3
    x[3] = t0123 + x[3] + x0 + x0; // 3*x0 + x1 + x2 + 2*x3
}

inline void p2_external_linear_layer(thread Fp* state) {
    // Apply M4 to each group of 4
    p2_m4(state);
    p2_m4(state + 4);
    p2_m4(state + 8);
    // Sum across groups
    Fp sums[4];
    for (uint i = 0; i < 4; i++) {
        sums[i] = state[i] + state[4 + i] + state[8 + i];
    }
    for (uint i = 0; i < 12; i++) {
        state[i] = state[i] + sums[i % 4];
    }
}

inline void p2_internal_linear_layer(thread Fp* state) {
    // sum = Σ state[i]
    Fp sum = Fp(0);
    for (uint i = 0; i < 12; i++) sum = sum + state[i];
    // state[i] = state[i] * diag[i] + sum
    for (uint i = 0; i < 12; i++) {
        state[i] = state[i] * Fp(P2_DIAG[i]) + sum;
    }
}

inline void p2_sbox(thread Fp* state) {
    for (uint i = 0; i < 12; i++) {
        state[i] = state[i].pow7();
    }
}

inline Fp p2_sbox_p(Fp x) {
    return x.pow7();
}

inline void eval_poseidon2_gate(
    thread const Fp* wires,
    thread Fp* constraints
) {
    uint c_idx = 0;

    // Wire layout from poseidon2.rs:
    // input(i) = i                          [0..12)
    // output(i) = WIDTH + i                 [12..24)
    // WIRE_SWAP = 2*WIDTH = 24
    // delta(i) = 25 + i                     [25..29)
    // START_ROUND_F_BEGIN = 29
    // full_sbox_0(r, i) = 29 + WIDTH*(r-1) + i  for r=1..3, i=0..11
    // START_PARTIAL = 29 + WIDTH*3 = 65
    // partial_sbox(r) = 65 + r              for r=0..21
    // START_ROUND_F_END = 65 + 22 = 87
    // full_sbox_1(r, i) = 87 + WIDTH*r + i  for r=0..3, i=0..11

    #define P2_WIRE_SWAP 24
    #define P2_START_DELTA 25
    #define P2_START_ROUND_F_BEGIN 29
    #define P2_START_PARTIAL 65
    #define P2_START_ROUND_F_END 87

    Fp swap = wires[P2_WIRE_SWAP];
    // swap is binary
    constraints[c_idx++] = swap * fp_sub(swap, Fp(1));

    // Delta constraints
    for (uint i = 0; i < 4; i++) {
        Fp input_lhs = wires[i];
        Fp input_rhs = wires[i + 4];
        Fp delta_i = wires[P2_START_DELTA + i];
        constraints[c_idx++] = fp_sub(swap * fp_sub(input_rhs, input_lhs), delta_i);
    }

    // Compute possibly-swapped state
    Fp state[12];
    for (uint i = 0; i < 4; i++) {
        Fp delta_i = wires[P2_START_DELTA + i];
        state[i] = wires[i] + delta_i;
        state[i + 4] = fp_sub(wires[i + 4], delta_i);
    }
    for (uint i = 8; i < 12; i++) {
        state[i] = wires[i];
    }

    // Initial linear layer
    p2_external_linear_layer(state);

    // First half of external rounds
    for (uint r = 0; r < P2_ROUNDS_F_HALF; r++) {
        // Add round constants
        for (uint i = 0; i < 12; i++) {
            state[i] = state[i] + Fp(P2_EXTERNAL_RC[r][i]);
        }
        // Constrain S-box inputs (except round 0 where they're implicit)
        if (r != 0) {
            for (uint i = 0; i < 12; i++) {
                Fp sbox_in = wires[P2_START_ROUND_F_BEGIN + 12 * (r - 1) + i];
                constraints[c_idx++] = fp_sub(state[i], sbox_in);
                state[i] = sbox_in;
            }
        }
        p2_sbox(state);
        p2_external_linear_layer(state);
    }

    // Internal (partial) rounds
    for (uint r = 0; r < P2_ROUNDS_P; r++) {
        state[0] = state[0] + Fp(P2_INTERNAL_RC[r]);
        Fp sbox_in = wires[P2_START_PARTIAL + r];
        constraints[c_idx++] = fp_sub(state[0], sbox_in);
        state[0] = p2_sbox_p(sbox_in);
        p2_internal_linear_layer(state);
    }

    // Second half of external rounds
    for (uint r = P2_ROUNDS_F_HALF; r < P2_ROUNDS_F; r++) {
        for (uint i = 0; i < 12; i++) {
            state[i] = state[i] + Fp(P2_EXTERNAL_RC[r][i]);
        }
        for (uint i = 0; i < 12; i++) {
            Fp sbox_in = wires[P2_START_ROUND_F_END + 12 * (r - P2_ROUNDS_F_HALF) + i];
            constraints[c_idx++] = fp_sub(state[i], sbox_in);
            state[i] = sbox_in;
        }
        p2_sbox(state);
        p2_external_linear_layer(state);
    }

    // Output constraints
    for (uint i = 0; i < 12; i++) {
        constraints[c_idx++] = fp_sub(state[i], wires[12 + i]);
    }
}

//===================================================================
// Gate: CosetInterpolation (coset_interpolation.rs)
// Barycentric interpolation on a coset of a multiplicative subgroup
// param0 = subgroup_bits, param1 = degree
// Barycentric weights and subgroup points passed in separate buffer
//===================================================================
inline void eval_coset_interpolation_gate(
    thread const Fp* wires,
    uint subgroup_bits,
    uint degree,
    device const unsigned long* subgroup_points,  // 2^subgroup_bits points
    device const unsigned long* bary_weights,     // 2^subgroup_bits weights
    thread Fp* constraints
) {
    uint num_points = 1u << subgroup_bits;
    uint num_intermediates = (num_points - 2) / (degree - 1);
    uint c_idx = 0;

    // Wire layout:
    // 0: shift (base field)
    // 1..1+num_points*D: values (FpE each = D wires)
    // 1+num_points*D..1+num_points*D+D: evaluation_point (FpE)
    // 1+num_points*D+D..1+num_points*D+2*D: evaluation_value (FpE)
    // after that: intermediates

    Fp shift = wires[0];
    uint val_start = 1;
    uint eval_pt_start = val_start + num_points * D;
    uint eval_val_start = eval_pt_start + D;
    uint inter_start = eval_val_start + D;

    FpE eval_point = FpE(wires[eval_pt_start], wires[eval_pt_start + 1]);

    // shifted_evaluation_point is the last D wires in intermediates section
    uint shifted_pt_start = inter_start + D * 2 * num_intermediates;
    FpE shifted_pt = FpE(wires[shifted_pt_start], wires[shifted_pt_start + 1]);

    // Constraint: eval_point - shifted_pt * shift = 0
    FpE diff0 = eval_point - shifted_pt.scalar_mul(shift);
    constraints[c_idx++] = diff0.c0;
    constraints[c_idx++] = diff0.c1;

    // Barycentric interpolation using partial_interpolate
    // partial_interpolate accumulates (eval, prod) over domain chunks
    FpE computed_eval = FpE::zero();
    FpE computed_prod = FpE::one();

    // Process first chunk: domain[0..degree]
    uint chunk_end = min(degree, num_points);
    for (uint j = 0; j < chunk_end; j++) {
        Fp domain_pt = Fp(subgroup_points[j]);
        Fp weight = Fp(bary_weights[j]);
        FpE val = FpE(wires[val_start + j*D], wires[val_start + j*D + 1]);
        FpE diff_x = shifted_pt - FpE(domain_pt);
        computed_eval = computed_eval * diff_x + val.scalar_mul(weight) * computed_prod;
        computed_prod = computed_prod * diff_x;
    }

    // Process intermediate chunks
    for (uint inter = 0; inter < num_intermediates; inter++) {
        uint ie_start = inter_start + D * inter;
        uint ip_start = inter_start + D * (num_intermediates + inter);
        FpE inter_eval = FpE(wires[ie_start], wires[ie_start + 1]);
        FpE inter_prod = FpE(wires[ip_start], wires[ip_start + 1]);

        FpE de = inter_eval - computed_eval;
        FpE dp = inter_prod - computed_prod;
        constraints[c_idx++] = de.c0;
        constraints[c_idx++] = de.c1;
        constraints[c_idx++] = dp.c0;
        constraints[c_idx++] = dp.c1;

        // Next chunk
        uint start_index = 1 + (degree - 1) * (inter + 1);
        uint end_index = min(start_index + degree - 1, num_points);
        computed_eval = inter_eval;
        computed_prod = inter_prod;
        for (uint j = start_index; j < end_index; j++) {
            Fp domain_pt = Fp(subgroup_points[j]);
            Fp weight = Fp(bary_weights[j]);
            FpE val = FpE(wires[val_start + j*D], wires[val_start + j*D + 1]);
            FpE diff_x = shifted_pt - FpE(domain_pt);
            computed_eval = computed_eval * diff_x + val.scalar_mul(weight) * computed_prod;
            computed_prod = computed_prod * diff_x;
        }
    }

    // Final constraint: evaluation_value - computed_eval = 0
    FpE eval_value = FpE(wires[eval_val_start], wires[eval_val_start + 1]);
    FpE final_diff = eval_value - computed_eval;
    constraints[c_idx++] = final_diff.c0;
    constraints[c_idx++] = final_diff.c1;
}

//===================================================================
// Gate: ByteDecomposition (lighter-prover custom gate)
// Decomposes a value into bytes and checks via aux limbs (base-4 digits).
// param0 = num_limbs, param1 = num_ops
// Wire layout per op i:
//   sum = i * (1 + num_limbs)
//   limbs = [1 + i*(1+num_limbs), 1 + i*(1+num_limbs) + num_limbs)
//   aux_limbs = [(1+num_limbs)*num_ops + i*4*num_limbs, ... + 4*num_limbs)
// Constraints per op:
//   4*num_limbs range checks: prod(aux - 0, aux - 1, aux - 2, aux - 3)
//   num_limbs sum checks: reduce_with_powers(aux_chunk[4], base=4) - byte
//   1 byte sum: reduce_with_powers(bytes, base=256) - expected_sum
//===================================================================
inline void eval_byte_decomposition_gate(
    thread const Fp* wires,
    uint num_limbs,
    uint num_ops,
    thread Fp* constraints
) {
    uint c_idx = 0;
    for (uint i = 0; i < num_ops; i++) {
        uint sum_wire = i * (1 + num_limbs);
        uint limbs_start = 1 + i * (1 + num_limbs);
        uint aux_start = (1 + num_limbs) * num_ops + i * 4 * num_limbs;

        // Range check aux limbs: each aux in {0,1,2,3}
        for (uint j = 0; j < 4 * num_limbs; j++) {
            Fp limb = wires[aux_start + j];
            Fp r = limb
                 * fp_sub(limb, Fp(1))
                 * fp_sub(limb, Fp(2))
                 * fp_sub(limb, Fp(3));
            constraints[c_idx++] = r;
        }

        // Check each byte = reduce_with_powers(aux_chunk, base=4)
        for (uint j = 0; j < num_limbs; j++) {
            Fp sum = Fp(0);
            Fp base_pow = Fp(1);
            Fp base4 = Fp(4);
            for (uint k = 0; k < 4; k++) {
                sum = sum + wires[aux_start + j * 4 + k] * base_pow;
                base_pow = base_pow * base4;
            }
            constraints[c_idx++] = fp_sub(sum, wires[limbs_start + j]);
        }

        // Check sum = reduce_with_powers(bytes, base=256)
        Fp byte_sum = Fp(0);
        Fp byte_pow = Fp(1);
        Fp base256 = Fp(256);
        for (uint j = 0; j < num_limbs; j++) {
            byte_sum = byte_sum + wires[limbs_start + j] * byte_pow;
            byte_pow = byte_pow * base256;
        }
        constraints[c_idx++] = fp_sub(byte_sum, wires[sum_wire]);
    }
}

//===================================================================
// Gate: U16SubtractionGate (subtraction_u16.rs)
// result = x - y - borrow_in, with output_borrow and 8 range-check limbs (2-bit each)
// param0 = num_ops
// Constraints per op: 11 (1 subtraction + 8 limb range checks + 1 combined + 1 borrow bit)
// Wire layout per op i:
//   5*i: input_x, 5*i+1: input_y, 5*i+2: input_borrow
//   5*i+3: output_result, 5*i+4: output_borrow
//   5*num_ops + 8*i + j: output limb j (j=0..7)
//===================================================================
inline void eval_u16_subtraction_gate(
    thread const Fp* wires,
    uint num_ops,
    thread Fp* constraints
) {
    uint c_idx = 0;
    Fp base = Fp(1UL << 16);
    Fp limb_base = Fp(1UL << 2); // 2-bit limbs

    for (uint i = 0; i < num_ops; i++) {
        Fp input_x = wires[5*i];
        Fp input_y = wires[5*i + 1];
        Fp input_borrow = wires[5*i + 2];
        Fp output_result = wires[5*i + 3];
        Fp output_borrow = wires[5*i + 4];

        // Constraint: output_result - (input_x - input_y - input_borrow + base * output_borrow) = 0
        Fp result_initial = fp_sub(fp_sub(input_x, input_y), input_borrow);
        constraints[c_idx++] = fp_sub(output_result, (result_initial + base * output_borrow));

        // Range check limbs (2-bit each, 8 limbs) — REVERSE order like U48/U32 subtraction
        Fp combined_limbs = Fp(0);
        for (int j = 7; j >= 0; j--) {
            Fp limb = wires[5*num_ops + 8*i + j];
            // prod(limb - k) for k=0..3
            Fp prod = limb
                * fp_sub(limb, Fp(1))
                * fp_sub(limb, Fp(2))
                * fp_sub(limb, Fp(3));
            constraints[c_idx++] = prod;

            combined_limbs = limb_base * combined_limbs + limb;
        }
        // Combined limbs = output_result
        constraints[c_idx++] = fp_sub(combined_limbs, output_result);

        // Borrow is boolean
        constraints[c_idx++] = output_borrow * fp_sub(Fp(1), output_borrow);
    }
}

//===================================================================
// Gate: QuinticMultiplicationGate (mul_quintic_ext_base.rs)
// Multiplies two elements of a quintic extension field (degree-5 over Fp)
// with reduction polynomial u^5 = 3.
// param0 = num_ops
// Constraints per op: 5
// Wire layout per op i:
//   15*i + 0..4: multiplicand_0 limbs (a[0..4])
//   15*i + 5..9: multiplicand_1 limbs (b[0..4])
//   15*i + 10..14: output limbs (c[0..4])
//===================================================================
inline void eval_quintic_multiplication_gate(
    thread const Fp* wires,
    uint num_ops,
    thread Fp* constraints
) {
    Fp const_3 = Fp(3);
    uint c_idx = 0;

    for (uint i = 0; i < num_ops; i++) {
        uint base_w = 15 * i;
        Fp a[5], b[5], c[5];
        for (uint j = 0; j < 5; j++) {
            a[j] = wires[base_w + j];
            b[j] = wires[base_w + 5 + j];
            c[j] = wires[base_w + 10 + j];
        }

        // Schoolbook multiplication: d[j+k] += a[j] * b[k]
        Fp d[9] = {Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0), Fp(0)};
        for (uint j = 0; j < 5; j++) {
            for (uint k = 0; k < 5; k++) {
                d[j + k] = d[j + k] + a[j] * b[k];
            }
        }

        // Reduction: u^5 = 3, so d[k+5] contributes 3*d[k+5] to d[k]
        for (uint k = 0; k < 5; k++) {
            Fp term = d[k];
            if (k + 5 <= 8) {
                term = term + const_3 * d[k + 5];
            }
            constraints[c_idx++] = fp_sub(term, c[k]);
        }
    }
}

#endif /* quotient_gates_metal */
