// Poseidon2 permutation for Goldilocks field (width=12)
// All constants in Metal constant address space (hardware L1-cached).
// Algorithm: 1 initial M_E + 4 full rounds + 22 partial rounds + 4 full rounds
//
// Constants from elliottech/plonky2 poseidon2/config.rs (Plonky3-derived)

#ifndef poseidon2_goldilocks
#define poseidon2_goldilocks
#include <metal_stdlib>
#include "goldilocks.metal"

using namespace metal;
namespace GoldilocksField {

// Poseidon2 round constants (from elliottech poseidon2/config.rs)
// 30 rounds: full[0..4], partial[4..26], full[26..30]
// Full rounds use EXTERNAL_CONSTANTS, partial rounds use INTERNAL_CONSTANTS (element[0] only).
constant ulong POSEIDON2_RC[360] = {
    // Round 0 (full) - EXTERNAL_CONSTANTS[0]
    15492826721047263190UL, 11728330187201910315UL, 8836021247773420868UL, 16777404051263952451UL,
    5510875212538051896UL, 6173089941271892285UL, 2927757366422211339UL, 10340958981325008808UL,
    8541987352684552425UL, 9739599543776434497UL, 15073950188101532019UL, 12084856431752384512UL,
    // Round 1 (full) - EXTERNAL_CONSTANTS[1]
    4584713381960671270UL, 8807052963476652830UL, 54136601502601741UL, 4872702333905478703UL,
    5551030319979516287UL, 12889366755535460989UL, 16329242193178844328UL, 412018088475211848UL,
    10505784623379650541UL, 9758812378619434837UL, 7421979329386275117UL, 375240370024755551UL,
    // Round 2 (full) - EXTERNAL_CONSTANTS[2]
    3331431125640721931UL, 15684937309956309981UL, 578521833432107983UL, 14379242000670861838UL,
    17922409828154900976UL, 8153494278429192257UL, 15904673920630731971UL, 11217863998460634216UL,
    3301540195510742136UL, 9937973023749922003UL, 3059102938155026419UL, 1895288289490976132UL,
    // Round 3 (full) - EXTERNAL_CONSTANTS[3]
    5580912693628927540UL, 10064804080494788323UL, 9582481583369602410UL, 10186259561546797986UL,
    247426333829703916UL, 13193193905461376067UL, 6386232593701758044UL, 17954717245501896472UL,
    1531720443376282699UL, 2455761864255501970UL, 11234429217864304495UL, 4746959618548874102UL,
    // Round 4 (partial) - INTERNAL_CONSTANTS[0]
    11921381764981422944UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 5 (partial) - INTERNAL_CONSTANTS[1]
    10318423381711320787UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 6 (partial) - INTERNAL_CONSTANTS[2]
    8291411502347000766UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 7 (partial) - INTERNAL_CONSTANTS[3]
    229948027109387563UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 8 (partial) - INTERNAL_CONSTANTS[4]
    9152521390190983261UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 9 (partial) - INTERNAL_CONSTANTS[5]
    7129306032690285515UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 10 (partial) - INTERNAL_CONSTANTS[6]
    15395989607365232011UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 11 (partial) - INTERNAL_CONSTANTS[7]
    8641397269074305925UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 12 (partial) - INTERNAL_CONSTANTS[8]
    17256848792241043600UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 13 (partial) - INTERNAL_CONSTANTS[9]
    6046475228902245682UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 14 (partial) - INTERNAL_CONSTANTS[10]
    12041608676381094092UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 15 (partial) - INTERNAL_CONSTANTS[11]
    12785542378683951657UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 16 (partial) - INTERNAL_CONSTANTS[12]
    14546032085337914034UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 17 (partial) - INTERNAL_CONSTANTS[13]
    3304199118235116851UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 18 (partial) - INTERNAL_CONSTANTS[14]
    16499627707072547655UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 19 (partial) - INTERNAL_CONSTANTS[15]
    10386478025625759321UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 20 (partial) - INTERNAL_CONSTANTS[16]
    13475579315436919170UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 21 (partial) - INTERNAL_CONSTANTS[17]
    16042710511297532028UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 22 (partial) - INTERNAL_CONSTANTS[18]
    1411266850385657080UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 23 (partial) - INTERNAL_CONSTANTS[19]
    9024840976168649958UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 24 (partial) - INTERNAL_CONSTANTS[20]
    14047056970978379368UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 25 (partial) - INTERNAL_CONSTANTS[21]
    838728605080212101UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    // Round 26 (full) - EXTERNAL_CONSTANTS[4]
    13571697342473846203UL, 17477857865056504753UL, 15963032953523553760UL, 16033593225279635898UL,
    14252634232868282405UL, 8219748254835277737UL, 7459165569491914711UL, 15855939513193752003UL,
    16788866461340278896UL, 7102224659693946577UL, 3024718005636976471UL, 13695468978618890430UL,
    // Round 27 (full) - EXTERNAL_CONSTANTS[5]
    8214202050877825436UL, 2670727992739346204UL, 16259532062589659211UL, 11869922396257088411UL,
    3179482916972760137UL, 13525476046633427808UL, 3217337278042947412UL, 14494689598654046340UL,
    15837379330312175383UL, 8029037639801151344UL, 2153456285263517937UL, 8301106462311849241UL,
    // Round 28 (full) - EXTERNAL_CONSTANTS[6]
    13294194396455217955UL, 17394768489610594315UL, 12847609130464867455UL, 14015739446356528640UL,
    5879251655839607853UL, 9747000124977436185UL, 8950393546890284269UL, 10765765936405694368UL,
    14695323910334139959UL, 16366254691123000864UL, 15292774414889043182UL, 10910394433429313384UL,
    // Round 29 (full) - EXTERNAL_CONSTANTS[7]
    17253424460214596184UL, 3442854447664030446UL, 3005570425335613727UL, 10859158614900201063UL,
    9763230642109343539UL, 6647722546511515039UL, 909012944955815706UL, 18101204076790399111UL,
    11588128829349125809UL, 15863878496612806566UL, 5201119062417750399UL, 176665553780565743UL,
};

// Internal diffusion matrix diagonal (MATRIX_DIAG_12_U64 from config.rs)
constant ulong POSEIDON2_DIAG[12] = {
    0xc3b6c08e23ba9300UL,
    0xd84b5de94a324fb6UL,
    0x0d0c371c5b35b84fUL,
    0x7964f570e7188037UL,
    0x5daf18bbd996604bUL,
    0x6743bc47b9595257UL,
    0x5528b9362c59bb70UL,
    0xac45e25b7127b68bUL,
    0xa2077d7dfbb606b5UL,
    0xf3faac6faee378aeUL,
    0x0c6388b51545e883UL,
    0xd27dbb6944917b60UL,
};

// ============================================================
// M4 matrix multiply: circ(2, 3, 1, 1) circulant matrix
// [2, 3, 1, 1]
// [1, 2, 3, 1]
// [1, 1, 2, 3]
// [3, 1, 1, 2]
// Matches elliottech/plonky2 Poseidon2 (Plonky3-derived)
// ============================================================
inline void apply_m4(thread Fp* x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1]; // x0 + 2*x1 + x2 + x3
    Fp t01233 = t0123 + x[3]; // x0 + x1 + x2 + 2*x3
    // Write x[3], x[1] first (they need original x[0], x[2])
    x[3] = t01233 + x[0] + x[0]; // 3*x0 + x1 + x2 + 2*x3
    x[1] = t01123 + x[2] + x[2]; // x0 + 2*x1 + 3*x2 + x3
    x[0] = t01123 + t01;          // 2*x0 + 3*x1 + x2 + x3
    x[2] = t01233 + t23;          // x0 + x1 + 2*x2 + 3*x3
}

// External linear layer: M_E = circ(2*M4, M4, ..., M4)
// Applies M4 to each group of 4 elements, then adds column sums
inline void poseidon2_external_linear_layer(thread Fp* state) {
    apply_m4(state);
    apply_m4(state + 4);
    apply_m4(state + 8);

    Fp sums[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        sums[k] = state[k] + state[4 + k] + state[8 + k];
    }

    #pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = state[i] + sums[i % 4];
    }
}

// Internal diffusion: state[i] = state[i] * diag[i] + sum(state)
inline void poseidon2_matmul_internal(thread Fp* state) {
    Fp sum = state[0];
    #pragma unroll
    for (int i = 1; i < 12; i++) {
        sum = sum + state[i];
    }
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = state[i] * Fp(POSEIDON2_DIAG[i]) + sum;
    }
}

// Add round constants (all 12 elements, for full rounds)
inline void poseidon2_add_rc(thread Fp* state, int round) {
    uint base = round * 12;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = state[i] + Fp(POSEIDON2_RC[base + i]);
    }
}

// S-box on all 12 elements (x^7)
inline void poseidon2_sbox(thread Fp* state) {
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = state[i].pow7();
    }
}

// ============================================================
// Poseidon2 permutation (constant address space, no TG memory)
// Structure: initial M_E + 4 full + 22 partial + 4 full
// ============================================================
inline void poseidon2_permute(thread Fp* state) {
    poseidon2_external_linear_layer(state);

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        poseidon2_add_rc(state, r);
        poseidon2_sbox(state);
        poseidon2_external_linear_layer(state);
    }

    #pragma unroll
    for (int r = 4; r < 26; r++) {
        state[0] = state[0] + Fp(POSEIDON2_RC[r * 12]);
        state[0] = state[0].pow7();
        poseidon2_matmul_internal(state);
    }

    #pragma unroll
    for (int r = 26; r < 30; r++) {
        poseidon2_add_rc(state, r);
        poseidon2_sbox(state);
        poseidon2_external_linear_layer(state);
    }
}

}
#endif
