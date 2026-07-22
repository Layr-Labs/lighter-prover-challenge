/*
 * Goldilocks Quadratic Extension Field (FpE)
 * Extension: F_p[α] / (α² - 7)  where p = 2^64 - 2^32 + 1
 * Element representation: c0 + c1·α
 */
#ifndef goldilocks_ext2_field
#define goldilocks_ext2_field

#include "goldilocks.metal"

namespace GoldilocksField
{
    // Non-residue W = 7 (α² = 7)
    static const constant unsigned long W_VAL = 7;
    // p - 1, used for negation: -x = (p-1)*x in GF(p)
    static const constant unsigned long NEG_ONE = 18446744069414584320UL;

    // Subtraction: a - b = a + (-b) = a + (p-1)*b
    inline Fp fp_sub(Fp a, Fp b) {
        return a + Fp(NEG_ONE) * b;
    }

    // Negation: -a = (p-1)*a
    inline Fp fp_neg(Fp a) {
        return Fp(NEG_ONE) * a;
    }

    class FpE
    {
    public:
        Fp c0, c1;  // c0 + c1·α

        FpE() = default;
        FpE(Fp a, Fp b) : c0(a), c1(b) {}
        explicit FpE(Fp a) : c0(a), c1(Fp(0)) {}
        explicit FpE(unsigned long a) : c0(Fp(a)), c1(Fp(0)) {}

        static FpE zero() { return FpE(Fp(0), Fp(0)); }
        static FpE one() { return FpE(Fp(1), Fp(0)); }

        FpE operator+(FpE r) const { return FpE(c0 + r.c0, c1 + r.c1); }

        FpE operator-(FpE r) const {
            return FpE(fp_sub(c0, r.c0), fp_sub(c1, r.c1));
        }

        FpE operator*(FpE r) const {
            // (a + bα)(c + dα) = (ac + W·bd) + (ad + bc)α
            Fp ac = c0 * r.c0;
            Fp bd = c1 * r.c1;
            Fp ad_bc = c0 * r.c1 + c1 * r.c0;
            Fp w_bd = Fp(W_VAL) * bd;
            return FpE(ac + w_bd, ad_bc);
        }

        // Scalar multiplication: FpE * Fp
        FpE scalar_mul(Fp s) const {
            return FpE(c0 * s, c1 * s);
        }

        // Negate
        FpE neg() const {
            return FpE(fp_neg(c0), fp_neg(c1));
        }

        // Sub one: self - 1
        FpE sub_one() const {
            return FpE(fp_sub(c0, Fp(1)), c1);
        }
    };
}

#endif /* goldilocks_ext2_field */
