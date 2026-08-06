#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::borrow::Borrow;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::packed::PackedField;
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::reducing::ReducingGate;
use crate::gates::reducing_extension::ReducingExtensionGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

/// When verifying the composition polynomial in FRI we have to compute sums of the form
/// `(sum_0^k a^i * x_i)/d_0 + (sum_k^r a^i * y_i)/d_1`
/// The most efficient way to do this is to compute both quotient separately using Horner's method,
/// scale the second one by `a^(r-1-k)`, and add them up.
/// This struct abstract away these operations by implementing Horner's method and keeping track
/// of the number of multiplications by `a` to compute the scaling factor.
/// See <https://github.com/0xPolygonZero/plonky2/pull/69> for more details and discussions.
#[derive(Debug, Clone)]
pub struct ReducingFactor<F: Field> {
    base: F,
    count: u64,
}

impl<F: Field> ReducingFactor<F> {
    pub const fn new(base: F) -> Self {
        Self { base, count: 0 }
    }

    fn mul(&mut self, x: F) -> F {
        self.count += 1;
        self.base * x
    }

    fn mul_ext<FE, P, const D: usize>(&mut self, x: P) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.count += 1;
        // TODO: Would like to use `FE::scalar_mul`, but it doesn't work with Packed currently.
        x * FE::from_basefield(self.base)
    }

    fn mul_poly(&mut self, p: &mut PolynomialCoeffs<F>) {
        self.count += 1;
        *p *= self.base;
    }

    pub fn reduce(&mut self, iter: impl DoubleEndedIterator<Item = impl Borrow<F>>) -> F {
        iter.rev()
            .fold(F::ZERO, |acc, x| self.mul(acc) + *x.borrow())
    }

    pub fn reduce_ext<FE, P, const D: usize>(
        &mut self,
        iter: impl DoubleEndedIterator<Item = impl Borrow<P>>,
    ) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        iter.rev()
            .fold(P::ZEROS, |acc, x| self.mul_ext(acc) + *x.borrow())
    }

    pub fn reduce_polys(
        &mut self,
        polys: impl DoubleEndedIterator<Item = impl Borrow<PolynomialCoeffs<F>>>,
    ) -> PolynomialCoeffs<F> {
        polys.rev().fold(PolynomialCoeffs::empty(), |mut acc, x| {
            self.mul_poly(&mut acc);
            acc += x.borrow();
            acc
        })
    }

    pub fn reduce_polys_base<BF: Extendable<D, Extension = F>, const D: usize>(
        &mut self,
        polys: impl IntoIterator<Item = impl Borrow<PolynomialCoeffs<BF>>>,
    ) -> PolynomialCoeffs<F>
    where
        F: FieldExtension<D, BaseField = BF>,
    {
        // Fused multiply-accumulate: one extension accumulator, each base
        // coefficient read exactly once. Equivalent to the old
        // `map(mul_extension).sum()` (field arithmetic is exact and the
        // per-power scalar products are accumulated in the same order), but
        // without one degree-sized temporary allocation + two clone passes per
        // polynomial.
        let mut weighted_polys = self.base.powers().zip(polys);
        let Some((base_power, poly)) = weighted_polys.next() else {
            return PolynomialCoeffs::empty();
        };

        self.count += 1;
        let first_poly = poly.borrow();
        let mut acc: Vec<F> = first_poly
            .coeffs
            .iter()
            .map(|&c| {
                let mut a = F::ZERO;
                a += <F as FieldExtension<D>>::scalar_mul(&base_power, c);
                a
            })
            .collect();

        for (base_power, poly) in weighted_polys {
            self.count += 1;
            let coeffs = &poly.borrow().coeffs;
            if coeffs.len() > acc.len() {
                acc.resize(coeffs.len(), F::ZERO);
            }
            for (a, &c) in acc.iter_mut().zip(coeffs.iter()) {
                *a += <F as FieldExtension<D>>::scalar_mul(&base_power, c);
            }
        }
        PolynomialCoeffs::new(acc)
    }

    pub fn shift(&mut self, x: F) -> F {
        let tmp = self.base.exp_u64(self.count) * x;
        self.count = 0;
        tmp
    }

    pub fn shift_poly(&mut self, p: &mut PolynomialCoeffs<F>) {
        *p *= self.base.exp_u64(self.count);
        self.count = 0;
    }

    /// Returns the factor `shift_poly` would multiply by (`base^count`) and
    /// resets the count, letting callers fuse the multiply into another pass.
    pub fn shift_factor(&mut self) -> F {
        let tmp = self.base.exp_u64(self.count);
        self.count = 0;
        tmp
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct ReducingFactorTarget<const D: usize> {
    base: ExtensionTarget<D>,
    count: u64,
}

impl<const D: usize> ReducingFactorTarget<D> {
    pub const fn new(base: ExtensionTarget<D>) -> Self {
        Self { base, count: 0 }
    }

    /// Reduces a vector of `Target`s using `ReducingGate`s.
    pub fn reduce_base<F>(
        &mut self,
        terms: &[Target],
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let l = terms.len();

        // For small reductions, use an arithmetic gate.
        if l <= ArithmeticExtensionGate::<D>::new_from_config(&builder.config).num_ops + 1 {
            let terms_ext = terms
                .iter()
                .map(|&t| builder.convert_to_ext(t))
                .collect::<Vec<_>>();
            return self.reduce_arithmetic(&terms_ext, builder);
        }

        let max_coeffs_len = ReducingGate::<D>::max_coeffs_len(
            builder.config.num_wires,
            builder.config.num_routed_wires,
        );
        self.count += l as u64;
        let zero = builder.zero();
        let zero_ext = builder.zero_extension();
        let mut acc = zero_ext;
        let mut reversed_terms = terms.to_vec();
        while !reversed_terms.len().is_multiple_of(max_coeffs_len) {
            reversed_terms.push(zero);
        }
        reversed_terms.reverse();
        for chunk in reversed_terms.chunks_exact(max_coeffs_len) {
            let gate = ReducingGate::new(max_coeffs_len);
            let row = builder.add_gate(gate.clone(), vec![]);

            builder.connect_extension(
                self.base,
                ExtensionTarget::from_range(row, ReducingGate::<D>::wires_alpha()),
            );
            builder.connect_extension(
                acc,
                ExtensionTarget::from_range(row, ReducingGate::<D>::wires_old_acc()),
            );
            for (&t, c) in chunk.iter().zip(gate.wires_coeffs()) {
                builder.connect(t, Target::wire(row, c));
            }

            acc = ExtensionTarget::from_range(row, ReducingGate::<D>::wires_output());
        }

        acc
    }

    /// Reduces a vector of `ExtensionTarget`s using `ReducingExtensionGate`s.
    pub fn reduce<F>(
        &mut self,
        terms: &[ExtensionTarget<D>], // Could probably work with a `DoubleEndedIterator` too.
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let l = terms.len();

        // For small reductions, use an arithmetic gate.
        if l <= ArithmeticExtensionGate::<D>::new_from_config(&builder.config).num_ops + 1 {
            return self.reduce_arithmetic(terms, builder);
        }

        let max_coeffs_len = ReducingExtensionGate::<D>::max_coeffs_len(
            builder.config.num_wires,
            builder.config.num_routed_wires,
        );
        self.count += l as u64;
        let zero_ext = builder.zero_extension();
        let mut acc = zero_ext;
        let mut reversed_terms = terms.to_vec();
        while !reversed_terms.len().is_multiple_of(max_coeffs_len) {
            reversed_terms.push(zero_ext);
        }
        reversed_terms.reverse();
        for chunk in reversed_terms.chunks_exact(max_coeffs_len) {
            let gate = ReducingExtensionGate::new(max_coeffs_len);
            let row = builder.add_gate(gate.clone(), vec![]);

            builder.connect_extension(
                self.base,
                ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_alpha()),
            );
            builder.connect_extension(
                acc,
                ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_old_acc()),
            );
            for (i, &t) in chunk.iter().enumerate() {
                builder.connect_extension(
                    t,
                    ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_coeff(i)),
                );
            }

            acc = ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_output());
        }

        acc
    }

    /// Reduces a vector of `ExtensionTarget`s using `ArithmeticGate`s.
    fn reduce_arithmetic<F>(
        &mut self,
        terms: &[ExtensionTarget<D>],
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        self.count += terms.len() as u64;
        terms
            .iter()
            .rev()
            .fold(builder.zero_extension(), |acc, &et| {
                builder.mul_add_extension(self.base, acc, et)
            })
    }

    pub fn shift<F>(
        &mut self,
        x: ExtensionTarget<D>,
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let zero_ext = builder.zero_extension();
        let exp = if x == zero_ext {
            // The result will get zeroed out, so don't actually compute the exponentiation.
            zero_ext
        } else {
            builder.exp_u64_extension(self.base, self.count)
        };

        self.count = 0;
        builder.mul_extension(exp, x)
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::quadratic::QuadraticExtension;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, PrimeField64, Sample};
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    type TestBaseField = GoldilocksField;
    type TestExtension = QuadraticExtension<TestBaseField>;

    fn reduce_polys_base_reference(
        reducer: &mut ReducingFactor<TestExtension>,
        polys: &[PolynomialCoeffs<TestBaseField>],
    ) -> PolynomialCoeffs<TestExtension> {
        let mut acc = Vec::new();
        for (base_power, poly) in reducer.base.powers().zip(polys) {
            reducer.count += 1;
            let coeffs = &poly.coeffs;
            if coeffs.len() > acc.len() {
                acc.resize(coeffs.len(), TestExtension::ZERO);
            }
            for (a, &c) in acc.iter_mut().zip(coeffs.iter()) {
                *a += <TestExtension as FieldExtension<2>>::scalar_mul(&base_power, c);
            }
        }
        PolynomialCoeffs::new(acc)
    }

    fn next_deterministic_u64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn make_raw_polys(lengths: &[usize], state: &mut u64) -> Vec<PolynomialCoeffs<TestBaseField>> {
        let fixed = [
            0,
            1,
            TestBaseField::ORDER - 1,
            TestBaseField::ORDER,
            TestBaseField::ORDER + 1,
            u64::MAX,
        ];
        lengths
            .iter()
            .enumerate()
            .map(|(poly_index, &len)| {
                let coeffs = (0..len)
                    .map(|coeff_index| {
                        let selector = poly_index.wrapping_mul(257).wrapping_add(coeff_index);
                        let raw = if selector % 3 == 0 {
                            fixed[selector % fixed.len()]
                        } else {
                            next_deterministic_u64(state)
                        };
                        GoldilocksField(raw)
                    })
                    .collect();
                PolynomialCoeffs::new(coeffs)
            })
            .collect()
    }

    fn assert_extension_raw_eq(left: TestExtension, right: TestExtension) {
        for (left_limb, right_limb) in left.0.into_iter().zip(right.0) {
            assert_eq!(
                left_limb.to_noncanonical_u64(),
                right_limb.to_noncanonical_u64()
            );
        }
    }

    fn assert_polynomial_raw_eq(
        left: &PolynomialCoeffs<TestExtension>,
        right: &PolynomialCoeffs<TestExtension>,
    ) {
        assert_eq!(left.coeffs.len(), right.coeffs.len());
        for (&left_coeff, &right_coeff) in left.coeffs.iter().zip(&right.coeffs) {
            assert_extension_raw_eq(left_coeff, right_coeff);
        }
    }

    #[test]
    fn first_term_seed_matches_reference_raw() {
        let mut state = 0x4652_495f_5345_4544;
        let mut alphas = vec![
            TestExtension::ZERO,
            TestExtension::ONE,
            QuadraticExtension([GoldilocksField(7), GoldilocksField(11)]),
            QuadraticExtension([
                GoldilocksField(TestBaseField::ORDER),
                GoldilocksField(TestBaseField::ORDER + 1),
            ]),
        ];
        for _ in 0..3 {
            alphas.push(QuadraticExtension([
                GoldilocksField(next_deterministic_u64(&mut state)),
                GoldilocksField(next_deterministic_u64(&mut state)),
            ]));
        }

        let length_patterns = vec![
            vec![],
            vec![0],
            vec![1],
            vec![0, 1],
            vec![1, 0],
            vec![32, 32],
            vec![33, 1, 257],
            vec![1, 257, 0, 31],
            vec![1; 17],
            vec![256; 260],
        ];

        for lengths in length_patterns {
            let polys = make_raw_polys(&lengths, &mut state);
            let followup = make_raw_polys(&[3, 0, 5], &mut state);
            for &alpha in &alphas {
                let mut reference = ReducingFactor::new(alpha);
                let mut candidate = ReducingFactor::new(alpha);

                let reference_poly = reduce_polys_base_reference(&mut reference, &polys);
                let candidate_poly = candidate.reduce_polys_base::<TestBaseField, 2>(&polys);
                assert_polynomial_raw_eq(&reference_poly, &candidate_poly);
                assert_eq!(reference.count, candidate.count);

                let reference_shift = reference.shift_factor();
                let candidate_shift = candidate.shift_factor();
                assert_extension_raw_eq(reference_shift, candidate_shift);
                assert_eq!(reference.count, 0);
                assert_eq!(candidate.count, 0);

                let reference_second = reduce_polys_base_reference(&mut reference, &followup);
                let candidate_second = candidate.reduce_polys_base::<TestBaseField, 2>(&followup);
                assert_polynomial_raw_eq(&reference_second, &candidate_second);
                assert_eq!(reference.count, candidate.count);
                assert_extension_raw_eq(reference.shift_factor(), candidate.shift_factor());
            }
        }
    }

    fn test_reduce_gadget_base(n: usize) -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let alpha = FF::rand();
        let vs = F::rand_vec(n);

        let manual_reduce = ReducingFactor::new(alpha).reduce(vs.iter().map(|&v| FF::from(v)));
        let manual_reduce = builder.constant_extension(manual_reduce);

        let mut alpha_t = ReducingFactorTarget::new(builder.constant_extension(alpha));
        let vs_t = builder.add_virtual_targets(vs.len());
        for (&v, &v_t) in vs.iter().zip(&vs_t) {
            pw.set_target(v_t, v)?;
        }
        let circuit_reduce = alpha_t.reduce_base(&vs_t, &mut builder);

        builder.connect_extension(manual_reduce, circuit_reduce);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    fn test_reduce_gadget(n: usize) -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let alpha = FF::rand();
        let vs = (0..n).map(FF::from_canonical_usize).collect::<Vec<_>>();

        let manual_reduce = ReducingFactor::new(alpha).reduce(vs.iter());
        let manual_reduce = builder.constant_extension(manual_reduce);

        let mut alpha_t = ReducingFactorTarget::new(builder.constant_extension(alpha));
        let vs_t = builder.add_virtual_extension_targets(vs.len());
        pw.set_extension_targets(&vs_t, &vs)?;
        let circuit_reduce = alpha_t.reduce(&vs_t, &mut builder);

        builder.connect_extension(manual_reduce, circuit_reduce);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_reduce_gadget_even() -> Result<()> {
        test_reduce_gadget(10)
    }

    #[test]
    fn test_reduce_gadget_odd() -> Result<()> {
        test_reduce_gadget(11)
    }

    #[test]
    fn test_reduce_gadget_base_100() -> Result<()> {
        test_reduce_gadget_base(100)
    }

    #[test]
    fn test_reduce_gadget_100() -> Result<()> {
        test_reduce_gadget(100)
    }
}
