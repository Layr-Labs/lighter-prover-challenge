// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gadgets/interleaved_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::builder::Builder;
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::uint::u32::gates::interleave_u32::U32InterleaveGate;
use crate::uint::u32::gates::uninterleave_to_b32::UninterleaveToB32Gate;
use crate::uint::u32::gates::uninterleave_to_u32::UninterleaveToU32Gate;

#[derive(Debug, Default, Clone, Copy)]
pub struct B32Target(pub Target);

/// Efficient binary operations for U32Target
/// Use a combination of arithmetic_u32 and a new interleaved representation
/// The interleaved representation allows for efficient and + xor (using 1 add)
pub trait CircuitBuilderB32<F: RichField + Extendable<D>, const D: usize> {
    // efficient methods that use arithmetic_u32
    fn not_u32(&mut self, x: U32Target) -> U32Target;
    fn lsh_u32(&mut self, x: U32Target, n: u8) -> U32Target;
    fn rsh_u32(&mut self, x: U32Target, n: u8) -> U32Target;
    fn lrot_u32(&mut self, x: U32Target, n: u8) -> U32Target;
    fn rrot_u32(&mut self, x: U32Target, n: u8) -> U32Target;
    fn conditional_u32(&mut self, x: U32Target, y: U32Target, x_or_y: BoolTarget) -> U32Target;

    // see U32InterleaveGate for documentation
    fn interleave_u32(&mut self, x: U32Target) -> B32Target;
    fn uninterleave_to_u32(&mut self, x: Target) -> (U32Target, U32Target);
    fn uninterleave_to_b32(&mut self, x: Target) -> (B32Target, B32Target);

    fn and_xor_u32(&mut self, x: U32Target, y: U32Target) -> (B32Target, B32Target);
    fn and_xor_b32(&mut self, x: B32Target, y: B32Target) -> (B32Target, B32Target);
    fn and_xor_b32_to_u32(&mut self, x: B32Target, y: B32Target) -> (U32Target, U32Target);
    fn and_xor_u32_to_u32(&mut self, x: U32Target, y: U32Target) -> (U32Target, U32Target);

    fn xor_u32(&mut self, x: U32Target, y: U32Target) -> U32Target;
    fn and_u32(&mut self, x: U32Target, y: U32Target) -> U32Target;

    fn not_u64(&mut self, x: &[U32Target; 2]) -> [U32Target; 2];
    fn lrot_u64(&mut self, a: &[U32Target; 2], n: u8) -> [U32Target; 2];
    fn xor_u64(&mut self, x: &[U32Target; 2], y: &[U32Target; 2]) -> [U32Target; 2];
    fn and_u64(&mut self, x: &[U32Target; 2], y: &[U32Target; 2]) -> [U32Target; 2];
    fn conditional_u64(
        &mut self,
        x: &[U32Target; 2],
        y: &[U32Target; 2],
        x_or_y: BoolTarget,
    ) -> [U32Target; 2];
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderB32<F, D> for Builder<F, D> {
    // not := 0xFFFFFFFF - x
    fn not_u32(&mut self, a: U32Target) -> U32Target {
        let zero = self.zero_u32();
        let ff = self.constant_u32(0xFFFFFFFF);
        self.sub_u32(ff, a, zero).0
    }

    // left shift := mul by power of 2, keep lower word
    fn lsh_u32(&mut self, a: U32Target, n: u8) -> U32Target {
        let power_of_two = self.constant_u32(0x1 << n);
        self.mul_u32(a, power_of_two).0
    }

    // right shift := mul by power of 2, keep higher word
    fn rsh_u32(&mut self, a: U32Target, n: u8) -> U32Target {
        if n == 0 {
            return a;
        }
        let power_of_two = self.constant_u32(0x1 << (32 - n));
        self.mul_u32(a, power_of_two).1
    }

    // left rotate := mul by power of 2, adding the two words (they don't overlap)
    fn lrot_u32(&mut self, a: U32Target, n: u8) -> U32Target {
        let power_of_two = self.constant_u32(0x1 << n);
        let (lo, hi) = self.mul_u32(a, power_of_two);
        self.add_u32(lo, hi).0
    }

    // right rotate := left rotate of 32-n
    fn rrot_u32(&mut self, a: U32Target, n: u8) -> U32Target {
        self.lrot_u32(a, 32 - n)
    }

    // convert U32Target -> B32Target by interleaving the bits
    fn interleave_u32(&mut self, x: U32Target) -> B32Target {
        if let Some(&result) = self.u32_interleave_results.get(&x) {
            return result;
        }

        let gate = U32InterleaveGate::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_x(copy)), x.0);

        let result = B32Target(Target::wire(row, gate.wire_ith_x_interleaved(copy)));

        self.u32_interleave_results.insert(x, result);

        result
    }

    fn uninterleave_to_u32(&mut self, x_dirty: Target) -> (U32Target, U32Target) {
        let gate = UninterleaveToU32Gate::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(
            Target::wire(row, gate.wire_ith_x_interleaved(copy)),
            x_dirty,
        );

        let x_evens = U32Target(Target::wire(row, gate.wire_ith_x_evens(copy)));
        let x_odds = U32Target(Target::wire(row, gate.wire_ith_x_odds(copy)));

        (x_evens, x_odds)
    }

    fn uninterleave_to_b32(&mut self, x_dirty: Target) -> (B32Target, B32Target) {
        let gate = UninterleaveToB32Gate::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(
            Target::wire(row, gate.wire_ith_x_interleaved(copy)),
            x_dirty,
        );

        let x_evens = B32Target(Target::wire(row, gate.wire_ith_x_evens(copy)));
        let x_odds = B32Target(Target::wire(row, gate.wire_ith_x_odds(copy)));

        (x_evens, x_odds)
    }

    fn and_xor_b32(&mut self, x: B32Target, y: B32Target) -> (B32Target, B32Target) {
        let sum = self.add(x.0, y.0);
        self.uninterleave_to_b32(sum)
    }

    fn and_xor_u32(&mut self, x: U32Target, y: U32Target) -> (B32Target, B32Target) {
        let x = self.interleave_u32(x);
        let y = self.interleave_u32(y);
        self.and_xor_b32(x, y)
    }

    fn and_xor_b32_to_u32(&mut self, x: B32Target, y: B32Target) -> (U32Target, U32Target) {
        let sum = self.add(x.0, y.0);
        self.uninterleave_to_u32(sum)
    }

    // x -> X [0 x 0 x 0 x 0 x]
    // y -> Y [0 y 0 y 0 y 0 y]
    // X+Y
    fn and_xor_u32_to_u32(&mut self, x: U32Target, y: U32Target) -> (U32Target, U32Target) {
        let x = self.interleave_u32(x);
        let y = self.interleave_u32(y);
        self.and_xor_b32_to_u32(x, y)
    }

    fn and_u32(&mut self, x: U32Target, y: U32Target) -> U32Target {
        self.and_xor_u32_to_u32(x, y).0
    }

    fn xor_u32(&mut self, x: U32Target, y: U32Target) -> U32Target {
        self.and_xor_u32_to_u32(x, y).1
    }

    fn lrot_u64(&mut self, a: &[U32Target; 2], n: u8) -> [U32Target; 2] {
        let (lo, hi) = if n < 32 { (a[0], a[1]) } else { (a[1], a[0]) };

        let power_of_two = self.constant_u32(0x1 << (n % 32));
        let (lo0, hi0) = self.mul_u32(lo, power_of_two);
        let (lo1, hi1) = self.mul_add_u32(hi, power_of_two, hi0);
        [self.add_u32(lo0, hi1).0, lo1]
    }

    fn xor_u64(&mut self, x: &[U32Target; 2], y: &[U32Target; 2]) -> [U32Target; 2] {
        [self.xor_u32(x[0], y[0]), self.xor_u32(x[1], y[1])]
    }

    fn and_u64(&mut self, x: &[U32Target; 2], y: &[U32Target; 2]) -> [U32Target; 2] {
        [self.and_u32(x[0], y[0]), self.and_u32(x[1], y[1])]
    }

    fn not_u64(&mut self, x: &[U32Target; 2]) -> [U32Target; 2] {
        [self.not_u32(x[0]), self.not_u32(x[1])]
    }

    // return if z { x } else { y }
    fn conditional_u32(&mut self, x: U32Target, y: U32Target, z: BoolTarget) -> U32Target {
        let not_z = U32Target(self.not(z).target);
        let maybe_x = self.mul_u32(x, U32Target(z.target)).0;
        self.mul_add_u32(y, not_z, maybe_x).0
    }

    fn conditional_u64(
        &mut self,
        x: &[U32Target; 2],
        y: &[U32Target; 2],
        z: BoolTarget,
    ) -> [U32Target; 2] {
        [
            self.conditional_u32(x[0], y[0], z),
            self.conditional_u32(x[1], y[1], z),
        ]
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;
    use crate::uint::u32::witness::WitnessU32;

    #[test]
    /// One hard-coded test case for now. Are there any weird edge cases that should also be explicitly covered?
    pub fn test_interleave_u32() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 135,
            ..CircuitConfig::standard_recursion_config()
        };

        let pw = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x_value: u32 = 0b1111_1111_1111_1111_1111_1111_1111_1100;
        let x_interleaved_value_expected: u64 =
            0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0000;

        let x_target = builder.constant_u32(x_value);
        let x_interleaved_target = builder.interleave_u32(x_target);

        builder.register_public_input(x_interleaved_target.0);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();

        let x_interleaved_value_actual = proof.public_inputs[0].to_canonical_u64();
        assert!(x_interleaved_value_actual == x_interleaved_value_expected);

        data.verify(proof)
    }

    #[test]
    /// One hard-coded test case for now. Are there any weird edge cases that should also be explicitly covered?
    pub fn test_uninterleave_to_u32() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x_interleaved_value: u64 =
            0b1111_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101;
        let x_evens_value_expected: u64 = 0b1100_0000_0000_0000_0000_0000_0000_0000;
        let x_odds_value_expected: u64 = 0b1111_1111_1111_1111_1111_1111_1111_1111;

        let x_interleaved_target = builder.constant(F::from_canonical_u64(x_interleaved_value));
        let (x_evens_target, x_odds_target) = builder.uninterleave_to_u32(x_interleaved_target);
        builder.register_public_input(x_evens_target.0);
        builder.register_public_input(x_odds_target.0);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();

        let x_evens_value_actual = proof.public_inputs[0].to_canonical_u64();
        let x_odds_value_actual = proof.public_inputs[1].to_canonical_u64();
        assert_eq!(x_evens_value_expected, x_evens_value_actual);
        assert_eq!(x_odds_value_expected, x_odds_value_actual);

        data.verify(proof)
    }

    #[test]
    /// One hard-coded test case for now. Are there any weird edge cases that should also be explicitly covered?
    pub fn test_uninterleave_to_b32() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x_interleaved_value: u64 =
            0b1111_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101;
        let x_evens_value_expected: u64 =
            0b0101_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000;
        let x_odds_value_expected: u64 =
            0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101;

        let x_interleaved_target = builder.constant(F::from_canonical_u64(x_interleaved_value));
        let (x_evens_target, x_odds_target) = builder.uninterleave_to_b32(x_interleaved_target);
        builder.register_public_input(x_evens_target.0);
        builder.register_public_input(x_odds_target.0);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();

        let x_evens_value_actual = proof.public_inputs[0].to_canonical_u64();
        let x_odds_value_actual = proof.public_inputs[1].to_canonical_u64();
        assert_eq!(x_evens_value_expected, x_evens_value_actual);
        assert_eq!(x_odds_value_expected, x_odds_value_actual);

        data.verify(proof)
    }

    #[test]
    fn test_not_u32() {
        #[rustfmt::skip]
        let tests = [
            0x0u32,
            0x1,
            0x01234567,
            0x89abcdef,
            0xffffffff
        ];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);
        let target_input = builder.add_virtual_u32_target_unsafe();
        let target_output = builder.add_virtual_u32_target_unsafe();
        let not_target = builder.not_u32(target_input);
        builder.connect_u32(not_target, target_output);
        let data = builder.build::<C>();

        for t in tests {
            let input = t;
            let output = !input;

            // test circuit
            let mut pw = PartialWitness::new();
            pw.set_u32_target(target_input, input).unwrap();
            pw.set_u32_target(target_output, output).unwrap();

            let proof = data.prove(pw).unwrap();
            assert!(data.verify(proof).is_ok());
        }
    }

    #[test]
    fn test_lrot_u32() {
        let tests = [0x0u32, 0x1, 0x01234567, 0x89abcdef, 0xffffffff];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        for n in 0..32u32 {
            let config = CircuitConfig::standard_recursion_config();
            let mut builder = Builder::<F, D>::new(config);
            let target_input = builder.add_virtual_u32_target_unsafe();
            let target_output = builder.add_virtual_u32_target_unsafe();
            let rot_target = builder.lrot_u32(target_input, n as u8);
            builder.connect_u32(rot_target, target_output);
            let data = builder.build::<C>();

            for t in tests {
                let input = t;
                let output = (input << n) | (input.wrapping_shr(32 - n));

                // test circuit
                let mut pw = PartialWitness::new();
                pw.set_u32_target(target_input, input).unwrap();
                pw.set_u32_target(target_output, output).unwrap();

                let proof = data.prove(pw).unwrap();
                assert!(data.verify(proof).is_ok());
            }
        }
    }

    #[test]
    fn test_lsh_u32() {
        let tests = [0x0u32, 0x1, 0x01234567, 0x89abcdef, 0xffffffff];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        for n in 0..32 {
            let config = CircuitConfig::standard_recursion_config();
            let mut builder = Builder::<F, D>::new(config);
            let target_input = builder.add_virtual_u32_target_unsafe();
            let target_output = builder.add_virtual_u32_target_unsafe();
            let shift_target = builder.lsh_u32(target_input, n);
            builder.connect_u32(shift_target, target_output);
            let data = builder.build::<C>();

            for t in tests {
                let input = t;
                let output = input << n;

                // test circuit
                let mut pw = PartialWitness::new();
                pw.set_u32_target(target_input, input).unwrap();
                pw.set_u32_target(target_output, output).unwrap();

                let proof = data.prove(pw).unwrap();
                assert!(data.verify(proof).is_ok());
            }
        }
    }

    #[test]
    fn test_rsh_u32() {
        let tests = [0x0u32, 0x1, 0x01234567, 0x89abcdef, 0xffffffff];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        for n in 0..32 {
            let config = CircuitConfig::standard_recursion_config();
            let mut builder = Builder::<F, D>::new(config);
            let target_input = builder.add_virtual_u32_target_unsafe();
            let target_output = builder.add_virtual_u32_target_unsafe();
            let shift_target = builder.rsh_u32(target_input, n);
            builder.connect_u32(shift_target, target_output);
            let data = builder.build::<C>();

            for t in tests {
                let input = t;
                let output = input >> n;

                // test circuit
                let mut pw = PartialWitness::new();
                pw.set_u32_target(target_input, input).unwrap();
                pw.set_u32_target(target_output, output).unwrap();

                let proof = data.prove(pw).unwrap();
                assert!(data.verify(proof).is_ok());
            }
        }
    }

    #[test]
    fn test_xor_u32() {
        #[rustfmt::skip]
        let tests = [
            [0x0u32, 0x0],
            [0x01234567, 0x01234567],
            [0x01234567, 0x0],
            [0x01234567, 0x89abcdef],
            [0xffffffff, 0xffffffff],
        ];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);
        let target_input1 = builder.add_virtual_u32_target_unsafe();
        let target_input2 = builder.add_virtual_u32_target_unsafe();
        let target_output = builder.add_virtual_u32_target_unsafe();
        let xor_target = builder.xor_u32(target_input1, target_input2);
        builder.connect_u32(xor_target, target_output);
        let data = builder.build::<C>();

        for t in tests {
            let input1 = t[0];
            let input2 = t[1];
            let output = input1 ^ input2;

            // test circuit
            let mut pw = PartialWitness::new();
            pw.set_u32_target(target_input1, input1).unwrap();
            pw.set_u32_target(target_input2, input2).unwrap();
            pw.set_u32_target(target_output, output).unwrap();

            let proof = data.prove(pw).unwrap();
            assert!(data.verify(proof).is_ok());
        }
    }

    #[test]
    fn test_and_u32() {
        #[rustfmt::skip]
        let tests = [
            [0x0u32, 0x0],
            [0x01234567, 0x01234567],
            [0x01234567, 0x0],
            [0x01234567, 0x89abcdef],
        ];

        // build circuit once
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);
        let target_input1 = builder.add_virtual_u32_target_unsafe();
        let target_input2 = builder.add_virtual_u32_target_unsafe();
        let target_output = builder.add_virtual_u32_target_unsafe();
        let and_target = builder.and_u32(target_input1, target_input2);
        builder.connect_u32(and_target, target_output);
        let data = builder.build::<C>();

        for t in tests {
            let input1 = t[0];
            let input2 = t[1];
            let output = input1 & input2;

            // test circuit
            let mut pw = PartialWitness::new();
            pw.set_u32_target(target_input1, input1).unwrap();
            pw.set_u32_target(target_input2, input2).unwrap();
            pw.set_u32_target(target_output, output).unwrap();

            let proof = data.prove(pw).unwrap();
            assert!(data.verify(proof).is_ok());
        }
    }
}
