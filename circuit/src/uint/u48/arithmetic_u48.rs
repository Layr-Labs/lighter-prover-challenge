// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gadgets/arithmetic_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use super::subtraction_u48::U48SubtractionGate;
use crate::builder::Builder;
pub trait CircuitBuilderU48<F: RichField + Extendable<D>, const D: usize> {
    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    // Inputs are NOT range-checked.
    fn sub_u48(&mut self, x: Target, y: Target, borrow: Target) -> (Target, Target);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderU48<F, D> for Builder<F, D> {
    fn sub_u48(&mut self, x: Target, y: Target, borrow: Target) -> (Target, Target) {
        let gate = U48SubtractionGate::<F, D>::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_input_x(copy)), x);
        self.connect(Target::wire(row, gate.wire_ith_input_y(copy)), y);
        self.connect(Target::wire(row, gate.wire_ith_input_borrow(copy)), borrow);

        let output_result = Target::wire(row, gate.wire_ith_output_result(copy));
        let output_borrow = Target::wire(row, gate.wire_ith_output_borrow(copy));

        (output_result, output_borrow)
    }
}
