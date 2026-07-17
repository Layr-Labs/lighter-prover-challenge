// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::iop::target::BoolTarget;
use plonky2::plonk::config::AlgebraicHasher;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::poseidon2::{Poseidon2, Poseidon2Hash};
use crate::utils::CircuitBuilderUtils;

pub trait CircuitBuilderHashUtils<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    fn conditional_assert_eq_hash(
        &mut self,
        is_enabled: BoolTarget,
        a: &HashOutTarget,
        b: &HashOutTarget,
    );

    #[must_use]
    fn is_equal_hash(&mut self, a: &HashOutTarget, b: &HashOutTarget) -> BoolTarget;

    fn hash_two_to_one(&mut self, left: &HashOutTarget, right: &HashOutTarget) -> HashOutTarget;
    fn hash_two_to_one_swap(
        &mut self,
        left: &HashOutTarget,
        right: &HashOutTarget,
        swap: BoolTarget,
    ) -> HashOutTarget
    where
        F: Poseidon2;
    fn hash_n_to_one(&mut self, elements: &[HashOutTarget]) -> HashOutTarget;

    #[must_use]
    fn select_hash(
        &mut self,
        b: BoolTarget,
        h0: &HashOutTarget,
        h1: &HashOutTarget,
    ) -> HashOutTarget;

    fn zero_hash_out(&mut self) -> HashOutTarget;

    #[must_use]
    fn is_zero_hash_out(&mut self, a: &HashOutTarget) -> BoolTarget;
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> CircuitBuilderHashUtils<F, D>
    for Builder<F, D>
{
    fn is_equal_hash(&mut self, a: &HashOutTarget, b: &HashOutTarget) -> BoolTarget {
        let assertions = [
            self.is_equal(a.elements[0], b.elements[0]),
            self.is_equal(a.elements[1], b.elements[1]),
            self.is_equal(a.elements[2], b.elements[2]),
            self.is_equal(a.elements[3], b.elements[3]),
        ];

        self.multi_and(&assertions)
    }

    #[track_caller]
    fn conditional_assert_eq_hash(
        &mut self,
        is_enabled: BoolTarget,
        a: &HashOutTarget,
        b: &HashOutTarget,
    ) {
        assert!(a.elements.len() == NUM_HASH_OUT_ELTS);
        assert!(b.elements.len() == NUM_HASH_OUT_ELTS);

        for i in 0..NUM_HASH_OUT_ELTS {
            self.conditional_assert_eq(is_enabled, a.elements[i], b.elements[i])
        }
    }

    /// Takes two hash outputs and returns their hash.
    fn hash_two_to_one(&mut self, left: &HashOutTarget, right: &HashOutTarget) -> HashOutTarget {
        let _false = self._false();

        self.hash_two_to_one_swap(left, right, _false)
    }

    /// Takes two hash outputs and swap left and right if `swap` is true then returns their hash.
    fn hash_two_to_one_swap(
        &mut self,
        left: &HashOutTarget,
        right: &HashOutTarget,
        swap: BoolTarget,
    ) -> HashOutTarget
    where
        F: Poseidon2,
    {
        let zero = self.zero();

        let mut perm_inputs =
            <Poseidon2Hash as AlgebraicHasher<F>>::AlgebraicPermutation::default();
        perm_inputs.set_from_slice(&left.elements, 0);
        perm_inputs.set_from_slice(&right.elements, NUM_HASH_OUT_ELTS);
        // Ensure the rest of the state, if any, is zero:
        perm_inputs.set_from_iter(core::iter::repeat(zero), 2 * NUM_HASH_OUT_ELTS);
        let perm_outs = Poseidon2Hash::permute_swapped(perm_inputs, swap, &mut self.builder);
        let hash_outs = perm_outs.squeeze()[0..NUM_HASH_OUT_ELTS]
            .try_into()
            .unwrap();

        HashOutTarget {
            elements: hash_outs,
        }
    }

    /// Takes list of hash outputs and returns their hash.
    fn hash_n_to_one(&mut self, elements: &[HashOutTarget]) -> HashOutTarget {
        assert!(!elements.is_empty());

        if elements.len() == 1 {
            return elements[0];
        }

        let mut result = self.hash_two_to_one(&elements[0], &elements[1]);

        for i in 2..elements.len() {
            result = self.hash_two_to_one(&result, &elements[i]);
        }

        result
    }

    fn select_hash(
        &mut self,
        b: BoolTarget,
        h0: &HashOutTarget,
        h1: &HashOutTarget,
    ) -> HashOutTarget {
        HashOutTarget {
            elements: core::array::from_fn(|i| self.select(b, h0.elements[i], h1.elements[i])),
        }
    }

    fn zero_hash_out(&mut self) -> HashOutTarget {
        HashOutTarget {
            elements: [self.zero(); NUM_HASH_OUT_ELTS],
        }
    }

    fn is_zero_hash_out(&mut self, a: &HashOutTarget) -> BoolTarget {
        let assertions = [
            self.is_zero(a.elements[0]),
            self.is_zero(a.elements[1]),
            self.is_zero(a.elements[2]),
            self.is_zero(a.elements[3]),
        ];

        self.multi_and(&assertions)
    }
}
