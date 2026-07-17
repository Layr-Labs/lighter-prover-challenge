// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use log::warn;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;

use super::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use super::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;

pub trait CircuitBuilderBiguintSubtractiveComparison<F: RichField + Extendable<D>, const D: usize> {
    fn cmp_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> SignTarget;

    fn is_lt_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget;
    fn is_lte_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget;
    fn is_gt_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget;
    fn is_gte_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget;

    fn is_lte_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget;

    fn conditional_assert_lt_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    );
    fn conditional_assert_lte_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    );

    fn max_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    fn max_biguint_multiple(&mut self, targets: &[&BigUintTarget]) -> BigUintTarget;
    fn min_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    fn min_biguint_multiple(&mut self, targets: &[&BigUintTarget]) -> BigUintTarget;

    fn conditional_assert_zero_biguint(&mut self, is_enabled: BoolTarget, a: &BigUintTarget);
    fn conditional_assert_not_zero_biguint(&mut self, is_enabled: BoolTarget, a: &BigUintTarget);
    fn conditional_assert_eq_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    );
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBiguintSubtractiveComparison<F, D>
    for Builder<F, D>
{
    fn is_lte_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget {
        let a_is_negative = self.is_sign_negative(a.sign);
        let b_is_negative = self.is_sign_negative(b.sign);

        // Positive - Positive -> a_abs_lte_b_abs
        // Positive - Zero     -> a_abs_lte_b_abs
        // Zero - Positive -> a_abs_lte_b_abs
        // Zero - Zero     -> a_abs_lte_b_abs
        let mut result_if_a_abs_neq_b_abs = self.is_lte_biguint(&a.abs, &b.abs);

        // Negative - Negative -> -a_abs_cmp_b_abs
        let a_neg_b_neg = self.and(a_is_negative, b_is_negative);
        let flipped_result = self.not(result_if_a_abs_neq_b_abs);
        result_if_a_abs_neq_b_abs =
            self.select_bool(a_neg_b_neg, flipped_result, result_if_a_abs_neq_b_abs);

        // Positive - Negative -> 0
        // Zero - Negative -> 0
        let a_not_neg_and_b_neg = self.and_not(b_is_negative, a_is_negative);
        result_if_a_abs_neq_b_abs = self.and_not(result_if_a_abs_neq_b_abs, a_not_neg_and_b_neg);

        // Negative - Positive -> 1
        // Negative - Zero     -> 1
        let a_neg_b_not_neg = self.and_not(a_is_negative, b_is_negative);
        result_if_a_abs_neq_b_abs = self.or(result_if_a_abs_neq_b_abs, a_neg_b_not_neg);

        let a_eq_b = self.is_equal_bigint(a, b);
        self.or(a_eq_b, result_if_a_abs_neq_b_abs)
    }

    fn cmp_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> SignTarget {
        let key = (a.clone(), b.clone());
        let key_reversed = (b.clone(), a.clone());

        if self.cmp_biguint_cache.contains_key(&key_reversed) {
            warn!("cmp_biguint is called with same parameters but reversed order!");
        }
        if self.is_lte_biguint_cache.contains_key(&key_reversed) {
            warn!(
                "cmp_biguint is called with same parameters that are already compared with is_lte_biguint"
            );
        }
        if self.is_lte_biguint_cache.contains_key(&key) {
            warn!(
                "cmp_biguint is called with same parameters that are already compared with is_lte_biguint"
            );
        }

        if let Some(&result) = self.cmp_biguint_cache.get(&key) {
            return result;
        }

        let (_, b_gt_a) = self.try_sub_biguint(a, b);
        let (_, a_gt_b) = self.try_sub_biguint(b, a);

        let result = SignTarget::new_unsafe(self.sub(a_gt_b.0, b_gt_a.0));

        self.cmp_biguint_cache.insert(key, result);

        result
    }

    fn is_lte_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget {
        let key = (a.clone(), b.clone());
        let key_reversed = (b.clone(), a.clone());

        if self.is_lte_biguint_cache.contains_key(&key_reversed) {
            warn!("is_lte_biguint is called with same parameters but reversed order!");
        }
        if self.cmp_biguint_cache.contains_key(&key_reversed) {
            warn!(
                "is_lte_biguint is called with same parameters that are already compared with cmp"
            );
        }
        if self.cmp_biguint_cache.contains_key(&key) {
            warn!(
                "is_lte_biguint is called with same parameters that are already compared with cmp"
            );
        }

        if let Some(&result) = self.is_lte_biguint_cache.get(&key) {
            return result;
        }

        let (_, a_gt_b) = self.try_sub_biguint(b, a);
        let result = self.not(BoolTarget::new_unsafe(a_gt_b.0));

        self.is_lte_biguint_cache.insert(key, result);

        result
    }

    fn is_gte_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget {
        self.is_lte_biguint(b, a)
    }

    fn is_lt_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget {
        let a_gte_b = self.is_gte_biguint(a, b);
        self.not(a_gte_b)
    }

    fn is_gt_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget {
        let a_lte_b = self.is_lte_biguint(a, b);
        self.not(a_lte_b)
    }

    fn conditional_assert_lt_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) {
        let a_gte_b = self.is_gte_biguint(a, b);
        let res = self.and(is_enabled, a_gte_b);
        self.assert_false(res);
    }

    fn conditional_assert_lte_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) {
        let a_lte_b = self.is_lte_biguint(a, b);
        let a_gt_b = self.not(a_lte_b);
        let should_be_false = self.and(is_enabled, a_gt_b);
        self.assert_false(should_be_false);
    }

    fn max_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let cmp = self.is_lte_biguint(a, b);
        self.select_biguint(cmp, b, a)
    }

    fn max_biguint_multiple(&mut self, targets: &[&BigUintTarget]) -> BigUintTarget {
        let max = &mut targets[0].clone();
        for target in &targets[1..] {
            *max = self.max_biguint(max, target);
        }
        max.clone()
    }

    fn min_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let cmp = self.is_lte_biguint(a, b);
        self.select_biguint(cmp, a, b)
    }

    fn min_biguint_multiple(&mut self, targets: &[&BigUintTarget]) -> BigUintTarget {
        let min = &mut targets[0].clone();
        for target in &targets[1..] {
            *min = self.min_biguint(min, target);
        }
        min.clone()
    }

    fn conditional_assert_zero_biguint(&mut self, is_enabled: BoolTarget, a: &BigUintTarget) {
        let is_enabled_times_a = self.mul_biguint_by_bool(a, is_enabled);
        self.assert_zero_biguint(&is_enabled_times_a);
    }

    fn conditional_assert_not_zero_biguint(&mut self, is_enabled: BoolTarget, a: &BigUintTarget) {
        let is_zero = self.is_zero_biguint(a);
        let res = self.and(is_enabled, is_zero);
        self.assert_false(res);
    }

    fn conditional_assert_eq_biguint(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) {
        let zero = self.zero_biguint();

        let lhs_selected = self.select_biguint(is_enabled, a, &zero);
        let rhs_selected = self.select_biguint(is_enabled, b, &zero);

        self.connect_biguint(&lhs_selected, &rhs_selected);
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[allow(unused_imports)]
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use num::{BigInt, FromPrimitive};
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::bigint::bigint::WitnessBigInt;
    #[allow(unused_imports)]
    use crate::circuit_logger::CircuitBuilderLogging;
    use crate::types::config::{BIG_U96_LIMBS, BIG_U128_LIMBS, C, CIRCUIT_CONFIG, F};

    #[test]
    fn test_is_lte_bigint() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let zero = builder.zero();
        let one = builder.one();

        let pos_big = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);
        let pos_small = builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS);
        let neg_big = builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS);
        let neg_small = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);
        let zero_bigint = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);

        let pos_big_pos_small = builder.is_lte_bigint(&pos_big, &pos_small);
        builder.connect(pos_big_pos_small.target, zero);
        let pos_small_pos_big = builder.is_lte_bigint(&pos_small, &pos_big);
        builder.connect(pos_small_pos_big.target, one);
        let pos_big_pos_big = builder.is_lte_bigint(&pos_big, &pos_big);
        builder.connect(pos_big_pos_big.target, one);
        let pos_neg = builder.is_lte_bigint(&pos_big, &neg_big);
        builder.connect(pos_neg.target, zero);
        let pos_zero = builder.is_lte_bigint(&pos_big, &zero_bigint);
        builder.connect(pos_zero.target, zero);
        let zero_pos = builder.is_lte_bigint(&zero_bigint, &pos_big);
        builder.connect(zero_pos.target, one);
        let zero_zero = builder.is_lte_bigint(&zero_bigint, &zero_bigint);
        builder.connect(zero_zero.target, one);
        let zero_neg = builder.is_lte_bigint(&zero_bigint, &neg_big);
        builder.connect(zero_neg.target, zero);
        let neg_big_neg_small = builder.is_lte_bigint(&neg_big, &neg_small);
        builder.connect(neg_big_neg_small.target, zero);
        let neg_small_neg_big = builder.is_lte_bigint(&neg_small, &neg_big);
        builder.connect(neg_small_neg_big.target, one);
        let neg_big_neg_big = builder.is_lte_bigint(&neg_big, &neg_big);
        builder.connect(neg_big_neg_big.target, one);
        let neg_pos = builder.is_lte_bigint(&neg_big, &pos_big);
        builder.connect(neg_pos.target, one);
        let neg_zero = builder.is_lte_bigint(&neg_big, &zero_bigint);
        builder.connect(neg_zero.target, one);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        pw.set_bigint_target(&pos_big, &BigInt::from_i64(100i64).unwrap())?;
        pw.set_bigint_target(&pos_small, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&neg_big, &BigInt::from_i64(-10i64).unwrap())?;
        pw.set_bigint_target(&neg_small, &BigInt::from_i64(-100i64).unwrap())?;
        pw.set_bigint_target(&zero_bigint, &BigInt::from_i64(0i64).unwrap())?;

        data.verify(data.prove(pw).unwrap())
    }
}
