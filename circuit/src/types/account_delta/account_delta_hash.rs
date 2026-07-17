// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::BoolTarget;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::account_delta::AccountDeltaTarget;
use crate::types::config::Builder;
use crate::types::constants::{EMPTY_ASSET_TREE_ROOT, EMPTY_POSITION_DELTA_TREE_ROOT};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

impl AccountDeltaTarget {
    pub fn aggregated_asset_delta_hash(
        &self,
        builder: &mut Builder,
        index: usize,
    ) -> HashOutTarget {
        let asset_balance = &self.aggregated_asset_deltas[index];
        let mut elements = vec![asset_balance.sign.target];
        elements.extend_from_slice(
            &asset_balance
                .abs
                .limbs
                .iter()
                .map(|x| x.0)
                .collect::<Vec<_>>(),
        );
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let empty_hash = builder.zero_hash_out();
        let is_empty = builder.is_zero_bigint(asset_balance);
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn hash_with_is_empty(&self, builder: &mut Builder) -> (HashOutTarget, BoolTarget) {
        let partial_hash = self.partial_hash(builder);
        self.hash_from_partial_hash(builder, &partial_hash)
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let partial_hash = self.partial_hash(builder);
        let (hash, _) = self.hash_from_partial_hash(builder, &partial_hash);
        hash
    }

    pub fn fee_account_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let (hash, _) = self.hash_from_partial_hash(builder, &self.partial_hash);
        hash
    }

    // Hash of fields that wouldn't change for the fee account
    fn partial_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut elements = vec![];
        let mut is_empty_flags = vec![];

        for share in &self.public_pool_shares_delta {
            elements.extend_from_slice(&[share.public_pool_index, share.shares_delta.target]);
            is_empty_flags.push(builder.is_zero(share.shares_delta.target));
        }

        elements.extend_from_slice(&[
            self.public_pool_info_delta.total_shares_delta.target,
            self.public_pool_info_delta.operator_shares_delta.target,
        ]);
        is_empty_flags.extend_from_slice(&[
            builder.is_zero(self.public_pool_info_delta.total_shares_delta.target),
            builder.is_zero(self.public_pool_info_delta.operator_shares_delta.target),
        ]);

        for &limb in self.l1_address.limbs.iter() {
            elements.push(limb.0);
            is_empty_flags.push(builder.is_zero_u32(limb));
        }

        elements.push(self.account_type);
        is_empty_flags.push(builder.is_zero(self.account_type));

        elements.extend_from_slice(&self.position_delta_root.elements);
        let empty_position_delta_tree_root = builder.constant_hash(EMPTY_POSITION_DELTA_TREE_ROOT);
        is_empty_flags.push(
            builder.is_equal_hash(&self.position_delta_root, &empty_position_delta_tree_root),
        );

        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let empty_hash = builder.zero_hash_out();

        let is_empty = builder.multi_and(&is_empty_flags);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    fn hash_from_partial_hash(
        &self,
        builder: &mut Builder,
        partial_hash: &HashOutTarget,
    ) -> (HashOutTarget, BoolTarget) {
        let mut elements = partial_hash.elements.to_vec();

        elements.extend_from_slice(&self.asset_delta_root.elements);

        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder, partial_hash);
        (
            builder.select_hash(is_empty, &empty_hash, &non_empty_hash),
            is_empty,
        )
    }

    fn is_empty(&self, builder: &mut Builder, partial_hash: &HashOutTarget) -> BoolTarget {
        let empty_hash = builder.zero_hash_out();

        let mut assertions = vec![builder.is_equal_hash(partial_hash, &empty_hash)];

        let empty_asset_delta_root = builder.constant_hash(EMPTY_ASSET_TREE_ROOT);
        assertions.push(builder.is_equal_hash(&self.asset_delta_root, &empty_asset_delta_root));

        builder.multi_and(&assertions)
    }
}
