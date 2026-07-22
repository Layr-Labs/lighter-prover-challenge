// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, FromPrimitive};
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct L1UpdateAssetTx {
    #[serde(rename = "ai")]
    pub asset_index: i16, // 6 bits
    #[serde(rename = "mta")]
    pub min_transfer_amount: i64, // 60 bits
    #[serde(rename = "mwa")]
    pub min_withdrawal_amount: i64, // 60 bits
    #[serde(rename = "mm", default)]
    pub margin_mode: u8,
}

#[derive(Debug)]
pub struct L1UpdateAssetTxTarget {
    pub asset_index: Target,
    pub min_transfer_amount: BigUintTarget,
    pub min_withdrawal_amount: BigUintTarget,
    pub margin_mode: Target,

    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1UpdateAssetTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L1UpdateAssetTxTarget {
            asset_index: builder.add_virtual_target(),
            min_transfer_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_withdrawal_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            margin_mode: builder.add_virtual_target(),

            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }
}

impl PriorityOperationsPubData for L1UpdateAssetTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let byte_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_UPDATE_ASSET),
            add_target(builder, bytes, self.asset_index, 16),
            add_big_uint_target(builder, bytes, &self.min_transfer_amount),
            add_big_uint_target(builder, bytes, &self.min_withdrawal_amount),
            add_byte_target_unsafe(bytes, self.margin_mode),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl Verify for L1UpdateAssetTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_types: &TxTypeTargets, tx_state: &TxState) {
        self.success = tx_types.is_l1_update_asset;
        self.is_enabled = tx_types.is_l1_update_asset;

        builder.conditional_assert_eq(
            self.is_enabled,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );
        ensure_valid_asset_index(builder, self.is_enabled, self.asset_index);

        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_transfer_amount);
        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_withdrawal_amount);

        builder.register_range_check(self.asset_index, ASSET_LIST_SIZE_BITS);
        builder.range_check_biguint(&self.min_transfer_amount, MAX_EXCHANGE_ASSET_BALANCE_BITS);
        builder.range_check_biguint(&self.min_withdrawal_amount, MAX_EXCHANGE_ASSET_BALANCE_BITS);
        builder.assert_bool(BoolTarget::new_unsafe(self.margin_mode));

        // Must be USDC if margin mode is enabled
        let is_margin_enabled =
            builder.is_equal_constant(self.margin_mode, ASSET_MARGIN_MODE_ENABLED);
        let is_usdc_asset = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);
        let should_be_false = builder.and_not(is_margin_enabled, is_usdc_asset);
        builder.conditional_assert_false(self.is_enabled, should_be_false);

        // USDC must have margin mode enabled
        let is_usdc_margin_mode_disabled = builder.and_not(is_usdc_asset, is_margin_enabled);
        builder.conditional_assert_false(self.is_enabled, is_usdc_margin_mode_disabled);

        // Ensure that asset is not empty
        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        self.success = builder.and_not(self.success, is_asset_empty);
    }
}

impl Apply for L1UpdateAssetTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.assets[TX_ASSET_ID].min_transfer_amount = builder.select_biguint(
            self.success,
            &self.min_transfer_amount,
            &tx_state.assets[TX_ASSET_ID].min_transfer_amount,
        );
        tx_state.assets[TX_ASSET_ID].min_withdrawal_amount = builder.select_biguint(
            self.success,
            &self.min_withdrawal_amount,
            &tx_state.assets[TX_ASSET_ID].min_withdrawal_amount,
        );
        tx_state.assets[TX_ASSET_ID].margin_mode = builder.select(
            self.success,
            self.margin_mode,
            tx_state.assets[TX_ASSET_ID].margin_mode,
        );

        self.success
    }
}

pub trait L1UpdateAssetTxTargetWitness<F: PrimeField64> {
    fn set_l1_update_asset_tx_target(
        &mut self,
        a: &L1UpdateAssetTxTarget,
        b: &L1UpdateAssetTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1UpdateAssetTxTargetWitness<F> for T {
    fn set_l1_update_asset_tx_target(
        &mut self,
        a: &L1UpdateAssetTxTarget,
        b: &L1UpdateAssetTx,
    ) -> Result<()> {
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_biguint_target(
            &a.min_transfer_amount,
            &BigUint::from_u64(b.min_transfer_amount as u64).unwrap(),
        )?;
        self.set_biguint_target(
            &a.min_withdrawal_amount,
            &BigUint::from_u64(b.min_withdrawal_amount as u64).unwrap(),
        )?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;

        Ok(())
    }
}
