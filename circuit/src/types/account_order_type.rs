// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::{BoolTarget, Target};

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::types::config::Builder;
use crate::types::constants::{
    LIMIT_ORDER, LIQUIDATION_ORDER, MARKET_ORDER, STOP_LOSS_LIMIT_ORDER, STOP_LOSS_ORDER,
    TAKE_PROFIT_LIMIT_ORDER, TAKE_PROFIT_ORDER, TWAP_ORDER, TWAP_SUB_ORDER,
};

#[derive(Debug)]
pub struct AccountOrderTypes {
    pub is_limit_order: BoolTarget,
    pub is_market_order: BoolTarget,
    pub is_stop_loss_order: BoolTarget,
    pub is_stop_loss_limit_order: BoolTarget,
    pub is_take_profit_order: BoolTarget,
    pub is_take_profit_limit_order: BoolTarget,
    pub is_conditional_order: BoolTarget,
    pub is_twap_order: BoolTarget,
    pub is_twap_sub_order: BoolTarget,
    pub is_liquidation_order: BoolTarget,
    pub is_stop_loss_variant: BoolTarget,
    pub is_take_profit_variant: BoolTarget,

    pub is_valid_l2_create_order: BoolTarget,
}

impl AccountOrderTypes {
    pub fn new(builder: &mut Builder, order_type: Target) -> Self {
        let limit_order = builder.constant_from_u8(LIMIT_ORDER);
        let market_order = builder.constant_from_u8(MARKET_ORDER);
        let stop_loss_order = builder.constant_from_u8(STOP_LOSS_ORDER);
        let stop_loss_limit_order = builder.constant_from_u8(STOP_LOSS_LIMIT_ORDER);
        let take_profit_order = builder.constant_from_u8(TAKE_PROFIT_ORDER);
        let take_profit_limit_order = builder.constant_from_u8(TAKE_PROFIT_LIMIT_ORDER);
        let twap_order = builder.constant_from_u8(TWAP_ORDER);
        let twap_sub_order = builder.constant_from_u8(TWAP_SUB_ORDER);
        let liquidation_order = builder.constant_from_u8(LIQUIDATION_ORDER);

        let is_limit_order = builder.is_equal(order_type, limit_order);
        let is_market_order = builder.is_equal(order_type, market_order);
        let is_stop_loss_order = builder.is_equal(order_type, stop_loss_order);
        let is_stop_loss_limit_order = builder.is_equal(order_type, stop_loss_limit_order);
        let is_take_profit_order = builder.is_equal(order_type, take_profit_order);
        let is_take_profit_limit_order = builder.is_equal(order_type, take_profit_limit_order);
        let is_conditional_order = builder.multi_or(&[
            is_stop_loss_order,
            is_stop_loss_limit_order,
            is_take_profit_order,
            is_take_profit_limit_order,
        ]);
        let is_twap_order = builder.is_equal(order_type, twap_order);
        let is_twap_sub_order = builder.is_equal(order_type, twap_sub_order);
        let is_liquidation_order = builder.is_equal(order_type, liquidation_order);

        let is_stop_loss_variant =
            builder.multi_or(&[is_stop_loss_order, is_stop_loss_limit_order]);

        let is_take_profit_variant =
            builder.multi_or(&[is_take_profit_order, is_take_profit_limit_order]);

        let is_valid_l2_create_order = builder.multi_or(&[
            is_limit_order,
            is_market_order,
            is_stop_loss_order,
            is_stop_loss_limit_order,
            is_take_profit_order,
            is_take_profit_limit_order,
            is_twap_order,
        ]);

        AccountOrderTypes {
            is_limit_order,
            is_market_order,
            is_stop_loss_order,
            is_stop_loss_limit_order,
            is_take_profit_order,
            is_take_profit_limit_order,
            is_conditional_order,
            is_twap_order,
            is_twap_sub_order,
            is_liquidation_order,
            is_stop_loss_variant,
            is_take_profit_variant,
            is_valid_l2_create_order,
        }
    }
}
