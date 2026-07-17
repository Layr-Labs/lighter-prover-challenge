// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;

use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::config::Hasher;

use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::{Builder, F};
use crate::types::constants::{ORDER_BOOK_MERKLE_LEVELS, ORDER_NONCE_BITS, ORDER_PRICE_BITS};
use crate::types::order::Order;
use crate::types::order_book_node::{OrderBookNode, OrderBookNodeTarget};

pub fn order_indexes_to_merkle_path(
    builder: &mut Builder,
    price_index: Target,
    nonce_index: Target,
) -> [BoolTarget; ORDER_BOOK_MERKLE_LEVELS] {
    let price_merkle_helpers = builder.split_le(price_index, ORDER_PRICE_BITS);
    let nonce_merkle_helpers = builder.split_le(nonce_index, ORDER_NONCE_BITS);

    nonce_merkle_helpers
        .into_iter()
        .chain(price_merkle_helpers)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn verify_order_book_tree_merkle_proof<const L: usize>(
    builder: &mut Builder,
    root: &HashOutTarget,
    leaf: HashOutTarget,
    proofs: &[OrderBookNodeTarget; L],
    helpers: [BoolTarget; L],
) {
    let mut state = leaf;
    for i in 0..L {
        let sibling_hash =
            builder.hash_two_to_one_swap(&state, &proofs[i].sibling_child_hash, helpers[i]);

        state = builder.hash_two_to_one(&sibling_hash, &proofs[i].internal_hash());
    }

    builder.connect_hashes(state, *root);
}

pub fn recalculate_order_book_tree_root<const L: usize>(
    builder: &mut Builder,
    leaf: HashOutTarget,
    proofs: &[OrderBookNodeTarget; L],
    helpers: [BoolTarget; L],
) -> HashOutTarget {
    let mut state = leaf;
    for i in 0..L {
        let sibling_hash =
            builder.hash_two_to_one_swap(&state, &proofs[i].sibling_child_hash, helpers[i]);
        let internal_hash = proofs[i].internal_hash();
        state = builder.hash_two_to_one(&sibling_hash, &internal_hash);
    }

    state
}

#[derive(Debug, Clone)]
pub struct OrderBookTree<const L: usize> {
    pub empty_hashes: Vec<HashOut<F>>,
    pub empty_internal_data: Vec<OrderBookNode<F>>,
    node_hashes: HashMap<u128, HashOut<F>>,
    internal_node_data: HashMap<u128, OrderBookNode<F>>,

    pub root: HashOut<F>,
}

impl<const L: usize> OrderBookTree<L> {
    pub fn new() -> Self {
        let mut empty_hashes = Vec::<HashOut<F>>::with_capacity(L + 1);

        let mut h = HashOut::ZERO;
        let nil_hash = h;

        empty_hashes.push(h);

        for _ in 0..L {
            h = Poseidon2Hash::two_to_one(h, h);
            h = Poseidon2Hash::two_to_one(h, nil_hash);
            empty_hashes.push(h);
        }

        Self {
            empty_hashes,
            empty_internal_data: vec![OrderBookNode::empty(); L + 1],
            root: h,
            node_hashes: HashMap::new(),
            internal_node_data: HashMap::new(),
        }
    }

    pub fn insert_leaf(&mut self, index: u128, leaf_data: Order) {
        let mut key = get_key_from_leaf_index::<L>(index);
        let mut node_hash = leaf_data.hash();
        self.node_hashes.insert(key, node_hash);
        self.internal_node_data.insert(
            key,
            OrderBookNode {
                ask_base_sum: leaf_data.ask_base_sum,
                ask_quote_sum: leaf_data.ask_quote_sum,
                ..OrderBookNode::empty()
            },
        );

        let mut ask_base_sum = leaf_data.ask_base_sum;
        let mut ask_quote_sum = leaf_data.ask_quote_sum;
        let mut bid_base_sum = leaf_data.bid_base_sum;
        let mut bid_quote_sum = leaf_data.bid_quote_sum;

        for i in 0..L {
            let bit = key & 1;
            let sibling_key = key ^ 1;
            let sibling_node_data = self
                .internal_node_data
                .get(&sibling_key)
                .unwrap_or(&self.empty_internal_data[i]);
            let sibling_node_hash = self
                .node_hashes
                .get(&sibling_key)
                .unwrap_or(&self.empty_hashes[i]);

            let parent_key = key >> 1;

            ask_base_sum += sibling_node_data.ask_base_sum;
            ask_quote_sum += sibling_node_data.ask_quote_sum;
            bid_base_sum += sibling_node_data.bid_base_sum;
            bid_quote_sum += sibling_node_data.bid_quote_sum;

            let new_parent_node = OrderBookNode {
                ask_base_sum,
                ask_quote_sum,
                bid_base_sum,
                bid_quote_sum,
                sibling_child_hash: HashOut::ZERO, // Placeholder
            };

            let parent_children_hash: HashOut<F> = if bit == 0 {
                Poseidon2Hash::two_to_one(node_hash, *sibling_node_hash)
            } else {
                Poseidon2Hash::two_to_one(*sibling_node_hash, node_hash)
            };
            let parent_hash =
                Poseidon2Hash::two_to_one(parent_children_hash, new_parent_node.internal_hash());

            self.node_hashes.insert(parent_key, parent_hash);
            self.internal_node_data.insert(parent_key, new_parent_node);

            node_hash = parent_hash;
            key = parent_key;
        }

        self.root = node_hash;
    }

    pub fn proof(&self, index: u128) -> [OrderBookNode<F>; L] {
        let mut proof = [self.empty_internal_data[0]; L];
        let mut key = get_key_from_leaf_index::<L>(index);

        for i in 0..L {
            let sibling_key = key ^ 1;
            let sibling_hash = self
                .node_hashes
                .get(&sibling_key)
                .unwrap_or(&self.empty_hashes[i]);
            key >>= 1;
            let node_data = self
                .internal_node_data
                .get(&key)
                .unwrap_or(&self.empty_internal_data[i]);

            proof[i] = OrderBookNode {
                ask_base_sum: node_data.ask_base_sum,
                ask_quote_sum: node_data.ask_quote_sum,
                bid_base_sum: node_data.bid_base_sum,
                bid_quote_sum: node_data.bid_quote_sum,
                sibling_child_hash: *sibling_hash,
            }
        }

        proof
    }
}

fn get_key_from_leaf_index<const L: usize>(index: u128) -> u128 {
    let converter: u128 = 1u128 << L;
    converter + index
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;
    use crate::bool_utils::CircuitBuilderBoolUtils;
    use crate::builder::Builder;

    #[test]
    fn test_order_index_to_merkle_path() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_ecc_config();
        let pw = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x0 = 8 + (1 << 24);
        let x1 = 11;
        let x0_target = builder.constant(F::from_canonical_u64(x0));
        let x1_target = builder.constant(F::from_canonical_u64(x1));
        let bits = order_indexes_to_merkle_path(&mut builder, x0_target, x1_target);

        // Result should be 11 + 2^48 * (8 + 2^24) = 2^0 + 2^1 + 2^3 + 2^51 + 2^72
        builder.assert_true(bits[0]);
        builder.assert_true(bits[1]);
        builder.assert_true(bits[3]);
        builder.assert_true(bits[51]);
        builder.assert_true(bits[72]);
        for i in 0..ORDER_BOOK_MERKLE_LEVELS {
            if i != 0 && i != 1 && i != 3 && i != 51 && i != 72 {
                builder.assert_false(bits[i]);
            }
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[test]
    fn test_order_book_tree() {
        const L: usize = ORDER_PRICE_BITS + ORDER_NONCE_BITS;

        let mut order_book_tree = OrderBookTree::<L>::new();

        let proof = order_book_tree.proof(2);
        for i in 0..L {
            assert_eq!(proof[i].sibling_child_hash, order_book_tree.empty_hashes[i]);
        }

        let order = Order {
            key_price: 8,
            key_nonce: 11,

            ask_base_sum: 10,
            ask_quote_sum: 80,
            bid_base_sum: 0,
            bid_quote_sum: 0,
        };
        let index = 11 + (1u128 << ORDER_NONCE_BITS) * 8;
        order_book_tree.insert_leaf(index, order);

        let proof = order_book_tree.proof(index);
        for i in 0..1 {
            assert_eq!(proof[i].sibling_child_hash, order_book_tree.empty_hashes[i]);
            assert_eq!(proof[i].ask_base_sum, 10i64);
            assert_eq!(proof[i].ask_quote_sum, 80i64);
            assert_eq!(proof[i].bid_base_sum, 0i64);
            assert_eq!(proof[i].bid_quote_sum, 0i64);
        }
    }
}
