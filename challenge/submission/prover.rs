//! This is the only contestant-editable file.

use circuit::block::Block;
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::types::config::F;

use crate::api::{Circuits, Proofs};

pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proofs {
    // Minimal E2E pipeline test: no-op edit to validate submit -> PR -> Actions -> scoring.
    let pre = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .unwrap();
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre.public_inputs);

    let mut assets = block.all_assets.clone();
    let mut markets = pre_output.new_market_details.clone();
    let mut system_config = block.old_system_config;
    let mut registers = block.register_stack_before;
    let mut account_root = block.old_account_tree_root;
    let mut account_data_root = block.old_account_pub_data_tree_root;
    let mut delta_root = block.old_account_delta_tree_root;
    let mut market_root = block.old_market_tree_root;
    let mut chain = BlockTxChainCircuit::cyclic_base_proof(
        &circuits.chain_data,
        &circuits.dummy_data,
        block.block_number,
        block.created_at,
        pre_output.new_state_root,
        pre_output.new_state_root,
        pre_output.new_validium_root,
        block.old_account_delta_tree_root,
        circuits.chain_witness_size,
        &pre_output.new_state_metadata,
    );

    for (index, txs) in block.txs.chunks(4).enumerate() {
        let input = BlockTx {
            created_at: block.created_at,
            old_system_config: system_config,
            register_stack_before: registers,
            all_assets_before: assets.clone(),
            all_market_details_before: markets.clone(),
            old_account_tree_root: account_root,
            old_account_pub_data_tree_root: account_data_root,
            old_account_delta_tree_root: delta_root,
            old_market_tree_root: market_root,
            txs: txs.to_vec(),
        };

        let tx = BlockTxCircuit::prove(&circuits.tx_data, &input, &circuits.tx_target).unwrap();
        let output = BlockTxWitness::from_public_inputs(&tx.public_inputs);
        assets = output.all_assets_after;
        markets = output.all_market_details_after;
        system_config = output.new_system_config;
        registers = output.register_stack_after;
        account_root = output.new_account_tree_root;
        account_data_root = output.new_account_pub_data_tree_root;
        delta_root = output.new_account_delta_tree_root;
        market_root = output.new_market_tree_root;

        chain = BlockTxChainCircuit::prove(
            &circuits.chain_target,
            &circuits.chain_data,
            index as u64,
            &chain,
            &circuits.dummy_proof,
            &tx,
        )
        .unwrap();
    }

    Proofs { pre, chain }
}
