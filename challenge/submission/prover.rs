//! This is the only contestant-editable file.

use std::sync::mpsc::sync_channel;
use std::thread;

use circuit::block::Block;
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::types::config::F;
use plonky2::field::types::Field;

use crate::api::{Circuits, Proof, Proofs};

pub fn prove_block(block: &Block<F>, circuits: &Circuits) -> Proofs {
    let pre = BlockPreExecutionCircuit::prove(
        &circuits.pre_data,
        &BlockPreExec::from_block(block),
        &circuits.pre_target,
    )
    .unwrap();
    let pre_output = BlockPreExecWitness::from_public_inputs(&pre.public_inputs);

    // The per-transaction proofs are a sequential chain: each chunk's input
    // state is the previous chunk's output (read from its proof). The cyclic
    // chain fold is a separate sequential chain: fold_i needs the previous
    // chain proof and tx proof i. Crucially, tx proof i+1 depends only on tx
    // proof i (its output state), NOT on the chain. So the two chains are
    // independent and overlap: while one thread proves the next tx, the other
    // folds the previous tx into the cyclic chain. Inputs, circuits, and the
    // final statement remain unchanged.
    // At tx_index == 0 the chain circuit verifies `circuits.dummy_proof`, not
    // the supplied cyclic proof. It still consumes the cyclic proof's public
    // inputs as the initial block state and checks their verifier-data suffix.
    // Reuse the already-built dummy proof body and replace the same state
    // inputs that `cyclic_base_proof` would set. The verifier-data suffix is
    // left byte-for-byte unchanged.
    let mut base_chain = circuits.dummy_proof.clone();
    base_chain.public_inputs[0] = F::from_canonical_u64(block.block_number);
    base_chain.public_inputs[1] = F::from_canonical_u64(block.created_at as u64);
    for (index, element) in [
        pre_output.new_state_root,
        pre_output.new_validium_root,
        pre_output.new_state_root,
        block.old_account_delta_tree_root,
    ]
    .iter()
    .flat_map(|hash| hash.elements)
    .enumerate()
    {
        base_chain.public_inputs[2 + index] = element;
    }
    let metadata = pre_output.new_state_metadata.to_public_inputs();
    base_chain.public_inputs
        [circuits.chain_witness_size..circuits.chain_witness_size + metadata.len()]
        .copy_from_slice(&metadata);

    // Bounded buffer keeps at most a couple of tx proofs in flight so the two
    // stages overlap without letting the producer race unboundedly ahead.
    let (tx_send, tx_recv) = sync_channel::<(u64, Proof)>(3);

    let chain = thread::scope(|scope| {
        // Producer: sequential per-chunk tx proofs, threading state forward.
        scope.spawn(move || {
            let mut assets = block.all_assets.clone();
            let mut markets = pre_output.new_market_details.clone();
            let mut system_config = block.old_system_config;
            let mut registers = block.register_stack_before;
            let mut account_root = block.old_account_tree_root;
            let mut account_data_root = block.old_account_pub_data_tree_root;
            let mut delta_root = block.old_account_delta_tree_root;
            let mut market_root = block.old_market_tree_root;

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

                let tx =
                    BlockTxCircuit::prove(&circuits.tx_data, &input, &circuits.tx_target).unwrap();
                let output = BlockTxWitness::from_public_inputs(&tx.public_inputs);
                assets = output.all_assets_after;
                markets = output.all_market_details_after;
                system_config = output.new_system_config;
                registers = output.register_stack_after;
                account_root = output.new_account_tree_root;
                account_data_root = output.new_account_pub_data_tree_root;
                delta_root = output.new_account_delta_tree_root;
                market_root = output.new_market_tree_root;

                // Fails the whole prove (panics the scope) only if the send
                // target is gone, i.e. the consumer already errored out.
                tx_send.send((index as u64, tx)).unwrap();
            }
            // Dropping tx_send here closes the channel and ends the consumer.
        });

        // Consumer: fold tx proofs into the cyclic chain, strictly in order.
        let mut chain = base_chain;
        for (index, tx) in tx_recv {
            chain = BlockTxChainCircuit::prove(
                &circuits.chain_target,
                &circuits.chain_data,
                index,
                &chain,
                &circuits.dummy_proof,
                &tx,
            )
            .unwrap();
        }
        chain
    });

    Proofs { pre, chain }
}
