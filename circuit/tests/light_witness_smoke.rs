use circuit::block::Block;
use circuit::tx_constraints::{TxTarget, TxTargetWitness};
use circuit::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::PartialWitness;

#[test]
fn light_tx_witness_supplies_every_generator_input() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let block = Block::<F>::from_json_with_empty_txs(
                include_bytes!("../../bench/bench_test.json"),
                1,
                1,
                1,
                1,
            )
            .expect("public fixture must parse");
            let tx = block
                .tx_chunks
                .into_iter()
                .flatten()
                .find(|tx| tx.tx_circuit_type == TX_LIGHT)
                .expect("fixture expansion must contain a light transaction");

            let mut builder = Builder::new(CIRCUIT_CONFIG);
            let target = TxTarget::new(&mut builder);
            builder.perform_registered_range_checks();
            let data = builder.build::<C>();

            let mut full_tx = tx.clone();
            full_tx.tx_circuit_type = TX_HEAVY;
            let mut full_witness = PartialWitness::new();
            full_witness
                .set_tx_target(&target, &full_tx)
                .expect("full witness assignment must succeed");

            let mut partial_witness = PartialWitness::new();
            partial_witness
                .set_tx_target(&target, &tx)
                .expect("light witness assignment must succeed");
            assert!(partial_witness.target_values.len() < full_witness.target_values.len());
            eprintln!(
                "light targets: {}, full targets: {}, removed: {}",
                partial_witness.target_values.len(),
                full_witness.target_values.len(),
                full_witness.target_values.len() - partial_witness.target_values.len()
            );
            generate_partial_witness::<F, C, D>(partial_witness, &data.prover_only, &data.common)
                .expect("every generator must have its required light witness inputs");
        })
        .expect("smoke-test thread must start")
        .join()
        .expect("smoke-test thread must finish");
}
