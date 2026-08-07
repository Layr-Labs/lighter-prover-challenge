fn main() {}

#[cfg(test)]
mod tests {
    use circuit::block::Block;
    use circuit::types::config::F;
    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

    const TX_COUNT: usize = 500;
    const HEAVY_TX_PER_PROOF: usize = 4;
    const LIGHT_TX_PER_PROOF: usize = 10;

    fn assert_public_500_tx_fixture_contract() {
        let light_tx_count = std::cmp::max(1, TX_COUNT * 98 / 100);
        let heavy_tx_count = std::cmp::max(1, TX_COUNT.saturating_sub(light_tx_count));
        assert_eq!(heavy_tx_count, 10);
        assert_eq!(light_tx_count, 490);

        let block = Block::<F>::from_json_with_empty_txs(
            include_bytes!("../../bench_test.json"),
            HEAVY_TX_PER_PROOF,
            LIGHT_TX_PER_PROOF,
            heavy_tx_count,
            light_tx_count,
        )
        .expect("public benchmark fixture must parse");

        assert!(
            block.tx_chunks.iter().all(|chunk| chunk
                .iter()
                .all(|tx| tx.tx_circuit_type == chunk[0].tx_circuit_type)),
            "every proof chunk must contain one circuit type"
        );

        let heavy_chunks: Vec<_> = block
            .tx_chunks
            .iter()
            .filter(|chunk| chunk[0].tx_circuit_type == TX_HEAVY)
            .collect();
        let light_chunks: Vec<_> = block
            .tx_chunks
            .iter()
            .filter(|chunk| chunk[0].tx_circuit_type == TX_LIGHT)
            .collect();

        assert_eq!(heavy_chunks.len(), 3);
        assert_eq!(light_chunks.len(), 49);
        assert!(
            heavy_chunks
                .iter()
                .all(|chunk| chunk.len() == HEAVY_TX_PER_PROOF)
        );
        assert!(
            light_chunks
                .iter()
                .all(|chunk| chunk.len() == LIGHT_TX_PER_PROOF)
        );

        let heavy_slots: usize = heavy_chunks.iter().map(|chunk| chunk.len()).sum();
        let light_slots: usize = light_chunks.iter().map(|chunk| chunk.len()).sum();
        assert_eq!(heavy_slots - heavy_tx_count, 2);
        assert_eq!(light_slots - light_tx_count, 0);
    }

    #[test]
    fn public_500_tx_fixture_has_expected_mixed_chunks_and_padding() {
        std::thread::Builder::new()
            .name("mixed-fixture-contract".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(assert_public_500_tx_fixture_contract)
            .expect("fixture contract thread must start")
            .join()
            .expect("fixture contract thread must finish");
    }
}
