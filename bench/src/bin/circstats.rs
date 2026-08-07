use std::collections::BTreeMap;
use std::time::Instant;

use circuit::block::Block;
use circuit::block_tx::{BlockTx, JumpState};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::block_pre_execution::BlockPreExec;
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::TX_LIGHT;
use plonky2::field::types::Field;
use plonky2::iop::witness::PartitionWitness;

fn main() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, 10, 304, TX_LIGHT);
    let tx_target = tx.target;
    let data = tx.builder.build::<C>();
    let po = &data.prover_only;
    println!("=== light tx circuit ===");
    println!("degree = {}", data.common.degree());
    println!("num_wires = {}", data.common.config.num_wires);
    println!("representative_map.len = {}", po.representative_map.len());
    println!("generators.len = {}", po.generators.len());
    let watch_pairs: usize = po
        .generator_indices_by_watches
        .values()
        .map(|v| v.len())
        .sum();
    println!(
        "distinct watched reps = {}",
        po.generator_indices_by_watches.len()
    );
    println!("total (rep,generator) watch pairs = {}", watch_pairs);

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for g in &po.generators {
        *counts.entry(g.0.id()).or_default() += 1;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (id, c) in v.iter().take(12) {
        println!("{c:>9}  {id}");
    }

    // --- timings on the real public fixture witness ---
    let block = Block::<F>::from_json_with_empty_txs(
        include_bytes!("../../bench_test.json"),
        4,
        10,
        10,
        490,
    )
    .expect("fixture parses");

    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pre_target = pre.target;
    let pre_data = pre.builder.build::<C>();
    let pre_proof = BlockPreExecutionCircuit::prove(
        &pre_data,
        &BlockPreExec::from_block(&block),
        &pre_target,
    )
    .expect("pre proof");
    let pre_output =
        circuit::block_pre_execution::BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
    let state_metadata_hash = pre_output.new_state_metadata.hash();

    let light_chunk = block
        .tx_chunks
        .iter()
        .find(|c| c[0].tx_circuit_type == TX_LIGHT)
        .expect("light chunk")
        .clone();

    let block_tx = BlockTx {
        created_at: block.created_at,
        state_metadata_hash,
        old_jump: JumpState::initial(pre_output.new_state_root, block.old_account_delta_tree_root),
        txs: light_chunk,
    };
    let partial = BlockTxCircuit::generate_witness(&block_tx, &tx_target).unwrap();
    println!("partial witness input targets = {}", partial.target_values.len());

    // Time PartitionWitness::new (the zero fill).
    let t = Instant::now();
    let mut w = PartitionWitness::<F>::new(
        data.common.config.num_wires,
        data.common.degree(),
        &po.representative_map,
    );
    println!("PartitionWitness::new = {:?}", t.elapsed());

    let t = Instant::now();
    let mut n_inputs = 0usize;
    for (&tt, &vv) in &partial.target_values {
        let _ = vv;
        let _ = tt;
        n_inputs += 1;
    }
    println!("iterate inputs {} = {:?}", n_inputs, t.elapsed());

    // Time the unresolved_watches init pass exactly as generator.rs does it.
    use plonky2::iop::witness::WitnessWrite;
    for (&tt, &vv) in &partial.target_values {
        w.set_target(tt, vv).unwrap();
    }
    let t = Instant::now();
    let mut unresolved = vec![0usize; po.generators.len()];
    for (&watch, watchers) in &po.generator_indices_by_watches {
        if !w.is_set_by_rep_index(watch) {
            for &g in watchers {
                unresolved[g] += 1;
            }
        }
    }
    println!("unresolved_watches init pass = {:?}", t.elapsed());
    let ready0 = unresolved.iter().filter(|&&c| c == 0).count();
    println!(
        "generators ready at start = {} / {}",
        ready0,
        po.generators.len()
    );
    let set_reps = (0..po.generator_indices_by_watches.len()).count();
    let _ = set_reps;
    let sum_unres: usize = unresolved.iter().sum();
    println!(
        "sum unresolved (= pushes under current scheme after start) = {}",
        sum_unres
    );

    let t = Instant::now();
    let mut z = vec![F::ZERO; po.representative_map.len()];
    println!("vec![F::ZERO; {}] = {:?}", z.len(), t.elapsed());
    z[0] = F::ONE;
    std::hint::black_box(&z);

    let t = Instant::now();
    let z2 = vec![0u64; po.representative_map.len()];
    println!("vec![0u64; n] (calloc) = {:?}", t.elapsed());
    std::hint::black_box(&z2);

    // Full witness generation timing.
    let t = Instant::now();
    let full = plonky2::iop::generator::generate_partial_witness::<F, C, D>(
        partial.clone(),
        po,
        &data.common,
    )
    .unwrap();
    println!("generate_partial_witness (all-empty fixture) = {:?}", t.elapsed());
    std::hint::black_box(&full);
}
