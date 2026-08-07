// Temporary audit tool: dumps per-circuit gate inventories.
#![feature(stmt_expr_attributes)]

#[path = "../api.rs"]
mod api;

use api::Circuits;
use circuit::types::config::{C, D, F};
use plonky2::plonk::circuit_data::CircuitData;

fn dump(name: &str, data: &CircuitData<F, C, D>) {
    let cd = &data.common;
    println!(
        "== {name}: degree_bits={} degree={} quotient_degree_factor={} num_gate_constraints={} num_gates={}",
        cd.degree_bits(),
        cd.degree(),
        cd.quotient_degree_factor,
        cd.num_gate_constraints,
        cd.gates.len()
    );
    let mut total = 0usize;
    for g in &cd.gates {
        let nc = g.0.num_constraints();
        total += nc;
        println!("   {:>5}  {}", nc, g.0.id());
    }
    println!("   sum(num_constraints) = {total}");
    let batches = (cd.degree() * cd.quotient_degree_factor).div_ceil(32);
    println!("   quotient batches (BATCH_SIZE=32) = {batches}");
}

fn main() {
    let c = Circuits::new();
    dump("pre_exec", &c.pre_data);
    dump("light_tx", &c.light_tx_data);
    dump("heavy_tx", &c.heavy_tx_data);
    dump("light_chain", &c.light_chain_data);
    dump("heavy_chain", &c.heavy_chain_data);
    let (_t, block) = c.build_block_circuit();
    dump("block", &block);
}
