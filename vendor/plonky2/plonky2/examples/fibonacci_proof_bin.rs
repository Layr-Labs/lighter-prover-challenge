use std::path::PathBuf;
use std::{env, fs};

use anyhow::{bail, Result};
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

fn main() -> Result<()> {
    const D: usize = 2;
    type C = Poseidon2GoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let mut args = env::args_os();
    let _bin = args.next();
    let output_path = match args.next() {
        Some(path) => PathBuf::from(path),
        None => bail!("usage: cargo run --example fibonacci_proof_bin -- <output_path>"),
    };

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..5000 {
        let next = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = next;
    }

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;
    data.verify(proof.clone())?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, proof.to_bytes())?;

    println!(
        "wrote {} bytes to {}",
        fs::metadata(&output_path)?.len(),
        output_path.display()
    );

    Ok(())
}
