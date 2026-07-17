// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;

use anyhow::Result;
use circuit::circuit_serializer::{
    BlockGateSerializer, BlockGeneratorSerializer, RecursionGateSerializer,
    RecursionGeneratorSerializer,
};
use circuit::ecdsa::curve::secp256k1::Secp256K1;
use circuit::recursion::cyclic_circuit::{Circuit, CyclicRecursionCircuit};
use circuit::types::config::{C, CIRCUIT_CONFIG, D};
use clap::Parser;
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::config::GenericHashOut;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    on_chain_operations_limit: usize,

    #[arg(long)]
    priority_operations_limit: usize,

    #[arg(long)]
    block_circuit_path: std::path::PathBuf,

    #[arg(long)]
    path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    let args = Args::parse();

    // Read block circuit
    let block_gate_serializer = BlockGateSerializer;
    let block_generator_serializer = BlockGeneratorSerializer::<C, D, Secp256K1>::default();
    let block_circuit_bytes = fs::read(args.block_circuit_path.clone())?;
    let block_circuit_data: CircuitData<GoldilocksField, C, 2> = CircuitData::from_bytes(
        &block_circuit_bytes,
        &block_gate_serializer,
        &block_generator_serializer,
    )
    .map_err(|err| {
        anyhow::Error::msg(format!(
            "Failed to read block circuit data from {:?}. err: {:?}",
            args.block_circuit_path.clone(),
            err
        ))
    })?;

    // Define cyclic circuit
    let circuit = CyclicRecursionCircuit::define(
        CIRCUIT_CONFIG,
        &block_circuit_data,
        args.on_chain_operations_limit,
    );
    // Build cyclic circuit
    let data = circuit.builder.build::<C>();

    // Write cyclic circuit data
    let cyclic_gate_serializer = RecursionGateSerializer;
    let cyclic_generator_serializer = RecursionGeneratorSerializer::<C, D>::default();
    let serialized_circuit = data
        .to_bytes(&cyclic_gate_serializer, &cyclic_generator_serializer)
        .map_err(|err| {
            anyhow::Error::msg(format!(
                "Failed to convert circuit data to bytes. {:?}",
                err
            ))
        })?;

    // Format: cyclic-circuit::<block-circuit-digest>::<digest>
    let path_name = format!(
        "cyclic-circuit::{}::{}",
        hex::encode(
            block_circuit_data
                .verifier_only
                .circuit_digest
                .to_bytes()
                .clone()
        ),
        hex::encode(data.verifier_only.circuit_digest.to_bytes().clone())
    );

    // If parent is given, append the file name
    let mut path = args.path.map_or_else(
        || Path::new(&path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(path_name.clone());
            v
        },
    );
    path.set_extension("bin");
    println!("{:?}", path);
    fs::write(path.clone(), serialized_circuit)?;

    // Read circuit data to validate the file
    let read_circuit = fs::read(path)?;
    let read_data: CircuitData<GoldilocksField, C, 2> = CircuitData::from_bytes(
        &read_circuit,
        &cyclic_gate_serializer,
        &cyclic_generator_serializer,
    )
    .map_err(|err| {
        anyhow::Error::msg(format!("Failed to read circuit data from bytes. {}", err))
    })?;
    assert_eq!(
        read_data, data,
        "read circuit data not match with orginal circuit"
    );

    Ok(())
}
