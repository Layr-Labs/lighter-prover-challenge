// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;

pub type Proof = ProofWithPublicInputs<F, C, D>;

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;
const CIRCUIT_BUILD_STACK_BYTES: usize = 64 * 1024 * 1024;

fn join_three_builds<FirstFn, SecondFn, ThirdFn, First, Second, Third>(
    first: FirstFn,
    second: SecondFn,
    third: ThirdFn,
) -> (First, Second, Third)
where
    FirstFn: FnOnce() -> First + Send,
    SecondFn: FnOnce() -> Second + Send,
    ThirdFn: FnOnce() -> Third,
    First: Send,
    Second: Send,
{
    std::thread::scope(|scope| {
        let first = std::thread::Builder::new()
            .name("heavy-circuit-build".to_owned())
            .stack_size(CIRCUIT_BUILD_STACK_BYTES)
            .spawn_scoped(scope, first)
            .expect("cannot start heavy circuit build");
        let second = std::thread::Builder::new()
            .name("light-circuit-build".to_owned())
            .stack_size(CIRCUIT_BUILD_STACK_BYTES)
            .spawn_scoped(scope, second)
            .expect("cannot start light circuit build");
        let third = third();
        (
            first.join().expect("heavy circuit build panicked"),
            second.join().expect("light circuit build panicked"),
            third,
        )
    })
}

struct TxPathCircuits {
    tx_target: BlockTxTarget,
    tx_data: CircuitData<F, C, D>,
    chain_target: BlockTxChainTarget,
    chain_data: CircuitData<F, C, D>,
    dummy_chain_circuit: CircuitData<F, C, D>,
    dummy_proof: Proof,
}

fn build_tx_path_circuits(tx_per_proof: usize, mode: u8) -> TxPathCircuits {
    let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, mode);
    let tx_target = tx.target;
    let tx_data = tx.builder.build::<C>();
    let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
    let chain_target = chain.target;
    let chain_data = chain.builder.build::<C>();
    let dummy_chain_circuit = dummy_circuit(&chain_data.common);
    let dummy_proof = cyclic_base_proof(
        &chain_data.common,
        &chain_data.verifier_only,
        &dummy_chain_circuit,
        [].into_iter().collect(),
    )
    .expect("cannot construct transaction chain dummy proof");

    TxPathCircuits {
        tx_target,
        tx_data,
        chain_target,
        chain_data,
        dummy_chain_circuit,
        dummy_proof,
    }
}

pub struct Circuits {
    pub heavy_tx_target: BlockTxTarget,
    pub heavy_tx_data: CircuitData<F, C, D>,
    pub light_tx_target: BlockTxTarget,
    pub light_tx_data: CircuitData<F, C, D>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    pub heavy_chain_data: CircuitData<F, C, D>,
    pub light_chain_target: BlockTxChainTarget,
    pub light_chain_data: CircuitData<F, C, D>,
    pub block_target: BlockTarget,
    pub block_data: CircuitData<F, C, D>,
    pub dummy_heavy_chain_circuit: CircuitData<F, C, D>,
    pub dummy_light_chain_circuit: CircuitData<F, C, D>,
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

impl Circuits {
    pub fn new() -> Self {
        let (heavy, light, (pre_target, pre_data)) = join_three_builds(
            || build_tx_path_circuits(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
            || build_tx_path_circuits(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, pre.builder.build::<C>())
            },
        );

        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &pre_data,
            &light.chain_data,
            &heavy.chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        let block_target = block.target;
        let block_data = block.builder.build::<C>();

        Self {
            heavy_tx_target: heavy.tx_target,
            heavy_tx_data: heavy.tx_data,
            light_tx_target: light.tx_target,
            light_tx_data: light.tx_data,
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: heavy.chain_data,
            light_chain_target: light.chain_target,
            light_chain_data: light.chain_data,
            block_target,
            block_data,
            dummy_heavy_chain_circuit: heavy.dummy_chain_circuit,
            dummy_light_chain_circuit: light.dummy_chain_circuit,
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

    use super::*;

    #[test]
    fn production_mixed_circuit_parameters_are_fixed() {
        assert_eq!(CHAIN_ID, 304);
        assert_eq!(HEAVY_TX_PER_PROOF, 4);
        assert_eq!(HEAVY_TX_MODE, TX_HEAVY);
        assert_eq!(LIGHT_TX_PER_PROOF, 10);
        assert_eq!(LIGHT_TX_MODE, TX_LIGHT);
        assert_eq!(ON_CHAIN_OPERATIONS_LIMIT, 1);
    }

    #[test]
    fn independent_circuit_builds_run_concurrently() {
        let ready = Arc::new(AtomicUsize::new(0));
        let build = |name| {
            let ready = Arc::clone(&ready);
            move || {
                ready.fetch_add(1, Ordering::SeqCst);
                let deadline = Instant::now() + Duration::from_secs(1);
                while ready.load(Ordering::SeqCst) != 3 && Instant::now() < deadline {
                    std::thread::yield_now();
                }
                (name, ready.load(Ordering::SeqCst) == 3)
            }
        };

        let (heavy, light, pre) = join_three_builds(build("heavy"), build("light"), build("pre"));

        assert_eq!(heavy, ("heavy", true));
        assert_eq!(light, ("light", true));
        assert_eq!(pre, ("pre", true));
    }
}
