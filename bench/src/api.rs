// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::path::PathBuf;
use std::sync::OnceLock;

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::circuit_serializer::{BlockGateSerializer, BlockGeneratorSerializer};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;

pub type Proof = ProofWithPublicInputs<F, C, D>;

type BlockGenSer = BlockGeneratorSerializer<C, D, circuit::ecdsa::curve::secp256k1::Secp256K1>;

/// Directory for cached preprocessed circuit data, set once by the worker from
/// its writable output directory before circuit construction. Circuit builds
/// are deterministic per binary, so cache entries are keyed by a fingerprint of
/// the running executable itself: any code change produces a different binary
/// hash and therefore can never read a stale entry. When the directory is
/// unset (tests) or entries are absent/unreadable (fresh run directory), every
/// circuit is built exactly as before and, on a best-effort basis, stored for
/// the next worker invocation in the same run directory.
static CIRCUIT_CACHE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn set_circuit_cache_dir(dir: Option<PathBuf>) {
    let _ = CIRCUIT_CACHE_DIR.set(dir);
}

fn binary_fingerprint() -> Option<&'static str> {
    static FINGERPRINT: OnceLock<Option<String>> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            use sha2::{Digest, Sha256};
            let exe = std::env::current_exe().ok()?;
            let bytes = std::fs::read(exe).ok()?;
            let digest = Sha256::digest(&bytes);
            Some(digest[..8].iter().map(|b| format!("{b:02x}")).collect())
        })
        .as_deref()
}

fn circuit_cache_path(name: &str) -> Option<PathBuf> {
    let dir = CIRCUIT_CACHE_DIR.get()?.as_ref()?;
    Some(dir.join(format!("circuit-cache-{}-{name}.bin", binary_fingerprint()?)))
}

fn cached_build(name: &str, build: impl FnOnce() -> CircuitData<F, C, D>) -> CircuitData<F, C, D> {
    let path = circuit_cache_path(name);
    if let Some(path) = &path {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(data) =
                CircuitData::from_bytes(&bytes, &BlockGateSerializer, &BlockGenSer::default())
            {
                return data;
            }
        }
    }
    let data = build();
    if let Some(path) = path {
        if let Ok(bytes) = data.to_bytes(&BlockGateSerializer, &BlockGenSer::default()) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }
    data
}

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;
pub const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

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
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

struct PathCircuits {
    tx_target: BlockTxTarget,
    tx_data: CircuitData<F, C, D>,
    chain_target: BlockTxChainTarget,
    chain_data: CircuitData<F, C, D>,
    dummy_proof: Proof,
}

impl PathCircuits {
    fn new(tx_per_proof: usize, tx_mode: u8) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
        let tx_target = tx.target;
        let tx_data = cached_build(&format!("tx-{tx_mode}"), || tx.builder.build::<C>());

        let chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_target = chain.target;
        let chain_data = cached_build(&format!("chain-{tx_mode}"), || chain.builder.build::<C>());

        let proof_bytes: &[u8] = match tx_mode {
            TX_HEAVY => include_bytes!("../dummy-heavy-chain-proof.bin"),
            TX_LIGHT => include_bytes!("../dummy-light-chain-proof.bin"),
            _ => panic!("unsupported block transaction mode {tx_mode}"),
        };
        let dummy_proof =
            bincode::deserialize(proof_bytes).expect("embedded chain dummy proof is invalid");

        Self {
            tx_target,
            tx_data,
            chain_target,
            chain_data,
            dummy_proof,
        }
    }
}

impl Circuits {
    pub fn new() -> Self {
        let ((pre_target, pre_data), (heavy, light)) = rayon::join(
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, cached_build("pre", || pre.builder.build::<C>()))
            },
            || {
                rayon::join(
                    || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
                    || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
                )
            },
        );
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
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }

    /// Builds the final block circuit, which depends on the pre-execution and
    /// both chain circuits but is only needed for the final proof. Callers run
    /// this concurrently with transaction/chain proving.
    pub fn build_block_circuit(&self) -> (BlockTarget, CircuitData<F, C, D>) {
        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &self.pre_data,
            &self.light_chain_data,
            &self.heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        (
            block.target,
            cached_build("block", || block.builder.build::<C>()),
        )
    }
}

#[cfg(test)]
mod tests {
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
    fn embedded_chain_dummy_proofs_deserialize() {
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-heavy-chain-proof.bin"))
            .expect("embedded heavy dummy proof is invalid");
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-light-chain-proof.bin"))
            .expect("embedded light dummy proof is invalid");
    }
}
