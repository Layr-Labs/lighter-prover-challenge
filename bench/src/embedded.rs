// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Embedded startup circuits.
//!
//! `build.rs` constructs the five startup circuits during the untimed compile
//! job and serializes them (see `circuit::embed`) into OUT_DIR blobs that are
//! compiled into this binary. [`Circuits::from_embedded`] reconstitutes the
//! exact `Circuits` value `Circuits::new` builds, several times faster than
//! rebuilding, moving that work out of the scored worker lifetime.
//!
//! [`Circuits::load_pre`] and [`Circuits::load_remaining`] are the production
//! entry points: embedded first, build fallback on any error,
//! `LIGHTER_BUILD_CIRCUITS=1` to force the build path (measurement A/B). The
//! `embedded_matches_rebuilt` ignored test is the value-equality oracle.

use anyhow::Context as _;
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::BlockTxChainTarget;
use circuit::block_tx_constraints::BlockTxTarget;
use circuit::embed::deserialize_embedded;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use plonky2::plonk::circuit_data::CircuitData;

use crate::api::{
    Circuits, HEAVY_TX_MODE, HEAVY_TX_PER_PROOF, LIGHT_TX_MODE, LIGHT_TX_PER_PROOF, PathCircuits,
};

static PRE_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pre.embed"));
static HEAVY_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_tx.embed"));
static HEAVY_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_chain.embed"));
static LIGHT_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_tx.embed"));
static LIGHT_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_chain.embed"));

fn load_blob<T: serde::de::DeserializeOwned>(
    name: &'static str,
    blob: &[u8],
) -> anyhow::Result<(T, CircuitData<F, C, D>)> {
    anyhow::ensure!(
        !blob.is_empty(),
        "embedded circuit blob {name} is an empty stub (compiled with LIGHTER_SKIP_EMBED=1)"
    );
    deserialize_embedded::<T>(blob)
        .map_err(|error| error.context(format!("loading embedded circuit {name}")))
}

pub(crate) struct RemainingCircuits {
    heavy: PathCircuits,
    light: PathCircuits,
}

fn load_embedded_path(
    tx_name: &'static str,
    tx_blob: &'static [u8],
    chain_name: &'static str,
    chain_blob: &'static [u8],
    dummy_proof: &'static [u8],
) -> anyhow::Result<PathCircuits> {
    let (tx, chain) = rayon::join(
        || load_blob::<BlockTxTarget>(tx_name, tx_blob),
        || load_blob::<BlockTxChainTarget>(chain_name, chain_blob),
    );
    let (tx_target, tx_data) = tx?;
    let (chain_target, chain_data) = chain?;
    let dummy_proof =
        bincode::deserialize(dummy_proof).context("embedded chain dummy proof is invalid")?;
    Ok(PathCircuits {
        tx_target,
        tx_data,
        chain_target,
        chain_data,
        dummy_proof,
    })
}

impl RemainingCircuits {
    fn from_embedded() -> anyhow::Result<Self> {
        let (heavy, light) = rayon::join(
            || {
                load_embedded_path(
                    "heavy_tx",
                    HEAVY_TX_BLOB,
                    "heavy_chain",
                    HEAVY_CHAIN_BLOB,
                    include_bytes!("../dummy-heavy-chain-proof.bin"),
                )
            },
            || {
                load_embedded_path(
                    "light_tx",
                    LIGHT_TX_BLOB,
                    "light_chain",
                    LIGHT_CHAIN_BLOB,
                    include_bytes!("../dummy-light-chain-proof.bin"),
                )
            },
        );
        Ok(Self {
            heavy: heavy?,
            light: light?,
        })
    }

    fn build() -> Self {
        let (heavy, light) = rayon::join(
            || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
            || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
        );
        Self { heavy, light }
    }

    pub(crate) fn with_pre(
        self,
        (pre_target, pre_data): (BlockPreExecutionTarget, CircuitData<F, C, D>),
    ) -> Circuits {
        Circuits {
            heavy_tx_target: self.heavy.tx_target,
            heavy_tx_data: self.heavy.tx_data,
            light_tx_target: self.light.tx_target,
            light_tx_data: self.light.tx_data,
            pre_target,
            pre_data,
            heavy_chain_target: self.heavy.chain_target,
            heavy_chain_data: self.heavy.chain_data,
            light_chain_target: self.light.chain_target,
            light_chain_data: self.light.chain_data,
            dummy_heavy_proof: self.heavy.dummy_proof,
            dummy_light_proof: self.light.dummy_proof,
        }
    }
}

fn force_circuit_build() -> bool {
    std::env::var_os("LIGHTER_BUILD_CIRCUITS").is_some_and(|value| value == "1")
}

fn build_pre() -> (BlockPreExecutionTarget, CircuitData<F, C, D>) {
    let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    (pre.target, pre.builder.build::<C>())
}

fn load_or_build_circuits<T>(
    force_build: bool,
    load: impl FnOnce() -> anyhow::Result<T>,
    build: impl FnOnce() -> T,
    on_load_error: impl FnOnce(anyhow::Error),
) -> T {
    if force_build {
        return build();
    }
    match load() {
        Ok(circuits) => circuits,
        Err(error) => {
            on_load_error(error);
            build()
        }
    }
}

impl Circuits {
    /// Loads only the pre-execution circuit. The startup-overlap path proves it
    /// while the other four circuits load, then reunites both owned parts.
    pub(crate) fn load_pre() -> (BlockPreExecutionTarget, CircuitData<F, C, D>) {
        load_or_build_circuits(
            force_circuit_build(),
            || load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB),
            build_pre,
            |error| {
                log::warn!("embedded pre circuit unavailable ({error:#}); building from scratch");
            },
        )
    }

    /// Loads the four transaction and chain circuits without touching the pre
    /// blob. This removes the duplicate pre deserialization from startup.
    pub(crate) fn load_remaining() -> RemainingCircuits {
        let force_build = force_circuit_build();
        if force_build {
            log::info!("LIGHTER_BUILD_CIRCUITS=1: building startup circuits from scratch");
        }
        load_or_build_circuits(
            force_build,
            RemainingCircuits::from_embedded,
            RemainingCircuits::build,
            |error| {
                log::warn!(
                    "remaining embedded circuits unavailable ({error:#}); building from scratch"
                );
            },
        )
    }

    /// Reconstructs all five startup circuits from the blobs embedded at
    /// compile time. Value-identical to [`Circuits::new`] (oracle:
    /// `embedded_matches_rebuilt`); errors if the blobs are absent, corrupt,
    /// or fail their internal commitment-cap check.
    pub fn from_embedded() -> anyhow::Result<Self> {
        // Same parallel layout as `Circuits::new`; the five loads are
        // independent (unlike builds, the chain loads do not wait on the
        // transaction circuits).
        let (pre, remaining) = rayon::join(
            || load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB),
            RemainingCircuits::from_embedded,
        );
        Ok(remaining?.with_pre(pre?))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use circuit::circuit_serializer::BlockGateSerializer;
    use circuit::embed::EmbedGeneratorSerializer;
    use plonky2::util::serialization::Write as _;

    use super::*;
    use crate::api::PROVER_THREAD_STACK_BYTES;

    #[test]
    fn circuit_load_policy_prefers_embedded_data() {
        let value = load_or_build_circuits(
            false,
            || Ok(7),
            || panic!("successful embedded load must not build"),
            |_| panic!("successful embedded load must not report an error"),
        );
        assert_eq!(value, 7);
    }

    #[test]
    fn circuit_load_policy_honors_forced_build() {
        let value = load_or_build_circuits(
            true,
            || panic!("forced build must not read embedded data"),
            || 11,
            |_| panic!("forced build must not report an embedded error"),
        );
        assert_eq!(value, 11);
    }

    #[test]
    fn circuit_load_policy_builds_after_embedded_failure() {
        let error_reported = Cell::new(false);
        let value = load_or_build_circuits(
            false,
            || anyhow::bail!("corrupt embedded data"),
            || 13,
            |_| error_reported.set(true),
        );
        assert_eq!(value, 13);
        assert!(error_reported.get());
    }

    fn on_big_stack(f: impl FnOnce() + Send + 'static) {
        // Mirror the prove binary's startup: circuit construction recurses
        // deeply on rayon workers, which need the production stack size (the
        // global pool can only be configured once per process; a second call
        // in the same test binary is fine to ignore).
        let _ = rayon::ThreadPoolBuilder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .build_global();
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(f)
            .expect("test thread must start")
            .join()
            .expect("test thread must finish");
    }

    fn generator_stream_bytes(data: &CircuitData<F, C, D>) -> Vec<u8> {
        let serializer = EmbedGeneratorSerializer {
            _phantom: Default::default(),
            _phantom2: Default::default(),
        };
        let mut bytes = Vec::new();
        for generator in &data.prover_only.generators {
            bytes
                .write_generator::<F, D>(generator, &serializer, &data.common)
                .expect("generator must serialize");
        }
        bytes
    }

    fn assert_circuit_pair_identical<T: serde::Serialize>(
        name: &str,
        rebuilt: (&T, &CircuitData<F, C, D>),
        embedded: (&T, &CircuitData<F, C, D>),
    ) {
        let (rebuilt_target, rebuilt_data) = rebuilt;
        let (embedded_target, embedded_data) = embedded;

        // Targets: byte-for-byte identical serialization.
        assert!(
            bincode::serialize(rebuilt_target).unwrap()
                == bincode::serialize(embedded_target).unwrap(),
            "{name}: target struct diverges"
        );

        // The named headline checks, each with its own message.
        assert!(
            rebuilt_data.prover_only.circuit_digest == embedded_data.prover_only.circuit_digest,
            "{name}: circuit digest diverges"
        );
        assert!(
            rebuilt_data.verifier_only == embedded_data.verifier_only,
            "{name}: verifier-only data diverges"
        );
        assert!(
            rebuilt_data
                .prover_only
                .constants_sigmas_commitment
                .merkle_tree
                .cap
                == embedded_data
                    .prover_only
                    .constants_sigmas_commitment
                    .merkle_tree
                    .cap,
            "{name}: constants/sigmas cap diverges"
        );
        assert!(
            rebuilt_data.prover_only.sigmas[777] == embedded_data.prover_only.sigmas[777],
            "{name}: sigmas[777] diverges"
        );
        assert!(
            rebuilt_data.prover_only.generators.len()
                == embedded_data.prover_only.generators.len(),
            "{name}: generator count diverges"
        );
        for index in [0, 1000, rebuilt_data.prover_only.generators.len() - 1] {
            assert!(
                rebuilt_data.prover_only.generators[index].0.id()
                    == embedded_data.prover_only.generators[index].0.id(),
                "{name}: generator #{index} id diverges"
            );
        }

        // Generators: full byte-level equality of the serialized streams
        // (`WitnessGeneratorRef::eq` compares ids only, so the whole-struct
        // equality below would not catch a divergent generator payload).
        assert!(
            generator_stream_bytes(rebuilt_data) == generator_stream_bytes(embedded_data),
            "{name}: serialized generator streams diverge"
        );

        // Common: full equality.
        assert!(
            rebuilt_data.common == embedded_data.common,
            "{name}: common circuit data diverges"
        );

        // Prover-only: full structural equality (commitment polynomials and
        // Merkle tree, sigmas, subgroup, public inputs, representative map,
        // watch index + counts, fft root table, lookups; generators by id).
        assert!(
            rebuilt_data.prover_only == embedded_data.prover_only,
            "{name}: prover-only circuit data diverges"
        );
    }

    fn assert_circuits_identical(expected: &Circuits, actual: &Circuits) {
        assert_circuit_pair_identical(
            "pre",
            (&expected.pre_target, &expected.pre_data),
            (&actual.pre_target, &actual.pre_data),
        );
        assert_circuit_pair_identical(
            "heavy_tx",
            (&expected.heavy_tx_target, &expected.heavy_tx_data),
            (&actual.heavy_tx_target, &actual.heavy_tx_data),
        );
        assert_circuit_pair_identical(
            "heavy_chain",
            (&expected.heavy_chain_target, &expected.heavy_chain_data),
            (&actual.heavy_chain_target, &actual.heavy_chain_data),
        );
        assert_circuit_pair_identical(
            "light_tx",
            (&expected.light_tx_target, &expected.light_tx_data),
            (&actual.light_tx_target, &actual.light_tx_data),
        );
        assert_circuit_pair_identical(
            "light_chain",
            (&expected.light_chain_target, &expected.light_chain_data),
            (&actual.light_chain_target, &actual.light_chain_data),
        );
    }

    /// Determinism oracle for the embed mechanism: builds all five circuits
    /// from scratch AND loads the embedded set, then asserts value identity.
    /// This is the gate for `Circuits::from_embedded` — if it fails, the
    /// mechanism is wrong. Run:
    /// `cargo test --release -p bench --bin prove -- --ignored embedded_matches_rebuilt --nocapture`
    #[test]
    #[ignore = "multi-second circuit rebuild; run explicitly"]
    fn embedded_matches_rebuilt() {
        on_big_stack(|| {
            let rebuilt = Circuits::new();
            let embedded = Circuits::from_embedded()
                .expect("embedded circuits must load when blobs are compiled in");

            assert_circuits_identical(&rebuilt, &embedded);

            // The gate serializer round trip below also pins the common data
            // encoding used by the blobs.
            let mut bytes = Vec::new();
            bytes
                .write_common_circuit_data(&rebuilt.light_tx_data.common, &BlockGateSerializer)
                .expect("common data must serialize");
            assert!(!bytes.is_empty());

            println!("embedded_matches_rebuilt: all five circuits are value-identical");
        });
    }

    #[test]
    fn split_embedded_load_matches_full_embedded_load() {
        on_big_stack(|| {
            let full = Circuits::from_embedded()
                .expect("embedded circuits must load when blobs are compiled in");
            let pre = load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
                .expect("embedded pre circuit must load");
            let split = RemainingCircuits::from_embedded()
                .expect("remaining embedded circuits must load")
                .with_pre(pre);

            assert_circuits_identical(&full, &split);
        });
    }

    /// Manual timing harness: embedded load vs fresh build, both under the
    /// production overlapped layout and per circuit sequentially. Run:
    /// `cargo test --release -p bench --bin prove -- --ignored embedded_load_timing --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn embedded_load_timing() {
        use std::time::Instant;

        on_big_stack(|| {
            // Process-cold embedded load first: this is the number the scored
            // worker pays (its first circuit work is exactly this call).
            let t = Instant::now();
            let embedded_cold = Circuits::from_embedded().expect("embedded circuits must load");
            let t_embedded_cold = t.elapsed();
            drop(embedded_cold);

            // Sequential per-circuit loads (inner rayon parallelism still
            // active, matching how a single lane would run).
            let seq = |name: &str, f: &dyn Fn()| {
                let t = Instant::now();
                f();
                println!("  sequential {name:<12} {:>10.1?}", t.elapsed());
            };
            let t = Instant::now();
            seq("pre", &|| {
                drop(load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB).unwrap())
            });
            seq("heavy_tx", &|| {
                drop(load_blob::<BlockTxTarget>("heavy_tx", HEAVY_TX_BLOB).unwrap())
            });
            seq("heavy_chain", &|| {
                drop(load_blob::<BlockTxChainTarget>("heavy_chain", HEAVY_CHAIN_BLOB).unwrap())
            });
            seq("light_tx", &|| {
                drop(load_blob::<BlockTxTarget>("light_tx", LIGHT_TX_BLOB).unwrap())
            });
            seq("light_chain", &|| {
                drop(load_blob::<BlockTxChainTarget>("light_chain", LIGHT_CHAIN_BLOB).unwrap())
            });
            let t_embedded_sequential = t.elapsed();

            // The build path under its production overlapped layout.
            let t = Instant::now();
            let rebuilt = Circuits::new();
            let t_rebuild = t.elapsed();
            drop(rebuilt);

            // Warm embedded load for comparison (allocator/GPU warmed by the
            // preceding work).
            let t = Instant::now();
            let embedded_warm = Circuits::from_embedded().expect("embedded circuits must load");
            let t_embedded_warm = t.elapsed();
            drop(embedded_warm);

            println!("\nembedded (overlapped, process-cold): {t_embedded_cold:>10.1?}");
            println!("embedded (sequential, one lane at a time): {t_embedded_sequential:>10.1?}");
            println!("rebuild  (overlapped `Circuits::new`):   {t_rebuild:>10.1?}");
            println!("embedded (overlapped, warm):             {t_embedded_warm:>10.1?}");
            println!(
                "net startup win (cold embedded vs rebuild): {:+.1} ms",
                (t_rebuild.as_secs_f64() - t_embedded_cold.as_secs_f64()) * 1e3
            );
        });
    }

    /// The embedded blobs must be present and non-empty in a normal compile
    /// (guards against LIGHTER_SKIP_EMBED stubs sneaking into a submission).
    #[test]
    fn embedded_blobs_are_compiled_in() {
        for (name, blob) in [
            ("pre", PRE_BLOB),
            ("heavy_tx", HEAVY_TX_BLOB),
            ("heavy_chain", HEAVY_CHAIN_BLOB),
            ("light_tx", LIGHT_TX_BLOB),
            ("light_chain", LIGHT_CHAIN_BLOB),
        ] {
            assert!(
                !blob.is_empty(),
                "embedded circuit blob {name} is an empty stub"
            );
        }
    }
}
