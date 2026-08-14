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
//! [`Circuits::load`] is the production entry point: embedded first, build
//! fallback on any error, `LIGHTER_BUILD_CIRCUITS=1` to force the build path
//! (measurement A/B). The `embedded_matches_rebuilt` ignored test is the
//! value-equality oracle between the two paths.

use circuit::block_pre_execution_constraints::BlockPreExecutionTarget;
use circuit::block_tx_chain_constraints::BlockTxChainTarget;
use circuit::block_tx_constraints::BlockTxTarget;
use circuit::embed::{deserialize_embedded, deserialize_embedded_with_commitment};
use circuit::types::config::{C, D, F};
use plonky2::plonk::circuit_data::CircuitData;

use crate::api::{Circuits, Proof};

static PRE_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pre.embed"));
const PRE_COMMITMENT_CACHE_PATH: &str =
    concat!(env!("OUT_DIR"), "/pre.commitment-cache");
static HEAVY_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_tx.embed"));
static HEAVY_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_chain.embed"));
static LIGHT_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_tx.embed"));
static LIGHT_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_chain.embed"));

/// The four startup circuits that do not participate in pre-execution. Keeping
/// this separate lets the worker start the pre-execution proof from its already
/// decoded circuit while these independent blobs load in parallel.
pub(crate) struct RemainingEmbeddedCircuits {
    heavy_tx: (BlockTxTarget, CircuitData<F, C, D>),
    heavy_chain: (BlockTxChainTarget, CircuitData<F, C, D>),
    light_tx: (BlockTxTarget, CircuitData<F, C, D>),
    light_chain: (BlockTxChainTarget, CircuitData<F, C, D>),
    dummy_heavy_proof: Proof,
    dummy_light_proof: Proof,
}

impl RemainingEmbeddedCircuits {
    pub(crate) fn into_circuits(
        self,
        pre: (BlockPreExecutionTarget, CircuitData<F, C, D>),
    ) -> Circuits {
        let (pre_target, pre_data) = pre;
        let (heavy_tx_target, heavy_tx_data) = self.heavy_tx;
        let (heavy_chain_target, heavy_chain_data) = self.heavy_chain;
        let (light_tx_target, light_tx_data) = self.light_tx;
        let (light_chain_target, light_chain_data) = self.light_chain;
        Circuits {
            heavy_tx_target,
            heavy_tx_data: std::sync::RwLock::new(heavy_tx_data),
            light_tx_target,
            light_tx_data: std::sync::RwLock::new(light_tx_data),
            pre_target,
            pre_data,
            heavy_chain_target,
            heavy_chain_data: std::sync::RwLock::new(heavy_chain_data),
            light_chain_target,
            light_chain_data: std::sync::RwLock::new(light_chain_data),
            dummy_heavy_proof: self.dummy_heavy_proof,
            dummy_light_proof: self.dummy_light_proof,
        }
    }
}

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

fn load_pre_cached() -> anyhow::Result<(BlockPreExecutionTarget, CircuitData<F, C, D>)> {
    anyhow::ensure!(
        !PRE_BLOB.is_empty(),
        "embedded circuit blob pre is an empty stub (compiled with LIGHTER_SKIP_EMBED=1)"
    );
    let cache = std::fs::read(PRE_COMMITMENT_CACHE_PATH).map_err(|error| {
        anyhow::Error::from(error).context("reading pre-execution commitment cache")
    })?;
    anyhow::ensure!(
        !cache.is_empty(),
        "pre-execution commitment cache is an empty stub (compiled with LIGHTER_SKIP_EMBED=1)"
    );
    deserialize_embedded_with_commitment::<BlockPreExecutionTarget>(PRE_BLOB, &cache)
        .map_err(|error| error.context("loading cached pre-execution commitment"))
}

impl Circuits {
    /// Loads only the pre-execution circuit blob. This is the fast path used
    /// by the startup overlap: the pre-execution proof can start (and hide)
    /// behind the remaining circuit loads.
    pub fn load_pre() -> anyhow::Result<(BlockPreExecutionTarget, CircuitData<F, C, D>)> {
        if std::env::var_os("LIGHTER_PRE_COMMITMENT_CACHE").is_some_and(|value| value == "0") {
            return load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB);
        }

        match load_pre_cached() {
            Ok(pre) => Ok(pre),
            Err(error) => {
                log::warn!(
                    "pre-execution commitment cache unavailable ({error:#}); using compact loader"
                );
                load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
            }
        }
    }

    /// Loads every embedded circuit except pre-execution. This is public to
    /// the worker startup path only; normal callers should keep using
    /// [`Self::load`].
    pub(crate) fn load_remaining_embedded() -> anyhow::Result<RemainingEmbeddedCircuits> {
        let (heavy, light) = rayon::join(
            || {
                rayon::join(
                    || load_blob::<BlockTxTarget>("heavy_tx", HEAVY_TX_BLOB),
                    || load_blob::<BlockTxChainTarget>("heavy_chain", HEAVY_CHAIN_BLOB),
                )
            },
            || {
                rayon::join(
                    || load_blob::<BlockTxTarget>("light_tx", LIGHT_TX_BLOB),
                    || load_blob::<BlockTxChainTarget>("light_chain", LIGHT_CHAIN_BLOB),
                )
            },
        );
        let (heavy_tx, heavy_chain) = (heavy.0?, heavy.1?);
        let (light_tx, light_chain) = (light.0?, light.1?);

        let dummy_heavy_proof: Proof =
            bincode::deserialize(include_bytes!("../dummy-heavy-chain-proof.bin"))
                .expect("embedded heavy chain dummy proof is invalid");
        let dummy_light_proof: Proof =
            bincode::deserialize(include_bytes!("../dummy-light-chain-proof.bin"))
                .expect("embedded light chain dummy proof is invalid");

        Ok(RemainingEmbeddedCircuits {
            heavy_tx,
            heavy_chain,
            light_tx,
            light_chain,
            dummy_heavy_proof,
            dummy_light_proof,
        })
    }

    /// Reconstructs all five startup circuits from the blobs embedded at
    /// compile time. Value-identical to [`Circuits::new`] (oracle:
    /// `embedded_matches_rebuilt`); errors if the blobs are absent, corrupt,
    /// or fail their internal commitment-cap check.
    pub fn from_embedded() -> anyhow::Result<Self> {
        // Same parallel layout as `Circuits::new`; the five loads are
        // independent (unlike builds, the chain loads do not wait on the
        // transaction circuits).
        let (pre, remaining) = rayon::join(Self::load_pre, Self::load_remaining_embedded);
        Ok(remaining?.into_circuits(pre?))
    }

    /// Production loader: embedded circuits when available, otherwise a fresh
    /// build. `LIGHTER_BUILD_CIRCUITS=1` forces the build path for A/B runs.
    pub fn load() -> Self {
        if std::env::var_os("LIGHTER_BUILD_CIRCUITS").is_some_and(|v| v == "1") {
            log::info!("LIGHTER_BUILD_CIRCUITS=1: building startup circuits from scratch");
            return Self::new();
        }
        match Self::from_embedded() {
            Ok(circuits) => circuits,
            Err(error) => {
                log::warn!("embedded circuits unavailable ({error:#}); building from scratch");
                Self::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use circuit::circuit_serializer::BlockGateSerializer;
    use circuit::embed::EmbedGeneratorSerializer;
    use plonky2::hash::merkle_tree::MerkleLeaves;
    use plonky2::util::serialization::Write as _;

    use super::*;
    use crate::api::PROVER_THREAD_STACK_BYTES;

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

    fn assert_pre_cache_identical(
        compact: &(BlockPreExecutionTarget, CircuitData<F, C, D>),
        cached: &(BlockPreExecutionTarget, CircuitData<F, C, D>),
    ) {
        let (compact_target, compact_data) = compact;
        let (cached_target, cached_data) = cached;
        assert_eq!(
            bincode::serialize(compact_target).unwrap(),
            bincode::serialize(cached_target).unwrap(),
            "pre target diverges"
        );
        assert_eq!(compact_data.common, cached_data.common, "pre common data");
        assert_eq!(
            compact_data.verifier_only, cached_data.verifier_only,
            "pre verifier data"
        );

        let compact_prover = &compact_data.prover_only;
        let cached_prover = &cached_data.prover_only;
        assert_eq!(
            generator_stream_bytes(compact_data),
            generator_stream_bytes(cached_data),
            "pre generator stream"
        );
        assert_eq!(
            compact_prover.generator_indices_by_watches,
            cached_prover.generator_indices_by_watches,
            "pre generator watch index"
        );
        assert_eq!(
            compact_prover.generator_watch_counts, cached_prover.generator_watch_counts,
            "pre generator watch counts"
        );
        assert_eq!(compact_prover.sigmas, cached_prover.sigmas, "pre sigmas");
        assert_eq!(compact_prover.subgroup, cached_prover.subgroup, "pre subgroup");
        assert_eq!(
            compact_prover.public_inputs, cached_prover.public_inputs,
            "pre public inputs"
        );
        assert_eq!(
            compact_prover.representative_map, cached_prover.representative_map,
            "pre representative map"
        );
        assert_eq!(
            compact_prover.fixed_routed_wires, cached_prover.fixed_routed_wires,
            "pre fixed routed wires"
        );
        assert_eq!(
            compact_prover.fft_root_table, cached_prover.fft_root_table,
            "pre FFT roots"
        );
        assert_eq!(
            compact_prover.circuit_digest, cached_prover.circuit_digest,
            "pre circuit digest"
        );
        assert_eq!(compact_prover.lookup_rows, cached_prover.lookup_rows, "pre lookups");
        assert_eq!(
            compact_prover.lut_to_lookups, cached_prover.lut_to_lookups,
            "pre lookup tables"
        );
        assert_eq!(
            compact_prover.constants_sigmas_quotient_cache,
            cached_prover.constants_sigmas_quotient_cache,
            "pre quotient cache"
        );
        assert_eq!(
            compact_prover.constants_sigmas_quotient_step,
            cached_prover.constants_sigmas_quotient_step,
            "pre quotient stride"
        );
        assert_eq!(
            compact_prover.constants_sigmas_quotient_domain,
            cached_prover.constants_sigmas_quotient_domain,
            "pre quotient domain"
        );

        let compact_commitment = &compact_prover.constants_sigmas_commitment;
        let cached_commitment = &cached_prover.constants_sigmas_commitment;
        assert_eq!(compact_commitment.degree_log, cached_commitment.degree_log);
        assert_eq!(compact_commitment.rate_bits, cached_commitment.rate_bits);
        assert_eq!(compact_commitment.blinding, cached_commitment.blinding);
        assert_eq!(
            compact_commitment.polynomials.len(),
            cached_commitment.polynomials.len()
        );
        for (compact_poly, cached_poly) in compact_commitment
            .polynomials
            .iter()
            .zip(&cached_commitment.polynomials)
        {
            assert_eq!(compact_poly.coeffs.len(), cached_poly.coeffs.len());
            for (&compact_value, &cached_value) in
                compact_poly.coeffs.iter().zip(&cached_poly.coeffs)
            {
                assert_eq!(compact_value.0, cached_value.0, "raw coefficient diverges");
            }
        }

        let compact_tree = &compact_commitment.merkle_tree;
        let cached_tree = &cached_commitment.merkle_tree;
        assert_eq!(compact_tree.num_leaves, cached_tree.num_leaves);
        assert_eq!(compact_tree.cap, cached_tree.cap);
        match (&compact_tree.leaves, &cached_tree.leaves) {
            (
                MerkleLeaves::Columns {
                    columns: compact_columns,
                    log_rows: compact_log_rows,
                },
                MerkleLeaves::Columns {
                    columns: cached_columns,
                    log_rows: cached_log_rows,
                },
            ) => {
                assert_eq!(compact_log_rows, cached_log_rows);
                assert_eq!(compact_columns.num_cols(), cached_columns.num_cols());
                assert_eq!(compact_columns.num_rows(), cached_columns.num_rows());
                for column in 0..compact_columns.num_cols() {
                    for (&compact_value, &cached_value) in compact_columns
                        .col(column)
                        .iter()
                        .zip(cached_columns.col(column))
                    {
                        assert_eq!(compact_value.0, cached_value.0, "raw LDE value diverges");
                    }
                }
            }
            (MerkleLeaves::Rows { data: compact, width: compact_width }, MerkleLeaves::Rows { data: cached, width: cached_width }) => {
                assert_eq!(compact_width, cached_width);
                for (&compact_value, &cached_value) in compact.iter().zip(cached) {
                    assert_eq!(compact_value.0, cached_value.0, "raw row value diverges");
                }
            }
            _ => panic!("pre commitment leaf layout diverges"),
        }
        for index in [0, compact_tree.num_leaves / 3, compact_tree.num_leaves - 1] {
            assert_eq!(compact_tree.leaf_vec(index), cached_tree.leaf_vec(index));
            assert_eq!(compact_tree.prove(index), cached_tree.prove(index));
        }
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

            assert_circuit_pair_identical(
                "pre",
                (&rebuilt.pre_target, &rebuilt.pre_data),
                (&embedded.pre_target, &embedded.pre_data),
            );
            assert_circuit_pair_identical(
                "heavy_tx",
                (
                    &rebuilt.heavy_tx_target,
                    &rebuilt.heavy_tx_data.read().unwrap(),
                ),
                (
                    &embedded.heavy_tx_target,
                    &embedded.heavy_tx_data.read().unwrap(),
                ),
            );
            assert_circuit_pair_identical(
                "heavy_chain",
                (
                    &rebuilt.heavy_chain_target,
                    &rebuilt.heavy_chain_data.read().unwrap(),
                ),
                (
                    &embedded.heavy_chain_target,
                    &embedded.heavy_chain_data.read().unwrap(),
                ),
            );
            assert_circuit_pair_identical(
                "light_tx",
                (
                    &rebuilt.light_tx_target,
                    &rebuilt.light_tx_data.read().unwrap(),
                ),
                (
                    &embedded.light_tx_target,
                    &embedded.light_tx_data.read().unwrap(),
                ),
            );
            assert_circuit_pair_identical(
                "light_chain",
                (
                    &rebuilt.light_chain_target,
                    &rebuilt.light_chain_data.read().unwrap(),
                ),
                (
                    &embedded.light_chain_target,
                    &embedded.light_chain_data.read().unwrap(),
                ),
            );

            // The gate serializer round trip below also pins the common data
            // encoding used by the blobs.
            let mut bytes = Vec::new();
            bytes
                .write_common_circuit_data(
                    &rebuilt.light_tx_data.read().unwrap().common,
                    &BlockGateSerializer,
                )
                .expect("common data must serialize");
            assert!(!bytes.is_empty());

            println!("embedded_matches_rebuilt: all five circuits are value-identical");
        });
    }

    /// Exact differential for the external pre commitment cache, including
    /// raw Goldilocks representations for every coefficient and LDE value.
    #[test]
    fn pre_commitment_cache_matches_compact() {
        on_big_stack(|| {
            let compact = load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
                .expect("compact pre circuit must load");
            let cached = load_pre_cached().expect("cached pre circuit must load");
            assert_pre_cache_identical(&compact, &cached);
        });
    }

    /// Isolated station harness. Each arm performs its production input path:
    /// compact reads linked bytes, while cache reads the external OUT_DIR file.
    #[test]
    #[ignore = "manual 7-sample-per-arm pre loader station"]
    fn pre_commitment_cache_load_timing() {
        use std::time::Instant;

        on_big_stack(|| {
            for round in 0..7 {
                let run = |arm: &str| {
                    let start = Instant::now();
                    let loaded = if arm == "compact" {
                        load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
                    } else {
                        load_pre_cached()
                    }
                    .expect("pre loader timing arm must succeed");
                    let elapsed = start.elapsed();
                    assert_eq!(
                        loaded.1.verifier_only.constants_sigmas_cap,
                        loaded
                            .1
                            .prover_only
                            .constants_sigmas_commitment
                            .merkle_tree
                            .cap
                    );
                    println!(
                        "pre_commitment_cache_load round={round} arm={arm} ns={}",
                        elapsed.as_nanos()
                    );
                    drop(loaded);
                };
                if round % 2 == 0 {
                    run("compact");
                    run("cached");
                } else {
                    run("cached");
                    run("compact");
                }
            }
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
        assert!(
            std::fs::metadata(PRE_COMMITMENT_CACHE_PATH)
                .is_ok_and(|metadata| metadata.len() > 0),
            "pre-execution commitment cache is missing or empty"
        );
    }
}
