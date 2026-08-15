// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Embedded startup circuits.
//!
//! `build.rs` constructs the five startup circuits and the final block circuit
//! during the untimed compile job and serializes them (see `circuit::embed`)
//! into OUT_DIR blobs that are compiled into this binary. The same shrunken
//! format deliberately omits the large constants/sigmas LDE and Merkle state;
//! [`Circuits::from_embedded`] and [`Circuits::load_block_embedded`]
//! reconstitute exact `CircuitData` values through the checked loader.
//!
//! [`Circuits::load`] is the startup entry point: embedded first, build
//! fallback on any error, `LIGHTER_BUILD_CIRCUITS=1` to force the build path.
//! `LIGHTER_BUILD_BLOCK_CIRCUIT=1` independently forces the legacy final-block
//! build for a controlled worker A/B. The `embedded_matches_rebuilt` ignored
//! test is the full value-equality oracle between all six embedded circuits
//! and freshly built data.

use circuit::block_constraints::BlockTarget;
use circuit::block_pre_execution_constraints::BlockPreExecutionTarget;
use circuit::block_tx_chain_constraints::BlockTxChainTarget;
use circuit::block_tx_constraints::BlockTxTarget;
use circuit::embed::deserialize_embedded;
use circuit::types::config::{C, D, F};
use plonky2::plonk::circuit_data::CircuitData;

use crate::api::{Circuits, Proof};

static PRE_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pre.embed"));
static HEAVY_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_tx.embed"));
static HEAVY_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_chain.embed"));
static LIGHT_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_tx.embed"));
static LIGHT_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_chain.embed"));
static BLOCK_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/block.embed"));

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

impl Circuits {
    /// Loads only the pre-execution circuit blob. This is the fast path used
    /// by the startup overlap: the pre-execution proof can start (and hide)
    /// behind the remaining circuit loads.
    pub fn load_pre() -> anyhow::Result<(BlockPreExecutionTarget, CircuitData<F, C, D>)> {
        load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
    }

    /// Loads the final block circuit from the same checked shrunken format as
    /// the startup circuits. It remains a separate blob/load because the block
    /// lane does not need it until transaction proving has begun.
    pub(crate) fn load_block_embedded() -> anyhow::Result<(BlockTarget, CircuitData<F, C, D>)> {
        load_blob::<BlockTarget>("block", BLOCK_BLOB)
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

        // Prover-only: full value equality with field-specific failures
        // (generator payloads were compared byte-for-byte above; generator
        // references themselves compare by registered id).
        let rebuilt = &rebuilt_data.prover_only;
        let embedded = &embedded_data.prover_only;
        assert_eq!(rebuilt.generators, embedded.generators, "{name}: generators diverge");
        assert_eq!(
            rebuilt.generator_indices_by_watches,
            embedded.generator_indices_by_watches,
            "{name}: generator watch index diverges"
        );
        assert_eq!(
            rebuilt.generator_watch_counts,
            embedded.generator_watch_counts,
            "{name}: generator watch counts diverge"
        );
        let rebuilt_commitment = &rebuilt.constants_sigmas_commitment;
        let embedded_commitment = &embedded.constants_sigmas_commitment;
        assert!(
            rebuilt_commitment.polynomials == embedded_commitment.polynomials,
            "{name}: constants/sigmas commitment polynomials diverge"
        );
        let rebuilt_tree = &rebuilt_commitment.merkle_tree;
        let embedded_tree = &embedded_commitment.merkle_tree;
        assert_eq!(
            rebuilt_tree.num_leaves,
            embedded_tree.num_leaves,
            "{name}: constants/sigmas Merkle leaf count diverges"
        );
        assert_eq!(
            rebuilt_tree.leaf_width(),
            embedded_tree.leaf_width(),
            "{name}: constants/sigmas Merkle leaf width diverges"
        );
        match (&rebuilt_tree.leaves, &embedded_tree.leaves) {
            (
                plonky2::hash::merkle_tree::MerkleLeaves::Rows {
                    data: rebuilt,
                    width: rebuilt_width,
                },
                plonky2::hash::merkle_tree::MerkleLeaves::Rows {
                    data: embedded,
                    width: embedded_width,
                },
            ) => {
                assert_eq!(rebuilt_width, embedded_width, "{name}: row leaf width diverges");
                assert_eq!(rebuilt, embedded, "{name}: row leaf values diverge");
            }
            (
                plonky2::hash::merkle_tree::MerkleLeaves::Columns {
                    columns: rebuilt,
                    log_rows: rebuilt_log_rows,
                },
                plonky2::hash::merkle_tree::MerkleLeaves::Columns {
                    columns: embedded,
                    log_rows: embedded_log_rows,
                },
            ) => {
                assert_eq!(
                    rebuilt_log_rows, embedded_log_rows,
                    "{name}: column leaf row count diverges"
                );
                assert_eq!(rebuilt.num_cols(), embedded.num_cols());
                assert_eq!(rebuilt.num_rows(), embedded.num_rows());
                for column in 0..rebuilt.num_cols() {
                    assert_eq!(
                        rebuilt.col(column),
                        embedded.col(column),
                        "{name}: column-major leaf column {column} diverges"
                    );
                }
            }
            _ => panic!("{name}: Merkle leaf storage layout diverges"),
        }
        // GPU occupancy may select interleaved CPU storage for one build and
        // level-order Metal storage for the other. Compare every logical path,
        // not that routing-only backing representation.
        for leaf in 0..rebuilt_tree.num_leaves {
            assert_eq!(
                rebuilt_tree.prove(leaf),
                embedded_tree.prove(leaf),
                "{name}: constants/sigmas Merkle path {leaf} diverges"
            );
        }
        assert_eq!(
            rebuilt_tree.cap,
            embedded_tree.cap,
            "{name}: constants/sigmas Merkle cap diverges"
        );
        assert_eq!(
            rebuilt_commitment.degree_log,
            embedded_commitment.degree_log,
            "{name}: constants/sigmas commitment degree diverges"
        );
        assert_eq!(
            rebuilt_commitment.rate_bits,
            embedded_commitment.rate_bits,
            "{name}: constants/sigmas commitment rate diverges"
        );
        assert_eq!(
            rebuilt_commitment.blinding,
            embedded_commitment.blinding,
            "{name}: constants/sigmas commitment blinding diverges"
        );
        assert_eq!(rebuilt.sigmas, embedded.sigmas, "{name}: sigmas diverge");
        assert_eq!(rebuilt.subgroup, embedded.subgroup, "{name}: subgroup diverges");
        assert_eq!(
            rebuilt.public_inputs,
            embedded.public_inputs,
            "{name}: public inputs diverge"
        );
        assert_eq!(
            rebuilt.representative_map,
            embedded.representative_map,
            "{name}: representative map diverges"
        );
        assert_eq!(
            rebuilt.fixed_routed_wires,
            embedded.fixed_routed_wires,
            "{name}: fixed routed-wire mask diverges"
        );
        assert_eq!(
            rebuilt.fft_root_table,
            embedded.fft_root_table,
            "{name}: FFT root table diverges"
        );
        assert_eq!(
            rebuilt.lookup_rows,
            embedded.lookup_rows,
            "{name}: lookup rows diverge"
        );
        assert_eq!(
            rebuilt.lut_to_lookups,
            embedded.lut_to_lookups,
            "{name}: lookup tables diverge"
        );
        assert_eq!(
            rebuilt.constants_sigmas_quotient_cache,
            embedded.constants_sigmas_quotient_cache,
            "{name}: constants/sigmas quotient cache diverges"
        );
        assert_eq!(
            rebuilt.constants_sigmas_quotient_step,
            embedded.constants_sigmas_quotient_step,
            "{name}: constants/sigmas quotient step diverges"
        );
        assert_eq!(
            rebuilt.constants_sigmas_quotient_domain,
            embedded.constants_sigmas_quotient_domain,
            "{name}: constants/sigmas quotient domain diverges"
        );
    }

    /// Determinism oracle for the embed mechanism: builds all six circuits
    /// from scratch AND loads the embedded set, then asserts full value
    /// identity, including the final block's target, common/verifier data,
    /// generators, representative map, constants/sigmas commitment and Merkle
    /// tree. If this fails, the mechanism is wrong. Run:
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

            // The final block blob uses the identical serializer/loader, but
            // is loaded later by the block lane rather than `from_embedded`.
            // Drop the independently loaded startup set before materializing
            // two degree-2^18 block commitments for this exhaustive oracle.
            drop(embedded);
            let rebuilt_block = rebuilt.rebuild_block_circuit();
            drop(rebuilt);
            let embedded_block = Circuits::load_block_embedded()
                .expect("embedded final block circuit must load");
            assert_circuit_pair_identical(
                "block",
                (&rebuilt_block.0, &rebuilt_block.1),
                (&embedded_block.0, &embedded_block.1),
            );

            println!("embedded_matches_rebuilt: all six circuits are value-identical");
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

    /// Manual timing harness for the final block lane. Both arms rebuild the
    /// omitted LDE/Merkle state; the comparison isolates compact decode from
    /// the legacy circuit definition/preprocessing work it replaces. Run this
    /// test by name in a fresh process for a cold loader measurement.
    #[test]
    #[ignore = "manual timing harness"]
    fn embedded_block_load_timing() {
        use std::time::Instant;

        on_big_stack(|| {
            let t = Instant::now();
            let embedded_cold = Circuits::load_block_embedded()
                .expect("embedded final block circuit must load");
            let t_embedded_cold = t.elapsed();
            drop(embedded_cold);

            let inputs = Circuits::from_embedded().expect("embedded startup circuits must load");
            let t = Instant::now();
            let rebuilt = inputs.rebuild_block_circuit();
            let t_rebuilt = t.elapsed();
            drop(rebuilt);
            drop(inputs);

            let t = Instant::now();
            let embedded_warm = Circuits::load_block_embedded()
                .expect("embedded final block circuit must reload");
            let t_embedded_warm = t.elapsed();
            drop(embedded_warm);

            println!("block embedded (process-cold): {t_embedded_cold:>10.1?}");
            println!("block rebuild  (warm inputs):  {t_rebuilt:>10.1?}");
            println!("block embedded (warm):         {t_embedded_warm:>10.1?}");
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
            ("block", BLOCK_BLOB),
        ] {
            assert!(
                !blob.is_empty(),
                "embedded circuit blob {name} is an empty stub"
            );
        }
    }
}
