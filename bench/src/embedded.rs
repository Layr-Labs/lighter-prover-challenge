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

use std::sync::Arc;

use circuit::block_pre_execution_constraints::BlockPreExecutionTarget;
use circuit::block_tx_chain_constraints::BlockTxChainTarget;
use circuit::block_tx_constraints::BlockTxTarget;
use circuit::embed::{ParsedEmbedded, deserialize_embedded, parse_embedded};
use circuit::types::config::{C, D, F};
use plonky2::plonk::circuit_data::CircuitData;

use crate::api::{CircuitSlot, Circuits, Proof};

static PRE_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pre.embed"));
static HEAVY_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_tx.embed"));
static HEAVY_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heavy_chain.embed"));
static LIGHT_TX_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_tx.embed"));
static LIGHT_CHAIN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/light_chain.embed"));

/// The four startup circuits that do not participate in pre-execution. Keeping
/// this separate lets the worker start the pre-execution proof from its already
/// decoded circuit while these independent blobs load in parallel.
///
/// Each circuit carries its [`CircuitSlot`] (commitment finalizing in the
/// background) and its leaked witness-generation facade.
pub(crate) struct RemainingEmbeddedCircuits {
    heavy_tx: (
        BlockTxTarget,
        Arc<CircuitSlot>,
        &'static CircuitData<F, C, D>,
    ),
    heavy_chain: (
        BlockTxChainTarget,
        Arc<CircuitSlot>,
        &'static CircuitData<F, C, D>,
    ),
    light_tx: (
        BlockTxTarget,
        Arc<CircuitSlot>,
        &'static CircuitData<F, C, D>,
    ),
    light_chain: (
        BlockTxChainTarget,
        Arc<CircuitSlot>,
        &'static CircuitData<F, C, D>,
    ),
    dummy_heavy_proof: Proof,
    dummy_light_proof: Proof,
}

impl RemainingEmbeddedCircuits {
    pub(crate) fn into_circuits(
        self,
        pre: (&'static BlockPreExecutionTarget, Arc<CircuitSlot>),
    ) -> Circuits {
        let (pre_target, pre_data) = pre;
        let (heavy_tx_target, heavy_tx_data, heavy_tx_facade) = self.heavy_tx;
        let (heavy_chain_target, heavy_chain_data, heavy_chain_facade) = self.heavy_chain;
        let (light_tx_target, light_tx_data, light_tx_facade) = self.light_tx;
        let (light_chain_target, light_chain_data, light_chain_facade) = self.light_chain;
        Circuits {
            heavy_tx_target,
            heavy_tx_data,
            light_tx_target,
            light_tx_data,
            pre_target,
            pre_data,
            heavy_chain_target,
            heavy_chain_data,
            light_chain_target,
            light_chain_data,
            dummy_heavy_proof: self.dummy_heavy_proof,
            dummy_light_proof: self.dummy_light_proof,
            heavy_tx_facade: Some(heavy_tx_facade),
            light_tx_facade: Some(light_tx_facade),
            heavy_chain_facade: Some(heavy_chain_facade),
            light_chain_facade: Some(light_chain_facade),
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

/// One deferred blob parse: target struct, the pending slot, the leaked
/// witness-generation facade, and the commitment left for the finalizer.
type DeferredBlob<T> = (
    T,
    Arc<CircuitSlot>,
    &'static CircuitData<F, C, D>,
    circuit::embed::PendingEmbeddedCommitment,
);

/// Parses one blob the deferred way: the returned slot's commitment is carried
/// in the returned [`PendingEmbeddedCommitment`](circuit::embed::PendingEmbeddedCommitment)
/// for the caller's finalizer to install, and the leaked facade carries
/// everything witness generation reads.
fn load_blob_deferred<T: serde::de::DeserializeOwned + Send + 'static>(
    name: &'static str,
    blob: &[u8],
) -> anyhow::Result<DeferredBlob<T>> {
    anyhow::ensure!(
        !blob.is_empty(),
        "embedded circuit blob {name} is an empty stub (compiled with LIGHTER_SKIP_EMBED=1)"
    );
    let ParsedEmbedded {
        target,
        circuit,
        commitment,
        witness_facade,
    } = parse_embedded::<T>(blob, true)
        .map_err(|error| error.context(format!("loading embedded circuit {name}")))?;
    let facade: &'static CircuitData<F, C, D> = Box::leak(Box::new(
        witness_facade.expect("deferred embedded loads always parse a facade"),
    ));
    let slot = Arc::new(CircuitSlot::pending(circuit));
    Ok((target, slot, facade, commitment))
}

/// Drives the deferred commitment finalizations in spine order on one thread.
/// Each `PolynomialBatch::from_values` is internally rayon-parallel, so a
/// single sequencer still saturates the machine; the ordering is what matters.
/// The light transaction circuit goes first because the first light chunk
/// proof is the earliest waiter on any commitment; the light chain is second
/// because the first fold follows one proof later. The heavy pair comes last:
/// its three-chunk path finishes with double-digit seconds of slack against
/// the 49-fold light spine, so it tolerates waiting behind the light work.
fn spawn_commitment_finalizer(
    mut pending: Vec<(
        &'static str,
        Arc<CircuitSlot>,
        circuit::embed::PendingEmbeddedCommitment,
    )>,
) {
    std::thread::Builder::new()
        .name("commitment-finalize".into())
        .stack_size(crate::api::PROVER_THREAD_STACK_BYTES)
        .spawn(move || {
            for (name, slot, commitment) in pending.drain(..) {
                // The slot's lock discipline (see `CircuitSlot`) lets this
                // `write()` only be delayed by bounded readers, never starved.
                let result = commitment.finalize_into(&mut slot.write());
                match result {
                    Ok(()) => slot.mark_ready(),
                    Err(error) => {
                        log::error!("embedded circuit {name} commitment finalization failed");
                        slot.mark_failed(error);
                    }
                }
            }
        })
        .expect("commitment finalizer thread must start");
}

impl Circuits {
    /// Loads only the pre-execution circuit blob. This is the fast path used
    /// by the startup overlap: the pre-execution proof can start (and hide)
    /// behind the remaining circuit loads.
    pub fn load_pre() -> anyhow::Result<(BlockPreExecutionTarget, CircuitData<F, C, D>)> {
        load_blob::<BlockPreExecutionTarget>("pre", PRE_BLOB)
    }

    /// Loads every embedded circuit except pre-execution. This is public to
    /// the worker startup path only; normal callers should keep using
    /// [`Self::load`].
    ///
    /// Deferred: each blob is *parsed* here (the fast half — decompression and
    /// section decoding), while the constants/sigmas commitment recompute (the
    /// expensive half) runs on a per-circuit finalizer thread started by
    /// [`load_blob_deferred`]. The pipeline can therefore start witness work
    /// against the returned facades as soon as this returns; the first proof of
    /// each circuit blocks in [`CircuitSlot::wait_ready`] until its finalizer
    /// finishes. Parse-order note: light first, so the spine's circuits are the
    /// first whose finalizers compete for the pool.
    pub(crate) fn load_remaining_embedded() -> anyhow::Result<RemainingEmbeddedCircuits> {
        let (light, heavy) = rayon::join(
            || {
                rayon::join(
                    || load_blob_deferred::<BlockTxTarget>("light_tx", LIGHT_TX_BLOB),
                    || load_blob_deferred::<BlockTxChainTarget>("light_chain", LIGHT_CHAIN_BLOB),
                )
            },
            || {
                rayon::join(
                    || load_blob_deferred::<BlockTxTarget>("heavy_tx", HEAVY_TX_BLOB),
                    || load_blob_deferred::<BlockTxChainTarget>("heavy_chain", HEAVY_CHAIN_BLOB),
                )
            },
        );
        let (light_tx, light_chain) = (light.0?, light.1?);
        let (heavy_tx, heavy_chain) = (heavy.0?, heavy.1?);

        // Spine-ordered finalization, one sequencer (see its doc comment).
        spawn_commitment_finalizer(vec![
            ("light_tx", Arc::clone(&light_tx.1), light_tx.3),
            ("light_chain", Arc::clone(&light_chain.1), light_chain.3),
            ("heavy_tx", Arc::clone(&heavy_tx.1), heavy_tx.3),
            ("heavy_chain", Arc::clone(&heavy_chain.1), heavy_chain.3),
        ]);
        let light_tx = (light_tx.0, light_tx.1, light_tx.2);
        let light_chain = (light_chain.0, light_chain.1, light_chain.2);
        let heavy_tx = (heavy_tx.0, heavy_tx.1, heavy_tx.2);
        let heavy_chain = (heavy_chain.0, heavy_chain.1, heavy_chain.2);

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
        let (pre_target, pre_data) = pre?;
        let pre = (
            Box::leak(Box::new(pre_target)) as &'static BlockPreExecutionTarget,
            Arc::new(CircuitSlot::ready(pre_data)),
        );
        let circuits = remaining?.into_circuits(pre);
        // This is the correctness-oracle / fallback entry point, not the scored
        // startup path: settle the deferred commitments before returning so the
        // result is exactly the value `Circuits::new` would have produced.
        for slot in [
            &circuits.light_tx_data,
            &circuits.light_chain_data,
            &circuits.heavy_tx_data,
            &circuits.heavy_chain_data,
        ] {
            slot.wait_ready();
        }
        Ok(circuits)
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
            rebuilt_data.prover_only.generators.len() == embedded_data.prover_only.generators.len(),
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
                (&*rebuilt.pre_target, &*rebuilt.pre_data.read()),
                (&*embedded.pre_target, &*embedded.pre_data.read()),
            );
            assert_circuit_pair_identical(
                "heavy_tx",
                (&rebuilt.heavy_tx_target, &*rebuilt.heavy_tx_data.read()),
                (&embedded.heavy_tx_target, &*embedded.heavy_tx_data.read()),
            );
            assert_circuit_pair_identical(
                "heavy_chain",
                (
                    &rebuilt.heavy_chain_target,
                    &*rebuilt.heavy_chain_data.read(),
                ),
                (
                    &embedded.heavy_chain_target,
                    &*embedded.heavy_chain_data.read(),
                ),
            );
            assert_circuit_pair_identical(
                "light_tx",
                (&rebuilt.light_tx_target, &*rebuilt.light_tx_data.read()),
                (&embedded.light_tx_target, &*embedded.light_tx_data.read()),
            );
            assert_circuit_pair_identical(
                "light_chain",
                (
                    &rebuilt.light_chain_target,
                    &*rebuilt.light_chain_data.read(),
                ),
                (
                    &embedded.light_chain_target,
                    &*embedded.light_chain_data.read(),
                ),
            );

            // The gate serializer round trip below also pins the common data
            // encoding used by the blobs.
            let mut bytes = Vec::new();
            bytes
                .write_common_circuit_data(
                    &rebuilt.light_tx_data.read().common,
                    &BlockGateSerializer,
                )
                .expect("common data must serialize");
            assert!(!bytes.is_empty());

            println!("embedded_matches_rebuilt: all five circuits are value-identical");
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

    /// Startup-gate decomposition for the deferred loader. Measures, per run:
    /// the eager overlapped load (the old pipeline-start gate), the deferred
    /// parse (the new earliest pipeline start), and the wall time at which each
    /// circuit's commitment becomes ready after a deferred load. The startup
    /// win is `eager_gate - max(parse, pre_proof)`; the per-circuit readiness
    /// times show the first proof of each circuit is never gated *later* than
    /// the eager gate. Run:
    /// `cargo test --release -p bench --bin prove -- --ignored startup_gate_timing --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn startup_gate_timing() {
        use std::time::Instant;

        on_big_stack(|| {
            for round in 0..3 {
                // Eager overlapped gate (warm).
                let t = Instant::now();
                let eager = Circuits::from_embedded().expect("embedded circuits must load");
                let t_eager = t.elapsed();
                drop(eager);

                // Deferred: parse returns immediately; readiness arrives later.
                let t = Instant::now();
                let remaining =
                    Circuits::load_remaining_embedded().expect("deferred loads must parse");
                let t_parse = t.elapsed();
                let (pre_target, pre_data) = Circuits::load_pre().expect("pre circuit must load");
                let circuits = remaining.into_circuits((
                    Box::leak(Box::new(pre_target)),
                    Arc::new(CircuitSlot::ready(pre_data)),
                ));
                let t_parse_total = t.elapsed();
                let mut readiness = Vec::new();
                for (name, slot) in [
                    ("light_tx", &circuits.light_tx_data),
                    ("light_chain", &circuits.light_chain_data),
                    ("heavy_tx", &circuits.heavy_tx_data),
                    ("heavy_chain", &circuits.heavy_chain_data),
                ] {
                    slot.wait_ready();
                    readiness.push((name, t.elapsed()));
                }
                drop(circuits);
                println!(
                    "round {round}: eager gate {t_eager:>9.1?} | deferred parse {t_parse:>9.1?} \
                     (+pre {t_parse_total:>9.1?}) | ready: {readiness:.1?}"
                );
            }
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
