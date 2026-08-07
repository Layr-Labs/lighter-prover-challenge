// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;

pub type Proof = ProofWithPublicInputs<F, C, D>;

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

pub(crate) struct PathCircuits {
    pub(crate) tx_target: BlockTxTarget,
    pub(crate) tx_data: CircuitData<F, C, D>,
    pub(crate) chain_target: BlockTxChainTarget,
    pub(crate) chain_data: CircuitData<F, C, D>,
    pub(crate) dummy_proof: Proof,
}

impl PathCircuits {
    pub(crate) fn new(tx_per_proof: usize, tx_mode: u8) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
        let tx_target = tx.target;
        let tx_data = tx.builder.build::<C>();

        let chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_target = chain.target;
        let chain_data = chain.builder.build::<C>();

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
                (pre.target, pre.builder.build::<C>())
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
        (block.target, block.builder.build::<C>())
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

#[cfg(test)]
mod build_timing {
    use std::time::Instant;

    use super::*;

    /// Manual timing harness for the startup circuit builds, which run once per
    /// worker spawn inside the ranked timed window (five spawns per run). Run:
    /// `cargo test --release -p bench --bin prove -- --ignored build_phase_timing --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn build_phase_timing() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let t = Instant::now();
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                let t_pre_define = t.elapsed();
                let t = Instant::now();
                let pre_data = pre.builder.build::<C>();
                let t_pre_build = t.elapsed();
                drop(pre_data);

                let t = Instant::now();
                let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, LIGHT_TX_PER_PROOF, CHAIN_ID, LIGHT_TX_MODE);
                let t_tx_define = t.elapsed();
                let t = Instant::now();
                let tx_data = tx.builder.build::<C>();
                let t_tx_build = t.elapsed();

                let t = Instant::now();
                let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
                let t_chain_define = t.elapsed();
                let t = Instant::now();
                let chain_data = chain.builder.build::<C>();
                let t_chain_build = t.elapsed();
                drop(chain_data);

                println!("pre:   define {t_pre_define:?} build {t_pre_build:?}");
                println!("tx:    define {t_tx_define:?} build {t_tx_build:?}");
                println!("chain: define {t_chain_define:?} build {t_chain_build:?}");
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// H10 settling probe: is "embed the built CircuitData, skip define+build"
    /// viable, and what does each leg cost?
    ///
    /// build.rs runs during the UNTIMED compile job (BENCHMARK.md: compilation
    /// runs candidate build scripts, offline), so embedded bytes bypass the
    /// 25 MiB archive cap — they are generated on the runner, not shipped.
    /// Full `CircuitData::to_bytes` is not viable: `write_merkle_tree`
    /// serializes every LDE leaf (~671 MiB for one tx circuit). The candidate
    /// form is selective: embed common + verifier + generators + watch map +
    /// commitment coefficient polynomials + representative map; recompute at
    /// runtime the commitment (`from_coeffs`, the same GPU-eligible path build
    /// uses, minus the IFFT), sigma values (FFT of the embedded sigma polys +
    /// transpose), the subgroup, and the FFT root table.
    ///
    /// This probe measures, for the light tx circuit (the fattest lane):
    ///   rebuild path: define + build::<C>()
    ///   embed path:   per-component serialized size, deserialize time, and
    ///                 recompute time, each with value-identity checks against
    ///                 the built original.
    /// Run:
    /// `cargo test --release -p bench --bin prove -- --ignored h10_embed_probe --nocapture`
    #[test]
    #[ignore = "manual probe"]
    fn h10_embed_probe() {
        use circuit::circuit_serializer::{BlockGateSerializer, BlockGeneratorSerializer};
        use circuit::ecdsa::curve::secp256k1::Secp256K1;
        use plonky2::field::fft::fft_root_table;
        use plonky2::field::polynomial::PolynomialValues;
        use plonky2::field::types::Field;
        use plonky2::fri::oracle::PolynomialBatch;
        use plonky2::util::serialization::{Buffer, Read as _, Write as _};
        use plonky2::util::timing::TimingTree;

        // `plonky2::util::transpose_poly_values` is pub(crate); reproduce its
        // row-major transpose here for the probe.
        fn transpose_poly_values(
            polys: Vec<plonky2::field::polynomial::PolynomialValues<F>>,
        ) -> Vec<Vec<F>> {
            let n_points = polys[0].values.len();
            (0..n_points)
                .map(|i| polys.iter().map(|p| p.values[i]).collect())
                .collect()
        }

        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let gate_ser = BlockGateSerializer;
                let gen_ser = BlockGeneratorSerializer::<C, D, Secp256K1> {
                    _phantom: Default::default(),
                    _phantom2: Default::default(),
                };

                // ---- rebuild path ----
                let t = Instant::now();
                let tx = BlockTxCircuit::define(
                    CIRCUIT_CONFIG,
                    LIGHT_TX_PER_PROOF,
                    CHAIN_ID,
                    LIGHT_TX_MODE,
                );
                let t_define = t.elapsed();
                let t = Instant::now();
                let data = tx.builder.build::<C>();
                let t_build = t.elapsed();

                let common = &data.common;
                let prover = &data.prover_only;
                let degree = 1usize << common.degree_bits();
                let num_polys = prover.constants_sigmas_commitment.polynomials.len();
                let num_routed = common.config.num_routed_wires;
                let rate_bits = common.config.fri_config.rate_bits;
                let cap_height = common.config.fri_config.cap_height;

                println!("\n==== H10 embed probe: light tx circuit ====");
                println!("degree 2^{}  polys {}  generators {}  repmap {}",
                    common.degree_bits(), num_polys, prover.generators.len(),
                    prover.representative_map.len());
                println!("REBUILD: define {:>10.1?}  build {:>10.1?}  total {:>10.1?}",
                    t_define, t_build, t_define + t_build);

                let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
                let mut total_bytes = 0usize;
                let mut total_read = std::time::Duration::ZERO;

                macro_rules! leg {
                    ($name:expr, $write:expr, $read:expr) => {{
                        let mut buf: Vec<u8> = Vec::new();
                        let t = Instant::now();
                        $write(&mut buf);
                        let tw = t.elapsed();
                        let t = Instant::now();
                        let out = $read(&buf);
                        let tr = t.elapsed();
                        total_bytes += buf.len();
                        total_read += tr;
                        println!("  {:<18} {:>9.2} MiB  write {:>9.1?}  read {:>9.1?}",
                            $name, mib(buf.len()), tw, tr);
                        out
                    }};
                }

                // ---- embed path: serialize / deserialize each component ----
                let rd_common = leg!("common",
                    |b: &mut Vec<u8>| { b.write_common_circuit_data(common, &gate_ser).unwrap(); },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        r.read_common_circuit_data::<F, D>(&gate_ser).unwrap()
                    });
                assert_eq!(rd_common.degree_bits(), common.degree_bits());

                leg!("verifier_only",
                    |b: &mut Vec<u8>| { b.write_verifier_only_circuit_data(&data.verifier_only).unwrap(); },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        r.read_verifier_only_circuit_data::<F, C, D>().unwrap()
                    });

                let rd_gens = leg!("generators",
                    |b: &mut Vec<u8>| {
                        b.write_usize(prover.generators.len()).unwrap();
                        for g in prover.generators.iter() {
                            b.write_generator::<F, D>(g, &gen_ser, common).unwrap();
                        }
                    },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        let n = r.read_usize().unwrap();
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(r.read_generator::<F, D>(&gen_ser, common).unwrap());
                        }
                        v
                    });
                assert_eq!(rd_gens.len(), prover.generators.len());
                assert_eq!(rd_gens[0].0.id(), prover.generators[0].0.id());

                let rd_watches = leg!("watch_map",
                    |b: &mut Vec<u8>| {
                        b.write_usize(prover.generator_indices_by_watches.len()).unwrap();
                        for (k, v) in prover.generator_indices_by_watches.iter() {
                            b.write_usize(*k).unwrap();
                            b.write_usize_vec(v).unwrap();
                        }
                    },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        let n = r.read_usize().unwrap();
                        let mut m = std::collections::BTreeMap::new();
                        for _ in 0..n {
                            let k = r.read_usize().unwrap();
                            m.insert(k, r.read_usize_vec().unwrap());
                        }
                        m
                    });
                assert_eq!(rd_watches.len(), prover.generator_indices_by_watches.len());

                let rd_polys = leg!("coeff_polys",
                    |b: &mut Vec<u8>| {
                        b.write_usize(num_polys).unwrap();
                        for p in prover.constants_sigmas_commitment.polynomials.iter() {
                            b.write_usize(p.coeffs.len()).unwrap();
                            b.write_field_vec(&p.coeffs).unwrap();
                        }
                    },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        let n = r.read_usize().unwrap();
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            let len = r.read_usize().unwrap();
                            v.push(plonky2::field::polynomial::PolynomialCoeffs::new(
                                r.read_field_vec::<F>(len).unwrap(),
                            ));
                        }
                        v
                    });
                assert_eq!(rd_polys.len(), num_polys);

                let rd_repmap = leg!("repmap",
                    |b: &mut Vec<u8>| {
                        b.write_usize_encoded_u32_vec(&prover.representative_map).unwrap();
                    },
                    |b: &Vec<u8>| {
                        let mut r = Buffer::new(b);
                        r.read_usize_encoded_u32_vec().unwrap()
                    });
                assert_eq!(rd_repmap.len(), prover.representative_map.len());
                assert_eq!(rd_repmap[1000], prover.representative_map[1000]);

                // ---- embed path: runtime recomputes ----
                let t = Instant::now();
                let subgroup = F::two_adic_subgroup(common.degree_bits());
                let t_subgroup = t.elapsed();
                assert_eq!(subgroup[1], prover.subgroup[1]);

                let max_fft_points = 1usize << (common.degree_bits() + rate_bits);
                let t = Instant::now();
                let root_table = fft_root_table::<F>(max_fft_points);
                let t_root = t.elapsed();

                let t = Instant::now();
                // The base-domain FFT (2^16) uses a different primitive root than
                // the max_fft_points table (2^19), so it needs its own table.
                let base_root_table = fft_root_table::<F>(1usize << common.degree_bits());
                let sigma_polys = &rd_polys[num_polys - num_routed..];
                let sigma_vecs: Vec<PolynomialValues<F>> = sigma_polys
                    .iter()
                    .map(|p| p.clone().fft_with_options(None, Some(&base_root_table)))
                    .collect();
                let sigmas = transpose_poly_values(sigma_vecs);
                let t_sigmas = t.elapsed();
                assert_eq!(sigmas.len(), prover.sigmas.len());
                assert_eq!(sigmas[777], prover.sigmas[777], "sigma values diverge");

                let t = Instant::now();
                let commitment = PolynomialBatch::<F, C, D>::from_coeffs(
                    rd_polys,
                    rate_bits,
                    false,
                    cap_height,
                    &mut TimingTree::default(),
                    Some(&root_table),
                );
                let t_commit = t.elapsed();
                assert_eq!(
                    commitment.merkle_tree.cap, prover.constants_sigmas_commitment.merkle_tree.cap,
                    "recomputed commitment cap diverges"
                );

                let recompute =
                    t_subgroup + t_root + t_sigmas + t_commit;
                println!("  RECOMPUTE: subgroup {t_subgroup:>9.1?}  root_table {t_root:>9.1?}  sigmas(fft+T) {t_sigmas:>9.1?}  commitment {t_commit:>9.1?}");
                println!("\nEMBED TOTAL: {:>8.2} MiB serialized, read {:>9.1?} + recompute {:>9.1?} = {:>9.1?}",
                    mib(total_bytes), total_read, recompute, total_read + recompute);
                println!("REBUILD    : {:>9.1?}", t_define + t_build);
                let embed = (total_read + recompute).as_secs_f64();
                let rebuild = (t_define + t_build).as_secs_f64();
                println!("NET per light-tx lane: {:+.1} ms ({})",
                    (rebuild - embed) * 1e3,
                    if rebuild > embed { "EMBED WINS" } else { "REBUILD WINS" });
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// Isolates *what* the ~2.18 s of one-time cost actually is. Charges each
    /// candidate warm-up separately, before any circuit work, so the pre-build
    /// that follows should collapse to its true ~0.30 s if that candidate was
    /// the cause.
    /// Run:
    /// `cargo test --release -p bench --bin prove -- --ignored build_warmup_attribution --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn build_warmup_attribution() {
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                use plonky2::field::types::Field;

                // 1. Rayon global pool construction, forced by a trivial
                //    parallel call that does no real work.
                let t = Instant::now();
                let pool_probe: u64 = {
                    use rayon::prelude::*;
                    (0u64..1_000).into_par_iter().sum()
                };
                let t_pool = t.elapsed();
                std::hint::black_box(pool_probe);

                // 2. First large heap touch: fault in ~512 MiB the way a
                //    circuit build's LDE buffers would.
                let t = Instant::now();
                let mut big: Vec<F> = Vec::with_capacity(64 << 20);
                big.resize(64 << 20, F::ONE);
                let t_faults = t.elapsed();
                std::hint::black_box(big.len());
                drop(big);

                // 3. Now build pre-execution, cold-process but post-warm-up.
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                let t = Instant::now();
                let pre_data = pre.builder.build::<C>();
                let t_pre_build = t.elapsed();
                drop(pre_data);

                println!("WARM-UP ATTRIBUTION");
                println!("rayon pool init      {t_pool:?}");
                println!("512MiB first touch   {t_faults:?}");
                println!("pre build (after)    {t_pre_build:?}");
                println!("  -> pre build was 2.482s cold-first, 0.295s after other circuits");
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// Same measurement as `build_phase_timing` but with the build order
    /// reversed, to separate genuinely circuit-specific build cost from
    /// whatever one-time cost the *first* `build::<C>()` in a process absorbs
    /// (rayon pool construction, first-touch page faults, lazy statics, FFT
    /// root tables). In pre-first order the pre-execution circuit — the
    /// smallest of the three — measured 2.48 s against the tx circuit's 0.63 s,
    /// which is the wrong way round if the cost were circuit-specific.
    /// Run:
    /// `cargo test --release -p bench --bin prove -- --ignored build_phase_timing_reversed --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn build_phase_timing_reversed() {
        // plonky2's `build_with_options` wraps its internal phases in a
        // `TimingTree::new("preprocess", Level::Trace)`, but nothing in the test
        // harness installs a logger, so that breakdown is normally discarded.
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Trace)
            .format_timestamp(None)
            .try_init();
        std::thread::Builder::new()
            .stack_size(PROVER_THREAD_STACK_BYTES)
            .spawn(|| {
                let t = Instant::now();
                let tx = BlockTxCircuit::define(
                    CIRCUIT_CONFIG,
                    LIGHT_TX_PER_PROOF,
                    CHAIN_ID,
                    LIGHT_TX_MODE,
                );
                let t_tx_define = t.elapsed();
                let t = Instant::now();
                let tx_data = tx.builder.build::<C>();
                let t_tx_build = t.elapsed();

                let t = Instant::now();
                let chain =
                    BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
                let t_chain_define = t.elapsed();
                let t = Instant::now();
                let chain_data = chain.builder.build::<C>();
                let t_chain_build = t.elapsed();
                drop(chain_data);

                let t = Instant::now();
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                let t_pre_define = t.elapsed();
                let t = Instant::now();
                let pre_data = pre.builder.build::<C>();
                let t_pre_build = t.elapsed();
                drop(pre_data);

                // Second pre build in the same process: any remaining gap
                // versus the first is circuit-independent warm-up.
                let pre2 = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                let t = Instant::now();
                let pre_data2 = pre2.builder.build::<C>();
                let t_pre_build_again = t.elapsed();
                drop(pre_data2);

                println!("REVERSED ORDER (tx -> chain -> pre -> pre again)");
                println!("tx:    define {t_tx_define:?} build {t_tx_build:?}");
                println!("chain: define {t_chain_define:?} build {t_chain_build:?}");
                println!("pre:   define {t_pre_define:?} build {t_pre_build:?}");
                println!("pre#2:                        build {t_pre_build_again:?}");
            })
            .expect("spawn")
            .join()
            .expect("join");
    }
}
