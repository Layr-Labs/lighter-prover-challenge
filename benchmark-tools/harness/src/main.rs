//! Trusted timer, verifier, sandbox launcher, and score writer.

#![feature(stmt_expr_attributes)]

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use std::{env, thread};

use bincode::Options;
use circuit::block::Block;
use circuit::block_pre_execution::BlockPreExecWitness;
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain::BlockTxChainWitness;
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TX_PER_PROOF: usize = 4;
const CHAIN_ID: u32 = 304;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_PROOF_BYTES: u64 = 256 * 1024 * 1024;
const VERIFIER_SOURCE_REV: &str = "8d7d809f925c479e1f18761566d9b94a7279e238";
const PROTOCOL_VERSION: &str = "lighter-proof-v1";

type Proof = ProofWithPublicInputs<F, C, D>;

#[derive(Deserialize)]
struct Proofs {
    pre: Proof,
    chain: Proof,
}

struct Circuits {
    pre_data: CircuitData<F, C, D>,
    chain_data: CircuitData<F, C, D>,
}

impl Circuits {
    fn new() -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, TX_PER_PROOF, CHAIN_ID);
        let tx_data = tx.builder.build::<C>();

        let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
        let pre_data = pre.builder.build::<C>();

        let chain = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, TX_PER_PROOF, 1);
        let chain_data = chain.builder.build::<C>();

        // Exercise the exact cyclic base-proof construction used by the worker
        // while the verifier is authored, catching incompatible circuit data.
        let dummy_data = dummy_circuit(&chain_data.common);
        let _ = cyclic_base_proof(
            &chain_data.common,
            &chain_data.verifier_only,
            &dummy_data,
            [].into_iter().collect(),
        )
        .expect("cannot construct verifier cyclic base proof");

        Self {
            pre_data,
            chain_data,
        }
    }
}

struct Config {
    worker: PathBuf,
    fixture: PathBuf,
    scratch: PathBuf,
    score: PathBuf,
    mode: String,
    transactions: usize,
    candidate_sha: String,
    sandbox_profile: Option<PathBuf>,
}

#[derive(Serialize)]
struct ScoreFile {
    score: f64,
    passed: bool,
    metrics: ScoreMetrics,
}

#[derive(Serialize)]
struct ScoreMetrics {
    runtime: String,
    timing_authority: &'static str,
    proving_seconds: f64,
    transactions: usize,
    transactions_per_second: f64,
    candidate_sha: String,
    verifier_source_rev: &'static str,
    verifier_sha256: String,
    protocol_version: &'static str,
    fixture_id: String,
    verified_proofs: usize,
    expected_proofs: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    prepare_scratch(&config.scratch)?;
    if let Some(parent) = config.score.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::remove_file(&config.score);

    let json = fs::read_to_string(&config.fixture)?;
    let block: Block<F> = serde_json::from_str(&json)?;
    if block.txs.len() != config.transactions {
        return Err(format!(
            "fixture contains {} transactions; expected {}",
            block.txs.len(),
            config.transactions
        )
        .into());
    }

    let proof_path = config.scratch.join("proof.bin");
    let timeout = env::var("LIGHTER_PROVE_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TIMEOUT);

    let started = Instant::now();
    let mut child = worker_command(&config, &proof_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_clear()
        .env("TMPDIR", &config.scratch)
        .spawn()?;
    let status = wait_for_exit(&mut child, timeout)?;
    let proving_seconds = started.elapsed().as_secs_f64();
    if !status.success() {
        return Err(format!("candidate worker failed with {status}").into());
    }

    let proofs = read_proofs(&proof_path)?;
    verify(&block, &Circuits::new(), &proofs)?;

    let verifier_sha256 = executable_sha256()?;
    let fixture_id = config
        .fixture
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("fixture")
        .to_owned();
    let throughput = config.transactions as f64 / proving_seconds;
    let score = ScoreFile {
        score: throughput,
        passed: true,
        metrics: ScoreMetrics {
            runtime: config.mode,
            timing_authority: "trusted verifier parent",
            proving_seconds,
            transactions: config.transactions,
            transactions_per_second: throughput,
            candidate_sha: config.candidate_sha,
            verifier_source_rev: VERIFIER_SOURCE_REV,
            verifier_sha256,
            protocol_version: PROTOCOL_VERSION,
            fixture_id,
            verified_proofs: 2,
            expected_proofs: 2,
        },
    };
    let rendered = serde_json::to_vec_pretty(&score)?;
    let temporary = config.score.with_extension("tmp");
    fs::write(&temporary, [&rendered[..], b"\n"].concat())?;
    fs::rename(temporary, &config.score)?;
    println!("{}", String::from_utf8(rendered)?);
    prepare_scratch(&config.scratch)?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let worker = PathBuf::from(args.next().ok_or("missing WORKER")?);
    let fixture = PathBuf::from(args.next().ok_or("missing FIXTURE")?);
    let scratch = PathBuf::from(args.next().ok_or("missing SCRATCH")?);
    let score = PathBuf::from(args.next().ok_or("missing SCORE")?);
    let mode = args.next().ok_or("missing MODE")?;
    let transactions = args.next().ok_or("missing TRANSACTIONS")?.parse()?;
    let candidate_sha = args.next().ok_or("missing CANDIDATE_SHA")?;
    let sandbox_profile = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(concat!(
            "usage: verifier WORKER FIXTURE SCRATCH SCORE MODE TRANSACTIONS ",
            "CANDIDATE_SHA [SANDBOX_PROFILE]"
        )
        .into());
    }
    if !worker.is_file() || !fixture.is_file() || transactions == 0 {
        return Err("invalid worker, fixture, or transaction count".into());
    }
    if sandbox_profile
        .as_ref()
        .is_some_and(|profile| !profile.is_file())
    {
        return Err("sandbox profile is not a file".into());
    }
    Ok(Config {
        worker,
        fixture,
        scratch,
        score,
        mode,
        transactions,
        candidate_sha,
        sandbox_profile,
    })
}

fn worker_command(config: &Config, proof: &Path) -> Command {
    let mut command = if let Some(profile) = &config.sandbox_profile {
        let mut sandbox = Command::new("/usr/bin/sandbox-exec");
        sandbox.arg("-f").arg(profile).arg(&config.worker);
        sandbox
    } else {
        Command::new(&config.worker)
    };
    command.arg(&config.fixture).arg(proof);
    command
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Result<ExitStatus, std::io::Error> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "candidate worker timed out",
            ));
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn read_proofs(path: &Path) -> Result<Proofs, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 || metadata.len() > MAX_PROOF_BYTES {
        return Err("proof output is empty or exceeds the trusted size limit".into());
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_PROOF_BYTES)
        .deserialize(&bytes)?)
}

fn verify(
    block: &Block<F>,
    circuits: &Circuits,
    proofs: &Proofs,
) -> Result<(), Box<dyn std::error::Error>> {
    circuits.pre_data.verify(proofs.pre.clone())?;
    circuits.chain_data.verify(proofs.chain.clone())?;

    let pre = BlockPreExecWitness::from_public_inputs(&proofs.pre.public_inputs);
    let chain = BlockTxChainWitness::from_public_inputs(&proofs.chain.public_inputs, 1, 1);
    if pre.block_number != block.block_number
        || pre.created_at != block.created_at
        || pre.old_state_root != block.old_state_root
        || chain.block_number != block.block_number
        || chain.created_at != block.created_at
        || chain.old_state_root != pre.new_state_root
        || chain.new_validium_root != block.new_validium_root
        || chain.new_state_root != block.new_state_root
        || chain.new_account_delta_tree_root != block.new_account_delta_tree_root
        || chain.on_chain_operations_count != block.on_chain_operations_count
        || chain.on_chain_operations_pub_data != block.on_chain_operations_pub_data
        || chain.priority_operations_count != block.priority_operations_count
        || chain.new_public_market_details != block.new_public_market_details
    {
        return Err("proof public outputs do not match the trusted fixture".into());
    }
    Ok(())
}

fn prepare_scratch(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.as_os_str().is_empty() || path == Path::new("/") {
        return Err("refusing unsafe scratch path".into());
    }
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn executable_sha256() -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(env::current_exe()?)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
