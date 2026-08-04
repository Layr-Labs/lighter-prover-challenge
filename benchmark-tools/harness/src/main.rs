//! Trusted timer, verifier, sandbox launcher, and score writer.

#![feature(stmt_expr_attributes)]

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use std::{env, thread};

use bincode::Options;
use circuit::block::{Block, BlockWitness};
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT, TX_TYPE_EMPTY};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use serde::Serialize;
use sha2::{Digest, Sha256};

const CHAIN_ID: u32 = 304;
const HEAVY_TX_PER_PROOF: usize = 4;
const HEAVY_TX_MODE: u8 = TX_HEAVY;
const LIGHT_TX_PER_PROOF: usize = 10;
const LIGHT_TX_MODE: u8 = TX_LIGHT;
const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
const PUBLIC_HEAVY_TX_COUNT: usize = 10;
const PUBLIC_LIGHT_TX_COUNT: usize = 490;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_PROOF_BYTES: u64 = 256 * 1024 * 1024;
const VERIFIER_SOURCE_REV: &str = "381fd529eb61dfff9ad245d94fce214a0a64d927";
const PROTOCOL_VERSION: &str = "lighter-mixed-block-proof-v1";
const PUBLIC_SYNTHETIC_MODE: &str = "public-synthetic-smoke";
const PUBLIC_SYNTHETIC_CASE: &str = "public-synthetic-provisional";
const PRIVATE_RANKED_MODE: &str = "official-throughput";
const PRIVATE_RANKED_CASE: &str = "private-active-ranked";
const PRIVATE_FIXTURE_ID: &str = "ranked-v1";
const PRIVATE_TRANSACTIONS_PER_FIXTURE: usize = 500;
const EXPECTED_FIXTURE_SHA256: &str =
    "6f1fbd2d5e64ed84f656b0c2dc299a8628801ac66488dfe021fdc4b2af53eb4b";
const EXPECTED_FIXTURE_SHA256_ENV: &str = "LIGHTER_EXPECTED_FIXTURE_SHA256";
const EXPECTED_FIXTURE_COUNT_ENV: &str = "LIGHTER_EXPECTED_FIXTURE_COUNT";

type Proof = ProofWithPublicInputs<F, C, D>;

struct Circuits {
    block_data: CircuitData<F, C, D>,
}

impl Circuits {
    fn new() -> Self {
        let heavy_tx =
            BlockTxCircuit::define(CIRCUIT_CONFIG, HEAVY_TX_PER_PROOF, CHAIN_ID, HEAVY_TX_MODE);
        let heavy_tx_data = heavy_tx.builder.build::<C>();

        let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
        let pre_data = pre.builder.build::<C>();

        let light_tx =
            BlockTxCircuit::define(CIRCUIT_CONFIG, LIGHT_TX_PER_PROOF, CHAIN_ID, LIGHT_TX_MODE);
        let light_tx_data = light_tx.builder.build::<C>();

        let heavy_chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &heavy_tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let heavy_chain_data = heavy_chain.builder.build::<C>();

        let light_chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &light_tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let light_chain_data = light_chain.builder.build::<C>();

        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &pre_data,
            &light_chain_data,
            &heavy_chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );

        Self {
            block_data: block.builder.build::<C>(),
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
    benchmark_case: &'static str,
    provisional: bool,
    competitive: bool,
    cacheable: bool,
    transaction_count_source: &'static str,
    timing_authority: &'static str,
    proving_seconds: f64,
    transactions: usize,
    transactions_per_second: f64,
    candidate_sha: String,
    verifier_source_rev: &'static str,
    verifier_sha256: String,
    protocol_version: &'static str,
    fixture_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixture_sha256: Option<String>,
    fixture_count: usize,
    verified_proofs: usize,
    expected_proofs: usize,
}

struct PrivateFixture {
    path: PathBuf,
    block: Block<F>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AggregateScore {
    proving_seconds: f64,
    transactions_per_second: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BenchmarkCase {
    name: &'static str,
    provisional: bool,
    competitive: bool,
    cacheable: bool,
    transaction_count_source: &'static str,
}

fn classify_benchmark_case(
    active_transactions: usize,
    mode: &str,
) -> Result<BenchmarkCase, &'static str> {
    match (active_transactions, mode) {
        (0, PUBLIC_SYNTHETIC_MODE) => Ok(BenchmarkCase {
            name: PUBLIC_SYNTHETIC_CASE,
            provisional: true,
            competitive: false,
            cacheable: true,
            transaction_count_source: "synthetic-configuration-unbound-by-proof",
        }),
        (1.., PRIVATE_RANKED_MODE) => Ok(BenchmarkCase {
            name: PRIVATE_RANKED_CASE,
            provisional: false,
            competitive: true,
            cacheable: false,
            transaction_count_source: "parsed-active-witness",
        }),
        (0, _) => Err("all-empty fixtures require public-synthetic-smoke mode"),
        (1.., _) => Err("active fixtures require official-throughput mode"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    prepare_scratch(&config.scratch)?;
    if let Some(parent) = config.score.parent() {
        fs::create_dir_all(parent)?;
    }
    remove_file_if_exists(&config.score)?;

    let timeout = env::var("LIGHTER_PROVE_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TIMEOUT);

    let verifier_sha256 = executable_sha256()?;
    let metrics = match config.mode.as_str() {
        PUBLIC_SYNTHETIC_MODE => run_public_benchmark(&config, timeout, verifier_sha256)?,
        PRIVATE_RANKED_MODE => run_private_benchmark(&config, timeout, verifier_sha256)?,
        _ => return Err("unsupported benchmark mode".into()),
    };
    let score = ScoreFile {
        score: metrics.transactions_per_second,
        passed: true,
        metrics,
    };
    let rendered = serde_json::to_vec_pretty(&score)?;
    let temporary = config.score.with_extension("tmp");
    fs::write(&temporary, [&rendered[..], b"\n"].concat())?;
    fs::rename(temporary, &config.score)?;
    println!("{}", String::from_utf8(rendered)?);
    prepare_scratch(&config.scratch)?;
    Ok(())
}

fn run_public_benchmark(
    config: &Config,
    timeout: Duration,
    verifier_sha256: String,
) -> Result<ScoreMetrics, Box<dyn std::error::Error>> {
    let expected_fixture_sha256 = env::var(EXPECTED_FIXTURE_SHA256_ENV)
        .unwrap_or_else(|_| EXPECTED_FIXTURE_SHA256.to_owned());
    let (fixture, fixture_sha256) =
        read_verified_fixture(&config.fixture, &expected_fixture_sha256)?;
    let block = parse_fixture(&fixture)?;
    let active_transactions = active_transaction_count(&block);
    let transactions = trusted_transaction_count(&block);
    let benchmark_case = classify_benchmark_case(active_transactions, &config.mode)?;
    if transactions != config.transactions {
        return Err(format!(
            "synthetic fixture configures {transactions} transactions; expected {}",
            config.transactions
        )
        .into());
    }

    let proof_path = config.scratch.join("proof.bin");
    let started = Instant::now();
    let mut child = spawn_worker(config, &config.fixture, &proof_path)?;
    let status = wait_for_exit(&mut child, started + timeout)?;
    let proving_seconds = started.elapsed().as_secs_f64();
    if !status.success() {
        return Err(format!("candidate worker failed with {status}").into());
    }

    let proof = read_proof(&proof_path)?;
    verify(&block, &Circuits::new(), &proof)?;

    let throughput = transactions as f64 / proving_seconds;
    let fixture_id = config
        .fixture
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("fixture")
        .to_owned();
    Ok(ScoreMetrics {
        runtime: config.mode.clone(),
        benchmark_case: benchmark_case.name,
        provisional: benchmark_case.provisional,
        competitive: benchmark_case.competitive,
        cacheable: benchmark_case.cacheable,
        transaction_count_source: benchmark_case.transaction_count_source,
        timing_authority: "trusted verifier parent",
        proving_seconds,
        transactions,
        transactions_per_second: throughput,
        candidate_sha: config.candidate_sha.clone(),
        verifier_source_rev: VERIFIER_SOURCE_REV,
        verifier_sha256,
        protocol_version: PROTOCOL_VERSION,
        fixture_id,
        fixture_sha256: Some(fixture_sha256),
        fixture_count: 1,
        verified_proofs: 1,
        expected_proofs: 1,
    })
}

fn run_private_benchmark(
    config: &Config,
    timeout: Duration,
    verifier_sha256: String,
) -> Result<ScoreMetrics, Box<dyn std::error::Error>> {
    let expected_count =
        parse_expected_fixture_count(env::var(EXPECTED_FIXTURE_COUNT_ENV).ok().as_deref())?;
    let fixture_paths = private_fixture_paths(&config.fixture, expected_count)?;
    let fixtures = read_private_fixtures(fixture_paths)?;
    let transaction_counts = fixtures
        .iter()
        .map(|fixture| active_transaction_count(&fixture.block))
        .collect::<Vec<_>>();
    let transactions = validate_private_transaction_counts(&transaction_counts)?;
    if transactions != config.transactions {
        return Err(format!(
            "private aggregate configures {transactions} transactions; expected {}",
            config.transactions
        )
        .into());
    }
    let benchmark_case = classify_benchmark_case(transactions, &config.mode)?;

    let circuits = Circuits::new();
    let deadline = Instant::now() + timeout;
    let aggregate = run_private_sequence(
        &fixtures,
        &config.scratch,
        deadline,
        transactions,
        |ordinal, fixture, proof_path, deadline| {
            spawn_and_time_worker(config, &fixture.path, proof_path, deadline)
                .map_err(|error| format!("private fixture {ordinal:02} {error}"))
        },
        |ordinal, fixture, proof_path| {
            let proof = read_proof(proof_path)
                .map_err(|_| format!("private fixture {ordinal:02} proof is invalid"))?;
            verify(&fixture.block, &circuits, &proof)
                .map_err(|_| format!("private fixture {ordinal:02} proof verification failed"))
        },
    )?;
    Ok(ScoreMetrics {
        runtime: config.mode.clone(),
        benchmark_case: benchmark_case.name,
        provisional: benchmark_case.provisional,
        competitive: benchmark_case.competitive,
        cacheable: benchmark_case.cacheable,
        transaction_count_source: benchmark_case.transaction_count_source,
        timing_authority: "trusted verifier parent",
        proving_seconds: aggregate.proving_seconds,
        transactions,
        transactions_per_second: aggregate.transactions_per_second,
        candidate_sha: config.candidate_sha.clone(),
        verifier_source_rev: VERIFIER_SOURCE_REV,
        verifier_sha256,
        protocol_version: PROTOCOL_VERSION,
        fixture_id: PRIVATE_FIXTURE_ID.to_owned(),
        fixture_sha256: None,
        fixture_count: fixtures.len(),
        verified_proofs: fixtures.len(),
        expected_proofs: fixtures.len(),
    })
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
    if !worker.is_file() || transactions == 0 {
        return Err("invalid worker or transaction count".into());
    }
    match mode.as_str() {
        PUBLIC_SYNTHETIC_MODE if !fixture.is_file() => {
            return Err("public fixture is not a file".into());
        }
        PRIVATE_RANKED_MODE if !fixture.is_dir() => {
            return Err("private fixture input is not a directory".into());
        }
        PUBLIC_SYNTHETIC_MODE | PRIVATE_RANKED_MODE => {}
        _ => return Err("unsupported benchmark mode".into()),
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

fn spawn_worker(config: &Config, fixture: &Path, proof: &Path) -> Result<Child, std::io::Error> {
    let mut command = if let Some(profile) = &config.sandbox_profile {
        let mut sandbox = Command::new("/usr/bin/sandbox-exec");
        sandbox.arg("-f").arg(profile).arg(&config.worker);
        sandbox
    } else {
        Command::new(&config.worker)
    };
    command
        .arg(fixture)
        .arg(proof)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_clear()
        .env("TMPDIR", &config.scratch)
        .spawn()
}

fn spawn_and_time_worker(
    config: &Config,
    fixture: &Path,
    proof: &Path,
    deadline: Instant,
) -> Result<Duration, String> {
    let started = Instant::now();
    let mut child =
        spawn_worker(config, fixture, proof).map_err(|_| "worker could not start".to_owned())?;
    let status = wait_for_exit(&mut child, deadline).map_err(|error| {
        if error.kind() == io::ErrorKind::TimedOut {
            format!("worker timed out: {error}")
        } else {
            "worker wait failed".to_owned()
        }
    })?;
    let duration = started.elapsed();
    if !status.success() {
        return Err(format!("worker failed with {status}"));
    }
    Ok(duration)
}

fn complete_timeout_cleanup(
    kill_result: io::Result<()>,
    wait: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    let wait_result = wait();
    match (kill_result, wait_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(kill_error), Err(wait_error)) => Err(io::Error::other(format!(
            "kill failed: {kill_error}; wait failed: {wait_error}"
        ))),
    }
}

fn wait_for_exit(child: &mut Child, deadline: Instant) -> Result<ExitStatus, std::io::Error> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            let kill_result = child.kill();
            let cleanup_result = complete_timeout_cleanup(kill_result, || child.wait().map(|_| ()));
            let message = match cleanup_result {
                Ok(()) => "candidate worker timed out".to_owned(),
                Err(error) => format!("candidate worker timed out; cleanup failed: {error}"),
            };
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, message));
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn remaining_time(deadline: Instant, now: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(now)
        .filter(|remaining| !remaining.is_zero())
}

fn private_proof_path(scratch: &Path) -> PathBuf {
    scratch.join("proof.bin")
}

fn run_private_sequence<T>(
    fixtures: &[T],
    scratch: &Path,
    deadline: Instant,
    transactions: usize,
    mut run_worker: impl FnMut(usize, &T, &Path, Instant) -> Result<Duration, String>,
    mut verify_proof: impl FnMut(usize, &T, &Path) -> Result<(), String>,
) -> Result<AggregateScore, Box<dyn std::error::Error>> {
    let mut proving_durations = Vec::with_capacity(fixtures.len());
    for (index, fixture) in fixtures.iter().enumerate() {
        let ordinal = index + 1;
        if remaining_time(deadline, Instant::now()).is_none() {
            return Err("candidate worker sequence timed out".into());
        }

        prepare_scratch(scratch)
            .map_err(|_| format!("private fixture {ordinal:02} scratch could not be reset"))?;
        let proof_path = private_proof_path(scratch);
        match fs::symlink_metadata(&proof_path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(format!(
                    "private fixture {ordinal:02} proof absence could not be verified"
                )
                .into());
            }
            Ok(_) => {
                return Err(format!(
                    "private fixture {ordinal:02} proof path survived scratch reset"
                )
                .into());
            }
        }
        if remaining_time(deadline, Instant::now()).is_none() {
            return Err("candidate worker sequence timed out".into());
        }

        let duration = run_worker(ordinal, fixture, &proof_path, deadline)?;
        proving_durations.push(duration);
        verify_proof(ordinal, fixture, &proof_path)?;
        if remaining_time(deadline, Instant::now()).is_none() {
            return Err("candidate worker sequence timed out".into());
        }
    }
    Ok(aggregate_score(transactions, &proving_durations)?)
}

fn parse_expected_fixture_count(value: Option<&str>) -> Result<usize, Box<dyn std::error::Error>> {
    let count = value
        .ok_or("LIGHTER_EXPECTED_FIXTURE_COUNT is required in private mode")?
        .parse::<usize>()
        .map_err(|_| "LIGHTER_EXPECTED_FIXTURE_COUNT must be a positive integer")?;
    if count == 0 {
        return Err("LIGHTER_EXPECTED_FIXTURE_COUNT must be a positive integer".into());
    }
    Ok(count)
}

fn is_private_fixture_filename(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".json") else {
        return false;
    };
    let mut characters = stem.chars();
    characters
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric())
        && characters
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn private_fixture_paths(
    directory: &Path,
    expected_count: usize,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if expected_count == 0 {
        return Err("private fixture count must be positive".into());
    }
    let directory_metadata =
        fs::symlink_metadata(directory).map_err(|_| "private fixture directory is unavailable")?;
    if !directory_metadata.file_type().is_dir() {
        return Err("private fixture directory is unsafe".into());
    }

    #[cfg(unix)]
    let mut file_identities = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for entry in
        fs::read_dir(directory).map_err(|_| "private fixture directory could not be read")?
    {
        let entry = entry.map_err(|_| "private fixture directory contains an unsafe entry")?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "private fixture directory contains an unsafe entry")?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|_| "private fixture directory contains an unsafe entry")?;
        if !is_private_fixture_filename(&name) || !metadata.file_type().is_file() {
            return Err("private fixture directory contains an unsafe entry".into());
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            if !file_identities.insert((metadata.dev(), metadata.ino())) {
                return Err("private fixture directory contains duplicate-equivalent files".into());
            }
        }
        paths.push(entry.path());
    }
    paths.sort();
    if paths.len() != expected_count {
        return Err(format!(
            "private fixture count mismatch: expected {expected_count}, discovered {}",
            paths.len()
        )
        .into());
    }
    Ok(paths)
}

fn read_validated_regular_file(path: &Path) -> Result<Vec<u8>, io::Error> {
    read_validated_regular_file_with(path, |path| File::open(path))
}

fn read_validated_regular_file_with(
    path: &Path,
    open: impl FnOnce(&Path) -> Result<File, io::Error>,
) -> Result<Vec<u8>, io::Error> {
    let validated_metadata = fs::symlink_metadata(path)?;
    if !validated_metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "private fixture is not a regular file",
        ));
    }

    let mut file = open(path)?;
    let opened_metadata = file.metadata()?;
    verify_opened_file_identity(&validated_metadata, &opened_metadata)?;

    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn verify_opened_file_identity(
    validated: &fs::Metadata,
    opened: &fs::Metadata,
) -> Result<(), io::Error> {
    if !validated.file_type().is_file() || !opened.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "private fixture is not a regular file",
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if validated.dev() == opened.dev() && validated.ino() == opened.ino() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "private fixture changed during validation",
            ))
        }
    }

    #[cfg(not(unix))]
    {
        let _ = (validated, opened);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "private fixture descriptor identity validation is unsupported",
        ))
    }
}

fn read_private_fixture_bytes(
    paths: Vec<PathBuf>,
) -> Result<Vec<(PathBuf, Vec<u8>)>, Box<dyn std::error::Error>> {
    let mut content_digests = std::collections::HashSet::<[u8; 32]>::new();
    let mut fixtures = Vec::with_capacity(paths.len());
    for (index, path) in paths.into_iter().enumerate() {
        let ordinal = index + 1;
        let bytes = read_validated_regular_file(&path)
            .map_err(|_| format!("private fixture {ordinal:02} could not be read"))?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if !content_digests.insert(digest) {
            return Err(format!("private fixture {ordinal:02} duplicates another fixture").into());
        }
        fixtures.push((path, bytes));
    }
    Ok(fixtures)
}

fn read_private_fixtures(
    paths: Vec<PathBuf>,
) -> Result<Vec<PrivateFixture>, Box<dyn std::error::Error>> {
    let fixture_bytes = read_private_fixture_bytes(paths)?;
    let mut fixtures = Vec::with_capacity(fixture_bytes.len());
    for (index, (path, bytes)) in fixture_bytes.into_iter().enumerate() {
        let ordinal = index + 1;
        let block = parse_fixture(&bytes)
            .map_err(|_| format!("private fixture {ordinal:02} could not be parsed"))?;
        fixtures.push(PrivateFixture { path, block });
    }
    Ok(fixtures)
}

fn validate_private_transaction_counts(
    counts: &[usize],
) -> Result<usize, Box<dyn std::error::Error>> {
    if counts.is_empty() {
        return Err("private fixture set is empty".into());
    }
    for (index, count) in counts.iter().enumerate() {
        if *count != PRIVATE_TRANSACTIONS_PER_FIXTURE {
            return Err(format!(
                "private fixture {:02} must contain exactly {PRIVATE_TRANSACTIONS_PER_FIXTURE} active transactions",
                index + 1
            )
            .into());
        }
    }
    counts
        .len()
        .checked_mul(PRIVATE_TRANSACTIONS_PER_FIXTURE)
        .ok_or_else(|| "private aggregate transaction count overflow".into())
}

fn aggregate_score(
    transactions: usize,
    durations: &[Duration],
) -> Result<AggregateScore, Box<dyn std::error::Error>> {
    let proving_duration = durations
        .iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total.checked_add(*duration)
        })
        .ok_or("aggregate proving duration overflow")?;
    let proving_seconds = proving_duration.as_secs_f64();
    if transactions == 0 || proving_seconds <= 0.0 {
        return Err("aggregate score requires positive transactions and duration".into());
    }
    Ok(AggregateScore {
        proving_seconds,
        transactions_per_second: transactions as f64 / proving_seconds,
    })
}

fn read_proof(path: &Path) -> Result<Proof, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 || metadata.len() > MAX_PROOF_BYTES {
        return Err("proof output is empty or exceeds the trusted size limit".into());
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(deserialize_proof(&bytes)?)
}

fn deserialize_proof(bytes: &[u8]) -> Result<Proof, Box<bincode::ErrorKind>> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_PROOF_BYTES)
        .deserialize(bytes)
}

fn verify(
    block: &Block<F>,
    circuits: &Circuits,
    proof: &Proof,
) -> Result<(), Box<dyn std::error::Error>> {
    circuits.block_data.verify(proof.clone())?;

    let actual =
        BlockWitness::from_public_inputs(&proof.public_inputs, ON_CHAIN_OPERATIONS_LIMIT, 1);
    let expected = expected_block_witness(block);
    let checks = [
        (
            actual.block_number == expected.block_number,
            "block.block_number",
        ),
        (actual.created_at == expected.created_at, "block.created_at"),
        (
            actual.old_state_root == expected.old_state_root,
            "block.old_state_root",
        ),
        (
            actual.new_validium_root == expected.new_validium_root,
            "block.new_validium_root",
        ),
        (
            actual.new_state_root == expected.new_state_root,
            "block.new_state_root",
        ),
        (
            actual.old_account_delta_tree_root == expected.old_account_delta_tree_root,
            "block.old_account_delta_tree_root",
        ),
        (
            actual.new_account_delta_tree_root == expected.new_account_delta_tree_root,
            "block.new_account_delta_tree_root",
        ),
        (
            actual.on_chain_operations_count == expected.on_chain_operations_count,
            "block.on_chain_operations_count",
        ),
        (
            actual.on_chain_operations_pub_data == expected.on_chain_operations_pub_data,
            "block.on_chain_operations_pub_data",
        ),
        (
            actual.priority_operations_count == expected.priority_operations_count,
            "block.priority_operations_count",
        ),
        (
            actual.old_prefix_priority_operation_hash
                == expected.old_prefix_priority_operation_hash,
            "block.old_prefix_priority_operation_hash",
        ),
        (
            actual.new_prefix_priority_operation_hash
                == expected.new_prefix_priority_operation_hash,
            "block.new_prefix_priority_operation_hash",
        ),
        (
            actual.new_public_market_details == expected.new_public_market_details,
            "block.new_public_market_details",
        ),
    ];
    if let Some((_, field)) = checks.into_iter().find(|(matches, _)| !matches) {
        return Err(format!("proof public output does not match trusted fixture: {field}").into());
    }
    Ok(())
}

fn expected_block_witness(block: &Block<F>) -> BlockWitness<F> {
    BlockWitness::from_block(block, ON_CHAIN_OPERATIONS_LIMIT)
}

fn parse_fixture(bytes: &[u8]) -> Result<Block<F>, serde_json::Error> {
    Block::from_json_with_empty_txs(
        bytes,
        HEAVY_TX_PER_PROOF,
        LIGHT_TX_PER_PROOF,
        PUBLIC_HEAVY_TX_COUNT,
        PUBLIC_LIGHT_TX_COUNT,
    )
}

fn active_transaction_count(block: &Block<F>) -> usize {
    block
        .tx_chunks
        .iter()
        .flatten()
        .filter(|tx| tx.tx_type != TX_TYPE_EMPTY)
        .count()
}

fn trusted_transaction_count(block: &Block<F>) -> usize {
    let active = active_transaction_count(block);
    if active == 0 {
        PUBLIC_HEAVY_TX_COUNT + PUBLIC_LIGHT_TX_COUNT
    } else {
        active
    }
}

#[cfg(unix)]
fn create_scratch_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700).create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() || metadata.permissions().mode() & 0o777 != 0o700 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "scratch directory was not atomically created with mode 0700",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn create_scratch_directory(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "secure scratch directory creation is unsupported",
    ))
}

fn prepare_scratch(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.is_absolute() || path == Path::new("/") {
        return Err("refusing unsafe scratch path".into());
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => fs::remove_dir_all(path)?,
        Ok(_) => return Err("refusing non-directory scratch path".into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    create_scratch_directory(path)?;
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> Result<(), io::Error> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn executable_sha256() -> Result<String, Box<dyn std::error::Error>> {
    Ok(file_sha256(&env::current_exe()?)?)
}

fn read_verified_fixture(
    path: &Path,
    expected_sha256: &str,
) -> Result<(Vec<u8>, String), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let digest = sha256(&bytes[..])?;
    if digest != expected_sha256 {
        return Err(
            format!("fixture SHA-256 mismatch: expected {expected_sha256}, got {digest}").into(),
        );
    }
    Ok((bytes, digest))
}

fn file_sha256(path: &Path) -> Result<String, std::io::Error> {
    sha256(File::open(path)?)
}

fn sha256(mut reader: impl Read) -> Result<String, std::io::Error> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicU64, Ordering};

    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};

    use super::*;

    static TEMP_DIR_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TempFixtureDir(PathBuf);

    impl TempFixtureDir {
        fn new(label: &str) -> Self {
            let sequence = TEMP_DIR_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "lighter-harness-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("temporary fixture directory must be creatable");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFixtureDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_private_fixture_files(directory: &Path, names: &[&str]) {
        for name in names {
            fs::write(directory.join(name), b"synthetic fixture\n")
                .expect("synthetic fixture file must be writable");
        }
    }

    fn score_metrics(
        runtime: &str,
        benchmark_case: &'static str,
        provisional: bool,
        competitive: bool,
        cacheable: bool,
        transaction_count_source: &'static str,
        transactions: usize,
        fixture_id: &str,
        fixture_sha256: Option<String>,
        fixture_count: usize,
        proof_count: usize,
    ) -> ScoreMetrics {
        ScoreMetrics {
            runtime: runtime.to_owned(),
            benchmark_case,
            provisional,
            competitive,
            cacheable,
            transaction_count_source,
            timing_authority: "trusted verifier parent",
            proving_seconds: 5.0,
            transactions,
            transactions_per_second: transactions as f64 / 5.0,
            candidate_sha: "0".repeat(40),
            verifier_source_rev: VERIFIER_SOURCE_REV,
            verifier_sha256: "0".repeat(64),
            protocol_version: PROTOCOL_VERSION,
            fixture_id: fixture_id.to_owned(),
            fixture_sha256,
            fixture_count,
            verified_proofs: proof_count,
            expected_proofs: proof_count,
        }
    }

    fn public_fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/bench_test.json")
    }

    #[test]
    fn private_fixture_directory_discovers_sorted_direct_child_json_files() {
        let directory = TempFixtureDir::new("sorted-private-set");
        write_private_fixture_files(
            directory.path(),
            &["charlie.json", "alpha.json", "bravo.json"],
        );

        let paths = private_fixture_paths(directory.path(), 3)
            .expect("any configured set of direct-child JSON files must be accepted");
        let expected =
            ["alpha.json", "bravo.json", "charlie.json"].map(|name| directory.path().join(name));

        assert_eq!(paths, expected);
    }

    #[test]
    fn private_fixture_directory_enforces_configured_count() {
        let directory = TempFixtureDir::new("configured-count");
        write_private_fixture_files(directory.path(), &["one.json", "two.json"]);

        assert!(private_fixture_paths(directory.path(), 0).is_err());
        assert!(private_fixture_paths(directory.path(), 1).is_err());
        assert!(private_fixture_paths(directory.path(), 3).is_err());
        assert_eq!(
            private_fixture_paths(directory.path(), 2)
                .expect("the discovered count must match configuration")
                .len(),
            2
        );
    }

    #[test]
    fn private_fixture_count_configuration_is_required_and_positive() {
        assert!(parse_expected_fixture_count(None).is_err());
        assert!(parse_expected_fixture_count(Some("")).is_err());
        assert!(parse_expected_fixture_count(Some("0")).is_err());
        assert!(parse_expected_fixture_count(Some("invalid")).is_err());
        assert_eq!(
            parse_expected_fixture_count(Some("7"))
                .expect("a positive configured fixture count must be accepted"),
            7
        );
    }

    #[test]
    fn private_fixture_directory_rejects_non_json_entries() {
        let directory = TempFixtureDir::new("non-json-private-entry");
        write_private_fixture_files(directory.path(), &["one.json", "two.json"]);
        fs::write(directory.path().join("notes.txt"), b"synthetic note\n")
            .expect("synthetic non-JSON entry must be writable");

        assert!(private_fixture_paths(directory.path(), 2).is_err());
    }

    #[test]
    fn private_fixture_directory_rejects_names_outside_the_extractor_pattern() {
        for (index, name) in [
            ".json",
            "-leading.json",
            "contains space.json",
            "contains.dot.json",
            "unicodé.json",
        ]
        .into_iter()
        .enumerate()
        {
            let directory = TempFixtureDir::new(&format!("unsafe-private-name-{index}"));
            write_private_fixture_files(directory.path(), &[name]);

            assert!(
                private_fixture_paths(directory.path(), 1).is_err(),
                "private fixture name must be rejected: {name:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn private_fixture_directory_rejects_symlinks_and_non_regular_entries() {
        use std::os::unix::fs::symlink;

        let symlinked = TempFixtureDir::new("symlinked-private-entry");
        write_private_fixture_files(symlinked.path(), &["one.json"]);
        symlink(
            symlinked.path().join("one.json"),
            symlinked.path().join("two.json"),
        )
        .expect("synthetic fixture symlink must be creatable");
        assert!(private_fixture_paths(symlinked.path(), 2).is_err());

        let non_regular = TempFixtureDir::new("non-regular-private-entry");
        write_private_fixture_files(non_regular.path(), &["one.json"]);
        fs::create_dir(non_regular.path().join("two.json"))
            .expect("synthetic non-regular entry must be creatable");
        assert!(private_fixture_paths(non_regular.path(), 2).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn validated_reader_rejects_symlink_swap_between_metadata_and_open() {
        use std::os::unix::fs::symlink;

        let directory = TempFixtureDir::new("symlink-swap");
        let fixture = directory.path().join("one.json");
        let original = directory.path().join("original");
        let replacement = directory.path().join("replacement");
        fs::write(&fixture, b"validated bytes").expect("synthetic fixture must be writable");
        fs::write(&replacement, b"replacement bytes")
            .expect("synthetic replacement must be writable");

        let result = read_validated_regular_file_with(&fixture, |path| {
            fs::rename(path, &original)?;
            symlink(&replacement, path)?;
            File::open(path)
        });

        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn private_fixture_directory_rejects_duplicate_equivalent_files() {
        let directory = TempFixtureDir::new("duplicate-private-entry");
        write_private_fixture_files(directory.path(), &["one.json"]);
        fs::hard_link(
            directory.path().join("one.json"),
            directory.path().join("two.json"),
        )
        .expect("synthetic hard-linked fixture must be creatable");

        assert!(private_fixture_paths(directory.path(), 2).is_err());
    }

    #[test]
    fn private_fixture_reader_rejects_duplicate_content_in_distinct_files() {
        let directory = TempFixtureDir::new("duplicate-private-content");
        write_private_fixture_files(directory.path(), &["one.json", "two.json"]);
        let paths = private_fixture_paths(directory.path(), 2)
            .expect("distinct regular fixture files must pass discovery");

        let error = read_private_fixture_bytes(paths)
            .expect_err("byte-identical fixture files must be rejected");

        assert_eq!(
            error.to_string(),
            "private fixture 02 duplicates another fixture"
        );
    }

    #[test]
    fn every_private_fixture_must_have_500_active_transactions_with_dynamic_total() {
        assert_eq!(
            validate_private_transaction_counts(&[500, 500])
                .expect("any positive number of valid private counts must be accepted"),
            1_000
        );
        assert_eq!(
            validate_private_transaction_counts(&[500; 6])
                .expect("aggregate count must derive from fixture count"),
            3_000
        );

        for ordinal in 0..3 {
            let mut counts = [500; 3];
            counts[ordinal] = 499;
            let error = validate_private_transaction_counts(&counts)
                .expect_err("an invalid private transaction count must be rejected");
            assert!(
                error
                    .to_string()
                    .contains(&format!("fixture {:02}", ordinal + 1)),
                "private error must identify only the generic ordinal: {error}"
            );
        }
        assert!(validate_private_transaction_counts(&[]).is_err());
    }

    #[test]
    fn aggregate_score_sums_dynamic_worker_durations_and_transactions() {
        let aggregate = aggregate_score(
            1_500,
            &[
                Duration::from_millis(1_250),
                Duration::from_millis(750),
                Duration::from_secs(1),
            ],
        )
        .expect("positive aggregate duration must produce a score");

        assert_eq!(aggregate.proving_seconds, 3.0);
        assert_eq!(aggregate.transactions_per_second, 500.0);
    }

    #[test]
    fn worker_sequence_uses_remaining_time_from_one_shared_deadline() {
        let started = Instant::now();
        let deadline = started + Duration::from_secs(10);

        assert_eq!(
            remaining_time(deadline, started + Duration::from_secs(3)),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            remaining_time(deadline, started + Duration::from_secs(9)),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            remaining_time(deadline, started + Duration::from_secs(10)),
            None
        );
    }

    #[test]
    fn deadline_check_observes_an_already_exited_worker_before_cleanup() {
        let mut child = Command::new("/usr/bin/true")
            .spawn()
            .expect("synthetic worker must start");
        child.wait().expect("synthetic worker must exit");

        let status = wait_for_exit(&mut child, Instant::now())
            .expect("an already-exited worker must not enter timeout cleanup");

        assert!(status.success());
    }

    #[test]
    fn timeout_cleanup_attempts_wait_after_kill_failure() {
        let wait_attempted = Cell::new(false);
        let kill_error = io::Error::other("synthetic kill failure");

        let error = complete_timeout_cleanup(Err(kill_error), || {
            wait_attempted.set(true);
            Err(io::Error::other("synthetic wait failure"))
        })
        .expect_err("cleanup failures must be propagated");

        assert!(wait_attempted.get(), "wait must run after kill failure");
        assert!(error.to_string().contains("synthetic kill failure"));
        assert!(error.to_string().contains("synthetic wait failure"));
    }

    #[test]
    fn deadline_cleanup_reaps_the_worker_and_preserves_timeout_semantics() {
        let mut child = Command::new("/bin/sleep")
            .arg("10")
            .spawn()
            .expect("synthetic timeout worker must start");

        let error = wait_for_exit(&mut child, Instant::now())
            .expect_err("expired deadline must report a timeout");

        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(
            child
                .try_wait()
                .expect("reaped worker state must be readable")
                .is_some(),
            "timed-out worker must be reaped"
        );
    }

    #[test]
    fn private_workers_reuse_only_the_proof_path_cleared_by_scratch_reset() {
        let scratch = Path::new("/synthetic/scratch");

        assert_eq!(private_proof_path(scratch), scratch.join("proof.bin"));
    }

    #[test]
    fn private_sequence_removes_stale_proof_and_cache_before_every_launch() {
        let scratch = TempFixtureDir::new("private-sequence-clean-scratch");
        fs::write(scratch.path().join("proof.bin"), b"stale proof")
            .expect("stale proof must be writable");
        fs::write(scratch.path().join("candidate.cache"), b"stale cache")
            .expect("stale cache must be writable");
        let fixtures = ["synthetic-a", "synthetic-b"];
        let mut launches = 0;

        let aggregate = run_private_sequence(
            &fixtures,
            scratch.path(),
            Instant::now() + Duration::from_secs(10),
            1_000,
            |_, _, proof_path, _| {
                let entries = fs::read_dir(scratch.path())
                    .expect("scratch must be readable at launch")
                    .collect::<Result<Vec<_>, _>>()
                    .expect("scratch entries must be readable");
                assert!(entries.is_empty(), "each worker must receive empty scratch");
                assert!(!proof_path.exists(), "proof path must be absent at launch");

                launches += 1;
                fs::write(proof_path, b"synthetic proof")
                    .expect("synthetic proof must be writable");
                fs::write(scratch.path().join("candidate.cache"), b"synthetic cache")
                    .expect("synthetic cache must be writable");
                Ok(Duration::from_secs(1))
            },
            |_, _, proof_path| {
                assert!(proof_path.is_file(), "worker proof must reach verification");
                Ok(())
            },
        )
        .expect("clean synthetic sequence must produce an aggregate");

        assert_eq!(launches, 2);
        assert_eq!(aggregate.proving_seconds, 2.0);
    }

    #[cfg(unix)]
    #[test]
    fn private_sequence_production_spawn_seam_resets_tmpdir_between_workers() {
        use std::os::unix::fs::PermissionsExt;

        let workspace = TempFixtureDir::new("private-sequence-production-spawn");
        let scratch = workspace.path().join("scratch");
        let worker = workspace.path().join("synthetic-worker.sh");
        let fixtures = [
            workspace.path().join("fixture-a"),
            workspace.path().join("fixture-b"),
        ];
        fs::write(&fixtures[0], b"fresh-a\n").expect("first synthetic fixture must be writable");
        fs::write(&fixtures[1], b"fresh-b\n").expect("second synthetic fixture must be writable");
        fs::write(
            &worker,
            br#"#!/bin/sh
set -eu
fixture="$1"
proof="$2"
[ "${proof}" = "${TMPDIR}/proof.bin" ]
for entry in "${TMPDIR}"/* "${TMPDIR}"/.[!.]* "${TMPDIR}"/..?*; do
  [ ! -e "${entry}" ] || exit 20
done
IFS= read -r marker < "${fixture}"
printf '%s\n' "${marker}" > "${proof}"
printf '%s\n' "cache from ${marker}" > "${TMPDIR}/candidate.cache"
"#,
        )
        .expect("synthetic worker must be writable");
        fs::set_permissions(&worker, fs::Permissions::from_mode(0o700))
            .expect("synthetic worker must be executable");
        let config = Config {
            worker,
            fixture: workspace.path().to_owned(),
            scratch: scratch.clone(),
            score: workspace.path().join("score.json"),
            mode: PRIVATE_RANKED_MODE.to_owned(),
            transactions: 1_000,
            candidate_sha: "0".repeat(40),
            sandbox_profile: None,
        };
        let verifications = Cell::new(0);

        let aggregate = run_private_sequence(
            &fixtures,
            &scratch,
            Instant::now() + Duration::from_secs(10),
            1_000,
            |ordinal, fixture, proof_path, deadline| {
                spawn_and_time_worker(&config, fixture, proof_path, deadline)
                    .map_err(|error| format!("private fixture {ordinal:02} {error}"))
            },
            |_, fixture, proof_path| {
                let expected =
                    fs::read(fixture).map_err(|error| format!("fixture read failed: {error}"))?;
                let actual =
                    fs::read(proof_path).map_err(|error| format!("proof read failed: {error}"))?;
                if actual != expected {
                    return Err("synthetic proof did not match its fixture".to_owned());
                }
                verifications.set(verifications.get() + 1);
                Ok(())
            },
        )
        .expect("actual spawn seam must isolate each synthetic worker");

        assert_eq!(verifications.get(), 2);
        assert!(aggregate.proving_seconds > 0.0);
        assert_eq!(
            fs::read(scratch.join("proof.bin")).expect("last fresh proof must remain"),
            b"fresh-b\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn private_sequence_recreates_scratch_with_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = TempFixtureDir::new("private-sequence-scratch-mode");

        run_private_sequence(
            &["synthetic"],
            scratch.path(),
            Instant::now() + Duration::from_secs(10),
            500,
            |_, _, proof_path, _| {
                let mode = fs::metadata(scratch.path())
                    .expect("scratch metadata must be readable")
                    .permissions()
                    .mode()
                    & 0o777;
                assert_eq!(mode, 0o700);
                fs::write(proof_path, b"synthetic proof")
                    .expect("synthetic proof must be writable");
                Ok(Duration::from_secs(1))
            },
            |_, _, _| Ok(()),
        )
        .expect("synthetic sequence must complete");
    }

    #[cfg(unix)]
    #[test]
    fn scratch_creation_seam_atomically_uses_owner_only_mode() {
        use std::os::unix::fs::PermissionsExt;

        let parent = TempFixtureDir::new("atomic-private-scratch");
        let scratch = parent.path().join("scratch");

        create_scratch_directory(&scratch).expect("scratch creation must succeed atomically");

        let mode = fs::metadata(&scratch)
            .expect("scratch metadata must be readable")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn private_sequence_aborts_before_launch_when_scratch_cleanup_fails() {
        let parent = TempFixtureDir::new("private-sequence-cleanup-failure");
        let scratch = parent.path().join("scratch-is-a-file");
        fs::write(&scratch, b"not a directory").expect("synthetic scratch file must be writable");
        let launches = Cell::new(0);

        let result = run_private_sequence(
            &["synthetic"],
            &scratch,
            Instant::now() + Duration::from_secs(10),
            500,
            |_, _, _, _| {
                launches.set(launches.get() + 1);
                Ok(Duration::from_secs(1))
            },
            |_, _, _| Ok(()),
        );

        let error = result.expect_err("scratch cleanup failure must abort the sequence");
        assert_eq!(
            error.to_string(),
            "private fixture 01 scratch could not be reset"
        );
        assert_eq!(launches.get(), 0);
    }

    #[test]
    fn scratch_reset_rejects_relative_paths() {
        let relative = PathBuf::from(format!(
            "lighter-relative-scratch-{}",
            TEMP_DIR_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));

        let result = prepare_scratch(&relative);
        if relative.is_dir() {
            fs::remove_dir_all(&relative).expect("relative scratch test path must be removable");
        }

        assert!(result.is_err(), "relative scratch paths must be rejected");
    }

    #[test]
    fn private_sequence_timeout_stops_later_fixtures_without_an_aggregate() {
        let scratch = TempFixtureDir::new("private-sequence-worker-timeout");
        let launches = RefCell::new(Vec::new());
        let verifications = RefCell::new(Vec::new());

        let result = run_private_sequence(
            &["synthetic-a", "synthetic-b", "synthetic-c"],
            scratch.path(),
            Instant::now() + Duration::from_secs(10),
            1_500,
            |ordinal, _, proof_path, _| {
                launches.borrow_mut().push(ordinal);
                if ordinal == 2 {
                    return Err("synthetic worker timeout".to_owned());
                }
                fs::write(proof_path, b"synthetic proof")
                    .expect("synthetic proof must be writable");
                Ok(Duration::from_secs(1))
            },
            |ordinal, _, _| {
                verifications.borrow_mut().push(ordinal);
                Ok(())
            },
        );

        let error = result.expect_err("worker timeout must yield no aggregate");
        assert_eq!(error.to_string(), "synthetic worker timeout");
        assert_eq!(&*launches.borrow(), &[1, 2]);
        assert_eq!(&*verifications.borrow(), &[1]);
    }

    #[test]
    fn stale_score_cleanup_propagates_errors_instead_of_continuing() {
        let directory = TempFixtureDir::new("stale-score-cleanup");
        let score = directory.path().join("score.json");
        fs::create_dir(&score).expect("synthetic invalid score directory must be creatable");

        let result = remove_file_if_exists(&score);

        assert!(result.is_err(), "score cleanup failure must be propagated");
        assert!(score.is_dir(), "failed cleanup must not disguise the path");
    }

    #[test]
    fn public_score_keeps_one_fixture_one_proof_and_the_fixed_hash() {
        let metrics = score_metrics(
            PUBLIC_SYNTHETIC_MODE,
            PUBLIC_SYNTHETIC_CASE,
            true,
            false,
            true,
            "synthetic-configuration-unbound-by-proof",
            500,
            "bench_test.json",
            Some(EXPECTED_FIXTURE_SHA256.to_owned()),
            1,
            1,
        );
        let serialized =
            serde_json::to_value(metrics).expect("public score metrics must serialize");

        assert_eq!(serialized["transactions"], 500);
        assert_eq!(serialized["fixture_count"], 1);
        assert_eq!(serialized["verified_proofs"], 1);
        assert_eq!(serialized["expected_proofs"], 1);
        assert_eq!(
            serialized["fixture_sha256"],
            serde_json::json!(EXPECTED_FIXTURE_SHA256)
        );
    }

    #[test]
    fn private_score_serialization_is_aggregate_only_and_omits_fixture_hashes() {
        let metrics = score_metrics(
            PRIVATE_RANKED_MODE,
            PRIVATE_RANKED_CASE,
            false,
            true,
            false,
            "parsed-active-witness",
            1_500,
            "ranked-v1",
            None,
            3,
            3,
        );
        let serialized =
            serde_json::to_value(metrics).expect("private score metrics must serialize");

        assert_eq!(serialized["fixture_id"], "ranked-v1");
        assert_eq!(serialized["transactions"], 1_500);
        assert_eq!(serialized["fixture_count"], 3);
        assert_eq!(serialized["verified_proofs"], 3);
        assert_eq!(serialized["expected_proofs"], 3);
        assert!(serialized.get("fixture_sha256").is_none());
        assert!(serialized.get("fixtures").is_none());
        assert!(serialized.get("per_fixture").is_none());
    }

    #[test]
    fn ranked_score_schema_distinguishes_public_and_private_aggregate_scores() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../ranked-score.schema.json"))
                .expect("ranked score schema must be valid JSON");
        let required = schema["properties"]["metrics"]["required"]
            .as_array()
            .expect("metrics required fields must be an array");
        assert!(required.contains(&serde_json::json!("fixture_count")));
        assert!(!required.contains(&serde_json::json!("fixture_sha256")));

        let branches = schema["properties"]["metrics"]["oneOf"]
            .as_array()
            .expect("score modes must use schema branches");
        let branch = |mode: &str| {
            branches
                .iter()
                .find(|branch| branch["properties"]["runtime"]["const"] == serde_json::json!(mode))
                .expect("score mode must have a schema branch")
        };
        let public = branch(PUBLIC_SYNTHETIC_MODE);
        assert_eq!(public["properties"]["transactions"]["const"], 500);
        assert_eq!(public["properties"]["fixture_count"]["const"], 1);
        assert_eq!(public["properties"]["verified_proofs"]["const"], 1);
        assert_eq!(public["properties"]["expected_proofs"]["const"], 1);
        assert_eq!(
            public["properties"]["fixture_sha256"]["const"],
            EXPECTED_FIXTURE_SHA256
        );

        let private = branch(PRIVATE_RANKED_MODE);
        assert_eq!(private["properties"]["transactions"]["minimum"], 500);
        assert_eq!(private["properties"]["transactions"]["multipleOf"], 500);
        assert_eq!(private["properties"]["fixture_id"]["const"], "ranked-v1");
        assert_eq!(private["properties"]["fixture_count"]["minimum"], 1);
        assert_eq!(private["properties"]["verified_proofs"]["minimum"], 1);
        assert_eq!(private["properties"]["expected_proofs"]["minimum"], 1);
        assert_eq!(
            private["not"]["required"],
            serde_json::json!(["fixture_sha256"])
        );
    }

    #[test]
    fn serialized_public_and_dynamic_private_scores_validate_against_ranked_schema() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../ranked-score.schema.json"))
                .expect("ranked score schema must be valid JSON");
        let validator =
            jsonschema::validator_for(&schema).expect("ranked score schema must compile");
        let scores = [
            (
                "public",
                ScoreFile {
                    score: 100.0,
                    passed: true,
                    metrics: score_metrics(
                        PUBLIC_SYNTHETIC_MODE,
                        PUBLIC_SYNTHETIC_CASE,
                        true,
                        false,
                        true,
                        "synthetic-configuration-unbound-by-proof",
                        500,
                        "bench_test.json",
                        Some(EXPECTED_FIXTURE_SHA256.to_owned()),
                        1,
                        1,
                    ),
                },
            ),
            (
                "dynamic private",
                ScoreFile {
                    score: 700.0,
                    passed: true,
                    metrics: score_metrics(
                        PRIVATE_RANKED_MODE,
                        PRIVATE_RANKED_CASE,
                        false,
                        true,
                        false,
                        "parsed-active-witness",
                        3_500,
                        PRIVATE_FIXTURE_ID,
                        None,
                        7,
                        7,
                    ),
                },
            ),
        ];

        for (label, score) in scores {
            let serialized =
                serde_json::to_value(score).expect("representative score must serialize");
            let errors = validator
                .iter_errors(&serialized)
                .map(|error| error.to_string())
                .collect::<Vec<_>>();
            assert!(
                errors.is_empty(),
                "{label} score failed ranked schema validation: {}",
                errors.join("; ")
            );
        }
    }

    #[test]
    fn trusted_proof_reader_returns_one_final_block_proof() {
        let read: fn(&Path) -> Result<Proof, Box<dyn std::error::Error>> = read_proof;
        let _ = read;
    }

    #[test]
    fn trusted_mixed_circuit_parameters_are_fixed() {
        assert_eq!(CHAIN_ID, 304);
        assert_eq!(HEAVY_TX_PER_PROOF, 4);
        assert_eq!(HEAVY_TX_MODE, TX_HEAVY);
        assert_eq!(LIGHT_TX_PER_PROOF, 10);
        assert_eq!(LIGHT_TX_MODE, TX_LIGHT);
        assert_eq!(ON_CHAIN_OPERATIONS_LIMIT, 1);
        assert_eq!(
            VERIFIER_SOURCE_REV,
            "381fd529eb61dfff9ad245d94fce214a0a64d927"
        );
    }

    #[test]
    fn public_synthetic_case_is_provisional_noncompetitive_and_cacheable() {
        let public = classify_benchmark_case(0, PUBLIC_SYNTHETIC_MODE)
            .expect("public synthetic mode must be accepted for an empty witness");
        assert_eq!(public.name, PUBLIC_SYNTHETIC_CASE);
        assert!(public.provisional);
        assert!(!public.competitive);
        assert!(public.cacheable);
        assert_eq!(
            public.transaction_count_source,
            "synthetic-configuration-unbound-by-proof"
        );
        assert!(classify_benchmark_case(0, PRIVATE_RANKED_MODE).is_err());

        let private = classify_benchmark_case(7, PRIVATE_RANKED_MODE)
            .expect("private ranked mode must accept active transactions");
        assert_eq!(private.name, PRIVATE_RANKED_CASE);
        assert!(!private.provisional);
        assert!(private.competitive);
        assert!(!private.cacheable);
        assert_eq!(private.transaction_count_source, "parsed-active-witness");
        assert!(classify_benchmark_case(7, PUBLIC_SYNTHETIC_MODE).is_err());
    }

    #[test]
    fn protected_fixture_deserializes_with_500_transactions() {
        let (transaction_count, heavy_chunks, light_chunks) = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let fixture =
                    fs::read(public_fixture()).expect("protected fixture must be readable");
                let block = parse_fixture(&fixture).expect("protected fixture must deserialize");
                let heavy_chunks = block
                    .tx_chunks
                    .iter()
                    .filter(|chunk| chunk[0].tx_circuit_type == TX_HEAVY)
                    .count();
                let light_chunks = block.tx_chunks.len() - heavy_chunks;

                (
                    trusted_transaction_count(&block),
                    heavy_chunks,
                    light_chunks,
                )
            })
            .expect("fixture deserialization thread must start")
            .join()
            .expect("fixture deserialization thread must complete");

        assert_eq!(transaction_count, 500);
        assert_eq!(heavy_chunks, 3);
        assert_eq!(light_chunks, 49);
    }

    #[test]
    fn fixture_public_outputs_use_circuit_padding() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let fixture =
                    fs::read(public_fixture()).expect("protected fixture must be readable");
                let block = parse_fixture(&fixture).expect("protected fixture must deserialize");
                let expected = expected_block_witness(&block);

                assert!(block.on_chain_operations_pub_data.is_empty());
                assert_eq!(expected.on_chain_operations_pub_data.len(), 1);
                assert!(
                    expected.on_chain_operations_pub_data[0]
                        .iter()
                        .all(|byte| *byte == 0)
                );
            })
            .expect("fixture normalization thread must start")
            .join()
            .expect("fixture normalization thread must complete");
    }

    #[test]
    fn approved_fixture_passes_sha256_validation() {
        let fixture = public_fixture();

        let (_, digest) = read_verified_fixture(&fixture, EXPECTED_FIXTURE_SHA256)
            .expect("approved fixture must pass SHA-256 validation");

        assert_eq!(digest, EXPECTED_FIXTURE_SHA256);
    }

    #[test]
    fn active_private_witness_uses_its_parsed_transaction_count() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let fixture =
                    fs::read(public_fixture()).expect("protected fixture must be readable");
                let mut json: serde_json::Value =
                    serde_json::from_slice(&fixture).expect("protected fixture JSON must parse");
                let txs = json["txs"]
                    .as_array_mut()
                    .expect("protected fixture must contain transactions");
                let trailing_empty = txs[0].clone();
                let mut active = trailing_empty.clone();
                active["tx_type"] = serde_json::json!(1);
                active["tct"] = serde_json::json!(TX_HEAVY);
                *txs = vec![active, trailing_empty];
                let active_fixture =
                    serde_json::to_vec(&json).expect("in-memory active fixture must serialize");
                let block = Block::<F>::from_json_with_empty_txs(
                    &active_fixture,
                    HEAVY_TX_PER_PROOF,
                    LIGHT_TX_PER_PROOF,
                    123,
                    456,
                )
                .expect("active fixture with trailing empty transaction must parse");

                assert_eq!(active_transaction_count(&block), 1);
                assert_eq!(trusted_transaction_count(&block), 1);
                assert_eq!(block.tx_chunks.len(), 2);
            })
            .expect("fixture transaction-count thread must start")
            .join()
            .expect("fixture transaction-count thread must complete");
    }

    #[test]
    fn exact_proof_parser_rejects_trailing_bytes() {
        let valid = include_bytes!("../../../bench/dummy-proof.bin");
        deserialize_proof(valid).expect("embedded proof must deserialize");

        let mut trailing = valid.to_vec();
        trailing.push(0);
        assert!(deserialize_proof(&trailing).is_err());
    }

    #[test]
    fn mismatching_fixture_is_rejected() {
        let fixture = env::temp_dir().join(format!(
            "lighter-harness-mismatching-fixture-{}.json",
            std::process::id()
        ));
        fs::write(&fixture, b"{}\n").expect("temporary fixture must be writable");

        let result = read_verified_fixture(&fixture, EXPECTED_FIXTURE_SHA256);
        fs::remove_file(&fixture).expect("temporary fixture must be removable");

        let error = result.expect_err("mismatching fixture must be rejected");
        assert!(
            error.to_string().contains("fixture SHA-256 mismatch"),
            "unexpected error: {error}"
        );
    }
}
