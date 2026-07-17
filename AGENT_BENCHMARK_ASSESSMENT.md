# Rapid Benchmark Integrity Assessment

You are reviewing the **current** Lighter prover challenge repository before it
is given to external competitors. Your task is to decide whether the benchmark
and challenge boundary are credible and sufficiently hardened for an initial
release.

This is an assessment only. Do **not** edit files, create commits, dispatch
GitHub workflows, or broaden the task into a general code audit.

## Timebox and standard of evidence

Spend at most **45 minutes**. Start with static tracing of the ranked path, then
run only small, high-value local checks if they can confirm or disprove a
specific concern. Do not run an expensive full proof merely to demonstrate that
the benchmark works.

Do not trust a README claim without tracing the relevant code. Do not report
speculative flaws as findings: distinguish verified issues from risks that need
an operator decision. A flaw is relevant if a contestant can improve a ranked
score without delivering the intended prover improvement, if the score is
unfair or non-reproducible, or if the setup can be abused or bypassed.

## Benchmark model

The intended task is to make proof generation faster while preserving proof
correctness:

```text
score = baseline time / candidate time
```

Contestants may change only `challenge/submission/`; the submitted source is
linked into the `prove` binary. The separate trusted `bench` binary starts
`prove`, measures its wall time, then reconstructs frozen verifier circuits and
checks the pre-execution and recursive proofs against fixture public outputs.

Ranked runs:

- use `.github/workflows/benchmark.yml` and
  `.github/scripts/run-ranked-benchmark.sh`;
- check out trusted `master` plus the dispatched candidate revision;
- reject committed candidate changes outside `challenge/submission/`;
- build trusted baseline and candidate using the locked challenge manifest;
- run both on the same ephemeral macOS runner; and
- sandbox the `prove` child using `sandbox-exec` with
  `.github/scripts/prover.sb`.

The present ranked case is the checked-in, public 500-transaction fixture
`bench/bench_test.json`. The documentation explicitly says that a rotating
private fixture pool and final private holdout are future work.

## Questions you must answer

Trace the relevant code and answer the following in priority order.

### 1. Is the ranked score meaningful?

- Does the timer measure the intended work, and can contestant-controlled code
  influence the timing, result, or score publication outside that work?
- Does verification bind the resulting proofs to all outputs that matter for
  this fixture and circuit version?
- Are baseline and candidate builds and measurements comparable, including
  compilation flags, fixture, environment, machine state, cache effects, and
  ordering?
- Does one fixed, public case allow fixture-specialized code or cached
  artifacts that satisfy the verifier but do not represent a generally faster
  prover? Assess this as a release blocker or an accepted limitation; do not
  merely repeat the documentation.

### 2. Is the edit boundary both enforceable and appropriate?

- Verify that a candidate cannot change the runner, Cargo manifests or lock
  files, fixtures, verifier, score output, workflow, or trusted baseline while
  still passing the scope check. Consider Git behavior, archives, paths,
  symlinks, generated files, build behavior, and uncommitted workspace state
  where relevant.
- Decide whether `challenge/submission/` is enough freedom for the intended
  optimization challenge while preventing contestants from redefining the task.
  Call out material restrictions that could make the challenge artificial or
  prevent legitimate optimization work.
- Check that code executed during build and benchmark phases has the intended
  trust level. Include the effects and limits of the macOS sandbox rather than
  treating “network denied” as complete containment.

### 3. Can the ranked environment be trusted and reproduced?

- Review dispatch/ref selection, ancestry validation, checkout behavior, GitHub
  permissions, runner pinning, timeout boundaries, artifact generation, and
  checksum handling.
- Review Rust toolchain and dependency pinning, the offline candidate build,
  and architecture-specific flags. Identify any uncontrolled input that changes
  a ranked result or lets a candidate affect a later run.
- Treat availability and runner damage separately from score manipulation. Only
  elevate them when they materially threaten a shared benchmark service.

### 4. Is the release gate realistic?

- Check that the documented local commands match the authoritative ranked path.
- Identify what must be in place before a public leaderboard is credible,
  especially fixture provenance, private cases/holdout, repeatability, and
  operational access to the runner.
- Separate “safe to run an internal beta” from “safe to rank public
  submissions.”

## Required inspection targets

Read these before reaching a verdict:

- `docs/BENCHMARK.md`
- `benchmark.json`, `setup.sh`, and `benchmark.sh`
- `.github/workflows/benchmark.yml` and `.github/workflows/ci.yml`
- `.github/scripts/run-ranked-benchmark.sh`
- `.github/scripts/sandbox-prove.sh` and `.github/scripts/prover.sb`
- `challenge/Cargo.toml`, `challenge/Cargo.lock`,
  `challenge/src/bin/bench.rs`, `challenge/src/bin/prove.rs`, and
  `challenge/src/api.rs`
- `challenge/submission/prover.rs`
- `challenge/ranked-score.schema.json`
- `rust-toolchain` and `.gitignore`

Use `git log` and recent diffs only when they clarify the intended security
property. Do not spend the timebox auditing the frozen circuit implementation;
focus on the benchmark harness and its boundary with contestant code.

## Report format

Return the report in chat, not as a repository change. Keep it concise and
decision-oriented:

```markdown
# Benchmark assessment — <date>

## Verdict: READY | READY WITH CONDITIONS | NOT READY

One paragraph distinguishing internal beta readiness from public ranking
readiness.

## Findings

### [BLOCKER|HIGH|MEDIUM|LOW] Short title
- **Evidence:** exact file and line range, plus command output if you ran one.
- **Exploit or failure path:** concrete, minimum steps and required access.
- **Impact:** what invalid score, unfairness, compromise, or outage results.
- **Recommendation:** smallest effective hardening step.

## Release gates
1. Ordered actions required before public ranked submissions.

## Positive controls verified
- Controls you traced that genuinely work, with file references.

## Open questions / accepted limitations
- Items that need an operator decision or evidence unavailable in the repo.
```

Use `READY` only if no blocker or high-severity issue prevents the intended
initial release. Use `READY WITH CONDITIONS` for an internal beta that is
adequately bounded but not yet suitable for a credible public leaderboard. Use
`NOT READY` if the current setup permits invalid ranked results, does not
enforce its stated trust boundary, or lacks a necessary operational control.

For every finding, include a precise path and line range. If an alleged issue
cannot be tied to a concrete path through the current code, omit it or place it
under open questions.
