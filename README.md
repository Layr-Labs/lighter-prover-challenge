# Lighter Prover and Circuits

## Telemetry: your coding-agent sessions are recorded and uploaded

**Cloning this repository and then running an AI coding agent inside it sends that agent's full
session transcript to Eigen Labs.** Capture is on by default, it happens on your own machine from
the moment the agent starts, and it covers the whole session — not only the code you end up
submitting. Read this section before you start work.

### What turns it on

This repository ships agent hook configuration that supported agents load automatically from the
working directory. There is no install step, prompt, or confirmation.

| File | Agent | Events that fire a hook |
| --- | --- | --- |
| `.claude/settings.local.json` | Claude Code | session start, subagent start, prompt submit, stop, stop-on-failure, session end, subagent stop |
| `.codex/config.toml` | Codex | session start, subagent start, prompt submit, stop, subagent stop |
| `.cursor/hooks.json` | Cursor | session start, prompt submit, stop, session end, subagent stop, plus a session-registration hook matched to shell commands that invoke the CLI |
| `opencode.json` and `.opencode/plugin/hilbert-trace.ts` | OpenCode | every native event except token deltas |
| `.omp/extensions/hilbert-trace.ts` | OMP | session start, before agent start, agent end, session shutdown |
| `.pi/extensions/hilbert-trace.ts` | Pi | session start, before agent start, agent end, session shutdown |

The OMP and Pi entries are the same extension file checked in under two paths; it detects at runtime
which of the two agents loaded it. Each of these hooks runs the same wrapper script,
`.hilbert/hooks/hilbert-trace.sh`. The wrapper is a thin transport: it pipes the hook payload into
`hilbert trace session` or `hilbert trace hook` and returns immediately. Reading transcripts,
redacting, retrying, and uploading are all done by the challenge CLI, in the background.

### What is captured

The hook payload identifies the agent session and tells the CLI where that agent keeps its native
transcript file. The CLI reads that file itself and uploads the bytes that are new since the last
checkpoint. The transcript is the entire session:

- your prompts;
- the model's replies, **including its reasoning / "thinking" content**;
- every tool call and every tool result — files the agent read, edits it made, shell commands it
  ran and their output;
- the working directory path, the agent name, and the agent's native session id.

Before upload the CLI applies best-effort regular-expression redaction to a short list of obvious
credential formats: `sk-…` and `sk-ant-…` API keys, `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` and
`github_pat_` GitHub tokens, `AKIA…` AWS key ids, JWTs, and PEM private-key blocks. **That is
pattern matching, not a guarantee.** Anything else that passes through the agent's context —
other credentials, private files, personal data, unrelated source code — is uploaded verbatim. Do
not use this checkout for unrelated work, and do not point an agent here at material you are not
willing to hand over.

### Where it goes

Transcripts are uploaded to the challenge API operated by Eigen Labs, Inc. They are keyed to the
challenge id stored in the clone's git config and to the account behind the API key you logged in
with, and are stored as metadata in the challenge database and as compressed transcript files in
Cloudflare R2. Under the challenge Privacy Policy this material is "Covered Data": it is used
internally to evaluate submissions and study how the challenge is solved, it is not currently
published, and the public leaderboard exposes none of it. The Privacy Policy on the challenge site
is the controlling statement of terms; this section describes the mechanism.

Uploading requires a stored challenge API key and a challenge checkout. If you have not logged in
with the CLI, or if capture is off, the hook still fires but the CLI exits without reading or
sending anything.

### Turning it off, and checking what is on

- `hilbert trace status` reports whether capture is enabled, whether this directory is recognised
  as a challenge checkout, and which hook files are installed.
- `hilbert trace off` disables capture for every challenge repository. The hook files stay on disk,
  but the CLI stops reading transcripts, stops uploading, and stops reinstalling hook files.
- Deleting the hook files while capture is enabled does not stick — the CLI rewrites them the next
  time it runs here. Use `hilbert trace off`.
- Disabling capture does not stop you from building, benchmarking, or submitting.

### Integrity of the hook files

None of these files are in the submission-editable path set, so a submission cannot change them.
CI additionally checks all eight of them — the four agent configs, the wrapper script, the OpenCode
plugin, and the OMP and Pi extensions — against what the CLI's trace installer produces
(`.github/scripts/verify-agent-trace-configs.sh`). The configs and the wrapper are regenerated and
compared byte for byte; the three TypeScript files, which the CLI emits verbatim, are pinned by
sha256. The check also requires each of those directories to hold exactly its one expected file, so
an unpinned second hook cannot be added beside a pinned one. They must never invoke anything other
than the trace wrapper. If you ever find that they do, treat it as a compromise and report it rather
than running the agent.

Those expected copies have been checked against a real CLI rather than only against themselves:
running `hilbert trace init --force` in a clean checkout reproduced every managed file byte for byte,
and `hilbert trace status` reports each adapter as `current`. The OMP and Pi extensions are also
byte-identical to the copies checked in to other Eigen Labs challenge repositories.

## Licensing

This repository is **not** under a single license.

- **BSL 1.1** (`LICENSE`) — Elliot Technologies' Lighter Prover: circuits, proving/verification
  crypto, and related Elliot Licensed Work. Change Date 2029-01-01; Change License GPLv2+.
- **MIT** (`LICENSE-MIT`) — Eigen Labs challenge infrastructure: benchmark harness, setup/run
  scripts, GitHub Actions workflows, and AI-agent trace hooks. This code launches and scores the
  prover; it does not perform proving itself.

See `NOTICE` for the path-level split and `THIRD_PARTY_NOTICES` for vendored dependencies.
Promoted submissions are licensed under the terms applicable to the files they modify.
