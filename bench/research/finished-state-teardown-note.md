# Skip finished prover-state teardown before `_exit`

## Context and attribution

This is a small structural optimization for the Lighter Prover Challenge. The
implementation and analysis used **GPT 5.6 Sol**, effort **max**, through
Codex. The local screening host is an Apple M1 Pro with 32 GB unified memory;
the ranked host is an Apple M4 Pro Mac mini with 48 GB and measures five serial
500-transaction workers. Local public-fixture timings are screening evidence,
while the ranked private-active result is the performance authority.

The base is promoted frontier 137, source `59c0155`, official score
`29.8785698468374 tx/s`. The rejected hybrid column-pool, FRI query-grain, and
four-state PoW experiments are absent. The candidate changes one location in
`bench/src/prover.rs` and no circuit, field arithmetic, constraint, transcript,
FRI parameter, Metal command, proof byte, verifier, dependency, or benchmark
boundary. The checked-in candidate checkpoint is `d1d839a`.

## Observation

The promoted worker already deliberately avoids most shutdown work. After it
serializes the final proof into a `BufWriter`, it calls `into_inner` to flush
and surface errors, explicitly drops the returned file descriptor, and enters
`_exit(2)`. Direct `_exit` bypasses Rust destructors and libc/image teardown;
the kernel reclaims the address space. This is safe because the trusted parent
verifier only consumes the closed proof file and no further userspace state is
observable.

One large teardown remained before serialization. `prove_block_after_pre`
takes `Block`, `Circuits`, and the pre-execution proof by value. It also owns
the completed light/heavy chain proofs and final block target. When the
function returns the final proof, Rust drops those finished input graphs before
the caller can serialize the return value. The later `_exit` cannot recover
that time because the drops already happened.

A diagnostic public-worker trace exposes this boundary without adding a new
timer. Relevant complete spans were:

| Span | Start (us) | Duration (us) | End (us) |
|---|---:|---:|---:|
| `final_block_proof` | 30,407,866.000 | 3,807,976.208 | 34,215,842.208 |
| `final_block_tail` | 30,395,200.625 | 3,820,643.166 | 34,215,843.791 |
| `prove_block_after_pre` | 824,948.333 | 33,391,013.542 | 34,215,961.875 |
| outer `block_pipeline` | 824,944.916 | 33,434,162.625 | 34,259,107.541 |
| `serialize_and_flush_proof` | 34,259,121.250 | 354.416 | 34,259,475.666 |

The inner function's work span ended `43,145.666 us` before the outer caller
received the return and closed the pipeline span. Serialization began only
`13.709 us` after that. Drop order explains the gap: the function-local profile
guard closes before by-value arguments and remaining owned state finish their
destructors, while the caller's pipeline guard includes that teardown.

This is a fixed per-worker cost. If the M4 cost is similar to the M1 trace,
five serial workers pay about `0.216 s`. Against the current promoted total of
roughly `83.67 s`, the structural ceiling is around `0.26%`, or approximately
`0.077 tx/s`. This estimate is intentionally a ceiling/transfer hypothesis,
not a ranked-score prediction.

## Change

After the final block proof completes and the exclusive-GPU phase is disabled,
the candidate calls `std::mem::forget` on one tuple containing:

- the consumed block;
- all loaded non-final circuit data;
- the pre-execution proof and its decoded output;
- the light and heavy chain proofs; and
- the final block target.

The returned final proof is not forgotten. The caller serializes it exactly as
before, flushes and closes the output file exactly as before, then calls the
existing `_exit(2)` path. Forgetting the tuple performs no allocation and does
not touch its contents; it merely suppresses recursive destructor traversal.

`LIGHTER_DROP_FINISHED_PROVER_STATE=1` retains the historical destructor path
inside the exact same release binary. The ranked/default candidate forgets the
dead state. This switch exists for local A/B only and does not inspect the
fixture or change proof work.

## Safety argument

All spawned proof threads use scoped threads and have joined before the final
block proof begins. The final block circuit data was deliberately leaked
earlier because the pending witness needs a stable reference; process exit was
already its reclamation mechanism. Every Metal command buffer is committed and
waited before its CPU-visible results are consumed, and the exclusive phase is
disabled before the new forget. There is no thread or callback that can access
the forgotten owners afterward.

The final proof owns its serialized data. It does not borrow from `Circuits`,
the input proofs, block, decoded pre-output, or target. Rust's type system would
reject returning such a borrow because the public `ProofWithPublicInputs` is an
owned structure. Suppressing drops therefore cannot invalidate the returned
proof.

The change does not conceal a required flush: circuit/proof destructors free
memory and release retained Rust/Metal objects, but the worker output is a
separate file whose `BufWriter` is created only after `prove_block_after_pre`
returns. That writer is still explicitly flushed and its descriptor closed
before `_exit`.

On error and panic paths, behavior is unchanged because the forget is reached
only after successful final proof construction. During ordinary execution the
kernel reclaims exactly the same virtual memory and handles milliseconds later
at process death. Peak memory does not increase: all forgotten objects were
already resident until this return boundary, and no subsequent proving
allocation occurs.

## Validation

`cargo check --locked --offline -p bench --bin prove` passed. The release build
used native Apple-silicon code generation and produced worker SHA-256
`2cf8789508fcaee07de9a14dfd8fe8aa61af1d81d26bd06b73ddfcd3b38695f1`.

The predeclared same-binary order was `B-C-C-B / C-B-B-C`, with B setting
`LIGHTER_DROP_FINISHED_PROVER_STATE=1` and C using the candidate default:

| Sample | Policy | Real | User | Sys |
|---:|---|---:|---:|---:|
| 1 | B, historical drop | 27.71 | 178.99 | 11.68 |
| 2 | C, forget | 26.29 | 179.72 | 10.89 |
| 3 | C, forget | 27.04 | 182.88 | 12.31 |
| 4 | B, historical drop | 27.07 | 176.36 | 11.08 |
| 5 | C, forget | 27.09 | 181.82 | 12.24 |
| 6 | B, historical drop | 26.71 | 181.26 | 10.84 |
| 7 | B, historical drop | 26.53 | 179.10 | 10.49 |
| 8 | C, forget | 26.71 | 179.39 | 11.00 |

Control mean was `27.005 s`; candidate mean was `26.7825 s`, a `0.8240%`
nominal runtime reduction. Control median was `26.890 s`; candidate median was
`26.875 s`, a `0.0558%` nominal reduction. Pairwise signs split 2-2. The first
block's `2.65%` apparent improvement exceeds the mechanism ceiling and is
treated as drift; the small positive median is directionally consistent with
the fixed teardown deletion but not independently resolvable from local noise.

The exact candidate then passed the pinned trusted public benchmark verifier:
one verified proof out of one expected proof. Its provisional public smoke was
`28.500265542 s` / `17.543696189888642 tx/s`; that synthetic score is not
compared with the private-active leaderboard.

## Official result and decision

The full submission was `7244133b-9f3a-4072-b84d-89993282260e`, archived by
Yukon as commit `23f3ca0`. It scored `25.8191460929401 tx/s` against promoted
frontier `29.8785698468374 tx/s`, a `-4.0594237538973 tx/s` delta, and was
officially rejected.

The run landed in the broad `25.x` low-service regime seen in nearby
submissions, so it does not statistically resolve the trace-derived maximum
effect of about `0.26%`. It also does not invalidate the proof-safety argument:
the worker still flushes the final proof before its existing `_exit(2)`, and
trusted verification passed. Nevertheless, the experiment's predeclared rule
required the official score to exceed the frontier. The candidate is therefore
rejected and the source deletion is reverted. Retain this note as evidence that
post-proof destruction is measurable but too small to justify stacking after a
terminal official rejection.
