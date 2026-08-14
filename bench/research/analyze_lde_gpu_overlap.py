#!/usr/bin/env python3
"""Measure phase-aligned GPU idle inside prover FFT/LDE trace spans.

The input is the diagnostic_profile Chrome trace.  Metal's GPUStartTime and
GPUEndTime counters use host time since boot, while Chrome events use time
since the profiler epoch.  We recover the constant offset from the earliest
completion callback (all callback latency is non-negative), then validate each
mapped GPU interval against its metal_submit_to_completed span.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def start_ns(event: dict) -> int:
    return round(event["ts"] * 1_000)


def duration_ns(event: dict) -> int:
    return round(event.get("dur", 0) * 1_000)


def end_ns(event: dict) -> int:
    return start_ns(event) + duration_ns(event)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    left, right = sorted(intervals)[0]
    for next_left, next_right in sorted(intervals)[1:]:
        if next_left > right:
            total += right - left
            left, right = next_left, next_right
        else:
            right = max(right, next_right)
    return total + right - left


def intersection_length(
    left: int, right: int, intervals: list[tuple[int, int]]
) -> int:
    return union_length(
        [(max(left, a), min(right, b)) for a, b in intervals if a < right and b > left]
    )


def proof_key(event: dict) -> tuple[str | None, int | None]:
    args = event.get("args", {})
    return args.get("context"), args.get("instance")


def map_gpu_intervals(events: list[dict]) -> tuple[list[dict], dict]:
    submit_sequences = sorted(
        (
            event
            for event in events
            if event.get("cat") == "metal_submit"
            and event.get("name") == "queue_sequence"
        ),
        key=lambda event: event["args"]["value"],
    )
    completed_spans = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "metal_submit_to_completed"
    ]
    expected = list(range(len(submit_sequences)))
    actual = [event["args"]["value"] for event in submit_sequences]
    if actual != expected or len(completed_spans) != len(submit_sequences):
        raise ValueError("Metal submissions are not a complete contiguous sequence")

    # The sequence atomic is incremented before its counter is recorded, so
    # simultaneous submitters can appear out of sequence in timestamp order.
    # Recover the command-name counter that immediately follows each sequence
    # on the same thread, then pair within each stable identity in local order.
    pending_submit_by_tid: dict[int, int] = {}
    command_by_sequence: dict[int, str] = {}
    for event in events:
        if event.get("cat") != "metal_submit":
            continue
        if event.get("name") == "queue_sequence":
            pending_submit_by_tid[event["tid"]] = event["args"]["value"]
        else:
            sequence = pending_submit_by_tid.pop(event["tid"], None)
            if sequence is None:
                raise ValueError("Metal work counter has no preceding sequence")
            command_by_sequence[sequence] = event["name"]
    if len(command_by_sequence) != len(submit_sequences):
        raise ValueError("Metal command-name counters are incomplete")

    submits_by_identity: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    spans_by_identity: dict[tuple, list[dict]] = defaultdict(list)
    for submit in submit_sequences:
        sequence = submit["args"]["value"]
        identity = (submit["tid"], proof_key(submit), command_by_sequence[sequence])
        submits_by_identity[identity].append((sequence, submit))
    for span in completed_spans:
        identity = (span["tid"], proof_key(span), span["name"])
        spans_by_identity[identity].append(span)

    spans_by_sequence: dict[int, dict] = {}
    if set(submits_by_identity) != set(spans_by_identity):
        raise ValueError("Metal submission/span identity groups differ")
    for identity, submits in submits_by_identity.items():
        spans = spans_by_identity[identity]
        submits.sort(key=lambda item: start_ns(item[1]))
        spans.sort(key=start_ns)
        if len(submits) != len(spans):
            raise ValueError(f"Metal identity count mismatch: {identity}")
        for (sequence, submit), span in zip(submits, spans):
            if start_ns(span) < start_ns(submit):
                raise ValueError(f"completion span predates submission {sequence}")
            spans_by_sequence[sequence] = span

    pending: dict[int, dict[str, int]] = defaultdict(dict)
    counters_by_sequence: dict[int, dict[str, int]] = {}
    for event in events:
        category = event.get("cat")
        name = event.get("name")
        if category == "metal_gpu" and name in {
            "execution_ns",
            "start_host_ns",
            "end_host_ns",
        }:
            pending[event["tid"]][name] = event["args"]["value"]
        elif category == "metal_complete" and name == "queue_sequence":
            sequence = event["args"]["value"]
            values = pending.pop(event["tid"], {})
            if set(values) != {"execution_ns", "start_host_ns", "end_host_ns"}:
                raise ValueError(f"incomplete GPU counters for sequence {sequence}")
            counters_by_sequence[sequence] = values

    if set(counters_by_sequence) != set(spans_by_sequence):
        raise ValueError("GPU counter sequences do not match completion spans")

    offset_candidates = [
        end_ns(spans_by_sequence[sequence]) - values["end_host_ns"]
        for sequence, values in counters_by_sequence.items()
    ]
    host_to_trace_offset_ns = min(offset_candidates)

    intervals = []
    duration_errors = []
    callback_lags = []
    pre_submit = 0
    after_completion = 0
    for sequence in expected:
        span = spans_by_sequence[sequence]
        values = counters_by_sequence[sequence]
        gpu_start = values["start_host_ns"] + host_to_trace_offset_ns
        gpu_end = values["end_host_ns"] + host_to_trace_offset_ns
        duration_errors.append(
            abs((gpu_end - gpu_start) - values["execution_ns"])
        )
        callback_lags.append(end_ns(span) - gpu_end)
        pre_submit += gpu_start < start_ns(span)
        after_completion += gpu_end > end_ns(span)
        intervals.append(
            {
                "sequence": sequence,
                "name": span["name"],
                "start": gpu_start,
                "end": gpu_end,
                "submit": start_ns(span),
                "context": proof_key(span),
                "args": span.get("args", {}),
            }
        )

    validation = {
        "commands": len(intervals),
        "host_to_trace_offset_ns": host_to_trace_offset_ns,
        "max_duration_error_ns": max(duration_errors),
        "pre_submit_intervals": pre_submit,
        "after_completion_intervals": after_completion,
        "callback_lag_min_us": min(callback_lags) / 1_000,
        "callback_lag_median_us": statistics.median(callback_lags) / 1_000,
        "callback_lag_p95_us": percentile(callback_lags, 0.95) / 1_000,
        "callback_lag_max_us": max(callback_lags) / 1_000,
    }
    return intervals, validation


PHASE_CONTAINERS = {
    "compute wires commitment": "wire_lde",
    "commit to partial products, Z's and, if any, lookup polynomials": "partial_products_lde",
    "commit to quotient polys": "quotient_lde",
}


def classify_fft(event: dict, complete_events: list[dict]) -> str:
    left, right = start_ns(event), end_ns(event)
    key = proof_key(event)
    matches = []
    for candidate in complete_events:
        phase = PHASE_CONTAINERS.get(candidate.get("name"))
        if (
            phase
            and proof_key(candidate) == key
            and start_ns(candidate) <= left
            and end_ns(candidate) >= right
        ):
            matches.append((duration_ns(candidate), phase))
    return min(matches)[1] if matches else "unattributed_fft"


def pipeline_group(args: dict) -> str:
    parent = args.get("parent_context", "process")
    if parent in {"heavy_tx_proof", "light_tx_proof"}:
        return "transaction_pipeline"
    if parent in {"heavy_chain", "light_chain"}:
        step = args.get("parent_chain_step", -1)
        return "chain_drain_19_plus" if step >= 19 else "chain_early_0_18"
    if parent == "final_block":
        return "final_block"
    if parent == "pre_execution":
        return "pre_execution"
    return parent


def annotate_interval(event: dict, phase: str, gpu: list[dict]) -> dict:
    left, right = start_ns(event), end_ns(event)
    overlaps = [command for command in gpu if command["start"] < right and command["end"] > left]
    busy = intersection_length(
        left, right, [(command["start"], command["end"]) for command in overlaps]
    )
    args = event.get("args", {})
    return {
        "event": event,
        "phase": phase,
        "start": left,
        "end": right,
        "duration": right - left,
        "busy": busy,
        "idle": right - left - busy,
        "idle_fraction": (right - left - busy) / (right - left),
        "degree": args.get("degree_bits"),
        "parent": args.get("parent_context", "process"),
        "pipeline": pipeline_group(args),
        "commands": Counter(command["name"] for command in overlaps),
    }


def summarize(rows: list[dict]) -> dict:
    duration = sum(row["duration"] for row in rows)
    busy = sum(row["busy"] for row in rows)
    interval_union = union_length([(row["start"], row["end"]) for row in rows])
    gpu_busy_union = union_length(
        [
            (max(row["start"], command["start"]), min(row["end"], command["end"]))
            for row in rows
            for command in GLOBAL_GPU
            if command["start"] < row["end"] and command["end"] > row["start"]
        ]
    )
    names = Counter()
    for row in rows:
        names.update(row["commands"])
    return {
        "n": len(rows),
        "summed_ms": duration / 1_000_000,
        "weighted_idle_pct": 100 * (duration - busy) / duration if duration else math.nan,
        "median_idle_pct": 100 * statistics.median(row["idle_fraction"] for row in rows),
        "p25_idle_pct": 100 * percentile([row["idle_fraction"] for row in rows], 0.25),
        "p75_idle_pct": 100 * percentile([row["idle_fraction"] for row in rows], 0.75),
        "cpu_union_ms": interval_union / 1_000_000,
        "gpu_idle_in_cpu_union_ms": (interval_union - gpu_busy_union) / 1_000_000,
        "commands": dict(names.most_common()),
    }


def idle_gap_stats(rows: list[dict], gpu: list[dict]) -> dict:
    cpu_union = []
    for left, right in sorted((row["start"], row["end"]) for row in rows):
        if not cpu_union or left > cpu_union[-1][1]:
            cpu_union.append([left, right])
        else:
            cpu_union[-1][1] = max(cpu_union[-1][1], right)
    gpu_union = []
    for left, right in sorted((command["start"], command["end"]) for command in gpu):
        if not gpu_union or left > gpu_union[-1][1]:
            gpu_union.append([left, right])
        else:
            gpu_union[-1][1] = max(gpu_union[-1][1], right)

    gaps = []
    for left, right in cpu_union:
        cursor = left
        for gpu_left, gpu_right in gpu_union:
            if gpu_right <= left:
                continue
            if gpu_left >= right:
                break
            clipped_left, clipped_right = max(left, gpu_left), min(right, gpu_right)
            if clipped_left > cursor:
                gaps.append(clipped_left - cursor)
            cursor = max(cursor, clipped_right)
        if cursor < right:
            gaps.append(right - cursor)

    gaps_ms = [gap / 1_000_000 for gap in gaps]
    return {
        "n": len(gaps),
        "total_ms": sum(gaps_ms),
        "median_ms": statistics.median(gaps_ms),
        "p75_ms": percentile(gaps_ms, 0.75),
        "p90_ms": percentile(gaps_ms, 0.90),
        "max_ms": max(gaps_ms),
        "at_least_30ms": sum(gap >= 30 for gap in gaps_ms),
        "time_in_at_least_30ms_gaps_ms": sum(gap for gap in gaps_ms if gap >= 30),
    }


def dependent_timings(rows: list[dict], events: list[dict], gpu: list[dict]) -> dict:
    complete = [event for event in events if event.get("ph") == "X"]
    gaps = defaultdict(list)
    final_merkle_names = {"merkle_tree", "merkle_parents"}
    for row in rows:
        key = proof_key(row["event"])
        fft_end = row["end"]
        wire_stage = next(
            (
                event
                for event in complete
                if event.get("name") == "compute wires commitment"
                and proof_key(event) == key
                and start_ns(event) <= row["start"]
                and end_ns(event) >= row["end"]
            ),
            None,
        )
        build = next(
            (
                event
                for event in complete
                if event.get("name") == "build Merkle tree"
                and proof_key(event) == key
                and start_ns(event) >= row["start"]
                and (wire_stage is None or start_ns(event) <= end_ns(wire_stage))
            ),
            None,
        )
        if build is not None:
            gaps["fft_end_to_build_start_ms"].append((start_ns(build) - fft_end) / 1_000_000)

        same_proof_merkle = [
            command
            for command in gpu
            if command["context"] == key
            and command["name"] in final_merkle_names
            and command["submit"] >= row["start"]
            and (wire_stage is None or command["submit"] <= end_ns(wire_stage) + 5_000_000)
        ]
        if same_proof_merkle:
            command = min(same_proof_merkle, key=lambda item: item["submit"])
            gaps["fft_end_to_final_merkle_submit_ms"].append(
                (command["submit"] - fft_end) / 1_000_000
            )
            gaps["fft_end_to_final_merkle_gpu_start_ms"].append(
                (command["start"] - fft_end) / 1_000_000
            )

    result = {}
    for name, values in gaps.items():
        result[name] = {
            "n": len(values),
            "median": statistics.median(values),
            "p25": percentile(values, 0.25),
            "p75": percentile(values, 0.75),
            "min": min(values),
            "max": max(values),
            "nonpositive": sum(value <= 0 for value in values),
            "le_1ms": sum(value <= 1 for value in values),
            "le_5ms": sum(value <= 5 for value in values),
            "le_20ms": sum(value <= 20 for value in values),
            "le_100ms": sum(value <= 100 for value in values),
        }
    return result


def print_table(title: str, groups: dict[tuple, list[dict]]) -> None:
    print(f"\n## {title}\n")
    print("| Group | n | summed ms | idle % weighted | idle % median [p25,p75] | CPU union ms | GPU idle in union ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = summarize(rows)
        label = "/".join(str(part) for part in key)
        print(
            f"| {label} | {summary['n']} | {summary['summed_ms']:.3f} | "
            f"{summary['weighted_idle_pct']:.2f} | {summary['median_idle_pct']:.2f} "
            f"[{summary['p25_idle_pct']:.2f},{summary['p75_idle_pct']:.2f}] | "
            f"{summary['cpu_union_ms']:.3f} | {summary['gpu_idle_in_cpu_union_ms']:.3f} |"
        )


GLOBAL_GPU: list[dict] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text())
    if not isinstance(events, list):
        events = events["traceEvents"]

    global GLOBAL_GPU
    GLOBAL_GPU, validation = map_gpu_intervals(events)
    complete = [event for event in events if event.get("ph") == "X"]
    fft_events = [event for event in complete if event.get("name") == "FFT + blinding"]
    fft_rows = [annotate_interval(event, classify_fft(event, complete), GLOBAL_GPU) for event in fft_events]

    final_fft_rows = [
        annotate_interval(event, "fri_final_fft", GLOBAL_GPU)
        for event in complete
        if event.get("name") == "perform final FFT"
    ]
    wire_stage_rows = [
        annotate_interval(event, "wire_commitment_stage", GLOBAL_GPU)
        for event in complete
        if event.get("name") == "compute wires commitment"
    ]

    print("# Phase-aligned CPU LDE / GPU overlap")
    print("\n## Clock-domain validation\n")
    for name, value in validation.items():
        print(f"- {name}: {value}")

    by_degree_phase = defaultdict(list)
    for row in fft_rows + final_fft_rows:
        by_degree_phase[(row["degree"], row["phase"])].append(row)
    print_table("FFT spans by degree and phase", by_degree_phase)

    by_pipeline_phase = defaultdict(list)
    for row in fft_rows + final_fft_rows:
        by_pipeline_phase[(row["pipeline"], row["degree"], row["phase"])].append(row)
    print_table("FFT spans by pipeline region", by_pipeline_phase)

    by_wire_stage = defaultdict(list)
    for row in wire_stage_rows:
        by_wire_stage[(row["pipeline"], row["degree"])].append(row)
    print_table("Whole wire-commitment stages", by_wire_stage)

    degree16_wire = [
        row for row in fft_rows if row["degree"] == 16 and row["phase"] == "wire_lde"
    ]
    chronological = sorted(degree16_wire, key=lambda row: row["start"])
    quartiles = []
    for index in range(4):
        rows = chronological[
            index * len(chronological) // 4 : (index + 1) * len(chronological) // 4
        ]
        summary = summarize(rows)
        quartiles.append(
            {
                "quartile": index + 1,
                "n": len(rows),
                "start_s": rows[0]["start"] / 1_000_000_000,
                "end_s": rows[-1]["end"] / 1_000_000_000,
                "weighted_idle_pct": summary["weighted_idle_pct"],
            }
        )
    print("\n## Degree-16 wire LDE chronological quartiles\n")
    print(json.dumps(quartiles, indent=2, sort_keys=True))
    print("\n## Degree-16 wire LDE idle-gap distribution\n")
    print(json.dumps(idle_gap_stats(degree16_wire, GLOBAL_GPU), indent=2, sort_keys=True))
    print("\n## Degree-16 wire LDE dependent timing\n")
    print(json.dumps(dependent_timings(degree16_wire, events, GLOBAL_GPU), indent=2, sort_keys=True))
    print("\n## Degree-16 wire LDE overlapping command counts\n")
    print(json.dumps(summarize(degree16_wire)["commands"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
