"""Evaluate confidence-gated navigation fixes from retrieval results.

The input is a result CSV that already contains one visual estimate per query
frame. This script does not change the visual localization algorithm. It adds an
abstention layer: publish a FIX only when visual evidence is strong enough,
otherwise output NO_FIX.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    max_rank: int
    min_inliers: int
    min_inlier_ratio: float
    min_dino_similarity: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def query_times(query_manifest: Path | None) -> dict[str, float]:
    if query_manifest is None:
        return {}
    rows = read_csv(query_manifest)
    return {row["frame_count"]: float(row["start_seconds"]) for row in rows}


def result_time(row: dict[str, str], times: dict[str, float], fallback_index: int) -> float:
    return times.get(str(row["query_frame_count"]), float(fallback_index))


def accepted(row: dict[str, str], thresholds: Thresholds) -> bool:
    return (
        int(row["motion_viterbi_rank"]) <= thresholds.max_rank
        and int(row["lg_inlier_count"]) >= thresholds.min_inliers
        and float(row["lg_inlier_ratio"]) >= thresholds.min_inlier_ratio
        and float(row["motion_viterbi_dino_similarity"]) >= thresholds.min_dino_similarity
    )


def summarize_policy(
    rows: list[dict[str, str]],
    times: dict[str, float],
    thresholds: Thresholds,
    good_error_m: float,
) -> dict[str, object]:
    accepted_rows: list[tuple[int, dict[str, str], float]] = []
    all_errors = [float(row["motion_viterbi_position_error_m"]) for row in rows]
    for index, row in enumerate(rows):
        if accepted(row, thresholds):
            accepted_rows.append((index, row, result_time(row, times, index)))

    accepted_errors = [
        float(row["motion_viterbi_position_error_m"])
        for _index, row, _time in accepted_rows
    ]
    accepted_times = [time for _index, _row, time in accepted_rows]
    gaps = [
        current - previous
        for previous, current in zip(accepted_times, accepted_times[1:])
    ]
    duration = result_time(rows[-1], times, len(rows) - 1) - result_time(rows[0], times, 0)
    if accepted_times:
        start_gap = accepted_times[0] - result_time(rows[0], times, 0)
        end_gap = result_time(rows[-1], times, len(rows) - 1) - accepted_times[-1]
        longest_gap = max([start_gap, end_gap, *gaps])
    else:
        longest_gap = duration

    good_fixes = sum(error <= good_error_m for error in accepted_errors)
    false_fixes = len(accepted_errors) - good_fixes
    coverage = len(accepted_rows) / max(len(rows), 1)

    return {
        "max_rank": thresholds.max_rank,
        "min_inliers": thresholds.min_inliers,
        "min_inlier_ratio": thresholds.min_inlier_ratio,
        "min_dino_similarity": thresholds.min_dino_similarity,
        "good_error_m": good_error_m,
        "total_frames": len(rows),
        "accepted_fixes": len(accepted_rows),
        "no_fix_frames": len(rows) - len(accepted_rows),
        "coverage": coverage,
        "coverage_percent": 100.0 * coverage,
        "accepted_mean_error_m": float(mean(accepted_errors)) if accepted_errors else None,
        "accepted_median_error_m": float(median(accepted_errors)) if accepted_errors else None,
        "accepted_p90_error_m": float(np.percentile(accepted_errors, 90)) if accepted_errors else None,
        "accepted_max_error_m": max(accepted_errors) if accepted_errors else None,
        "good_fixes": good_fixes,
        "false_fixes": false_fixes,
        "good_fix_rate_among_accepted": good_fixes / max(len(accepted_errors), 1),
        "good_fix_rate_among_all_frames": good_fixes / max(len(rows), 1),
        "mean_time_between_fixes_s": float(mean(gaps)) if gaps else None,
        "median_time_between_fixes_s": float(median(gaps)) if gaps else None,
        "longest_gap_without_fix_s": float(longest_gap),
        "always_output_mean_error_m": float(mean(all_errors)),
        "always_output_median_error_m": float(median(all_errors)),
        "always_output_good_fix_rate": sum(error <= good_error_m for error in all_errors) / max(len(all_errors), 1),
    }


def policy_score(summary: dict[str, object], min_coverage: float, max_longest_gap_s: float) -> tuple[float, float, float, float]:
    coverage = float(summary["coverage"])
    longest_gap = float(summary["longest_gap_without_fix_s"])
    if coverage < min_coverage or longest_gap > max_longest_gap_s:
        return (-1.0, -longest_gap, coverage, 0.0)
    return (
        float(summary["good_fix_rate_among_accepted"]),
        float(summary["good_fix_rate_among_all_frames"]),
        -longest_gap,
        coverage,
    )


def build_sweep(
    rows: list[dict[str, str]],
    times: dict[str, float],
    good_error_m: float,
) -> list[dict[str, object]]:
    sweep: list[dict[str, object]] = []
    for max_rank in [1, 2, 3, 4, 6]:
        for min_inliers in [50, 100, 150, 200, 300, 500]:
            for min_ratio in [0.60, 0.70, 0.80, 0.90, 0.95]:
                for min_similarity in [0.90, 0.94, 0.96, 0.98, 0.99]:
                    sweep.append(
                        summarize_policy(
                            rows,
                            times,
                            Thresholds(max_rank, min_inliers, min_ratio, min_similarity),
                            good_error_m,
                        )
                    )
    return sweep


def best_policy(
    sweep: list[dict[str, object]],
    min_coverage: float,
    max_longest_gap_s: float,
) -> dict[str, object]:
    return max(
        sweep,
        key=lambda summary: policy_score(summary, min_coverage, max_longest_gap_s),
    )


def write_frame_decisions(
    rows: list[dict[str, str]],
    times: dict[str, float],
    summary: dict[str, object],
    output_csv: Path,
) -> None:
    thresholds = Thresholds(
        max_rank=int(summary["max_rank"]),
        min_inliers=int(summary["min_inliers"]),
        min_inlier_ratio=float(summary["min_inlier_ratio"]),
        min_dino_similarity=float(summary["min_dino_similarity"]),
    )
    output_rows: list[dict[str, object]] = []
    last_fix_time: float | None = None
    for index, row in enumerate(rows):
        time_s = result_time(row, times, index)
        is_fix = accepted(row, thresholds)
        if is_fix:
            seconds_since_previous_fix = None if last_fix_time is None else time_s - last_fix_time
            last_fix_time = time_s
        else:
            seconds_since_previous_fix = None
        output_rows.append(
            {
                "query_frame_count": row["query_frame_count"],
                "time_s": time_s,
                "decision": "FIX" if is_fix else "NO_FIX",
                "seconds_since_previous_fix": seconds_since_previous_fix,
                "position_error_m": float(row["motion_viterbi_position_error_m"]),
                "is_good_fix": is_fix and float(row["motion_viterbi_position_error_m"]) <= float(summary["good_error_m"]),
                "motion_viterbi_rank": int(row["motion_viterbi_rank"]),
                "motion_viterbi_dino_similarity": float(row["motion_viterbi_dino_similarity"]),
                "lg_inlier_count": int(row["lg_inlier_count"]),
                "lg_inlier_ratio": float(row["lg_inlier_ratio"]),
                "motion_viterbi_reference_dataset": row["motion_viterbi_reference_dataset"],
                "motion_viterbi_reference_frame_count": row["motion_viterbi_reference_frame_count"],
            }
        )
    write_csv(output_rows, output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--sweep-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--good-error-m", type=float, default=20.0)
    parser.add_argument("--min-coverage", type=float, default=0.30)
    parser.add_argument("--max-longest-gap-s", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.results_csv)
    times = query_times(args.query_manifest)
    sweep = build_sweep(rows, times, args.good_error_m)
    write_csv(sweep, args.sweep_csv)
    best = best_policy(sweep, args.min_coverage, args.max_longest_gap_s)
    best["selection_min_coverage"] = args.min_coverage
    best["selection_max_longest_gap_s"] = args.max_longest_gap_s
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_frame_decisions(rows, times, best, args.decisions_csv)

    for key, value in best.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.sweep_csv}")
    print(f"wrote: {args.decisions_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
