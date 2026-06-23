"""Summarize the worst localization errors from a retrieval result CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def error_column(row: dict[str, str]) -> str:
    for name in (
        "motion_viterbi_position_error_m",
        "temporal_position_error_m",
        "reranked_position_error_m",
        "position_error_m",
    ):
        if name in row:
            return name
    raise KeyError("No known error column found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise SystemExit("No rows found")

    err_col = error_column(rows[0])
    rows.sort(key=lambda row: float(row[err_col]), reverse=True)
    fields = [
        field
        for field in (
            "query_dataset",
            "query_frame_count",
            "query_frame_path",
            "dino_reference_dataset",
            "dino_reference_frame_count",
            "dino_position_error_m",
            "temporal_reference_dataset",
            "temporal_reference_frame_count",
            "temporal_reference_frame_path",
            "temporal_rank",
            "motion_viterbi_reference_dataset",
            "motion_viterbi_reference_frame_count",
            "motion_viterbi_reference_frame_path",
            "motion_viterbi_rank",
            "lg_match_count",
            "lg_inlier_count",
            "lg_inlier_ratio",
            err_col,
        )
        if field in rows[0]
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows[: args.top_n]:
            writer.writerow({field: row[field] for field in fields})

    print(f"error_column: {err_col}")
    print(f"wrote: {args.output_csv}")
    for row in rows[: args.top_n]:
        print(
            f"{row.get('query_frame_count')} -> "
            f"{row.get('motion_viterbi_reference_dataset', row.get('temporal_reference_dataset', row.get('reference_dataset')))}:"
            f"{row.get('motion_viterbi_reference_frame_count', row.get('temporal_reference_frame_count', row.get('reference_frame_count')))} "
            f"error={float(row[err_col]):.2f}m"
        )


if __name__ == "__main__":
    main()
