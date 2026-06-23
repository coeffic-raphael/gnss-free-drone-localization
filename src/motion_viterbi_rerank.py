"""Second-order Viterbi reranking with a constant-velocity motion prior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np


def parse_manifest_arg(value: str) -> tuple[Path, str]:
    if "=" not in value:
        path = Path(value)
        return path, path.stem
    dataset_id, path_text = value.split("=", 1)
    return Path(path_text), dataset_id


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        row["dataset_id"] = dataset_id
    return rows


def local_xy_from_latlon(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    earth_radius_m = 6_378_137.0
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    return (
        (lon - lon0) * math.cos(lat0) * earth_radius_m,
        (lat - lat0) * earth_radius_m,
    )


def ground_xy(row: dict[str, str], origin: tuple[float, float]) -> tuple[float, float]:
    return local_xy_from_latlon(
        float(row["ground_latitude"]),
        float(row["ground_longitude"]),
        origin[0],
        origin[1],
    )


def position_error_m(query: dict[str, str], reference: dict[str, str], origin: tuple[float, float]) -> float:
    qx, qy = ground_xy(query, origin)
    rx, ry = ground_xy(reference, origin)
    return math.hypot(qx - rx, qy - ry)


def unary_cost(candidate: dict[str, float | int], dino_weight: float, inlier_weight: float, ratio_weight: float) -> float:
    return (
        -dino_weight * float(candidate["dino_similarity"])
        -inlier_weight * math.log1p(float(candidate["lg_inlier_count"]))
        -ratio_weight * float(candidate["lg_inlier_ratio"])
    )


def step_cost(
    previous_reference: dict[str, str],
    current_reference: dict[str, str],
    origin: tuple[float, float],
    max_step_m: float,
    transition_weight: float,
) -> float:
    px, py = ground_xy(previous_reference, origin)
    cx, cy = ground_xy(current_reference, origin)
    distance = math.hypot(cx - px, cy - py)
    excess = max(0.0, distance - max_step_m)
    return transition_weight * (excess / max(max_step_m, 1e-6)) ** 2


def acceleration_cost(
    older_reference: dict[str, str],
    previous_reference: dict[str, str],
    current_reference: dict[str, str],
    origin: tuple[float, float],
    acceleration_scale_m: float,
    acceleration_weight: float,
) -> float:
    ox, oy = ground_xy(older_reference, origin)
    px, py = ground_xy(previous_reference, origin)
    cx, cy = ground_xy(current_reference, origin)
    prev_dx = px - ox
    prev_dy = py - oy
    curr_dx = cx - px
    curr_dy = cy - py
    acceleration = math.hypot(curr_dx - prev_dx, curr_dy - prev_dy)
    return acceleration_weight * (acceleration / max(acceleration_scale_m, 1e-6)) ** 2


def direction_change_cost(
    older_reference: dict[str, str],
    previous_reference: dict[str, str],
    current_reference: dict[str, str],
    origin: tuple[float, float],
    direction_scale_degrees: float,
    direction_weight: float,
    min_direction_step_m: float,
) -> float:
    if direction_weight <= 0.0:
        return 0.0
    ox, oy = ground_xy(older_reference, origin)
    px, py = ground_xy(previous_reference, origin)
    cx, cy = ground_xy(current_reference, origin)
    prev_dx = px - ox
    prev_dy = py - oy
    curr_dx = cx - px
    curr_dy = cy - py
    prev_norm = math.hypot(prev_dx, prev_dy)
    curr_norm = math.hypot(curr_dx, curr_dy)
    if prev_norm < min_direction_step_m or curr_norm < min_direction_step_m:
        return 0.0
    cosine = (prev_dx * curr_dx + prev_dy * curr_dy) / (prev_norm * curr_norm)
    cosine = min(1.0, max(-1.0, cosine))
    angle_degrees = math.degrees(math.acos(cosine))
    return direction_weight * (angle_degrees / max(direction_scale_degrees, 1e-6)) ** 2


def first_order_viterbi(
    candidates: list[list[dict[str, float | int]]],
    reference_rows: list[dict[str, str]],
    origin: tuple[float, float],
    dino_weight: float,
    inlier_weight: float,
    ratio_weight: float,
    max_step_m: float,
    transition_weight: float,
) -> list[int]:
    costs: list[list[float]] = []
    parents: list[list[int]] = []
    for query_idx, query_candidates in enumerate(candidates):
        current_costs: list[float] = []
        current_parents: list[int] = []
        for cand_idx, candidate in enumerate(query_candidates):
            base = unary_cost(candidate, dino_weight, inlier_weight, ratio_weight)
            if query_idx == 0:
                current_costs.append(base)
                current_parents.append(-1)
                continue
            current_reference = reference_rows[int(candidate["reference_index"])]
            best_cost = math.inf
            best_parent = -1
            for prev_idx, previous in enumerate(candidates[query_idx - 1]):
                previous_reference = reference_rows[int(previous["reference_index"])]
                cost = (
                    costs[query_idx - 1][prev_idx]
                    + base
                    + step_cost(
                        previous_reference,
                        current_reference,
                        origin,
                        max_step_m,
                        transition_weight,
                    )
                )
                if cost < best_cost:
                    best_cost = cost
                    best_parent = prev_idx
            current_costs.append(best_cost)
            current_parents.append(best_parent)
        costs.append(current_costs)
        parents.append(current_parents)

    last_idx = int(np.argmin(np.array(costs[-1])))
    selected: list[int] = []
    for query_idx in range(len(candidates) - 1, -1, -1):
        selected.append(last_idx)
        last_idx = parents[query_idx][last_idx]
    selected.reverse()
    return selected


def second_order_viterbi(
    candidates: list[list[dict[str, float | int]]],
    reference_rows: list[dict[str, str]],
    origin: tuple[float, float],
    dino_weight: float,
    inlier_weight: float,
    ratio_weight: float,
    max_step_m: float,
    transition_weight: float,
    acceleration_scale_m: float,
    acceleration_weight: float,
    direction_scale_degrees: float,
    direction_weight: float,
    min_direction_step_m: float,
) -> list[int]:
    if len(candidates) < 3:
        return first_order_viterbi(
            candidates,
            reference_rows,
            origin,
            dino_weight,
            inlier_weight,
            ratio_weight,
            max_step_m,
            transition_weight,
        )

    pair_costs: dict[tuple[int, int], float] = {}
    pair_parents: list[dict[tuple[int, int], int]] = [{} for _ in candidates]

    for i, cand0 in enumerate(candidates[0]):
        ref0 = reference_rows[int(cand0["reference_index"])]
        cost0 = unary_cost(cand0, dino_weight, inlier_weight, ratio_weight)
        for j, cand1 in enumerate(candidates[1]):
            ref1 = reference_rows[int(cand1["reference_index"])]
            cost1 = unary_cost(cand1, dino_weight, inlier_weight, ratio_weight)
            pair_costs[(i, j)] = cost0 + cost1 + step_cost(
                ref0,
                ref1,
                origin,
                max_step_m,
                transition_weight,
            )

    for query_idx in range(2, len(candidates)):
        next_pair_costs: dict[tuple[int, int], float] = {}
        for prev_pair, prev_cost in pair_costs.items():
            older_idx, previous_idx = prev_pair
            older_reference = reference_rows[int(candidates[query_idx - 2][older_idx]["reference_index"])]
            previous_reference = reference_rows[int(candidates[query_idx - 1][previous_idx]["reference_index"])]
            for current_idx, current_candidate in enumerate(candidates[query_idx]):
                current_reference = reference_rows[int(current_candidate["reference_index"])]
                cost = (
                    prev_cost
                    + unary_cost(current_candidate, dino_weight, inlier_weight, ratio_weight)
                    + step_cost(
                        previous_reference,
                        current_reference,
                        origin,
                        max_step_m,
                        transition_weight,
                    )
                    + acceleration_cost(
                        older_reference,
                        previous_reference,
                        current_reference,
                        origin,
                        acceleration_scale_m,
                        acceleration_weight,
                    )
                    + direction_change_cost(
                        older_reference,
                        previous_reference,
                        current_reference,
                        origin,
                        direction_scale_degrees,
                        direction_weight,
                        min_direction_step_m,
                    )
                )
                next_pair = (previous_idx, current_idx)
                if cost < next_pair_costs.get(next_pair, math.inf):
                    next_pair_costs[next_pair] = cost
                    pair_parents[query_idx][next_pair] = older_idx
        pair_costs = next_pair_costs

    last_pair = min(pair_costs, key=pair_costs.get)
    selected = [0 for _ in candidates]
    selected[-2], selected[-1] = last_pair
    for query_idx in range(len(candidates) - 1, 1, -1):
        parent = pair_parents[query_idx][(selected[query_idx - 1], selected[query_idx])]
        selected[query_idx - 2] = parent
    return selected


def summarize(results: list[dict[str, object]]) -> dict[str, float | int]:
    dino_errors = np.array([float(row["dino_position_error_m"]) for row in results])
    temporal_errors = np.array([float(row["motion_viterbi_position_error_m"]) for row in results])
    improved = int((temporal_errors < dino_errors).sum())
    worsened = int((temporal_errors > dino_errors).sum())
    unchanged = len(results) - improved - worsened
    return {
        "queries": len(results),
        "dino_mean_error_m": float(dino_errors.mean()) if len(dino_errors) else 0.0,
        "dino_median_error_m": float(np.median(dino_errors)) if len(dino_errors) else 0.0,
        "dino_p90_error_m": float(np.percentile(dino_errors, 90)) if len(dino_errors) else 0.0,
        "motion_viterbi_mean_error_m": float(temporal_errors.mean()) if len(temporal_errors) else 0.0,
        "motion_viterbi_median_error_m": float(np.median(temporal_errors)) if len(temporal_errors) else 0.0,
        "motion_viterbi_p90_error_m": float(np.percentile(temporal_errors, 90)) if len(temporal_errors) else 0.0,
        "motion_viterbi_max_error_m": float(temporal_errors.max()) if len(temporal_errors) else 0.0,
        "improved_queries": improved,
        "worsened_queries": worsened,
        "unchanged_queries": unchanged,
    }


def write_results(results: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", action="append", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--candidates-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--dino-weight", type=float, default=4.0)
    parser.add_argument("--inlier-weight", type=float, default=1.0)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=35.0)
    parser.add_argument("--transition-weight", type=float, default=4.0)
    parser.add_argument("--acceleration-scale-m", type=float, default=20.0)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--direction-scale-degrees", type=float, default=45.0)
    parser.add_argument("--direction-weight", type=float, default=0.0)
    parser.add_argument("--min-direction-step-m", type=float, default=3.0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    args = parser.parse_args()

    reference_rows: list[dict[str, str]] = []
    for value in args.reference_manifest:
        path, dataset_id = parse_manifest_arg(value)
        reference_rows.extend(load_manifest(path, dataset_id))
    query_path, query_dataset_id = parse_manifest_arg(args.query_manifest)
    query_rows = load_manifest(query_path, query_dataset_id)

    with args.candidates_cache.open("rb") as cache_file:
        candidates = pickle.load(cache_file)
    if args.candidate_limit > 0:
        candidates = [query_candidates[: args.candidate_limit] for query_candidates in candidates]

    origin = (
        float(reference_rows[0]["ground_latitude"]),
        float(reference_rows[0]["ground_longitude"]),
    )
    selected_indices = second_order_viterbi(
        candidates,
        reference_rows,
        origin,
        args.dino_weight,
        args.inlier_weight,
        args.ratio_weight,
        args.max_step_m,
        args.transition_weight,
        args.acceleration_scale_m,
        args.acceleration_weight,
        args.direction_scale_degrees,
        args.direction_weight,
        args.min_direction_step_m,
    )

    results: list[dict[str, object]] = []
    for query_index, (query, selected_idx) in enumerate(zip(query_rows, selected_indices)):
        dino_reference = reference_rows[int(candidates[query_index][0]["reference_index"])]
        candidate = candidates[query_index][selected_idx]
        selected_reference = reference_rows[int(candidate["reference_index"])]
        results.append(
            {
                "query_dataset": query["dataset_id"],
                "query_frame_count": int(query["frame_count"]),
                "query_frame_path": query["frame_path"],
                "dino_reference_dataset": dino_reference["dataset_id"],
                "dino_reference_frame_count": int(dino_reference["frame_count"]),
                "dino_position_error_m": position_error_m(query, dino_reference, origin),
                "motion_viterbi_reference_dataset": selected_reference["dataset_id"],
                "motion_viterbi_reference_frame_count": int(selected_reference["frame_count"]),
                "motion_viterbi_reference_frame_path": selected_reference["frame_path"],
                "motion_viterbi_rank": int(candidate["rank"]),
                "motion_viterbi_dino_similarity": float(candidate["dino_similarity"]),
                "lg_match_count": int(candidate["lg_match_count"]),
                "lg_inlier_count": int(candidate["lg_inlier_count"]),
                "lg_inlier_ratio": float(candidate["lg_inlier_ratio"]),
                "motion_viterbi_position_error_m": position_error_m(query, selected_reference, origin),
            }
        )

    write_results(results, args.output_csv)
    summary = summarize(results)
    summary.update(
        {
            "dino_weight": args.dino_weight,
            "inlier_weight": args.inlier_weight,
            "ratio_weight": args.ratio_weight,
            "max_step_m": args.max_step_m,
            "transition_weight": args.transition_weight,
            "acceleration_scale_m": args.acceleration_scale_m,
            "acceleration_weight": args.acceleration_weight,
            "direction_scale_degrees": args.direction_scale_degrees,
            "direction_weight": args.direction_weight,
            "min_direction_step_m": args.min_direction_step_m,
            "candidate_limit": args.candidate_limit,
        }
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
