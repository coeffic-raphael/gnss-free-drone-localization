"""Temporal reranking over DINOv2 top-k candidates scored by LightGlue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from tqdm import tqdm


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def extract_features(
    image_path: Path,
    extractor: SuperPoint,
    device: torch.device,
    image_resize: int,
    cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    key = str(image_path)
    if key not in cache:
        image = load_image(image_path, resize=image_resize).to(device)
        with torch.no_grad():
            cache[key] = extractor.extract(image)
    return cache[key]


def lightglue_score(
    query_path: Path,
    reference_path: Path,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    image_resize: int,
    feature_cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    feats0 = extract_features(query_path, extractor, device, image_resize, feature_cache)
    feats1 = extract_features(reference_path, extractor, device, image_resize, feature_cache)
    with torch.no_grad():
        matches01 = matcher({"image0": feats0, "image1": feats1})

    matches = matches01["matches"][0].detach().cpu().numpy()
    scores = matches01["scores"][0].detach().cpu().numpy()
    match_count = int(len(matches))
    mean_score = float(scores.mean()) if len(scores) else 0.0
    inlier_count = 0
    inlier_ratio = 0.0

    if match_count >= 4:
        keypoints0 = feats0["keypoints"][0].detach().cpu().numpy()
        keypoints1 = feats1["keypoints"][0].detach().cpu().numpy()
        pts0 = keypoints0[matches[:, 0]]
        pts1 = keypoints1[matches[:, 1]]
        _homography, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_count = int(mask.ravel().sum())
            inlier_ratio = float(inlier_count / max(match_count, 1))

    return {
        "lg_match_count": float(match_count),
        "lg_mean_score": mean_score,
        "lg_inlier_count": float(inlier_count),
        "lg_inlier_ratio": inlier_ratio,
    }


def build_candidates(
    reference_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    similarities: np.ndarray,
    top_k: int,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    image_resize: int,
) -> list[list[dict[str, float | int]]]:
    feature_cache: dict[str, dict[str, torch.Tensor]] = {}
    all_candidates: list[list[dict[str, float | int]]] = []
    for query_index, query in enumerate(tqdm(query_rows, desc="LightGlue candidates")):
        top_positions = np.argsort(similarities[query_index])[::-1][: max(1, top_k)]
        query_candidates: list[dict[str, float | int]] = []
        for position in top_positions:
            reference = reference_rows[int(position)]
            score = lightglue_score(
                Path(query["frame_path"]),
                Path(reference["frame_path"]),
                extractor,
                matcher,
                device,
                image_resize,
                feature_cache,
            )
            query_candidates.append(
                {
                    "reference_index": int(position),
                    "rank": len(query_candidates) + 1,
                    "dino_similarity": float(similarities[query_index, int(position)]),
                    **score,
                }
            )
        all_candidates.append(query_candidates)
    return all_candidates


def load_or_build_candidates(
    candidates_cache: Path | None,
    recompute: bool,
    reference_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    similarities: np.ndarray,
    top_k: int,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    image_resize: int,
) -> list[list[dict[str, float | int]]]:
    if candidates_cache is not None and candidates_cache.exists() and not recompute:
        with candidates_cache.open("rb") as cache_file:
            return pickle.load(cache_file)
    candidates = build_candidates(
        reference_rows,
        query_rows,
        similarities,
        top_k,
        extractor,
        matcher,
        device,
        image_resize,
    )
    if candidates_cache is not None:
        candidates_cache.parent.mkdir(parents=True, exist_ok=True)
        with candidates_cache.open("wb") as cache_file:
            pickle.dump(candidates, cache_file)
    return candidates


def unary_cost(candidate: dict[str, float | int], dino_weight: float, inlier_weight: float, ratio_weight: float) -> float:
    return (
        -dino_weight * float(candidate["dino_similarity"])
        -inlier_weight * math.log1p(float(candidate["lg_inlier_count"]))
        -ratio_weight * float(candidate["lg_inlier_ratio"])
    )


def transition_cost(
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


def dynamic_programming(
    candidates: list[list[dict[str, float | int]]],
    reference_rows: list[dict[str, str]],
    origin: tuple[float, float],
    dino_weight: float,
    inlier_weight: float,
    ratio_weight: float,
    max_step_m: float,
    transition_weight: float,
) -> list[dict[str, float | int]]:
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

            best_cost = math.inf
            best_parent = -1
            current_reference = reference_rows[int(candidate["reference_index"])]
            for prev_idx, previous in enumerate(candidates[query_idx - 1]):
                previous_reference = reference_rows[int(previous["reference_index"])]
                cost = (
                    costs[query_idx - 1][prev_idx]
                    + base
                    + transition_cost(
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
    selected: list[dict[str, float | int]] = []
    for query_idx in range(len(candidates) - 1, -1, -1):
        selected.append(candidates[query_idx][last_idx])
        last_idx = parents[query_idx][last_idx]
    selected.reverse()
    return selected


def summarize(results: list[dict[str, object]]) -> dict[str, float | int]:
    dino_errors = np.array([float(row["dino_position_error_m"]) for row in results])
    temporal_errors = np.array([float(row["temporal_position_error_m"]) for row in results])
    improved = int((temporal_errors < dino_errors).sum())
    worsened = int((temporal_errors > dino_errors).sum())
    unchanged = len(results) - improved - worsened
    return {
        "queries": len(results),
        "dino_mean_error_m": float(dino_errors.mean()) if len(dino_errors) else 0.0,
        "dino_median_error_m": float(np.median(dino_errors)) if len(dino_errors) else 0.0,
        "dino_p90_error_m": float(np.percentile(dino_errors, 90)) if len(dino_errors) else 0.0,
        "temporal_mean_error_m": float(temporal_errors.mean()) if len(temporal_errors) else 0.0,
        "temporal_median_error_m": float(np.median(temporal_errors)) if len(temporal_errors) else 0.0,
        "temporal_p90_error_m": float(np.percentile(temporal_errors, 90)) if len(temporal_errors) else 0.0,
        "temporal_max_error_m": float(temporal_errors.max()) if len(temporal_errors) else 0.0,
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
    parser.add_argument("--descriptor-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates-cache", type=Path)
    parser.add_argument("--recompute-candidates", action="store_true")
    parser.add_argument("--image-resize", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--dino-weight", type=float, default=4.0)
    parser.add_argument("--inlier-weight", type=float, default=1.0)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=35.0)
    parser.add_argument("--transition-weight", type=float, default=4.0)
    args = parser.parse_args()

    reference_rows: list[dict[str, str]] = []
    for value in args.reference_manifest:
        path, dataset_id = parse_manifest_arg(value)
        reference_rows.extend(load_manifest(path, dataset_id))
    query_path, query_dataset_id = parse_manifest_arg(args.query_manifest)
    query_rows = load_manifest(query_path, query_dataset_id)

    descriptors = np.load(args.descriptor_cache)
    reference_descriptors = descriptors[: len(reference_rows)]
    query_descriptors = descriptors[len(reference_rows) :]
    similarities = query_descriptors @ reference_descriptors.T

    device = choose_device()
    print(f"device: {device}")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    candidates = load_or_build_candidates(
        args.candidates_cache,
        args.recompute_candidates,
        reference_rows,
        query_rows,
        similarities,
        args.top_k,
        extractor,
        matcher,
        device,
        args.image_resize,
    )
    origin = (
        float(reference_rows[0]["ground_latitude"]),
        float(reference_rows[0]["ground_longitude"]),
    )
    selected = dynamic_programming(
        candidates,
        reference_rows,
        origin,
        args.dino_weight,
        args.inlier_weight,
        args.ratio_weight,
        args.max_step_m,
        args.transition_weight,
    )

    results: list[dict[str, object]] = []
    for query_index, (query, candidate) in enumerate(zip(query_rows, selected)):
        dino_reference = reference_rows[int(candidates[query_index][0]["reference_index"])]
        temporal_reference = reference_rows[int(candidate["reference_index"])]
        results.append(
            {
                "query_dataset": query["dataset_id"],
                "query_frame_count": int(query["frame_count"]),
                "query_frame_path": query["frame_path"],
                "dino_reference_dataset": dino_reference["dataset_id"],
                "dino_reference_frame_count": int(dino_reference["frame_count"]),
                "dino_position_error_m": position_error_m(query, dino_reference, origin),
                "temporal_reference_dataset": temporal_reference["dataset_id"],
                "temporal_reference_frame_count": int(temporal_reference["frame_count"]),
                "temporal_reference_frame_path": temporal_reference["frame_path"],
                "temporal_rank": int(candidate["rank"]),
                "temporal_dino_similarity": float(candidate["dino_similarity"]),
                "lg_match_count": int(candidate["lg_match_count"]),
                "lg_inlier_count": int(candidate["lg_inlier_count"]),
                "lg_inlier_ratio": float(candidate["lg_inlier_ratio"]),
                "temporal_position_error_m": position_error_m(query, temporal_reference, origin),
            }
        )

    write_results(results, args.output_csv)
    summary = summarize(results)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
