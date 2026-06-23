"""Cross-flight GNSS-denied retrieval with frozen DINOv2 descriptors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from anyloc_dino_retrieval import compute_descriptors


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


def position_error_m(query: dict[str, str], reference: dict[str, str], origin: tuple[float, float]) -> float:
    qx, qy = local_xy_from_latlon(
        float(query["ground_latitude"]),
        float(query["ground_longitude"]),
        origin[0],
        origin[1],
    )
    rx, ry = local_xy_from_latlon(
        float(reference["ground_latitude"]),
        float(reference["ground_longitude"]),
        origin[0],
        origin[1],
    )
    return math.hypot(qx - rx, qy - ry)


def run_cross_retrieval(
    reference_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    reference_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    top_k: int,
) -> list[dict[str, object]]:
    similarities = query_descriptors @ reference_descriptors.T
    origin = (
        float(reference_rows[0]["ground_latitude"]),
        float(reference_rows[0]["ground_longitude"]),
    )
    results: list[dict[str, object]] = []
    for query_index, query in enumerate(query_rows):
        top_positions = np.argsort(similarities[query_index])[::-1][: max(1, top_k)]
        best_ref = reference_rows[int(top_positions[0])]
        top_errors = [
            position_error_m(query, reference_rows[int(position)], origin)
            for position in top_positions
        ]
        results.append(
            {
                "query_dataset": query["dataset_id"],
                "query_frame_count": int(query["frame_count"]),
                "query_frame_path": query["frame_path"],
                "reference_dataset": best_ref["dataset_id"],
                "reference_frame_count": int(best_ref["frame_count"]),
                "reference_frame_path": best_ref["frame_path"],
                "similarity": float(similarities[query_index, int(top_positions[0])]),
                "top_k": len(top_errors),
                "oracle_top_k_error_m": min(top_errors),
                "query_ground_latitude": float(query["ground_latitude"]),
                "query_ground_longitude": float(query["ground_longitude"]),
                "estimated_ground_latitude": float(best_ref["ground_latitude"]),
                "estimated_ground_longitude": float(best_ref["ground_longitude"]),
                "position_error_m": position_error_m(query, best_ref, origin),
            }
        )
    return results


def write_results(results: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summarize(results: list[dict[str, object]]) -> dict[str, float | int | dict[str, int]]:
    errors = np.array([float(row["position_error_m"]) for row in results])
    oracle_errors = np.array([float(row["oracle_top_k_error_m"]) for row in results])
    reference_hits: dict[str, int] = {}
    for row in results:
        reference_hits[str(row["reference_dataset"])] = reference_hits.get(str(row["reference_dataset"]), 0) + 1
    return {
        "queries": len(results),
        "mean_error_m": float(errors.mean()) if len(errors) else 0.0,
        "median_error_m": float(np.median(errors)) if len(errors) else 0.0,
        "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else 0.0,
        "max_error_m": float(errors.max()) if len(errors) else 0.0,
        "oracle_top_k_mean_error_m": float(oracle_errors.mean()) if len(oracle_errors) else 0.0,
        "oracle_top_k_median_error_m": float(np.median(oracle_errors)) if len(oracle_errors) else 0.0,
        "reference_hits": reference_hits,
    }


def parse_manifest_arg(value: str) -> tuple[Path, str]:
    if "=" not in value:
        path = Path(value)
        return path, path.stem
    dataset_id, path_text = value.split("=", 1)
    return Path(path_text), dataset_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-manifest",
        action="append",
        required=True,
        help="Reference manifest as DATASET_ID=path.csv. Can be repeated.",
    )
    parser.add_argument("--query-manifest", required=True, help="Query manifest as DATASET_ID=path.csv.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--descriptor-cache", type=Path, required=True)
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--dinov2-repo", type=Path, default=Path("third_party/dinov2"))
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("outputs/models/dinov2/dinov2_vits14_pretrain.pth"),
    )
    parser.add_argument("--max-size", type=int, default=518)
    parser.add_argument("--aggregation", choices=["mean", "gem", "vlad"], default="mean")
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--max-train-patches", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    reference_rows: list[dict[str, str]] = []
    for value in args.reference_manifest:
        path, dataset_id = parse_manifest_arg(value)
        reference_rows.extend(load_manifest(path, dataset_id))
    query_path, query_dataset_id = parse_manifest_arg(args.query_manifest)
    query_rows = load_manifest(query_path, query_dataset_id)
    all_rows = reference_rows + query_rows

    args.descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.descriptor_cache.exists() and not args.recompute:
        descriptors = np.load(args.descriptor_cache)
    else:
        weights_path = args.weights_path if args.weights_path.exists() else None
        descriptors = compute_descriptors(
            all_rows,
            args.model_name,
            args.max_size,
            args.dinov2_repo,
            weights_path,
            aggregation=args.aggregation,
            num_clusters=args.num_clusters,
            max_train_patches=args.max_train_patches,
            random_seed=args.random_seed,
            gem_p=args.gem_p,
        )
        np.save(args.descriptor_cache, descriptors)

    reference_descriptors = descriptors[: len(reference_rows)]
    query_descriptors = descriptors[len(reference_rows) :]
    results = run_cross_retrieval(
        reference_rows,
        query_rows,
        reference_descriptors,
        query_descriptors,
        top_k=args.top_k,
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
