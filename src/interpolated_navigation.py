"""Interpolated navigation: FIX frames use retrieval position, NO_FIX frames
are linearly interpolated between the nearest confident FIX neighbours.

This improves on two failure modes of the base pipeline:
  1. Always-output (18.83 m mean): includes many wrong retrievals.
  2. Confidence-gated (13.67 m, 30% coverage): leaves 46 s gaps.

With interpolation, all 115 frames get a position:
  - FIX  frames → retrieval position  (~13.67 m mean)
  - NO_FIX frames → linear interpolation between surrounding FIX frames
    (error bounded by drone displacement during the gap)

Usage:
    python src/interpolated_navigation.py \\
        outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv \\
        outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \\
        data/processed/DJI_v14_frame_manifest_1fps.csv \\
        --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \\
        --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \\
        --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \\
        --output-csv outputs/anyloc/dji_mini3_interpolated_results.csv \\
        --summary-json outputs/anyloc/dji_mini3_interpolated_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EARTH_RADIUS_M = 6_378_137.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = EARTH_RADIUS_M
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def lerp_latlon(
    lat0: float, lon0: float,
    lat1: float, lon1: float,
    t: float,
) -> tuple[float, float]:
    """Linear interpolation between two lat/lon points. t in [0, 1]."""
    return lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0)


def build_ref_index(ref_manifests: dict[str, list[dict]]) -> dict[tuple[str, int], tuple[float, float]]:
    """Map (dataset, frame_count) → (ground_lat, ground_lon)."""
    index: dict[tuple[str, int], tuple[float, float]] = {}
    for dataset, rows in ref_manifests.items():
        for row in rows:
            fc = int(row["frame_count"])
            lat = float(row["ground_latitude"])
            lon = float(row["ground_longitude"])
            index[(dataset, fc)] = (lat, lon)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions_csv", type=Path,
                        help="Output of confidence_gate_results.py (decisions CSV)")
    parser.add_argument("results_csv", type=Path,
                        help="Motion Viterbi results CSV")
    parser.add_argument("query_manifest", type=Path,
                        help="Query frame manifest (ground truth)")
    parser.add_argument("--reference-manifest", action="append", dest="ref_manifests",
                        metavar="NAME=PATH",
                        help="Reference manifests, e.g. v11=data/processed/DJI_v11_frame_manifest_1fps.csv")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_interpolated_results.csv"))
    parser.add_argument("--summary-json", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_interpolated_summary.json"))
    args = parser.parse_args()

    # Load reference manifests
    ref_manifests: dict[str, list[dict]] = {}
    for spec in (args.ref_manifests or []):
        name, path = spec.split("=", 1)
        ref_manifests[name] = load_csv(Path(path))
    ref_index = build_ref_index(ref_manifests)

    # Load query manifest (ground truth)
    query_rows = load_csv(args.query_manifest)
    gt_by_fc: dict[int, tuple[float, float]] = {
        int(r["frame_count"]): (float(r["ground_latitude"]), float(r["ground_longitude"]))
        for r in query_rows
    }

    # Load results CSV (estimated position via reference frame lookup)
    results_rows = load_csv(args.results_csv)
    est_by_fc: dict[int, tuple[float, float]] = {}
    for r in results_rows:
        fc = int(r["query_frame_count"])
        ref_ds = r["motion_viterbi_reference_dataset"]
        ref_fc = int(r["motion_viterbi_reference_frame_count"])
        key = (ref_ds, ref_fc)
        if key in ref_index:
            est_by_fc[fc] = ref_index[key]

    # Load confidence gate decisions
    decisions_rows = load_csv(args.decisions_csv)
    fix_fcs: set[int] = set()
    all_fcs_ordered: list[int] = []
    for r in decisions_rows:
        fc = int(r["query_frame_count"])
        all_fcs_ordered.append(fc)
        if r["decision"] == "FIX":
            fix_fcs.add(fc)

    all_fcs_ordered.sort()

    print(f"Total query frames: {len(all_fcs_ordered)}")
    print(f"FIX frames:         {len(fix_fcs)}")
    print(f"NO_FIX frames:      {len(all_fcs_ordered) - len(fix_fcs)}")

    # For each frame, determine interpolated position
    # FIX: use retrieval estimate directly
    # NO_FIX: interpolate between nearest FIX before and after

    fix_list = sorted(fix_fcs)  # ordered list of FIX frame indices

    def get_interpolated(fc: int) -> tuple[float, float] | None:
        if fc in fix_fcs:
            return est_by_fc.get(fc)

        # Find nearest FIX before and after
        prev_fix = None
        next_fix = None
        for f in fix_list:
            if f < fc:
                prev_fix = f
            elif f > fc and next_fix is None:
                next_fix = f
                break

        if prev_fix is not None and next_fix is not None:
            lat0, lon0 = est_by_fc[prev_fix]
            lat1, lon1 = est_by_fc[next_fix]
            t = (fc - prev_fix) / (next_fix - prev_fix)
            return lerp_latlon(lat0, lon0, lat1, lon1, t)
        elif prev_fix is not None:
            # After last fix: hold last known position
            return est_by_fc[prev_fix]
        elif next_fix is not None:
            # Before first fix: use first fix position
            return est_by_fc[next_fix]
        return None

    # Compute errors and write output
    output_rows = []
    errors_fix: list[float] = []
    errors_interp: list[float] = []
    errors_all: list[float] = []

    for fc in all_fcs_ordered:
        gt = gt_by_fc.get(fc)
        est = get_interpolated(fc)
        method = "FIX" if fc in fix_fcs else "INTERPOLATED"

        error = None
        if gt and est:
            error = haversine_m(gt[0], gt[1], est[0], est[1])
            errors_all.append(error)
            if method == "FIX":
                errors_fix.append(error)
            else:
                errors_interp.append(error)

        output_rows.append({
            "query_frame_count": fc,
            "method": method,
            "est_latitude": f"{est[0]:.8f}" if est else "",
            "est_longitude": f"{est[1]:.8f}" if est else "",
            "gt_latitude": f"{gt[0]:.8f}" if gt else "",
            "gt_longitude": f"{gt[1]:.8f}" if gt else "",
            "position_error_m": f"{error:.4f}" if error is not None else "",
        })

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        w.writeheader()
        w.writerows(output_rows)

    def stats(errs: list[float]) -> dict:
        if not errs:
            return {}
        s = sorted(errs)
        n = len(s)
        return {
            "count": n,
            "mean_m": sum(s) / n,
            "median_m": s[n // 2],
            "p90_m": s[int(n * 0.9)],
            "max_m": s[-1],
        }

    summary = {
        "all_frames": stats(errors_all),
        "fix_frames_only": stats(errors_fix),
        "interpolated_frames_only": stats(errors_interp),
        "baseline_always_output": {
            "mean_m": 18.83, "median_m": 15.21, "p90_m": 36.05, "max_m": 72.53
        },
        "baseline_confidence_gated": {
            "mean_m": 13.67, "median_m": 10.58, "coverage": 0.304
        },
    }

    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== Interpolated Navigation Results ===")
    print(f"{'Method':<30} {'Mean':>8} {'Median':>8} {'P90':>8} {'Max':>8}")
    print("-" * 66)

    a = summary["all_frames"]
    print(f"{'All frames (interp+fix)':<30} {a['mean_m']:>7.2f}m {a['median_m']:>7.2f}m {a['p90_m']:>7.2f}m {a['max_m']:>7.2f}m")

    fi = summary["fix_frames_only"]
    print(f"{'FIX frames only':<30} {fi['mean_m']:>7.2f}m {fi['median_m']:>7.2f}m {fi['p90_m']:>7.2f}m {fi['max_m']:>7.2f}m")

    ii = summary["interpolated_frames_only"]
    if ii:
        print(f"{'Interpolated frames only':<30} {ii['mean_m']:>7.2f}m {ii['median_m']:>7.2f}m {ii['p90_m']:>7.2f}m {ii['max_m']:>7.2f}m")

    print()
    print("Baselines:")
    print(f"  Always output (100% cov):   mean 18.83 m, median 15.21 m, P90 36.05 m")
    print(f"  Confidence gated (30% cov): mean 13.67 m, median 10.58 m")
    print()
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
