"""Apply Gaussian-weighted path smoothing to the Viterbi estimated trajectory.

Instead of always trusting the single best-retrieved frame, this script
averages each estimated position with its temporal neighbours using a
Gaussian window. Isolated wrong retrievals (outlier spikes) get pulled
toward the correct neighbourhood; correct retrievals near correct
neighbours are barely affected.

Tries multiple window sizes and reports which one minimises mean error.

Usage:
    python src/smooth_path.py \\
        outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \\
        data/processed/DJI_v14_frame_manifest_1fps.csv \\
        --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \\
        --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \\
        --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \\
        --output-csv outputs/anyloc/dji_mini3_smoothed_results.csv \\
        --summary-json outputs/anyloc/dji_mini3_smoothed_summary.json
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
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def gaussian_weights(half_window: int, sigma: float = 1.0) -> list[float]:
    """Symmetric Gaussian weights for indices -half_window..+half_window."""
    w = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-half_window, half_window + 1)]
    total = sum(w)
    return [x / total for x in w]


def smooth_path(
    lats: list[float],
    lons: list[float],
    half_window: int,
    sigma: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Gaussian-weighted moving average over lat/lon lists."""
    n = len(lats)
    weights = gaussian_weights(half_window, sigma)
    out_lats, out_lons = [], []
    for i in range(n):
        w_sum = lat_sum = lon_sum = 0.0
        for j, w in enumerate(weights):
            idx = i + j - half_window
            if 0 <= idx < n:
                lat_sum += w * lats[idx]
                lon_sum += w * lons[idx]
                w_sum += w
        out_lats.append(lat_sum / w_sum)
        out_lons.append(lon_sum / w_sum)
    return out_lats, out_lons


def smooth_path_causal(
    lats: list[float],
    lons: list[float],
    window: int,
    sigma: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Causal Gaussian-weighted moving average using only the current frame
    and up to ``window - 1`` *past* frames (no look-ahead at all). This is
    the real-time-deployable counterpart of ``smooth_path``: each output
    point can be produced the instant its own frame arrives, with zero
    added latency, because it never needs frames that haven't happened yet.
    """
    n = len(lats)
    weights = [math.exp(-0.5 * (j / sigma) ** 2) for j in range(window)]
    out_lats, out_lons = [], []
    for i in range(n):
        w_sum = lat_sum = lon_sum = 0.0
        for j, w in enumerate(weights):
            idx = i - j
            if idx >= 0:
                lat_sum += w * lats[idx]
                lon_sum += w * lons[idx]
                w_sum += w
        out_lats.append(lat_sum / w_sum)
        out_lons.append(lon_sum / w_sum)
    return out_lats, out_lons


def stats(errors: list[float]) -> dict:
    s = sorted(errors)
    n = len(s)
    return {
        "count": n,
        "mean_m": round(sum(s) / n, 4),
        "median_m": round(s[n // 2], 4),
        "p90_m": round(s[int(n * 0.9)], 4),
        "max_m": round(s[-1], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("query_manifest", type=Path)
    parser.add_argument("--reference-manifest", action="append", dest="ref_manifests",
                        metavar="NAME=PATH")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_smoothed_results.csv"))
    parser.add_argument("--summary-json", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_smoothed_summary.json"))
    parser.add_argument(
        "--causal",
        action="store_true",
        help=(
            "Use one-sided (past-only) Gaussian smoothing instead of the "
            "symmetric window. Zero added latency, real-time-deployable, "
            "at the cost of some accuracy versus the batch/symmetric mode."
        ),
    )
    args = parser.parse_args()

    # Load reference manifests → (dataset, frame_count) → (lat, lon)
    ref_index: dict[tuple[str, int], tuple[float, float]] = {}
    for spec in (args.ref_manifests or []):
        name, path = spec.split("=", 1)
        for row in load_csv(Path(path)):
            ref_index[(name, int(row["frame_count"]))] = (
                float(row["ground_latitude"]),
                float(row["ground_longitude"]),
            )

    # Load query manifest → ground truth
    gt_by_fc: dict[int, tuple[float, float]] = {
        int(r["frame_count"]): (float(r["ground_latitude"]), float(r["ground_longitude"]))
        for r in load_csv(args.query_manifest)
    }

    # Load Viterbi results → estimated positions (ordered by frame)
    results_rows = sorted(load_csv(args.results_csv), key=lambda r: int(r["query_frame_count"]))
    fcs: list[int] = []
    raw_lats: list[float] = []
    raw_lons: list[float] = []

    for r in results_rows:
        fc = int(r["query_frame_count"])
        ref_ds = r["motion_viterbi_reference_dataset"]
        ref_fc = int(r["motion_viterbi_reference_frame_count"])
        pos = ref_index.get((ref_ds, ref_fc))
        if pos:
            fcs.append(fc)
            raw_lats.append(pos[0])
            raw_lons.append(pos[1])

    # Baseline errors (no smoothing)
    baseline_errors = [
        haversine_m(raw_lats[i], raw_lons[i], *gt_by_fc[fcs[i]])
        for i in range(len(fcs))
        if fcs[i] in gt_by_fc
    ]
    baseline = stats(baseline_errors)

    print(f"Frames: {len(fcs)}")
    print(f"\nBaseline (no smoothing):  mean {baseline['mean_m']:.2f} m  "
          f"median {baseline['median_m']:.2f} m  P90 {baseline['p90_m']:.2f} m  "
          f"max {baseline['max_m']:.2f} m")

    # Try different window sizes
    sweep: list[dict] = []
    best_mean = float("inf")
    best_hw = 0

    print(f"\n{'Window':>8}  {'Mean':>8}  {'Median':>8}  {'P90':>8}  {'Max':>8}")
    print("-" * 50)

    for half_window in range(0, 13):
        if half_window == 0:
            slats, slons = raw_lats, raw_lons
        elif args.causal:
            window = 2 * half_window + 1
            slats, slons = smooth_path_causal(raw_lats, raw_lons, window, sigma=half_window * 0.6)
        else:
            slats, slons = smooth_path(raw_lats, raw_lons, half_window, sigma=half_window * 0.6)

        errs = [
            haversine_m(slats[i], slons[i], *gt_by_fc[fcs[i]])
            for i in range(len(fcs))
            if fcs[i] in gt_by_fc
        ]
        s = stats(errs)
        sweep.append({"half_window": half_window, "full_window": 2 * half_window + 1, **s})

        marker = " ← best" if s["mean_m"] < best_mean else ""
        if s["mean_m"] < best_mean:
            best_mean = s["mean_m"]
            best_hw = half_window

        label = f"w={2*half_window+1} (hw={half_window})"
        print(f"{label:>8}  {s['mean_m']:>7.2f}m  {s['median_m']:>7.2f}m  "
              f"{s['p90_m']:>7.2f}m  {s['max_m']:>7.2f}m{marker}")

    # Write best smoothed output
    if best_hw == 0:
        best_lats, best_lons = raw_lats, raw_lons
    elif args.causal:
        best_lats, best_lons = smooth_path_causal(
            raw_lats, raw_lons, 2 * best_hw + 1, sigma=best_hw * 0.6
        )
    else:
        best_lats, best_lons = smooth_path(raw_lats, raw_lons, best_hw, sigma=best_hw * 0.6)

    output_rows = []
    best_errors = []
    for i, fc in enumerate(fcs):
        gt = gt_by_fc.get(fc)
        err = haversine_m(best_lats[i], best_lons[i], *gt) if gt else None
        if err is not None:
            best_errors.append(err)
        output_rows.append({
            "query_frame_count": fc,
            "raw_lat": f"{raw_lats[i]:.8f}",
            "raw_lon": f"{raw_lons[i]:.8f}",
            "smoothed_lat": f"{best_lats[i]:.8f}",
            "smoothed_lon": f"{best_lons[i]:.8f}",
            "gt_lat": f"{gt[0]:.8f}" if gt else "",
            "gt_lon": f"{gt[1]:.8f}" if gt else "",
            "raw_error_m": f"{haversine_m(raw_lats[i], raw_lons[i], *gt):.4f}" if gt else "",
            "smoothed_error_m": f"{err:.4f}" if err is not None else "",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        w.writeheader()
        w.writerows(output_rows)

    summary = {
        "causal": args.causal,
        "best_half_window": best_hw,
        "best_full_window": 2 * best_hw + 1,
        "best_smoothed": stats(best_errors),
        "baseline_no_smoothing": baseline,
        "sweep": sweep,
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest window: {2*best_hw+1} (half={best_hw})")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
