"""Causal post-processing for the satellite-first hybrid output.

Two real-time-deployable, zero-added-latency improvements applied to the
per-frame CSV produced by run_satellite_first_hybrid.sh:

  1. Gap filling: NO_FIX frames get no position at all today. Carry the last
     known fix forward (simple causal dead-reckoning placeholder) so the
     output trajectory has no holes.
  2. Causal Gaussian smoothing: average each (carried-forward) position with
     its *past* neighbours only (smooth_path.smooth_path_causal). This pulls
     isolated bad fixes (e.g. a noisy VPR_FALLBACK estimate) toward the
     surrounding trajectory without needing any future frame, so it adds
     ZERO latency and stays strictly real-time.

Sweeps a few causal window sizes and reports the one that minimises mean
error, exactly like smooth_path.py does for the original VPR-only path.

Usage:
    python src/smooth_hybrid_path.py \\
        outputs/hybrid/satellite_first_v14.csv \\
        data/processed/DJI_v14_frame_manifest_1fps.csv \\
        --output-csv outputs/hybrid/satellite_first_v14_smoothed.csv \\
        --summary-json outputs/hybrid/satellite_first_v14_smoothed_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smooth_path import haversine_m, smooth_path_causal, stats


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fill_gaps(fcs: list[int], lats: list[float | None], lons: list[float | None]) -> tuple[list[float], list[float]]:
    """Carry the last known fix forward into NO_FIX frames (causal, no
    look-ahead). The very first frame(s), if missing, fall back to the first
    available fix (can't dead-reckon before any data exists)."""
    out_lats: list[float] = []
    out_lons: list[float] = []
    first_known = next((i for i, v in enumerate(lats) if v is not None), None)
    last_lat = lats[first_known] if first_known is not None else 0.0
    last_lon = lons[first_known] if first_known is not None else 0.0
    for lat, lon in zip(lats, lons):
        if lat is not None:
            last_lat, last_lon = lat, lon
        out_lats.append(last_lat)
        out_lons.append(last_lon)
    return out_lats, out_lons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hybrid_csv", type=Path)
    parser.add_argument("query_manifest", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-half-window", type=int, default=6)
    args = parser.parse_args()

    rows = sorted(load_csv(args.hybrid_csv), key=lambda r: int(r["frame_count"]))
    gt_by_fc = {
        int(r["frame_count"]): (float(r["ground_latitude"]), float(r["ground_longitude"]))
        for r in load_csv(args.query_manifest)
    }

    fcs = [int(r["frame_count"]) for r in rows]
    sources = [r["source"] for r in rows]
    raw_lats: list[float | None] = [float(r["final_lat"]) if r["final_lat"] else None for r in rows]
    raw_lons: list[float | None] = [float(r["final_lon"]) if r["final_lon"] else None for r in rows]

    n_no_fix = sum(1 for s in sources if s == "NO_FIX")
    print(f"Frames: {len(fcs)}  (NO_FIX before gap-filling: {n_no_fix})")

    # Step 1: causal gap-filling (carry last fix forward)
    filled_lats, filled_lons = fill_gaps(fcs, raw_lats, raw_lons)

    filled_errors = [haversine_m(filled_lats[i], filled_lons[i], *gt_by_fc[fcs[i]]) for i in range(len(fcs)) if fcs[i] in gt_by_fc]
    filled_stats = stats(filled_errors)
    print(f"\nGap-filled, no smoothing:  mean {filled_stats['mean_m']:.2f} m  "
          f"median {filled_stats['median_m']:.2f} m  P90 {filled_stats['p90_m']:.2f} m  "
          f"max {filled_stats['max_m']:.2f} m")

    # Step 2: sweep causal smoothing windows on top of the gap-filled path
    sweep: list[dict] = []
    best_mean = float("inf")
    best_hw = 0

    print(f"\n{'Window':>10}  {'Mean':>8}  {'Median':>8}  {'P90':>8}  {'Max':>8}")
    print("-" * 52)

    for half_window in range(0, args.max_half_window + 1):
        if half_window == 0:
            slats, slons = filled_lats, filled_lons
        else:
            window = 2 * half_window + 1
            slats, slons = smooth_path_causal(filled_lats, filled_lons, window, sigma=half_window * 0.6)

        errs = [haversine_m(slats[i], slons[i], *gt_by_fc[fcs[i]]) for i in range(len(fcs)) if fcs[i] in gt_by_fc]
        s = stats(errs)
        sweep.append({"half_window": half_window, "full_window": 2 * half_window + 1, **s})

        marker = " <- best" if s["mean_m"] < best_mean else ""
        if s["mean_m"] < best_mean:
            best_mean = s["mean_m"]
            best_hw = half_window

        label = f"w={2*half_window+1}"
        print(f"{label:>10}  {s['mean_m']:>7.2f}m  {s['median_m']:>7.2f}m  "
              f"{s['p90_m']:>7.2f}m  {s['max_m']:>7.2f}m{marker}")

    if best_hw == 0:
        best_lats, best_lons = filled_lats, filled_lons
    else:
        best_lats, best_lons = smooth_path_causal(filled_lats, filled_lons, 2 * best_hw + 1, sigma=best_hw * 0.6)

    output_rows = []
    best_errors = []
    for i, fc in enumerate(fcs):
        gt = gt_by_fc.get(fc)
        err = haversine_m(best_lats[i], best_lons[i], *gt) if gt else None
        if err is not None:
            best_errors.append(err)
        output_rows.append({
            "frame_count": fc,
            "source": sources[i],
            "raw_final_lat": raw_lats[i] if raw_lats[i] is not None else "",
            "raw_final_lon": raw_lons[i] if raw_lons[i] is not None else "",
            "smoothed_lat": f"{best_lats[i]:.8f}",
            "smoothed_lon": f"{best_lons[i]:.8f}",
            "gt_lat": f"{gt[0]:.8f}" if gt else "",
            "gt_lon": f"{gt[1]:.8f}" if gt else "",
            "smoothed_error_m": f"{err:.4f}" if err is not None else "",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        w.writeheader()
        w.writerows(output_rows)

    summary = {
        "frames_total": len(fcs),
        "no_fix_frames_before_gap_fill": n_no_fix,
        "gap_filled_no_smoothing": filled_stats,
        "best_half_window": best_hw,
        "best_full_window": 2 * best_hw + 1,
        "best_smoothed": stats(best_errors),
        "sweep": sweep,
        "thresholds_best": {
            t: sum(1 for e in best_errors if e <= t) for t in [5, 10, 15, 20, 30]
        },
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest causal window: {2*best_hw+1} (half={best_hw})  "
          f"mean {summary['best_smoothed']['mean_m']:.2f} m "
          f"(vs gap-filled-only {filled_stats['mean_m']:.2f} m, "
          f"vs raw-with-holes baseline not computed here)")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
