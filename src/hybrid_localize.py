"""Hybrid VPR + satellite localization pipeline.

Merges per-frame decisions from the VPR confidence gate with satellite
tile-matching results to produce a single localisation estimate per frame:

  FIX        → VPR smoothed position  (VPR was confident)
  NO_FIX     → satellite position if available, else VPR fallback

Outputs:
  outputs/hybrid/hybrid_results_v14.csv   — per-frame combined output
  outputs/hybrid/hybrid_summary_v14.json  — aggregate statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def load_csv_index(path: Path, key: str) -> dict[str, dict]:
    """Load a CSV and index rows by `key` column."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r[key]: r for r in rows}


# ---------------------------------------------------------------------------
# Reference manifest lat/lon lookup
# ---------------------------------------------------------------------------

def build_reference_latlon(manifests: list[str]) -> dict[str, tuple[float, float]]:
    """
    Return {frame_path → (ground_lat, ground_lon)} from reference manifests.
    """
    index: dict[str, tuple[float, float]] = {}
    for spec in manifests:
        _, path = spec.split("=", 1)
        with open(path) as f:
            for row in csv.DictReader(f):
                index[row["frame_path"]] = (
                    float(row["ground_latitude"]),
                    float(row["ground_longitude"]),
                )
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpr-decisions", type=Path, required=True,
                        help="CSV from confidence_gate_results.py")
    parser.add_argument("--vpr-results", type=Path, required=True,
                        help="CSV from motion_viterbi_rerank.py (with reference frame paths)")
    parser.add_argument("--smoothed-csv", type=Path, default=None,
                        help="CSV from smooth_path.py (smoothed_lat/lon)")
    parser.add_argument("--satellite-csv", type=Path, required=True,
                        help="CSV from test_satellite_match.sh evaluation")
    parser.add_argument("--query-manifest", type=Path, required=True,
                        help="Query frame manifest (for GT positions)")
    parser.add_argument("--reference-manifest", action="append", default=[],
                        metavar="NAME=PATH",
                        help="Reference manifests (name=path), repeatable")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/hybrid"))
    parser.add_argument("--version", default="v14")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all sources ─────────────────────────────────────────────────────

    # Query manifest: ground-truth lat/lon per frame
    with open(args.query_manifest) as f:
        query_rows = list(csv.DictReader(f))
    query_index = {r["frame_count"]: r for r in query_rows}

    # VPR decisions (confidence gate)
    decisions = load_csv_index(args.vpr_decisions, "query_frame_count")

    # VPR results (reference frame paths for lat/lon lookup)
    vpr_results = load_csv_index(args.vpr_results, "query_frame_count")

    # Reference lat/lon lookup
    ref_latlon = build_reference_latlon(args.reference_manifest)

    # Smoothed VPR path (optional)
    smoothed: dict[str, tuple[float, float]] = {}
    if args.smoothed_csv and args.smoothed_csv.exists():
        with open(args.smoothed_csv) as f:
            for row in csv.DictReader(f):
                smoothed[row["query_frame_count"]] = (
                    float(row["smoothed_lat"]),
                    float(row["smoothed_lon"]),
                )

    # Satellite results
    sat = load_csv_index(args.satellite_csv, "frame_count")

    # ── Merge ────────────────────────────────────────────────────────────────

    out_rows = []

    for q_row in query_rows:
        fc = q_row["frame_count"]
        gt_lat = float(q_row["ground_latitude"])
        gt_lon = float(q_row["ground_longitude"])

        # VPR position (smoothed if available, else raw reference match)
        if fc in smoothed:
            vpr_lat, vpr_lon = smoothed[fc]
        elif fc in vpr_results:
            ref_path = vpr_results[fc].get("motion_viterbi_reference_frame_path", "")
            if ref_path and ref_path in ref_latlon:
                vpr_lat, vpr_lon = ref_latlon[ref_path]
            else:
                vpr_lat, vpr_lon = gt_lat, gt_lon   # shouldn't happen
        else:
            vpr_lat, vpr_lon = gt_lat, gt_lon

        vpr_err = haversine_m(gt_lat, gt_lon, vpr_lat, vpr_lon)

        # Decision from confidence gate
        dec_row = decisions.get(fc, {})
        vpr_decision = dec_row.get("decision", "NO_FIX")

        # Satellite result
        sat_row = sat.get(fc, {})
        sat_status = sat_row.get("status", "")
        sat_ok = (sat_status == "ok")
        if sat_ok:
            sat_lat = float(sat_row["est_lat"])
            sat_lon = float(sat_row["est_lon"])
            sat_err = haversine_m(gt_lat, gt_lon, sat_lat, sat_lon)
            sat_inliers = int(sat_row.get("inliers", 0))
        else:
            sat_lat = sat_lon = None
            sat_err = None
            sat_inliers = 0

        # ── Fusion decision ──────────────────────────────────────────────────
        if vpr_decision == "FIX":
            final_status = "VPR_FIX"
            final_lat, final_lon = vpr_lat, vpr_lon
            final_err = vpr_err
        elif sat_ok:
            final_status = "SAT_FIX"
            final_lat, final_lon = sat_lat, sat_lon
            final_err = sat_err
        else:
            final_status = "VPR_FALLBACK"
            final_lat, final_lon = vpr_lat, vpr_lon
            final_err = vpr_err

        out_rows.append({
            "frame_count":    fc,
            "time_s":         q_row["start_seconds"],
            "gt_lat":         gt_lat,
            "gt_lon":         gt_lon,
            "vpr_decision":   vpr_decision,
            "vpr_lat":        round(vpr_lat, 7),
            "vpr_lon":        round(vpr_lon, 7),
            "vpr_error_m":    round(vpr_err, 2),
            "sat_status":     sat_status,
            "sat_lat":        round(sat_lat, 7) if sat_lat else "",
            "sat_lon":        round(sat_lon, 7) if sat_lon else "",
            "sat_error_m":    round(sat_err, 2) if sat_err is not None else "",
            "sat_inliers":    sat_inliers,
            "final_status":   final_status,
            "final_lat":      round(final_lat, 7),
            "final_lon":      round(final_lon, 7),
            "final_error_m":  round(final_err, 2),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────

    out_csv = args.output_dir / f"hybrid_results_{args.version}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ── Summary ───────────────────────────────────────────────────────────────

    def stats(errs: list[float]) -> dict:
        if not errs:
            return {}
        return {
            "count":  len(errs),
            "median": round(statistics.median(errs), 2),
            "mean":   round(statistics.mean(errs), 2),
            "min":    round(min(errs), 2),
            "max":    round(max(errs), 2),
            "pct_5m":  round(100 * sum(1 for e in errs if e <= 5)  / len(errs), 1),
            "pct_10m": round(100 * sum(1 for e in errs if e <= 10) / len(errs), 1),
            "pct_15m": round(100 * sum(1 for e in errs if e <= 15) / len(errs), 1),
            "pct_20m": round(100 * sum(1 for e in errs if e <= 20) / len(errs), 1),
        }

    by_status: dict[str, list[float]] = {}
    for r in out_rows:
        s = r["final_status"]
        by_status.setdefault(s, []).append(r["final_error_m"])

    all_errors = [r["final_error_m"] for r in out_rows]

    summary = {
        "version": args.version,
        "total_frames": len(out_rows),
        "counts": {s: len(errs) for s, errs in by_status.items()},
        "overall": stats(all_errors),
        "by_status": {s: stats(errs) for s, errs in by_status.items()},
        "thresholds_overall": {
            f"pct_le_{t}m": round(
                100 * sum(1 for e in all_errors if e <= t) / len(all_errors), 1
            )
            for t in [5, 10, 15, 20, 30]
        },
    }

    out_json = args.output_dir / f"hybrid_summary_{args.version}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # ── Print ─────────────────────────────────────────────────────────────────

    print(f"\n{'═'*60}")
    print(f"Hybrid localisation — {args.version}  ({len(out_rows)} frames)")
    print(f"{'═'*60}")
    for s, errs in sorted(by_status.items()):
        pct = 100 * len(errs) / len(out_rows)
        med = statistics.median(errs) if errs else float("nan")
        print(f"  {s:<16} {len(errs):3d} frames ({pct:5.1f}%)  "
              f"median err = {med:.1f} m")

    print(f"\n  Overall ({len(out_rows)} frames):")
    for t in [5, 10, 15, 20, 30]:
        n = sum(1 for e in all_errors if e <= t)
        print(f"    ≤ {t:2d} m : {n:3d}/{len(out_rows)}  "
              f"({100*n/len(out_rows):5.1f}%)")
    print(f"  Median: {statistics.median(all_errors):.1f} m  "
          f"Mean: {statistics.mean(all_errors):.1f} m")
    print(f"\n  CSV  : {out_csv}")
    print(f"  JSON : {out_json}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
