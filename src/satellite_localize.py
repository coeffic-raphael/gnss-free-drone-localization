"""Satellite-based localization for NO_FIX drone frames.

For each frame:
1. Apply IPM to warp the tilted drone view to pseudo-nadir.
2. Find the satellite tile covering the approximate position.
3. Run SuperPoint + LightGlue to match the warped frame against the tile.
4. Estimate lat/lon via RANSAC homography.

Usage:
    python src/satellite_localize.py \\
        --results-csv outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \\
        --confidence-csv outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv \\
        --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \\
        --satellite-dir data/satellite \\
        --zoom 18 \\
        --output-csv outputs/anyloc/dji_mini3_satellite_localize_results.csv \\
        --summary-json outputs/anyloc/dji_mini3_satellite_localize_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add parent dir for LightGlue imports
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "dinov2"))

from ipm_warp import ipm_warp, DEFAULT_HFOV_DEG
from satellite_tiles import (
    find_tile,
    load_tile_meta,
    tile_pixel_to_latlon,
    latlon_to_tile_pixel,
)
from geometry import gps_to_local_xy, local_xy_to_gps


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# LightGlue matcher
# ---------------------------------------------------------------------------

def load_matcher(device: torch.device):
    from lightglue import LightGlue, SuperPoint
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def extract_features(img_bgr: np.ndarray, extractor, device: torch.device):
    from lightglue.utils import numpy_image_to_torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = numpy_image_to_torch(img_rgb).to(device)
    with torch.no_grad():
        feats = extractor.extract(tensor)
    return feats


def match_pair(feats0, feats1, matcher, device: torch.device):
    from lightglue import LightGlue
    with torch.no_grad():
        result = matcher({"image0": feats0, "image1": feats1})
    kpts0 = feats0["keypoints"][0].cpu().numpy()
    kpts1 = feats1["keypoints"][0].cpu().numpy()
    matches = result["matches"][0].cpu().numpy()
    valid = matches > -1
    matched0 = kpts0[valid]
    matched1 = kpts1[matches[valid]]
    return matched0, matched1


# ---------------------------------------------------------------------------
# Single-frame satellite localization
# ---------------------------------------------------------------------------

def localize_frame(
    frame_path: Path,
    approx_lat: float,
    approx_lon: float,
    altitude_m: float,
    camera_angle_deg: float,
    heading_deg: float,
    satellite_dir: Path,
    zoom: int,
    extractor,
    matcher,
    device: torch.device,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    ipm_size: int = 512,
    ipm_gsd_m: float = 0.5,
    min_inliers: int = 15,
) -> dict:
    """
    Attempt to localize one frame using satellite tile matching.

    Returns a dict with keys: status, lat, lon, inlier_count, inlier_ratio
    status is 'SAT_FIX' or 'SAT_NOFIX'.
    """
    result = {
        "status": "SAT_NOFIX",
        "lat": float("nan"),
        "lon": float("nan"),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
    }

    # --- Load frame ---
    img = cv2.imread(str(frame_path))
    if img is None:
        return result

    # --- IPM warp ---
    warped, gsd, (center_east, center_north) = ipm_warp(
        img,
        altitude_m=altitude_m,
        camera_angle_deg=camera_angle_deg,
        heading_deg=heading_deg,
        hfov_deg=hfov_deg,
        output_size=ipm_size,
        output_gsd_m=ipm_gsd_m,
    )

    # Approximate centre of warped image on ground (lat/lon)
    lat0, lon0 = approx_lat, approx_lon
    lat_centre, lon_centre = _offset_latlon(lat0, lon0, center_east, center_north)

    # --- Find satellite tile ---
    tx, ty = find_tile(lat_centre, lon_centre, zoom)
    tile_path = satellite_dir / f"{zoom}_{tx}_{ty}.jpg"
    if not tile_path.exists():
        return result

    sat_img = cv2.imread(str(tile_path))
    if sat_img is None:
        return result
    meta = load_tile_meta(tile_path)

    # Resize satellite tile to same size as IPM output for matching
    sat_resized = cv2.resize(sat_img, (ipm_size, ipm_size))

    # Scale factor from tile pixels to our resized pixels
    tile_px_orig = sat_img.shape[1]  # typically 256
    scale = ipm_size / tile_px_orig

    # --- LightGlue matching ---
    feats0 = extract_features(warped, extractor, device)
    feats1 = extract_features(sat_resized, extractor, device)
    pts0, pts1 = match_pair(feats0, feats1, matcher, device)

    if len(pts0) < min_inliers:
        return result

    # --- RANSAC homography ---
    H_mat, inlier_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=4.0)
    if H_mat is None:
        return result

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(pts0) if len(pts0) > 0 else 0.0

    if inlier_count < min_inliers:
        return result

    # --- Map IPM centre to satellite pixel, then to lat/lon ---
    # Centre of IPM image (the ground point the camera looks at)
    ipm_centre = np.array([[[ipm_size / 2.0, ipm_size / 2.0]]], dtype=np.float32)
    sat_pt = cv2.perspectiveTransform(ipm_centre, H_mat)[0][0]

    # Convert back to original tile pixel coords
    sat_px = sat_pt[0] / scale
    sat_py = sat_pt[1] / scale

    lat_est, lon_est = tile_pixel_to_latlon(sat_px, sat_py, meta, tile_px=tile_px_orig)

    result.update({
        "status": "SAT_FIX",
        "lat": lat_est,
        "lon": lon_est,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
    })
    return result


def _offset_latlon(
    lat: float, lon: float, east_m: float, north_m: float
) -> tuple[float, float]:
    """Offset a lat/lon by east_m / north_m metres."""
    lat_new = lat + north_m / 111_320.0
    lon_new = lon + east_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat_new, lon_new


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


OUTPUT_FIELDS = [
    "query_frame_count",
    "query_frame_path",
    "vpr_status",
    "sat_status",
    "final_status",
    "est_lat",
    "est_lon",
    "gt_lat",
    "gt_lon",
    "error_m",
    "sat_inlier_count",
    "sat_inlier_ratio",
    "altitude_m",
    "camera_angle_deg",
    "heading_deg",
]


def run(
    results_csv: Path,
    confidence_csv: Path | None,
    query_manifest: Path,
    satellite_dir: Path,
    zoom: int,
    output_csv: Path,
    summary_json: Path,
    min_inliers: int = 15,
    ipm_size: int = 512,
    ipm_gsd_m: float = 0.5,
) -> None:
    device = choose_device()
    print(f"Device: {device}")

    extractor, matcher = load_matcher(device)

    # Load VPR results
    vpr_rows = {r["query_frame_count"]: r for r in load_csv(results_csv)}

    # Load confidence decisions (optional)
    if confidence_csv and confidence_csv.exists():
        conf_rows = {r["query_frame_count"]: r for r in load_csv(confidence_csv)}
    else:
        conf_rows = {}

    # Load query manifest (ground truth)
    manifest_rows = {r["frame_count"]: r for r in load_csv(query_manifest)}

    output_rows = []
    errors = []

    for frame_count, manifest in sorted(manifest_rows.items(), key=lambda x: int(x[0])):
        vpr = vpr_rows.get(frame_count, {})
        conf = conf_rows.get(frame_count, {})

        vpr_status = conf.get("decision", "UNKNOWN")  # FIX / NO_FIX
        gt_lat = float(manifest.get("ground_latitude") or "nan")
        gt_lon = float(manifest.get("ground_longitude") or "nan")
        altitude_m = float(manifest.get("rel_alt_m") or 0)
        camera_angle_deg = float(manifest.get("camera_angle_deg") or 60)
        heading_deg = float(manifest.get("heading_deg") or 0)
        frame_path = Path(manifest.get("frame_path", ""))

        # VPR estimate (always available from motion viterbi)
        vpr_lat = float(vpr.get("ref_ground_latitude") or "nan")
        vpr_lon = float(vpr.get("ref_ground_longitude") or "nan")

        sat_result = {"status": "SAT_SKIP", "lat": float("nan"), "lon": float("nan"),
                      "inlier_count": 0, "inlier_ratio": 0.0}

        # Run satellite localization for NO_FIX frames (or always if no confidence info)
        should_run_sat = (vpr_status == "NO_FIX") or (not conf_rows)
        if should_run_sat and frame_path.exists() and satellite_dir.exists():
            approx_lat = vpr_lat if math.isfinite(vpr_lat) else gt_lat
            approx_lon = vpr_lon if math.isfinite(vpr_lon) else gt_lon
            if math.isfinite(approx_lat) and math.isfinite(approx_lon):
                sat_result = localize_frame(
                    frame_path=frame_path,
                    approx_lat=approx_lat,
                    approx_lon=approx_lon,
                    altitude_m=altitude_m,
                    camera_angle_deg=camera_angle_deg,
                    heading_deg=heading_deg,
                    satellite_dir=satellite_dir,
                    zoom=zoom,
                    extractor=extractor,
                    matcher=matcher,
                    device=device,
                    ipm_size=ipm_size,
                    ipm_gsd_m=ipm_gsd_m,
                    min_inliers=min_inliers,
                )

        # Choose final estimate: FIX → VPR, NO_FIX + SAT_FIX → satellite, else VPR
        if vpr_status == "FIX":
            final_lat, final_lon, final_status = vpr_lat, vpr_lon, "VPR_FIX"
        elif sat_result["status"] == "SAT_FIX":
            final_lat, final_lon, final_status = sat_result["lat"], sat_result["lon"], "SAT_FIX"
        else:
            final_lat, final_lon, final_status = vpr_lat, vpr_lon, "VPR_NOFIX"

        # Error
        if math.isfinite(gt_lat) and math.isfinite(final_lat):
            error_m = haversine_m(gt_lat, gt_lon, final_lat, final_lon)
            errors.append((error_m, final_status))
        else:
            error_m = float("nan")

        output_rows.append({
            "query_frame_count": frame_count,
            "query_frame_path": str(frame_path),
            "vpr_status": vpr_status,
            "sat_status": sat_result["status"],
            "final_status": final_status,
            "est_lat": f"{final_lat:.8f}" if math.isfinite(final_lat) else "",
            "est_lon": f"{final_lon:.8f}" if math.isfinite(final_lon) else "",
            "gt_lat": f"{gt_lat:.8f}" if math.isfinite(gt_lat) else "",
            "gt_lon": f"{gt_lon:.8f}" if math.isfinite(gt_lon) else "",
            "error_m": f"{error_m:.2f}" if math.isfinite(error_m) else "",
            "sat_inlier_count": sat_result["inlier_count"],
            "sat_inlier_ratio": f"{sat_result['inlier_ratio']:.3f}",
            "altitude_m": f"{altitude_m:.1f}",
            "camera_angle_deg": f"{camera_angle_deg:.1f}",
            "heading_deg": f"{heading_deg:.1f}",
        })

        print(f"  frame {frame_count:4s}: {final_status:<12s}  "
              f"err={error_m:6.1f} m  sat_inliers={sat_result['inlier_count']}")

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    # Summary
    all_errors = [e for e, _ in errors]
    vpr_fix_errors = [e for e, s in errors if s == "VPR_FIX"]
    sat_fix_errors = [e for e, s in errors if s == "SAT_FIX"]
    vpr_nofix_errors = [e for e, s in errors if s == "VPR_NOFIX"]

    def stats(errs: list[float]) -> dict:
        if not errs:
            return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
        errs_s = sorted(errs)
        return {
            "count": len(errs_s),
            "mean": round(sum(errs_s) / len(errs_s), 2),
            "median": round(errs_s[len(errs_s) // 2], 2),
            "p90": round(errs_s[int(0.9 * len(errs_s))], 2),
            "max": round(max(errs_s), 2),
        }

    summary = {
        "total_frames": len(output_rows),
        "overall": stats(all_errors),
        "vpr_fix": stats(vpr_fix_errors),
        "sat_fix": stats(sat_fix_errors),
        "vpr_nofix_fallback": stats(vpr_nofix_errors),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Hybrid pipeline summary ===")
    print(f"Total frames : {summary['total_frames']}")
    print(f"VPR FIX      : {summary['vpr_fix']['count']}  mean={summary['vpr_fix']['mean']} m")
    print(f"SAT FIX      : {summary['sat_fix']['count']}  mean={summary['sat_fix']['mean']} m")
    print(f"VPR fallback : {summary['vpr_nofix_fallback']['count']}  mean={summary['vpr_nofix_fallback']['mean']} m")
    print(f"Overall mean : {summary['overall']['mean']} m")
    print(f"\nResults → {output_csv}")
    print(f"Summary → {summary_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True,
                        help="Motion Viterbi results CSV")
    parser.add_argument("--confidence-csv", type=Path, default=None,
                        help="Confidence gate decisions CSV (optional)")
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--satellite-dir", type=Path, default=Path("data/satellite"))
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_satellite_localize_results.csv"))
    parser.add_argument("--summary-json", type=Path,
                        default=Path("outputs/anyloc/dji_mini3_satellite_localize_summary.json"))
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--ipm-size", type=int, default=512)
    parser.add_argument("--ipm-gsd", type=float, default=0.5)
    args = parser.parse_args()

    run(
        results_csv=args.results_csv,
        confidence_csv=args.confidence_csv,
        query_manifest=args.query_manifest,
        satellite_dir=args.satellite_dir,
        zoom=args.zoom,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        min_inliers=args.min_inliers,
        ipm_size=args.ipm_size,
        ipm_gsd_m=args.ipm_gsd,
    )


if __name__ == "__main__":
    main()
