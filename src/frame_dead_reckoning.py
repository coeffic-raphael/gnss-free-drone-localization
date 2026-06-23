"""Estimate frame-to-frame drone displacement from optical flow (GNSS-free).

Uses SuperPoint + LightGlue to match consecutive query frames, then converts
the median pixel translation to ground meters using barometric altitude and
DJI Mini 3 Pro camera geometry.

Output: a CSV with one row per consecutive pair, containing estimated
dx_m, dy_m, speed_m_s and cumulative dead-reckoning position.

Usage:
    python src/frame_dead_reckoning.py \
        data/processed/DJI_v14_frame_manifest_1fps.csv \
        outputs/dead_reckoning/v14_dead_reckoning.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DJI Mini 3 Pro camera constants
# ---------------------------------------------------------------------------
# Horizontal field of view: 82.1 degrees (from DJI specs)
# Image width at 1080p extraction: 1920 px
HFOV_DEG = 82.1
IMAGE_WIDTH_PX = 1920
IMAGE_HEIGHT_PX = 1080
FOCAL_PX = (IMAGE_WIDTH_PX / 2) / math.tan(math.radians(HFOV_DEG / 2))  # ~1105 px

# Camera angle for Mini 3 Pro flights: 60 deg below horizon = 30 deg from nadir
CAMERA_ANGLE_FROM_NADIR_DEG = 30.0


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gsd_at_center(altitude_m: float, camera_from_nadir_deg: float, focal_px: float) -> float:
    """Ground sampling distance (m/px) at the image center for a tilted camera.

    Approximates the slant range to the ground point directly in front of the
    camera center ray, then divides by focal length.
    """
    if altitude_m <= 0:
        return 0.0
    slant_range = altitude_m / math.cos(math.radians(camera_from_nadir_deg))
    return slant_range / focal_px


def pixel_translation_from_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
) -> tuple[float, float, int]:
    """Return (dx_px, dy_px, n_matches) as median keypoint displacement."""
    if len(kpts0) < 4:
        return 0.0, 0.0, len(kpts0)
    dx = float(np.median(kpts1[:, 0] - kpts0[:, 0]))
    dy = float(np.median(kpts1[:, 1] - kpts0[:, 1]))
    return dx, dy, len(kpts0)


def run(manifest_path: Path, output_csv: Path, max_keypoints: int = 512) -> None:
    device = choose_device()
    print(f"device: {device}")

    rows = load_manifest(manifest_path)
    if len(rows) < 2:
        print("Need at least 2 frames.")
        return

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    cum_x, cum_y = 0.0, 0.0

    for i in tqdm(range(len(rows) - 1), desc="dead reckoning"):
        row0 = rows[i]
        row1 = rows[i + 1]

        path0 = Path(row0["frame_path"])
        path1 = Path(row1["frame_path"])

        if not path0.exists() or not path1.exists():
            continue

        alt0 = float(row0["rel_alt_m"])
        alt1 = float(row1["rel_alt_m"])
        alt_mean = (alt0 + alt1) / 2.0
        dt = float(row1["start_seconds"]) - float(row0["start_seconds"])
        if dt <= 0:
            dt = 1.0

        gsd = gsd_at_center(alt_mean, CAMERA_ANGLE_FROM_NADIR_DEG, FOCAL_PX)

        # Extract and match features
        with torch.no_grad():
            img0 = load_image(path0).to(device)
            img1 = load_image(path1).to(device)
            feats0 = extractor.extract(img0)
            feats1 = extractor.extract(img1)
            matches = matcher({"image0": feats0, "image1": feats1})

        kpts0_all = feats0["keypoints"][0].cpu().numpy()
        kpts1_all = feats1["keypoints"][0].cpu().numpy()
        match_indices = matches["matches"][0].cpu().numpy()  # shape (N, 2)

        if len(match_indices) >= 4:
            matched_kpts0 = kpts0_all[match_indices[:, 0]]
            matched_kpts1 = kpts1_all[match_indices[:, 1]]
            dx_px, dy_px, n_matches = pixel_translation_from_matches(matched_kpts0, matched_kpts1)
        else:
            dx_px, dy_px, n_matches = 0.0, 0.0, 0

        # Convert pixel shift to ground meters
        # dx_px > 0 means the scene moved right => drone moved left (west) in camera frame
        # We approximate: camera points forward (heading direction), x = right, y = down in image
        # Ground x (east-ish) ≈ -dx_px * gsd (scene moving right = drone moving left)
        # Ground y (forward)  ≈ -dy_px * gsd (scene moving down = drone moving forward)
        # Note: this is a first-order approximation ignoring rotation between frames
        dr_dx = -dx_px * gsd
        dr_dy = -dy_px * gsd
        speed = math.hypot(dr_dx, dr_dy) / dt

        cum_x += dr_dx
        cum_y += dr_dy

        results.append({
            "frame_a": row0["frame_count"],
            "frame_b": row1["frame_count"],
            "dt_s": f"{dt:.3f}",
            "alt_mean_m": f"{alt_mean:.1f}",
            "gsd_m_px": f"{gsd:.4f}",
            "dx_px": f"{dx_px:.2f}",
            "dy_px": f"{dy_px:.2f}",
            "dr_dx_m": f"{dr_dx:.3f}",
            "dr_dy_m": f"{dr_dy:.3f}",
            "speed_m_s": f"{speed:.2f}",
            "n_matches": n_matches,
            "cum_x_m": f"{cum_x:.2f}",
            "cum_y_m": f"{cum_y:.2f}",
        })

    fieldnames = list(results[0].keys()) if results else []
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {len(results)} rows to {output_csv}")

    # Print summary
    speeds = [float(r["speed_m_s"]) for r in results]
    drs = [math.hypot(float(r["dr_dx_m"]), float(r["dr_dy_m"])) for r in results]
    matches_list = [int(r["n_matches"]) for r in results]
    print(f"mean speed:        {sum(speeds)/len(speeds):.2f} m/s")
    print(f"mean step:         {sum(drs)/len(drs):.2f} m/frame")
    print(f"mean n_matches:    {sum(matches_list)/len(matches_list):.0f}")
    print(f"total path (DR):   {sum(drs):.1f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--max-keypoints", type=int, default=512)
    args = parser.parse_args()
    run(args.manifest, args.output_csv, args.max_keypoints)


if __name__ == "__main__":
    main()
