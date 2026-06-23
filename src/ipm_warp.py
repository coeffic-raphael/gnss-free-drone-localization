"""Inverse Perspective Mapping (IPM) for tilted drone frames.

Warps a drone frame taken at a tilt angle into a pseudo-nadir (top-down) view
using the known altitude, camera angle, and heading.

Usage (standalone test):
    python src/ipm_warp.py \\
        data/processed/frames_v14_1fps/frame_000001.jpg \\
        --altitude 119 --camera-angle 60 --heading 45 \\
        --output /tmp/ipm_test.jpg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

# DJI Mini 3 Pro / Air 3 / Air 3S: 24 mm equiv, ~82° HFOV at 4:3 or 16:9
DEFAULT_HFOV_DEG = 82.0


def camera_matrix(img_w: int, img_h: int, hfov_deg: float = DEFAULT_HFOV_DEG) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for a pinhole camera."""
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * img_h / img_w)
    fx = img_w / (2.0 * math.tan(hfov / 2.0))
    fy = img_h / (2.0 * math.tan(vfov / 2.0))
    cx = img_w / 2.0
    cy = img_h / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------

def rotation_cam_to_enu(camera_angle_deg: float, heading_deg: float) -> np.ndarray:
    """
    Return 3×3 rotation matrix R such that  d_enu = R @ d_cam.

    Conventions:
    - ENU world frame: x = East, y = North, z = Up
    - Camera frame:    x = Right, y = Down, z = Forward
    - camera_angle_deg: angle below the horizon (0 = horizontal, 90 = nadir)
    - heading_deg: clockwise from north (0 = north, 90 = east)

    Camera axes in ENU:
        cam_x = ( cos(psi),  -sin(psi),   0          )   East when psi=0
        cam_y = (-sin(th)*sin(psi), -sin(th)*cos(psi), -cos(th))
        cam_z = ( cos(th)*sin(psi),  cos(th)*cos(psi), -sin(th))
    """
    th = math.radians(camera_angle_deg)
    psi = math.radians(heading_deg)
    st, ct = math.sin(th), math.cos(th)
    sp, cp = math.sin(psi), math.cos(psi)

    # Columns = camera axes expressed in ENU
    R = np.array([
        [ cp,  -st * sp,  ct * sp],   # row = ENU-x component
        [-sp,  -st * cp,  ct * cp],   # row = ENU-y component
        [ 0.0, -ct,      -st      ],   # row = ENU-z component
    ], dtype=np.float64)
    return R


# ---------------------------------------------------------------------------
# IPM core
# ---------------------------------------------------------------------------

def ipm_warp(
    img: np.ndarray,
    altitude_m: float,
    camera_angle_deg: float,
    heading_deg: float,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    output_size: int = 512,
    output_gsd_m: float = 0.5,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Warp a tilted drone frame to a top-down (nadir) view.

    Parameters
    ----------
    img : BGR image (H × W × 3)
    altitude_m : drone altitude above ground in metres
    camera_angle_deg : camera tilt below horizon in degrees (0 = horizontal, 90 = nadir)
    heading_deg : drone heading, clockwise from north
    hfov_deg : horizontal field of view of the camera
    output_size : side length of the square output image in pixels
    output_gsd_m : ground sample distance of the output (metres per pixel)

    Returns
    -------
    warped : top-down image (output_size × output_size × 3), blank where no coverage
    output_gsd_m : metres per pixel of the output (same as input arg)
    center_enu : (east_m, north_m) position of the image centre in drone-centric coords
                 (i.e. the ground point the camera is looking at)
    """
    H_img, W_img = img.shape[:2]
    fx, fy, cx, cy = camera_matrix(W_img, H_img, hfov_deg)

    # Camera-to-ENU rotation
    R_c2w = rotation_cam_to_enu(camera_angle_deg, heading_deg)
    R_w2c = R_c2w.T  # R is orthogonal → inverse = transpose

    # Ground point the camera centre is looking at (drone-centric ENU)
    th = math.radians(camera_angle_deg)
    if abs(math.tan(th)) < 1e-6:
        ground_dist = 0.0
    else:
        ground_dist = altitude_m / math.tan(th)
    psi = math.radians(heading_deg)
    center_east  = ground_dist * math.sin(psi)
    center_north = ground_dist * math.cos(psi)
    center_enu = (center_east, center_north)

    # Build output pixel grid
    half = output_size / 2.0
    # Output rows → north (row 0 = north edge), cols → east (col 0 = west edge)
    col_idx, row_idx = np.meshgrid(
        np.arange(output_size, dtype=np.float64),
        np.arange(output_size, dtype=np.float64),
    )
    # Ground coords in drone-centric ENU (metres)
    gx = (col_idx - half) * output_gsd_m + center_east   # east
    gy = (half - row_idx) * output_gsd_m + center_north  # north (inverted row)
    gz = np.full_like(gx, -altitude_m)                   # below drone

    # Stack to (3, N)
    d_world = np.stack([gx, gy, gz], axis=0).reshape(3, -1)

    # Rotate to camera frame
    d_cam = R_w2c @ d_world  # (3, N)

    # Keep only points in front of the camera (positive z_cam)
    valid = d_cam[2] > 1e-3

    map_x = np.where(valid, fx * d_cam[0] / d_cam[2] + cx, -1.0).reshape(output_size, output_size)
    map_y = np.where(valid, fy * d_cam[1] / d_cam[2] + cy, -1.0).reshape(output_size, output_size)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    warped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, output_gsd_m, center_enu


def ground_footprint_m(
    altitude_m: float,
    camera_angle_deg: float,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    img_w: int = 1920,
    img_h: int = 1080,
) -> tuple[float, float]:
    """Estimate ground footprint (width_m, height_m) of the camera view at this altitude."""
    th = math.radians(camera_angle_deg)
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * img_h / img_w)

    if abs(math.tan(th)) < 1e-6:
        return float("inf"), float("inf")

    # Width: perpendicular to heading, at centre-row distance
    centre_dist = altitude_m / math.tan(th)

    # Horizontal half-angle
    half_w = math.atan(math.tan(hfov / 2.0))
    width = 2.0 * centre_dist * math.tan(half_w)

    # Vertical (near/far) extent
    half_v = vfov / 2.0
    near_dist = altitude_m / math.tan(th + half_v) if th + half_v < math.pi / 2 else 0.0
    far_dist  = altitude_m / math.tan(th - half_v) if th - half_v > 0 else float("inf")
    height = min(far_dist, 2 * centre_dist) - near_dist

    return width, height


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--altitude", type=float, required=True, help="metres")
    parser.add_argument("--camera-angle", type=float, default=60.0,
                        help="degrees below horizon (default: 60)")
    parser.add_argument("--heading", type=float, default=0.0,
                        help="degrees clockwise from north (default: 0)")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--gsd", type=float, default=0.5,
                        help="output ground sample distance in m/pixel (default: 0.5)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/ipm_warped.jpg"))
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Cannot read {args.image}")

    warped, gsd, center = ipm_warp(
        img,
        altitude_m=args.altitude,
        camera_angle_deg=args.camera_angle,
        heading_deg=args.heading,
        hfov_deg=args.hfov,
        output_size=args.output_size,
        output_gsd_m=args.gsd,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), warped)
    print(f"Wrote {args.output}  ({args.output_size}×{args.output_size} px, {gsd} m/px)")
    print(f"Camera centre on ground: east={center[0]:.1f} m, north={center[1]:.1f} m from drone")

    w, h = ground_footprint_m(args.altitude, args.camera_angle, args.hfov)
    print(f"Estimated ground footprint: {w:.0f} m wide × {h:.0f} m tall")


if __name__ == "__main__":
    main()
