"""Visualise dead reckoning vs GNSS ground truth.

Reads the dead reckoning CSV produced by frame_dead_reckoning.py and the
query frame manifest (which contains the GNSS ground truth), then generates
an SVG comparing:
  - Cumulative dead reckoning path  (from optical flow)
  - GNSS drone path                 (ground truth, used only for evaluation)
  - GNSS camera-center path         (what we are actually trying to estimate)

This tells us how much drift the optical flow accumulates over the flight,
and whether it is good enough to tighten the Viterbi motion prior.

Usage:
    python src/dead_reckoning_report.py \
        outputs/dead_reckoning/v14_dead_reckoning.csv \
        data/processed/DJI_v14_frame_manifest_1fps.csv \
        outputs/figures/dead_reckoning_v14.svg
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EARTH_RADIUS_M = 6_378_137.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    x = (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0)) * EARTH_RADIUS_M
    y = (math.radians(lat) - math.radians(lat0)) * EARTH_RADIUS_M
    return x, y


def path_length(pts: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def final_drift(dr: list[tuple[float, float]], gnss: list[tuple[float, float]]) -> float:
    n = min(len(dr), len(gnss))
    if n == 0:
        return 0.0
    return math.hypot(dr[n - 1][0] - gnss[n - 1][0], dr[n - 1][1] - gnss[n - 1][1])


def mean_error(dr: list[tuple[float, float]], gnss: list[tuple[float, float]]) -> float:
    n = min(len(dr), len(gnss))
    if n == 0:
        return 0.0
    return sum(math.hypot(dr[i][0] - gnss[i][0], dr[i][1] - gnss[i][1]) for i in range(n)) / n


def write_svg(
    dr_path: list[tuple[float, float]],
    gnss_drone: list[tuple[float, float]],
    gnss_center: list[tuple[float, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W, H, M = 960, 760, 80

    all_pts = dr_path + gnss_drone + gnss_center
    xs, ys = [p[0] for p in all_pts], [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = min((W - 2 * M) / max(max_x - min_x, 1), (H - 2 * M - 110) / max(max_y - min_y, 1))

    def proj(p: tuple[float, float]) -> tuple[float, float]:
        return M + (p[0] - min_x) * scale, H - M - 110 - (p[1] - min_y) * scale

    def polyline(pts: list[tuple[float, float]], color: str, w: float, dash: str = "", opacity: float = 1.0) -> str:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        coords = " ".join(f"{proj(p)[0]:.1f},{proj(p)[1]:.1f}" for p in pts)
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{d} opacity="{opacity}"/>'

    n = min(len(dr_path), len(gnss_center))
    err_mean = mean_error(dr_path, gnss_center)
    err_final = final_drift(dr_path, gnss_center)

    lines = [
        f'<rect width="100%" height="100%" fill="#f9f9f6"/>',
        f'<rect x="{M}" y="{M}" width="{W-2*M}" height="{H-2*M-110}" fill="#fff" stroke="#ccc" stroke-width="1"/>',
        f'<text x="{M}" y="38" font-family="Arial, sans-serif" font-size="19" font-weight="700" fill="#202020">Dead reckoning vs GNSS — v14 (optical flow)</text>',
        f'<text x="{M}" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">{n} frames | mean DR drift vs camera-center: {err_mean:.1f} m | final drift: {err_final:.1f} m</text>',
        polyline(gnss_drone, "#1b6ca8", 2.0, dash="6 4", opacity=0.6),
        polyline(gnss_center, "#1a9850", 2.5),
        polyline(dr_path, "#e07b00", 2.5),
    ]

    # Drift lines every 5 frames
    for i in range(0, min(len(dr_path), len(gnss_center)), 5):
        ax, ay = proj(gnss_center[i])
        bx, by = proj(dr_path[i])
        err = math.hypot(dr_path[i][0] - gnss_center[i][0], dr_path[i][1] - gnss_center[i][1])
        alpha = min(0.7, 0.1 + err / 40.0)
        lines.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#c00" stroke-width="1" opacity="{alpha:.2f}"/>')

    # Start/end dots
    for path, color in [(gnss_drone, "#1b6ca8"), (gnss_center, "#1a9850"), (dr_path, "#e07b00")]:
        if path:
            px, py = proj(path[0])
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
            px, py = proj(path[-1])
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')

    # Scale bar 50m
    bar_px = 50 * scale
    bx = W - M - bar_px - 10
    by = H - M - 110 - 20
    lines += [
        f'<line x1="{bx:.1f}" y1="{by}" x2="{bx+bar_px:.1f}" y2="{by}" stroke="#444" stroke-width="2"/>',
        f'<text x="{bx+bar_px/2:.1f}" y="{by-6}" font-family="Arial, sans-serif" font-size="12" fill="#444" text-anchor="middle">50 m</text>',
    ]

    # Legend
    ly = H - 80
    for i, (color, label, dash) in enumerate([
        ("#1b6ca8", "Drone GNSS (ground truth, eval only)", "6 4"),
        ("#1a9850", "Camera-center GNSS (ground truth)", ""),
        ("#e07b00", "Dead reckoning from optical flow", ""),
    ]):
        lx = M + i * 310
        d = f' stroke-dasharray="{dash}"' if dash else ""
        lines += [
            f'<line x1="{lx}" y1="{ly-4}" x2="{lx+24}" y2="{ly-4}" stroke="{color}" stroke-width="3"{d}/>',
            f'<text x="{lx+32}" y="{ly}" font-family="Arial, sans-serif" font-size="12" fill="#333">{label}</text>',
        ]

    lines.append(f'<text x="{M}" y="{H-52}" font-family="Arial, sans-serif" font-size="12" fill="#777">Red ticks = drift between optical flow estimate and ground-truth camera-center. Darker = larger drift.</text>')

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    svg += "\n".join(f"  {l}" for l in lines)
    svg += "\n</svg>\n"
    output_path.write_text(svg, encoding="utf-8")
    print(f"wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dr_csv", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    dr_rows = load_csv(args.dr_csv)
    manifest = load_csv(args.manifest)

    if not manifest:
        print("Empty manifest.")
        return

    origin_lat = float(manifest[0]["drone_latitude"])
    origin_lon = float(manifest[0]["drone_longitude"])

    # GNSS paths from manifest
    gnss_drone = [latlon_to_xy(float(r["drone_latitude"]), float(r["drone_longitude"]), origin_lat, origin_lon) for r in manifest]
    gnss_center = [latlon_to_xy(float(r["ground_latitude"]), float(r["ground_longitude"]), origin_lat, origin_lon) for r in manifest]

    # Dead reckoning path — cumulative from first frame position
    start_x, start_y = gnss_center[0]  # anchor DR to first known point
    dr_path = [(start_x, start_y)]
    for row in dr_rows:
        prev = dr_path[-1]
        dr_path.append((prev[0] + float(row["dr_dx_m"]), prev[1] + float(row["dr_dy_m"])))

    print(f"DR path length:    {path_length(dr_path):.1f} m")
    print(f"GNSS center len:   {path_length(gnss_center):.1f} m")
    print(f"Mean DR drift:     {mean_error(dr_path, gnss_center):.2f} m")
    print(f"Final DR drift:    {final_drift(dr_path, gnss_center):.2f} m")

    write_svg(dr_path, gnss_drone, gnss_center, args.output_svg)


if __name__ == "__main__":
    main()
