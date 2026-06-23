"""Preliminary experiment report: three-path comparison for Direction 4.

Generates an SVG that overlays:
  1. Drone GNSS path        — raw drone position from SRT telemetry
  2. Ground-truth center path — camera-center projected from angle + heading (SRT geometry)
  3. Estimated center path  — pipeline output (DINOv2 + LightGlue + Motion Viterbi + smoothing)

Usage:
    python src/preliminary_experiment_report.py \
        data/processed/DJI_v14_ground_projection_60deg.csv \
        data/processed/DJI_v14_frame_manifest_1fps.csv \
        outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \
        data/processed/DJI_v11_frame_manifest_1fps.csv \
        data/processed/DJI_v12_frame_manifest_1fps.csv \
        data/processed/DJI_v13_frame_manifest_1fps.csv \
        --smoothed-csv outputs/anyloc/dji_mini3_smoothed_results.csv \
        --output outputs/figures/preliminary_experiment_v14.svg
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EARTH_RADIUS_M = 6_378_137.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    x = (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0)) * EARTH_RADIUS_M
    y = (math.radians(lat) - math.radians(lat0)) * EARTH_RADIUS_M
    return x, y


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


# ---------------------------------------------------------------------------
# Build reference lookup {(dataset_id, frame_count) -> (ground_lat, ground_lon)}
# ---------------------------------------------------------------------------

def build_reference_lookup(manifest_paths: list[Path]) -> dict[tuple[str, int], tuple[float, float]]:
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for path in manifest_paths:
        rows = load_csv(path)
        # dataset id = stem up to first underscore after "DJI_", e.g. "v11"
        dataset_id = path.stem.split("_frame_manifest")[0].replace("DJI_", "")
        for row in rows:
            key = (dataset_id, int(row["frame_count"]))
            lookup[key] = (float(row["ground_latitude"]), float(row["ground_longitude"]))
    return lookup


# ---------------------------------------------------------------------------
# Build the three paths
# ---------------------------------------------------------------------------

def build_drone_path(proj_rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    """Raw drone GNSS positions (local XY metres, deduplicated)."""
    seen: set[tuple[float, float]] = set()
    points: list[tuple[float, float]] = []
    for row in proj_rows:
        pt = (float(row["drone_x_m"]), float(row["drone_y_m"]))
        if pt not in seen:
            seen.add(pt)
            points.append(pt)
    return points


def build_gt_center_path(proj_rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    """Camera-center ground point from SRT geometry (local XY metres, 1fps samples)."""
    # proj_rows is the full 30fps projection CSV; keep only 1-fps rows by
    # matching the frame counts present in the query manifest.
    return [(float(r["ground_x_m"]), float(r["ground_y_m"])) for r in proj_rows]


def build_estimated_path(
    results_rows: list[dict[str, str]],
    ref_lookup: dict[tuple[str, int], tuple[float, float]],
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float]]:
    """Estimated camera-center from pipeline retrieval output (Viterbi lookup)."""
    points: list[tuple[float, float]] = []
    for row in results_rows:
        dataset = row["motion_viterbi_reference_dataset"]
        frame = int(row["motion_viterbi_reference_frame_count"])
        key = (dataset, frame)
        if key not in ref_lookup:
            continue
        lat, lon = ref_lookup[key]
        points.append(latlon_to_xy(lat, lon, origin_lat, origin_lon))
    return points


def build_smoothed_path(
    smoothed_rows: list[dict[str, str]],
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float]]:
    """Estimated camera-center from Gaussian-smoothed Viterbi output."""
    rows = sorted(smoothed_rows, key=lambda r: int(r["query_frame_count"]))
    return [
        latlon_to_xy(float(r["smoothed_lat"]), float(r["smoothed_lon"]), origin_lat, origin_lon)
        for r in rows
        if r.get("smoothed_lat")
    ]


def build_gt_center_path_1fps(
    query_manifest: list[dict[str, str]],
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float]]:
    """Ground-truth camera-center at 1fps (from the frame manifest)."""
    return [
        latlon_to_xy(float(r["ground_latitude"]), float(r["ground_longitude"]), origin_lat, origin_lon)
        for r in query_manifest
    ]


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------

def compute_errors(
    gt: list[tuple[float, float]],
    est: list[tuple[float, float]],
) -> dict[str, float]:
    if not gt or not est:
        return {}
    n = min(len(gt), len(est))
    errors = [math.hypot(est[i][0] - gt[i][0], est[i][1] - gt[i][1]) for i in range(n)]
    errors_sorted = sorted(errors)
    return {
        "n": n,
        "mean": sum(errors) / n,
        "median": errors_sorted[n // 2],
        "p90": errors_sorted[int(0.9 * n)],
        "max": max(errors),
    }


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def svg_polyline(
    points: list[tuple[float, float]],
    project,  # callable
    color: str,
    width: float,
    dash: str = "",
    opacity: float = 1.0,
) -> str:
    pts = " ".join(f"{project(p)[0]:.1f},{project(p)[1]:.1f}" for p in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr} opacity="{opacity}"/>'


def svg_dot(cx: float, cy: float, r: float, color: str) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}"/>'


def svg_legend_item(
    x: float, y: float, color: str, label: str, dash: str = "", font_size: int = 13
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x}" y1="{y - 4}" x2="{x + 24}" y2="{y - 4}" stroke="{color}" '
        f'stroke-width="3"{dash_attr}/>'
        f'<text x="{x + 32}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{font_size}" fill="#333">{label}</text>'
    )


def write_svg(
    drone_path: list[tuple[float, float]],
    gt_path: list[tuple[float, float]],
    est_path: list[tuple[float, float]],
    stats: dict[str, float],
    output_path: Path,
    camera_angle_deg: float = 60.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W, H, M = 980, 780, 80

    all_points = drone_path + gt_path + est_path
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((W - 2 * M) / span_x, (H - 2 * M - 120) / span_y)

    def project(p: tuple[float, float]) -> tuple[float, float]:
        return (
            M + (p[0] - min_x) * scale,
            H - M - 110 - (p[1] - min_y) * scale,
        )

    lines: list[str] = []

    # Background
    lines.append(f'<rect width="100%" height="100%" fill="#f9f9f6"/>')
    lines.append(
        f'<rect x="{M}" y="{M}" width="{W - 2*M}" height="{H - 2*M - 110}" '
        f'fill="#ffffff" stroke="#cccccc" stroke-width="1"/>'
    )

    # Title & subtitle
    n = int(stats.get("n", 0))
    mean_e = stats.get("mean", 0)
    median_e = stats.get("median", 0)
    p90_e = stats.get("p90", 0)
    lines.append(
        f'<text x="{M}" y="38" font-family="Arial, sans-serif" font-size="20" '
        f'font-weight="700" fill="#202020">'
        f'Preliminary experiment — v14 path comparison (camera angle {camera_angle_deg:.0f}°)</text>'
    )
    lines.append(
        f'<text x="{M}" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">'
        f'{n} query frames | mean error {mean_e:.1f} m | median {median_e:.1f} m | P90 {p90_e:.1f} m'
        f'</text>'
    )

    # Paths
    lines.append(svg_polyline(drone_path, project, "#1b6ca8", 2.5, dash="6 4", opacity=0.7))
    lines.append(svg_polyline(gt_path, project, "#1a9850", 2.5))
    lines.append(svg_polyline(est_path, project, "#d73027", 2.5))

    # Start/end dots
    for path, color in [(drone_path, "#1b6ca8"), (gt_path, "#1a9850"), (est_path, "#d73027")]:
        if path:
            px, py = project(path[0])
            lines.append(svg_dot(px, py, 5, color))
            px, py = project(path[-1])
            lines.append(svg_dot(px, py, 5, color))

    # Error lines between GT and estimated
    n_lines = min(len(gt_path), len(est_path))
    for i in range(n_lines):
        gx, gy = project(gt_path[i])
        ex, ey = project(est_path[i])
        err = math.hypot(est_path[i][0] - gt_path[i][0], est_path[i][1] - gt_path[i][1])
        alpha = min(0.6, 0.1 + err / 60.0)
        lines.append(
            f'<line x1="{gx:.1f}" y1="{gy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="#888" stroke-width="1" opacity="{alpha:.2f}"/>'
        )

    # Legend
    legend_y = H - 80
    lines.append(svg_legend_item(M, legend_y, "#1b6ca8", "Drone GNSS path (SRT)", dash="6 4"))
    lines.append(svg_legend_item(M + 260, legend_y, "#1a9850", "Ground-truth camera-center (geometry)"))
    lines.append(svg_legend_item(M + 560, legend_y, "#d73027", "Estimated camera-center (Viterbi + smoothing w=19)"))
    lines.append(
        f'<text x="{M}" y="{H - 52}" font-family="Arial, sans-serif" font-size="12" fill="#777">'
        f'Grey lines connect ground-truth to estimated camera-center per frame. '
        f'Darker = larger error. Local XY coordinates in metres (origin = first SRT point).'
        f'</text>'
    )

    # Scale bar — 50 m
    bar_px = 50 * scale
    bar_x = W - M - bar_px - 10
    bar_y = H - M - 120 - 20
    lines.append(
        f'<line x1="{bar_x:.1f}" y1="{bar_y}" x2="{bar_x + bar_px:.1f}" y2="{bar_y}" '
        f'stroke="#444" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{bar_x + bar_px/2:.1f}" y="{bar_y - 6}" font-family="Arial, sans-serif" '
        f'font-size="12" fill="#444" text-anchor="middle">50 m</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">\n'
        + "\n".join(f"  {line}" for line in lines)
        + "\n</svg>\n"
    )
    output_path.write_text(svg, encoding="utf-8")
    print(f"wrote: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection_csv", type=Path,
                        help="Full-resolution projection CSV, e.g. DJI_v14_ground_projection_60deg.csv")
    parser.add_argument("query_manifest", type=Path,
                        help="1fps frame manifest for the query flight, e.g. DJI_v14_frame_manifest_1fps.csv")
    parser.add_argument("results_csv", type=Path,
                        help="Motion Viterbi results CSV from the pipeline")
    parser.add_argument("ref_manifests", type=Path, nargs="+",
                        help="1fps frame manifests for the reference flights (v11, v12, v13)")
    parser.add_argument("--smoothed-csv", type=Path, default=None,
                        help="Smoothed results CSV from smooth_path.py; if given, uses smoothed_lat/lon instead of Viterbi lookup")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/figures/preliminary_experiment_v14.svg"))
    args = parser.parse_args()

    proj_rows = load_csv(args.projection_csv)
    query_manifest = load_csv(args.query_manifest)
    results_rows = load_csv(args.results_csv)
    ref_lookup = build_reference_lookup(args.ref_manifests)

    # Origin = first point of the projection CSV
    origin_lat = float(proj_rows[0]["drone_latitude"])
    origin_lon = float(proj_rows[0]["drone_longitude"])

    # Convert projection CSV paths to local XY
    for row in proj_rows:
        x, y = latlon_to_xy(float(row["drone_latitude"]), float(row["drone_longitude"]), origin_lat, origin_lon)
        row["drone_x_m"] = str(x)
        row["drone_y_m"] = str(y)
        x, y = latlon_to_xy(float(row["ground_latitude"]), float(row["ground_longitude"]), origin_lat, origin_lon)
        row["ground_x_m"] = str(x)
        row["ground_y_m"] = str(y)

    drone_path = build_drone_path(proj_rows)
    gt_path = build_gt_center_path_1fps(query_manifest, origin_lat, origin_lon)

    if args.smoothed_csv is not None:
        smoothed_rows = load_csv(args.smoothed_csv)
        est_path = build_smoothed_path(smoothed_rows, origin_lat, origin_lon)
    else:
        est_path = build_estimated_path(results_rows, ref_lookup, origin_lat, origin_lon)

    stats = compute_errors(gt_path, est_path)
    print(f"query frames:   {len(gt_path)}")
    print(f"estimated:      {len(est_path)}")
    print(f"mean error:     {stats.get('mean', 0):.2f} m")
    print(f"median error:   {stats.get('median', 0):.2f} m")
    print(f"P90 error:      {stats.get('p90', 0):.2f} m")
    print(f"max error:      {stats.get('max', 0):.2f} m")
    print(f"drone path:     {path_length(drone_path):.1f} m")
    print(f"gt path length: {path_length(gt_path):.1f} m")
    print(f"est path len:   {path_length(est_path):.1f} m")

    write_svg(drone_path, gt_path, est_path, stats, args.output)


if __name__ == "__main__":
    main()
