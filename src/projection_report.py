"""Create an SVG report for drone path and projected camera-center path."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_paths(csv_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], set[str], set[str]]:
    drone: list[tuple[float, float]] = []
    ground: list[tuple[float, float]] = []
    heading_sources: set[str] = set()
    camera_angle_sources: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            drone.append((float(row["drone_x_m"]), float(row["drone_y_m"])))
            ground.append((float(row["ground_x_m"]), float(row["ground_y_m"])))
            if row.get("heading_source"):
                heading_sources.add(row["heading_source"])
            if row.get("camera_angle_source"):
                camera_angle_sources.add(row["camera_angle_source"])
    return drone, ground, heading_sources, camera_angle_sources


def path_length_m(points: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def simplify(points: list[tuple[float, float]], max_points: int = 1200) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    return points[::stride]


def write_svg(
    drone: list[tuple[float, float]],
    ground: list[tuple[float, float]],
    heading_sources: set[str],
    camera_angle_sources: set[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_points = drone + ground
    width = 950
    height = 750
    margin = 80

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return px, py

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in map(project, simplify(points)))

    drone_points = polyline(drone)
    ground_points = polyline(ground)
    start_x, start_y = project(drone[0])
    end_x, end_y = project(drone[-1])
    ground_start_x, ground_start_y = project(ground[0])
    ground_end_x, ground_end_y = project(ground[-1])
    mean_offset = sum(
        math.hypot(gx - dx, gy - dy)
        for (dx, dy), (gx, gy) in zip(drone, ground)
    ) / len(drone)
    heading_text = ", ".join(sorted(heading_sources)) if heading_sources else "unknown"
    camera_angle_text = ", ".join(sorted(camera_angle_sources)) if camera_angle_sources else "unknown"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <text x="{margin}" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#202020">Drone path and projected camera-center ground path</text>
  <text x="{margin}" y="64" font-family="Arial, sans-serif" font-size="14" fill="#555">heading source: {heading_text} | camera angle source: {camera_angle_text} | mean drone-to-ground-center offset: {mean_offset:.1f} m</text>
  <rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#d0d0c8"/>
  <polyline points="{drone_points}" fill="none" stroke="#1b6ca8" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{ground_points}" fill="none" stroke="#d95f02" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>
  <line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{ground_start_x:.1f}" y2="{ground_start_y:.1f}" stroke="#777" stroke-width="1.5" stroke-dasharray="5 5"/>
  <line x1="{end_x:.1f}" y1="{end_y:.1f}" x2="{ground_end_x:.1f}" y2="{ground_end_y:.1f}" stroke="#777" stroke-width="1.5" stroke-dasharray="5 5"/>
  <circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="5" fill="#1a9850"/>
  <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="5" fill="#d73027"/>
  <circle cx="{ground_start_x:.1f}" cy="{ground_start_y:.1f}" r="5" fill="#1a9850"/>
  <circle cx="{ground_end_x:.1f}" cy="{ground_end_y:.1f}" r="5" fill="#d73027"/>
  <rect x="{margin}" y="{height - 58}" width="18" height="4" fill="#1b6ca8"/>
  <text x="{margin + 28}" y="{height - 51}" font-family="Arial, sans-serif" font-size="13" fill="#333">drone GNSS path, length {path_length_m(drone):.1f} m</text>
  <rect x="{margin + 260}" y="{height - 58}" width="18" height="4" fill="#d95f02"/>
  <text x="{margin + 288}" y="{height - 51}" font-family="Arial, sans-serif" font-size="13" fill="#333">projected camera-center ground path, length {path_length_m(ground):.1f} m</text>
  <text x="{margin}" y="{height - 25}" font-family="Arial, sans-serif" font-size="13" fill="#555">Projection assumes flat local ground and uses the camera center ray from DJI metadata when available.</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    drone, ground, heading_sources, camera_angle_sources = load_paths(args.input_csv)
    write_svg(drone, ground, heading_sources, camera_angle_sources, args.output_svg)
    print(f"drone_path_length_m: {path_length_m(drone):.2f}")
    print(f"ground_path_length_m: {path_length_m(ground):.2f}")
    print(f"wrote: {args.output_svg}")


if __name__ == "__main__":
    main()
