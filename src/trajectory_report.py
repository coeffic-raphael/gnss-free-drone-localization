"""Create a lightweight trajectory report from parsed telemetry CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EARTH_RADIUS_M = 6_378_137.0


def load_points(csv_path: Path) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row.get("latitude") or not row.get("longitude"):
                continue
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if abs(latitude) <= 1e-9 or abs(longitude) <= 1e-9:
                continue
            points.append(
                {
                    "time": float(row["start_seconds"]),
                    "lat": latitude,
                    "lon": longitude,
                    "rel_alt": float(row["rel_alt"]) if row.get("rel_alt") else math.nan,
                }
            )
    return points


def to_local_xy(points: list[dict[str, float]]) -> list[tuple[float, float]]:
    if not points:
        return []

    lat0 = math.radians(points[0]["lat"])
    lon0 = math.radians(points[0]["lon"])
    cos_lat0 = math.cos(lat0)

    xy: list[tuple[float, float]] = []
    for point in points:
        lat = math.radians(point["lat"])
        lon = math.radians(point["lon"])
        x = (lon - lon0) * cos_lat0 * EARTH_RADIUS_M
        y = (lat - lat0) * EARTH_RADIUS_M
        xy.append((x, y))
    return xy


def path_length_m(xy: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(xy, xy[1:])
    )


def simplify_for_svg(xy: list[tuple[float, float]], max_points: int = 1200) -> list[tuple[float, float]]:
    if len(xy) <= max_points:
        return xy
    stride = math.ceil(len(xy) / max_points)
    return xy[::stride]


def write_svg(points: list[dict[str, float]], xy: list[tuple[float, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not xy:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" />\n", encoding="utf-8")
        return

    width = 900
    height = 700
    margin = 70
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
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

    svg_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(project, simplify_for_svg(xy)))
    start_x, start_y = project(xy[0])
    end_x, end_y = project(xy[-1])
    distance = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
    length = path_length_m(xy)
    duration = points[-1]["time"] - points[0]["time"]
    rel_alts = [p["rel_alt"] for p in points if not math.isnan(p["rel_alt"])]
    alt_text = (
        f"{min(rel_alts):.1f} m to {max(rel_alts):.1f} m"
        if rel_alts
        else "not available"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <text x="{margin}" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#202020">GNSS trajectory from SRT telemetry</text>
  <text x="{margin}" y="64" font-family="Arial, sans-serif" font-size="14" fill="#555">duration: {duration:.2f}s | samples: {len(points)} | path length: {length:.1f}m | displacement: {distance:.1f}m | relative altitude: {alt_text}</text>
  <rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#d0d0c8"/>
  <polyline points="{svg_points}" fill="none" stroke="#1b6ca8" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="6" fill="#1a9850"/>
  <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="6" fill="#d73027"/>
  <text x="{start_x + 10:.1f}" y="{start_y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#1a9850">start</text>
  <text x="{end_x + 10:.1f}" y="{end_y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#d73027">end</text>
  <text x="{margin}" y="{height - 25}" font-family="Arial, sans-serif" font-size="13" fill="#555">Local coordinates are approximated from the first GNSS sample. X: east-west meters, Y: north-south meters.</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    points = load_points(args.input_csv)
    xy = to_local_xy(points)
    write_svg(points, xy, args.output_svg)
    print(f"samples: {len(points)}")
    print(f"path_length_m: {path_length_m(xy):.2f}")
    if xy:
        print(f"displacement_m: {math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]):.2f}")
    print(f"wrote: {args.output_svg}")


if __name__ == "__main__":
    main()
