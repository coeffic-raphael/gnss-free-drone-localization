"""Project the video-frame center point onto the ground plane."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from geometry import (
    gps_to_local_xy,
    ground_distance_from_camera_angle,
    heading_from_delta,
    heading_unit_vector,
    local_xy_to_gps,
)


OUTPUT_FIELDS = [
    "frame_count",
    "timestamp",
    "start_seconds",
    "drone_latitude",
    "drone_longitude",
    "drone_x_m",
    "drone_y_m",
    "rel_alt_m",
    "heading_deg",
    "heading_source",
    "camera_angle_deg",
    "camera_angle_source",
    "ground_distance_m",
    "ground_x_m",
    "ground_y_m",
    "ground_latitude",
    "ground_longitude",
]


def load_telemetry(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row.get("latitude") or not row.get("longitude"):
                continue
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if abs(latitude) <= 1e-9 or abs(longitude) <= 1e-9:
                continue
            rows.append(
                {
                    "frame_count": int(row["frame_count"]) if row.get("frame_count") else "",
                    "timestamp": row.get("timestamp") or row.get("gps_datetime", ""),
                    "start_seconds": float(row.get("start_seconds") or row.get("sample_time") or 0),
                    "latitude": latitude,
                    "longitude": longitude,
                    "rel_alt": float(row["rel_alt"]) if row.get("rel_alt") else math.nan,
                    "drone_yaw": _optional_float(row.get("drone_yaw")),
                    "gimbal_pitch": _optional_float(row.get("gimbal_pitch")),
                    "gimbal_yaw": _optional_float(row.get("gimbal_yaw")),
                }
            )
    return rows


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def normalize_heading(value: float) -> float:
    return (value + 360.0) % 360.0


def estimate_headings(
    xy: list[tuple[float, float]],
    window: int = 30,
    min_displacement_m: float = 2.0,
) -> list[float | None]:
    """Estimate heading from local trajectory using a centered frame window."""
    headings: list[float | None] = []
    for index, _point in enumerate(xy):
        before = max(0, index - window)
        after = min(len(xy) - 1, index + window)
        dx = xy[after][0] - xy[before][0]
        dy = xy[after][1] - xy[before][1]
        if math.hypot(dx, dy) < min_displacement_m:
            headings.append(None)
        else:
            headings.append(heading_from_delta(dx, dy))
    return fill_missing_headings(headings)


def fill_missing_headings(headings: list[float | None]) -> list[float | None]:
    last: float | None = None
    filled: list[float | None] = []
    for heading in headings:
        if heading is not None:
            last = heading
        filled.append(last)

    next_heading: float | None = None
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] is not None:
            next_heading = filled[index]
        elif next_heading is not None:
            filled[index] = next_heading
    return filled


def project_ground_points(
    rows: list[dict[str, float | str]],
    camera_angle_deg: float,
    angle_reference: str,
    heading_window: int,
    min_heading_displacement_m: float,
    heading_source: str,
    camera_angle_source: str,
) -> list[dict[str, float | str]]:
    if not rows:
        return []

    origin_lat = float(rows[0]["latitude"])
    origin_lon = float(rows[0]["longitude"])
    drone_xy = [
        gps_to_local_xy(
            float(row["latitude"]),
            float(row["longitude"]),
            origin_lat,
            origin_lon,
        )
        for row in rows
    ]
    headings = estimate_headings(
        drone_xy,
        window=heading_window,
        min_displacement_m=min_heading_displacement_m,
    )

    output_rows: list[dict[str, float | str]] = []
    for row, (x, y), estimated_heading in zip(rows, drone_xy, headings):
        altitude = float(row["rel_alt"])
        heading = select_heading(row, estimated_heading, heading_source)
        camera_angle = select_camera_angle(row, camera_angle_deg, camera_angle_source)
        ground_distance = ground_distance_from_camera_angle(
            altitude,
            camera_angle,
            angle_reference=angle_reference,
        )

        if heading is None or not math.isfinite(ground_distance):
            ground_x = math.nan
            ground_y = math.nan
            ground_lat = math.nan
            ground_lon = math.nan
        else:
            ux, uy = heading_unit_vector(heading)
            ground_x = x + ground_distance * ux
            ground_y = y + ground_distance * uy
            ground_lat, ground_lon = local_xy_to_gps(
                ground_x,
                ground_y,
                origin_lat,
                origin_lon,
            )

        output_rows.append(
            {
                "frame_count": row["frame_count"],
                "timestamp": row["timestamp"],
                "start_seconds": row["start_seconds"],
                "drone_latitude": row["latitude"],
                "drone_longitude": row["longitude"],
                "drone_x_m": x,
                "drone_y_m": y,
                "rel_alt_m": altitude,
                "heading_deg": heading if heading is not None else "",
                "heading_source": resolved_heading_source(row, heading_source),
                "camera_angle_deg": camera_angle,
                "camera_angle_source": resolved_camera_angle_source(row, camera_angle_source),
                "ground_distance_m": ground_distance,
                "ground_x_m": ground_x,
                "ground_y_m": ground_y,
                "ground_latitude": ground_lat,
                "ground_longitude": ground_lon,
            }
        )
    return output_rows


def select_heading(
    row: dict[str, float | str],
    estimated_heading: float | None,
    heading_source: str,
) -> float | None:
    if heading_source == "gimbal_yaw" and row.get("gimbal_yaw") is not None:
        return normalize_heading(float(row["gimbal_yaw"]))
    if heading_source == "drone_yaw" and row.get("drone_yaw") is not None:
        return normalize_heading(float(row["drone_yaw"]))
    if heading_source == "trajectory":
        return estimated_heading
    if row.get("gimbal_yaw") is not None:
        return normalize_heading(float(row["gimbal_yaw"]))
    if row.get("drone_yaw") is not None:
        return normalize_heading(float(row["drone_yaw"]))
    return estimated_heading


def select_camera_angle(
    row: dict[str, float | str],
    default_camera_angle_deg: float,
    camera_angle_source: str,
) -> float:
    if camera_angle_source == "gimbal_pitch" and row.get("gimbal_pitch") is not None:
        return abs(float(row["gimbal_pitch"]))
    if camera_angle_source == "fixed":
        return default_camera_angle_deg
    if row.get("gimbal_pitch") is not None:
        return abs(float(row["gimbal_pitch"]))
    return default_camera_angle_deg


def resolved_heading_source(row: dict[str, float | str], heading_source: str) -> str:
    if heading_source != "auto":
        return heading_source
    if row.get("gimbal_yaw") is not None:
        return "gimbal_yaw"
    if row.get("drone_yaw") is not None:
        return "drone_yaw"
    return "trajectory"


def resolved_camera_angle_source(row: dict[str, float | str], camera_angle_source: str) -> str:
    if camera_angle_source != "auto":
        return camera_angle_source
    if row.get("gimbal_pitch") is not None:
        return "gimbal_pitch"
    return "fixed"


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    distances = [float(row["ground_distance_m"]) for row in rows]
    headings = [float(row["heading_deg"]) for row in rows if row["heading_deg"] != ""]
    return {
        "rows": len(rows),
        "ground_distance_min_m": min(distances) if distances else 0.0,
        "ground_distance_max_m": max(distances) if distances else 0.0,
        "heading_samples": len(headings),
        "heading_min_deg": min(headings) if headings else 0.0,
        "heading_max_deg": max(headings) if headings else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--camera-angle-deg", type=float, default=45.0)
    parser.add_argument(
        "--angle-reference",
        choices=["below_horizon", "from_nadir"],
        default="below_horizon",
    )
    parser.add_argument("--heading-window", type=int, default=30)
    parser.add_argument("--min-heading-displacement-m", type=float, default=2.0)
    parser.add_argument(
        "--heading-source",
        choices=["auto", "gimbal_yaw", "drone_yaw", "trajectory"],
        default="auto",
    )
    parser.add_argument(
        "--camera-angle-source",
        choices=["auto", "gimbal_pitch", "fixed"],
        default="auto",
    )
    args = parser.parse_args()

    telemetry = load_telemetry(args.input_csv)
    projected = project_ground_points(
        telemetry,
        camera_angle_deg=args.camera_angle_deg,
        angle_reference=args.angle_reference,
        heading_window=args.heading_window,
        min_heading_displacement_m=args.min_heading_displacement_m,
        heading_source=args.heading_source,
        camera_angle_source=args.camera_angle_source,
    )
    write_csv(projected, args.output_csv)
    for key, value in summarize(projected).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
