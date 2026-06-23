"""Extract per-frame DJI MP4 metadata with ExifTool."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


FIELDS = [
    "sample_time",
    "gps_datetime",
    "latitude",
    "longitude",
    "rel_alt",
    "drone_roll",
    "drone_pitch",
    "drone_yaw",
    "gimbal_pitch",
    "gimbal_yaw",
]


EXIFTOOL_FORMAT = ",".join(
    [
        "$SampleTime",
        "$GPSDateTime",
        "$GPSLatitude",
        "$GPSLongitude",
        "$RelativeAltitude",
        "$DroneRoll",
        "$DronePitch",
        "$DroneYaw",
        "$GimbalPitch",
        "$GimbalYaw",
    ]
)


def require_exiftool() -> str:
    exiftool = shutil.which("exiftool")
    if not exiftool:
        raise SystemExit("exiftool is required. Install it with: brew install exiftool")
    return exiftool


def parse_value(value: str) -> float | str:
    value = value.strip()
    if value == "-":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def extract_metadata(mp4_path: Path) -> list[dict[str, float | int | str]]:
    command = [
        require_exiftool(),
        "-n",
        "-ee",
        "-p",
        EXIFTOOL_FORMAT,
        str(mp4_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    rows: list[dict[str, float | int | str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) != len(FIELDS):
            continue
        row = {field: parse_value(value) for field, value in zip(FIELDS, parts)}
        row["frame_count"] = len(rows) + 1
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_count", *FIELDS]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    def numeric(field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) != ""]

    summary: dict[str, object] = {"rows": len(rows)}
    for field in [
        "latitude",
        "longitude",
        "rel_alt",
        "drone_yaw",
        "gimbal_pitch",
        "gimbal_yaw",
    ]:
        values = numeric(field)
        summary[f"{field}_min"] = min(values) if values else None
        summary[f"{field}_max"] = max(values) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mp4", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = extract_metadata(args.input_mp4)
    write_csv(rows, args.output_csv)
    for key, value in summarize(rows).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
