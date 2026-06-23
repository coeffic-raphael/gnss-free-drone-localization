"""Parse DJI SRT telemetry files into structured CSV rows."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
FRAME_RE = re.compile(
    r"(?:FrameCnt|SrtCnt)\s*:?\s*(?P<frame>\d+),\s*DiffTime\s*:?\s*(?P<diff>\d+)ms"
)
BRACKET_RE = re.compile(r"\[(?P<body>[^\]]+)\]")
DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+")


@dataclass(frozen=True)
class TelemetryRow:
    subtitle_index: int
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    frame_count: int | None
    diff_time_ms: int | None
    timestamp: str | None
    iso: float | None
    shutter: str | None
    fnum: float | None
    ev: float | None
    color_md: str | None
    focal_len: float | None
    latitude: float | None
    longitude: float | None
    rel_alt: float | None
    abs_alt: float | None
    ct: float | None


CSV_FIELDS = [field for field in TelemetryRow.__dataclass_fields__]


def srt_time_to_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def parse_scalar(value: str) -> float | str:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def parse_metadata_line(line: str) -> dict[str, float | str]:
    metadata: dict[str, float | str] = {}
    for match in BRACKET_RE.finditer(line):
        body = match.group("body").strip()
        parts = body.split()
        if len(parts) >= 4 and parts[0].endswith(":") and parts[2].endswith(":"):
            metadata[parts[0][:-1]] = parse_scalar(parts[1])
            metadata[parts[2][:-1]] = parse_scalar(parts[3])
        elif ":" in body:
            key, value = body.split(":", 1)
            metadata[key.strip()] = parse_scalar(value)
    return metadata


def parse_block(block: str) -> TelemetryRow | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[0].isdigit():
        return None

    time_match = TIME_RANGE_RE.search(lines[1])
    if not time_match:
        return None

    body = " ".join(lines[2:])
    frame_match = FRAME_RE.search(body)
    timestamp_match = DATETIME_RE.search(body)
    metadata = parse_metadata_line(body)

    start_time = time_match.group("start")
    end_time = time_match.group("end")

    return TelemetryRow(
        subtitle_index=int(lines[0]),
        start_time=start_time,
        end_time=end_time,
        start_seconds=srt_time_to_seconds(start_time),
        end_seconds=srt_time_to_seconds(end_time),
        frame_count=int(frame_match.group("frame")) if frame_match else None,
        diff_time_ms=int(frame_match.group("diff")) if frame_match else None,
        timestamp=timestamp_match.group(0) if timestamp_match else None,
        iso=_as_float(metadata.get("iso")),
        shutter=_as_str(metadata.get("shutter")),
        fnum=_as_float(metadata.get("fnum")),
        ev=_as_float(metadata.get("ev")),
        color_md=_as_str(metadata.get("color_md")),
        focal_len=_as_float(metadata.get("focal_len")),
        latitude=_as_float(metadata.get("latitude")),
        longitude=_as_float(metadata.get("longitude")),
        rel_alt=_as_float(metadata.get("rel_alt")),
        abs_alt=_as_float(metadata.get("abs_alt")),
        ct=_as_float(metadata.get("ct")),
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_srt(path: Path) -> list[TelemetryRow]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    return [row for block in blocks if (row := parse_block(block)) is not None]


def write_csv(rows: Iterable[TelemetryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[TelemetryRow]) -> dict[str, object]:
    latitudes = [row.latitude for row in rows if row.latitude is not None]
    longitudes = [row.longitude for row in rows if row.longitude is not None]
    valid_gps = [
        row
        for row in rows
        if row.latitude is not None
        and row.longitude is not None
        and abs(row.latitude) > 1e-9
        and abs(row.longitude) > 1e-9
    ]
    rel_alts = [row.rel_alt for row in rows if row.rel_alt is not None]
    timestamps = [
        datetime.fromisoformat(row.timestamp)
        for row in rows
        if row.timestamp is not None
    ]
    return {
        "rows": len(rows),
        "valid_gps_rows": len(valid_gps),
        "first_valid_gps_time": valid_gps[0].start_time if valid_gps else None,
        "duration_seconds": rows[-1].end_seconds if rows else 0,
        "first_timestamp": timestamps[0].isoformat(sep=" ") if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat(sep=" ") if timestamps else None,
        "latitude_min": min(latitudes) if latitudes else None,
        "latitude_max": max(latitudes) if latitudes else None,
        "longitude_min": min(longitudes) if longitudes else None,
        "longitude_max": max(longitudes) if longitudes else None,
        "rel_alt_min": min(rel_alts) if rel_alts else None,
        "rel_alt_max": max(rel_alts) if rel_alts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = parse_srt(args.input_srt)
    write_csv(rows, args.output_csv)

    for key, value in summarize(rows).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
