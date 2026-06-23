"""Build a frame manifest by joining extracted frames with projection metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "frame_count",
    "frame_path",
    "timestamp",
    "start_seconds",
    "drone_latitude",
    "drone_longitude",
    "rel_alt_m",
    "heading_deg",
    "heading_source",
    "camera_angle_deg",
    "camera_angle_source",
    "ground_latitude",
    "ground_longitude",
]


def load_projection_rows(projection_csv: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with projection_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows[int(row["frame_count"])] = row
    return rows


def frame_number_from_name(path: Path) -> int | None:
    digits = "".join(char for char in path.stem if char.isdigit())
    if not digits:
        return None
    return int(digits)


def build_manifest(
    frames_dir: Path,
    projection_csv: Path,
    output_csv: Path,
    frame_ext: str,
    fps: float,
    time_offset_s: float,
) -> int:
    projection_rows = load_projection_rows(projection_csv)
    min_seconds = min(float(row["start_seconds"]) for row in projection_rows.values())
    max_seconds = max(float(row["start_seconds"]) for row in projection_rows.values())
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for frame_path in sorted(frames_dir.glob(f"*.{frame_ext.lstrip('.')}")):
            extracted_index = frame_number_from_name(frame_path)
            if extracted_index is None:
                continue
            if fps > 0:
                seconds = time_offset_s + (extracted_index - 1) / fps
                if seconds < min_seconds or seconds > max_seconds:
                    continue
                frame_count = min(
                    projection_rows,
                    key=lambda key: abs(float(projection_rows[key]["start_seconds"]) - seconds),
                )
            else:
                frame_count = extracted_index
            if frame_count not in projection_rows:
                continue
            row = projection_rows[frame_count]
            writer.writerow(
                {
                    "frame_count": frame_count,
                    "frame_path": str(frame_path),
                    "timestamp": row["timestamp"],
                    "start_seconds": row["start_seconds"],
                    "drone_latitude": row["drone_latitude"],
                    "drone_longitude": row["drone_longitude"],
                    "rel_alt_m": row["rel_alt_m"],
                    "heading_deg": row["heading_deg"],
                    "heading_source": row["heading_source"],
                    "camera_angle_deg": row["camera_angle_deg"],
                    "camera_angle_source": row["camera_angle_source"],
                    "ground_latitude": row["ground_latitude"],
                    "ground_longitude": row["ground_longitude"],
                }
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("projection_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--frame-ext", default="jpg")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Extraction FPS. If set, frame names are treated as extracted-frame indices.",
    )
    parser.add_argument(
        "--time-offset-s",
        type=float,
        default=0.0,
        help="Video timestamp of the first extracted frame.",
    )
    args = parser.parse_args()

    count = build_manifest(
        args.frames_dir,
        args.projection_csv,
        args.output_csv,
        args.frame_ext,
        args.fps,
        args.time_offset_s,
    )
    print(f"manifest_rows: {count}")
    print(f"wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
