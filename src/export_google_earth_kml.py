"""Export navigation experiment paths to a Google Earth KML file."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def valid_coord(latitude: str, longitude: str) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return lat != 0.0 and lon != 0.0


def kml_coord(longitude: str, latitude: str, altitude: str = "0") -> str:
    return f"{float(longitude):.8f},{float(latitude):.8f},{float(altitude):.2f}"


def line_string(name: str, style_id: str, coords: list[str]) -> str:
    joined_coords = "\n            ".join(coords)
    return f"""    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style_id}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>
            {joined_coords}
        </coordinates>
      </LineString>
    </Placemark>"""


def point_placemark(
    name: str,
    style_id: str,
    latitude: str,
    longitude: str,
    description: str,
) -> str:
    return f"""    <Placemark>
      <name>{html.escape(name)}</name>
      <description>{html.escape(description)}</description>
      <styleUrl>#{style_id}</styleUrl>
      <Point>
        <coordinates>{kml_coord(longitude, latitude)}</coordinates>
      </Point>
    </Placemark>"""


def style(style_id: str, color: str, width: int = 4) -> str:
    return f"""    <Style id="{style_id}">
      <LineStyle>
        <color>{color}</color>
        <width>{width}</width>
      </LineStyle>
      <IconStyle>
        <scale>0.9</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href>
        </Icon>
      </IconStyle>
    </Style>"""


def build_reference_index(reference_manifests: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for manifest in reference_manifests:
        dataset = manifest.stem.replace("DJI_", "").replace("_frame_manifest_1fps", "")
        for row in read_csv(manifest):
            index[(dataset, row["frame_count"])] = row
    return index


def result_schema(result_rows: list[dict[str, str]]) -> tuple[str, str]:
    if not result_rows:
        raise ValueError("Retrieval results are empty.")

    columns = set(result_rows[0])
    if "motion_viterbi_reference_dataset" in columns:
        return "motion_viterbi", "Motion-Viterbi"
    if "temporal_reference_dataset" in columns:
        return "temporal", "Temporal"
    raise ValueError(
        "Unsupported retrieval result schema. Expected temporal_* or motion_viterbi_* columns."
    )


def export_kml(
    query_manifest: Path,
    retrieval_results: Path,
    reference_manifests: list[Path],
    output: Path,
    max_error_points: int,
) -> None:
    query_rows = read_csv(query_manifest)
    result_rows = read_csv(retrieval_results)
    result_prefix, result_label = result_schema(result_rows)
    reference_dataset_key = f"{result_prefix}_reference_dataset"
    reference_frame_key = f"{result_prefix}_reference_frame_count"
    error_key = f"{result_prefix}_position_error_m"
    query_by_frame = {row["frame_count"]: row for row in query_rows}
    reference_index = build_reference_index(reference_manifests)

    drone_path: list[str] = []
    true_center_path: list[str] = []
    estimated_center_path: list[str] = []
    matched_rows: list[tuple[dict[str, str], dict[str, str], dict[str, str]]] = []

    for row in query_rows:
        if valid_coord(row["drone_latitude"], row["drone_longitude"]):
            drone_path.append(kml_coord(row["drone_longitude"], row["drone_latitude"], "0"))
        if valid_coord(row["ground_latitude"], row["ground_longitude"]):
            true_center_path.append(kml_coord(row["ground_longitude"], row["ground_latitude"], "0"))

    for result in result_rows:
        query = query_by_frame.get(result["query_frame_count"])
        if query is None:
            continue
        reference = reference_index.get(
            (result[reference_dataset_key], result[reference_frame_key])
        )
        if reference is None:
            continue
        if not valid_coord(reference["ground_latitude"], reference["ground_longitude"]):
            continue
        estimated_center_path.append(
            kml_coord(reference["ground_longitude"], reference["ground_latitude"], "0")
        )
        matched_rows.append((result, query, reference))

    worst_rows = sorted(
        matched_rows,
        key=lambda items: float(items[0].get(error_key, "0") or 0),
        reverse=True,
    )[:max_error_points]

    placemarks = [
        line_string("Captured drone path from SRT GNSS", "dronePath", drone_path),
        line_string("Ground-truth video-center path from SRT projection", "trueCenterPath", true_center_path),
        line_string(
            f"Estimated video-center path from visual retrieval ({result_label})",
            "estimatedCenterPath",
            estimated_center_path,
        ),
    ]

    for index, (result, query, reference) in enumerate(worst_rows, start=1):
        error_m = float(result[error_key])
        estimated_latitude = reference["ground_latitude"]
        estimated_longitude = reference["ground_longitude"]
        description = (
            f"Query frame: {result['query_frame_count']}\\n"
            f"Matched reference: {result[reference_dataset_key]} "
            f"frame {result[reference_frame_key]}\\n"
            f"{result_label} error: {error_m:.2f} m\\n"
            f"True center: {query['ground_latitude']}, {query['ground_longitude']}\\n"
            f"Estimated center: {estimated_latitude}, {estimated_longitude}"
        )
        placemarks.append(
            point_placemark(
                f"Worst error #{index}: {error_m:.1f} m",
                "errorPoint",
                estimated_latitude,
                estimated_longitude,
                description,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>DJI Mini 3 Pro GNSS-denied navigation experiment</name>
{style("dronePath", "ffff7f00", 4)}
{style("trueCenterPath", "ff00aa00", 4)}
{style("estimatedCenterPath", "ff0000ff", 4)}
{style("errorPoint", "ff00ffff", 2)}
{chr(10).join(placemarks)}
  </Document>
</kml>
""",
        encoding="utf-8",
    )

    print(f"Wrote {output}")
    print(f"Drone path points: {len(drone_path)}")
    print(f"True center points: {len(true_center_path)}")
    print(f"Estimated center points: {len(estimated_center_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-error-points", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_kml(
        query_manifest=args.query_manifest,
        retrieval_results=args.retrieval_results,
        reference_manifests=args.reference_manifest,
        output=args.output,
        max_error_points=args.max_error_points,
    )


if __name__ == "__main__":
    main()
