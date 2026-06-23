"""Build an HTML debug page for the worst visual-retrieval errors."""

from __future__ import annotations

import argparse
import csv
import html
import math
import pickle
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_manifest_arg(value: str) -> tuple[Path, str]:
    if "=" not in value:
        path = Path(value)
        return path, path.stem
    dataset_id, path_text = value.split("=", 1)
    return Path(path_text), dataset_id


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    for row in rows:
        row["dataset_id"] = dataset_id
    return rows


def local_xy_from_latlon(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    earth_radius_m = 6_378_137.0
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    return (
        (lon - lon0) * math.cos(lat0) * earth_radius_m,
        (lat - lat0) * earth_radius_m,
    )


def ground_xy(row: dict[str, str], origin: tuple[float, float]) -> tuple[float, float]:
    return local_xy_from_latlon(
        float(row["ground_latitude"]),
        float(row["ground_longitude"]),
        origin[0],
        origin[1],
    )


def position_error_m(query: dict[str, str], reference: dict[str, str], origin: tuple[float, float]) -> float:
    qx, qy = ground_xy(query, origin)
    rx, ry = ground_xy(reference, origin)
    return math.hypot(qx - rx, qy - ry)


def image_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def format_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def candidate_map_svg(
    query: dict[str, str],
    candidates: list[tuple[dict[str, float | int], dict[str, str], float]],
    selected_frame_count: str,
    origin: tuple[float, float],
) -> str:
    points = [("true", *ground_xy(query, origin))]
    for candidate, reference, _error in candidates:
        label = "selected" if str(reference["frame_count"]) == selected_frame_count else "candidate"
        points.append((label, *ground_xy(reference, origin)))

    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    width = 320
    height = 220
    pad = 26

    def project(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - min_x) / span_x * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        return px, py

    marks = []
    for idx, (label, x, y) in enumerate(points):
        px, py = project(x, y)
        if label == "true":
            color = "#16a34a"
            radius = 7
            text = "GT"
        elif label == "selected":
            color = "#dc2626"
            radius = 6
            text = "SEL"
        else:
            color = "#2563eb"
            radius = 4
            text = str(idx)
        marks.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" />'
            f'<text x="{px + 7:.1f}" y="{py - 7:.1f}">{html.escape(text)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        '<rect x="0" y="0" width="320" height="220" rx="6" fill="#f8fafc" />'
        '<text x="12" y="20">local map: green=truth, red=selected, blue=top candidates</text>'
        + "".join(marks)
        + "</svg>"
    )


def build_page(
    reference_manifests: list[Path],
    query_manifest: Path,
    candidates_cache: Path,
    results_csv: Path,
    output_html: Path,
    worst_count: int,
    top_k: int,
) -> None:
    reference_rows: list[dict[str, str]] = []
    for manifest in reference_manifests:
        path, dataset_id = parse_manifest_arg(str(manifest))
        reference_rows.extend(load_manifest(path, dataset_id))

    query_path, query_dataset_id = parse_manifest_arg(str(query_manifest))
    query_rows = load_manifest(query_path, query_dataset_id)
    query_by_frame = {str(row["frame_count"]): row for row in query_rows}

    with candidates_cache.open("rb") as cache_file:
        candidates_by_query = pickle.load(cache_file)

    results = sorted(
        read_csv(results_csv),
        key=lambda row: float(row.get("motion_viterbi_position_error_m", "0") or 0),
        reverse=True,
    )[:worst_count]
    result_index_by_query_frame = {
        str(row["frame_count"]): idx for idx, row in enumerate(query_rows)
    }
    origin = (
        float(reference_rows[0]["ground_latitude"]),
        float(reference_rows[0]["ground_longitude"]),
    )

    sections: list[str] = []
    for worst_rank, result in enumerate(results, start=1):
        query = query_by_frame[str(result["query_frame_count"])]
        query_idx = result_index_by_query_frame[str(result["query_frame_count"])]
        top_candidates = candidates_by_query[query_idx][:top_k]
        selected_frame_count = str(result["motion_viterbi_reference_frame_count"])
        candidate_rows = []
        candidate_triplets = []

        for candidate in top_candidates:
            reference = reference_rows[int(candidate["reference_index"])]
            error_m = position_error_m(query, reference, origin)
            selected_class = " selected" if str(reference["frame_count"]) == selected_frame_count else ""
            candidate_triplets.append((candidate, reference, error_m))
            candidate_rows.append(
                f"""
                <article class="candidate{selected_class}">
                  <img src="{image_uri(reference['frame_path'])}" alt="reference candidate">
                  <div class="meta">
                    <strong>rank {int(candidate['rank'])} - {html.escape(reference['dataset_id'])}
                    frame {html.escape(reference['frame_count'])}</strong>
                    <span>error {error_m:.2f} m</span>
                    <span>DINO {format_float(candidate['dino_similarity'])}</span>
                    <span>LG inliers {int(candidate['lg_inlier_count'])}</span>
                    <span>LG ratio {format_float(candidate['lg_inlier_ratio'])}</span>
                  </div>
                </article>
                """
            )

        sections.append(
            f"""
            <section class="case">
              <header>
                <div>
                  <h2>Worst #{worst_rank} - query frame {html.escape(result['query_frame_count'])}</h2>
                  <p>
                    selected {html.escape(result['motion_viterbi_reference_dataset'])}
                    frame {html.escape(result['motion_viterbi_reference_frame_count'])},
                    error {float(result['motion_viterbi_position_error_m']):.2f} m,
                    selected rank {html.escape(result['motion_viterbi_rank'])}
                  </p>
                </div>
              </header>
              <div class="case-grid">
                <div class="query">
                  <img src="{image_uri(query['frame_path'])}" alt="query frame">
                  <p>Query / ground truth center:
                  {float(query['ground_latitude']):.7f}, {float(query['ground_longitude']):.7f}</p>
                </div>
                <div class="map">
                  {candidate_map_svg(query, candidate_triplets, selected_frame_count, origin)}
                </div>
              </div>
              <div class="candidates">
                {''.join(candidate_rows)}
              </div>
            </section>
            """
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Visual Retrieval Debug</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111827;
      background: #f1f5f9;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, p {{
      margin: 0;
    }}
    .intro {{
      margin-bottom: 22px;
    }}
    .intro p {{
      margin-top: 8px;
      color: #475569;
    }}
    .case {{
      margin: 0 0 28px;
      padding: 18px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #fff;
    }}
    .case header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }}
    .case header p {{
      margin-top: 6px;
      color: #475569;
    }}
    .case-grid {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 340px;
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
    }}
    .query p {{
      margin-top: 8px;
      color: #475569;
      font-size: 13px;
    }}
    svg {{
      width: 100%;
      height: auto;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
    }}
    svg text {{
      font-size: 10px;
      fill: #334155;
    }}
    .candidates {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .candidate {{
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
      background: #f8fafc;
    }}
    .candidate.selected {{
      border: 3px solid #dc2626;
    }}
    .meta {{
      display: grid;
      gap: 4px;
      padding: 10px;
      font-size: 13px;
      color: #334155;
    }}
    .meta strong {{
      color: #111827;
    }}
  </style>
</head>
<body>
  <main>
    <div class="intro">
      <h1>Worst visual-retrieval errors</h1>
      <p>Top {top_k} DINO candidates with LightGlue scores. Red border means the Motion-Viterbi selected candidate.</p>
    </div>
    {''.join(sections)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", action="append", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--candidates-cache", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--worst-count", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_page(
        reference_manifests=args.reference_manifest,
        query_manifest=args.query_manifest,
        candidates_cache=args.candidates_cache,
        results_csv=args.results_csv,
        output_html=args.output_html,
        worst_count=args.worst_count,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
