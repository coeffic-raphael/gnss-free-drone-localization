"""Export hybrid localisation results to a Google Earth KML file.

Reads outputs/hybrid/hybrid_results_v14.csv and produces a KML with:
  - Ground-truth path (green dashed)
  - Estimated path coloured by status:
      VPR_FIX      → green   (#ff00aa00)
      SAT_FIX      → blue    (#ffff5500)  KML uses AABBGGRR
      VPR_FALLBACK → orange  (#ff0055ff)
  - Error lines connecting GT to estimate for the worst frames
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


# ---------------------------------------------------------------------------
# KML helpers
# ---------------------------------------------------------------------------

def kml_style(style_id: str, abgr: str, width: int = 3) -> str:
    return f"""    <Style id="{style_id}">
      <LineStyle><color>{abgr}</color><width>{width}</width></LineStyle>
      <PolyStyle><fill>0</fill></PolyStyle>
    </Style>"""


def kml_icon_style(style_id: str, abgr: str, scale: float = 0.6) -> str:
    return f"""    <Style id="{style_id}">
      <IconStyle>
        <color>{abgr}</color>
        <scale>{scale}</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
      <LabelStyle><scale>0</scale></LabelStyle>
    </Style>"""


def line_string(name: str, style_id: str, coords: list[str]) -> str:
    joined = "\n            ".join(coords)
    return f"""    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style_id}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>
            {joined}
        </coordinates>
      </LineString>
    </Placemark>"""


def point_placemark(name: str, style_id: str, lat: str, lon: str, desc: str) -> str:
    return f"""    <Placemark>
      <name>{html.escape(name)}</name>
      <description>{html.escape(desc)}</description>
      <styleUrl>#{style_id}</styleUrl>
      <Point>
        <coordinates>{float(lon):.8f},{float(lat):.8f},0</coordinates>
      </Point>
    </Placemark>"""


def error_line(lat1: str, lon1: str, lat2: str, lon2: str, style_id: str) -> str:
    c1 = f"{float(lon1):.8f},{float(lat1):.8f},0"
    c2 = f"{float(lon2):.8f},{float(lat2):.8f},0"
    return f"""    <Placemark>
      <styleUrl>#{style_id}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{c1} {c2}</coordinates>
      </LineString>
    </Placemark>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STATUS_STYLE = {
    "VPR_FIX":      "vprFix",
    "SAT_FIX":      "satFix",
    "VPR_FALLBACK": "vprFallback",
    "INTERP":       "vprFallback",
}

# KML colours are AABBGGRR (alpha, blue, green, red)
STYLES = [
    kml_style("gtPath",       "ff88aa88", width=2),   # grey-green dashed GT
    kml_style("vprFixLine",   "ff00aa00", width=3),   # green
    kml_style("satFixLine",   "ffff5500", width=3),   # blue (AABBGGRR: ff+55+00+ff → orange in KML means blue)
    kml_style("vprFbLine",    "ff0055ff", width=3),   # orange
    kml_style("errorLine",    "44ffffff", width=1),   # faint white
    kml_icon_style("vprFix",      "ff00aa00"),
    kml_icon_style("satFix",      "ffff5500"),
    kml_icon_style("vprFallback", "ff0055ff"),
    kml_icon_style("gtDot",       "ff888888", scale=0.4),
]


def export_hybrid_kml(hybrid_csv: Path, output: Path, max_error_points: int = 15) -> None:
    rows = list(csv.DictReader(hybrid_csv.open()))

    gt_coords, est_by_status, all_points = [], {}, []
    for r in rows:
        gt_coords.append(f"{float(r['gt_lon']):.8f},{float(r['gt_lat']):.8f},0")
        s = r["final_status"]
        est_by_status.setdefault(s, []).append(
            f"{float(r['final_lon']):.8f},{float(r['final_lat']):.8f},0"
        )
        all_points.append(r)

    # Build per-status paths
    placemarks = []
    placemarks.append(line_string("Ground truth path", "gtPath", gt_coords))

    label_map = {
        "VPR_FIX":      ("VPR_FIX path (smoothed VPR, 9.1 m median)", "vprFixLine"),
        "SAT_FIX":      ("SAT_FIX path (satellite, 12.1 m median)",    "satFixLine"),
        "VPR_FALLBACK": ("VPR_FALLBACK path (raw VPR, 14.8 m median)", "vprFbLine"),
        "INTERP":       ("INTERP path",                                  "vprFbLine"),
    }
    for status, (name, sty) in label_map.items():
        if status in est_by_status:
            placemarks.append(line_string(name, sty, est_by_status[status]))

    # Individual points coloured by status
    for r in all_points:
        s = r["final_status"]
        sty = STATUS_STYLE.get(s, "vprFallback")
        err = float(r["final_error_m"])
        desc = (
            f"Frame: {r['frame_count']}  t={r['time_s']}s\n"
            f"Status: {s}\n"
            f"Error: {err:.1f} m\n"
            f"GT: {r['gt_lat']}, {r['gt_lon']}\n"
            f"Est: {r['final_lat']}, {r['final_lon']}"
        )
        placemarks.append(point_placemark(
            f"{s} #{r['frame_count']} ({err:.1f} m)", sty,
            r["final_lat"], r["final_lon"], desc
        ))

    # Error lines for worst frames
    worst = sorted(all_points, key=lambda r: float(r["final_error_m"]), reverse=True)
    for r in worst[:max_error_points]:
        placemarks.append(error_line(
            r["gt_lat"], r["gt_lon"], r["final_lat"], r["final_lon"], "errorLine"
        ))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Hybrid localisation — v14 (VPR + satellite)</name>
{chr(10).join(STYLES)}
{chr(10).join(placemarks)}
  </Document>
</kml>
""",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"  {len(gt_coords)} GT points")
    for s, pts in est_by_status.items():
        print(f"  {s}: {len(pts)} estimated points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-csv", type=Path,
                        default=Path("outputs/hybrid/hybrid_results_v14.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/maps/dji_mini3_v14_hybrid.kml"))
    parser.add_argument("--max-error-points", type=int, default=15)
    args = parser.parse_args()
    export_hybrid_kml(args.hybrid_csv, args.output, args.max_error_points)


if __name__ == "__main__":
    main()
