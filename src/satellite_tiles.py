"""Download and manage satellite tiles from Esri World Imagery.

Tiles are cached in data/satellite/z_x_y.jpg with a companion JSON file
(data/satellite/z_x_y.json) containing the geographic bounding box.

Usage:
    python src/satellite_tiles.py --center-lat 32.1047 --center-lon 35.2077 \
        --radius-m 600 --zoom 18 --output-dir data/satellite
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Tile math (Web Mercator / Slippy Map)
# ---------------------------------------------------------------------------

def lon_to_tile_x(lon_deg: float, zoom: int) -> int:
    return int((lon_deg + 180.0) / 360.0 * (1 << zoom))


def lat_to_tile_y(lat_deg: float, zoom: int) -> int:
    lat_r = math.radians(lat_deg)
    return int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * (1 << zoom))


def tile_to_lon(tile_x: int, zoom: int) -> float:
    """West edge longitude of tile."""
    return tile_x / (1 << zoom) * 360.0 - 180.0


def tile_to_lat(tile_y: int, zoom: int) -> float:
    """North edge latitude of tile."""
    n = math.pi - 2.0 * math.pi * tile_y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))


def tile_bbox(tile_x: int, tile_y: int, zoom: int) -> dict[str, float]:
    """Return {north, south, west, east} lat/lon bounding box of a tile."""
    return {
        "north": tile_to_lat(tile_y, zoom),
        "south": tile_to_lat(tile_y + 1, zoom),
        "west":  tile_to_lon(tile_x, zoom),
        "east":  tile_to_lon(tile_x + 1, zoom),
    }


def tile_gsd_m(lat_deg: float, zoom: int, tile_px: int = 256) -> float:
    """Ground sample distance in metres/pixel for a tile at this latitude and zoom."""
    circumference_m = 2 * math.pi * 6_378_137.0
    return circumference_m * math.cos(math.radians(lat_deg)) / (tile_px * (1 << zoom))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; drone-nav-research/1.0)",
    "Referer": "https://www.arcgis.com/",
}


def download_tile(tile_x: int, tile_y: int, zoom: int, output_dir: Path,
                  retry: int = 3, delay: float = 0.5) -> Path:
    """Download one tile if not already cached. Returns local path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jpg_path = output_dir / f"{zoom}_{tile_x}_{tile_y}.jpg"
    json_path = output_dir / f"{zoom}_{tile_x}_{tile_y}.json"

    if jpg_path.exists() and json_path.exists():
        return jpg_path

    url = TILE_URL.format(z=zoom, x=tile_x, y=tile_y)
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            jpg_path.write_bytes(data)
            bbox = tile_bbox(tile_x, tile_y, zoom)
            bbox["zoom"] = zoom
            bbox["tile_x"] = tile_x
            bbox["tile_y"] = tile_y
            bbox["gsd_m"] = tile_gsd_m((bbox["north"] + bbox["south"]) / 2, zoom)
            json_path.write_text(json.dumps(bbox, indent=2))
            print(f"  Downloaded {jpg_path.name}")
            time.sleep(delay)
            return jpg_path
        except Exception as exc:
            print(f"  Attempt {attempt + 1}/{retry} failed for tile {zoom}/{tile_x}/{tile_y}: {exc}")
            time.sleep(delay * (attempt + 1))

    raise RuntimeError(f"Failed to download tile {zoom}/{tile_x}/{tile_y}")


def download_region(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int,
    output_dir: Path,
) -> list[Path]:
    """Download all tiles covering a circular region. Returns list of tile paths."""
    # Convert radius to degrees (approximate)
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))
    margin_lat = radius_m * lat_deg_per_m
    margin_lon = radius_m * lon_deg_per_m

    x_min = lon_to_tile_x(center_lon - margin_lon, zoom)
    x_max = lon_to_tile_x(center_lon + margin_lon, zoom)
    y_min = lat_to_tile_y(center_lat + margin_lat, zoom)  # smaller y = more north
    y_max = lat_to_tile_y(center_lat - margin_lat, zoom)

    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"Downloading {total} tiles at zoom {zoom} covering ±{radius_m:.0f}m around "
          f"({center_lat:.5f}, {center_lon:.5f})")

    paths = []
    for tx in range(x_min, x_max + 1):
        for ty in range(y_min, y_max + 1):
            paths.append(download_tile(tx, ty, zoom, output_dir))
    return paths


# ---------------------------------------------------------------------------
# Lookup: which tile covers a lat/lon?
# ---------------------------------------------------------------------------

def find_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    return lon_to_tile_x(lon, zoom), lat_to_tile_y(lat, zoom)


def load_tile_meta(tile_path: Path) -> dict:
    return json.loads(tile_path.with_suffix(".json").read_text())


def load_tile_mosaic(
    lat: float,
    lon: float,
    zoom: int,
    tile_dir: Path,
    grid: int = 3,
    tile_px: int = 256,
) -> tuple | None:
    """
    Load a grid×grid mosaic of tiles centred on (lat, lon).

    Returns (mosaic_img, meta) where meta has the same keys as a single tile
    (north/south/west/east/gsd_m/zoom) but covers the full mosaic extent,
    and pixel coordinates are relative to the full mosaic image.
    Returns None if the centre tile is missing.
    """
    import numpy as np
    import cv2

    cx, cy = find_tile(lat, lon, zoom)
    half = grid // 2                        # e.g. 1 for 3×3

    # Gather all required tile images
    rows = []
    for row_ty in range(cy - half, cy - half + grid):
        cols = []
        for col_tx in range(cx - half, cx - half + grid):
            p = tile_dir / f"{zoom}_{col_tx}_{row_ty}.jpg"
            if p.exists():
                img = cv2.imread(str(p))
                if img is None:
                    img = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
            else:
                img = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
            cols.append(img)
        rows.append(np.hstack(cols))
    mosaic = np.vstack(rows)

    # Bounding box of the mosaic
    tx0 = cx - half
    ty0 = cy - half
    tx1 = cx - half + grid      # exclusive column
    ty1 = cy - half + grid      # exclusive row
    meta = {
        "north":  tile_to_lat(ty0, zoom),
        "south":  tile_to_lat(ty1, zoom),
        "west":   tile_to_lon(tx0, zoom),
        "east":   tile_to_lon(tx1, zoom),
        "zoom":   zoom,
        "tile_px": tile_px * grid,
        "gsd_m":  tile_gsd_m(lat, zoom, tile_px),
    }
    return mosaic, meta


def latlon_to_tile_pixel(
    lat: float,
    lon: float,
    meta: dict,
    tile_px: int = 256,
) -> tuple[float, float]:
    """Convert a lat/lon to pixel coordinates within the tile."""
    px = (lon - meta["west"]) / (meta["east"] - meta["west"]) * tile_px
    py = (meta["north"] - lat) / (meta["north"] - meta["south"]) * tile_px
    return px, py


def tile_pixel_to_latlon(
    px: float,
    py: float,
    meta: dict,
    tile_px: int = 256,
) -> tuple[float, float]:
    """Convert tile pixel coordinates to lat/lon."""
    lon = meta["west"] + px / tile_px * (meta["east"] - meta["west"])
    lat = meta["north"] - py / tile_px * (meta["north"] - meta["south"])
    return lat, lon


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--radius-m", type=float, default=600.0,
                        help="Radius in metres around the centre point")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=Path("data/satellite"))
    args = parser.parse_args()

    paths = download_region(
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        radius_m=args.radius_m,
        zoom=args.zoom,
        output_dir=args.output_dir,
    )
    print(f"\nDone. {len(paths)} tiles in {args.output_dir}")
    gsd = tile_gsd_m(args.center_lat, args.zoom)
    print(f"GSD at this latitude: {gsd:.3f} m/pixel")


if __name__ == "__main__":
    main()
