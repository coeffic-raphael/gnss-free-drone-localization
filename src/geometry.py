"""Geometric helpers for drone telemetry and ground projection."""

from __future__ import annotations

import math


EARTH_RADIUS_M = 6_378_137.0


def gps_to_local_xy(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    """Approximate WGS84 coordinates as local east/north meters."""
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    x = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return x, y


def local_xy_to_gps(
    x: float,
    y: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    """Approximate local east/north meters as WGS84 coordinates."""
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    lat = y / EARTH_RADIUS_M + lat0
    lon = x / (math.cos(lat0) * EARTH_RADIUS_M) + lon0
    return math.degrees(lat), math.degrees(lon)


def heading_from_delta(dx: float, dy: float) -> float | None:
    """Return heading in degrees clockwise from north for an east/north vector."""
    if math.hypot(dx, dy) < 1e-6:
        return None
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def heading_unit_vector(heading_deg: float) -> tuple[float, float]:
    """Return an east/north unit vector from a heading angle."""
    radians = math.radians(heading_deg)
    return math.sin(radians), math.cos(radians)


def ground_distance_from_camera_angle(
    altitude_m: float,
    camera_angle_deg: float,
    angle_reference: str = "below_horizon",
) -> float:
    """Estimate horizontal ground distance from drone to camera center point.

    `below_horizon` means 0 degrees at the horizon and 90 degrees straight down.
    `from_nadir` means 0 degrees straight down and 90 degrees at the horizon.
    """
    if altitude_m <= 0:
        return 0.0

    if angle_reference == "below_horizon":
        angle = math.radians(camera_angle_deg)
        tangent = math.tan(angle)
        if abs(tangent) < 1e-9:
            return math.inf
        return altitude_m / tangent

    if angle_reference == "from_nadir":
        return altitude_m * math.tan(math.radians(camera_angle_deg))

    raise ValueError(f"Unsupported angle reference: {angle_reference}")
