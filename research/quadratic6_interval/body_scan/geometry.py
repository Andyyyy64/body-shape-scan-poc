"""Metric 2D cross-sections. Width integration returns convex-hull perimeter."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from .io import InputError


def number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise InputError(f"Invalid {name}")
    return value


def polygon_perimeter(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        raise InputError("A closed polygon needs at least three points")
    return sum(math.dist(a, b) for a, b in zip(points, points[1:] + points[:1]))


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set(points))
    if len(points) < 3:
        raise InputError("At least three distinct points required")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    hull = half(points)[:-1] + half(list(reversed(points)))[:-1]
    if len(hull) < 3:
        raise InputError("Collinear cross-section")
    return hull


def support_width(points: list[tuple[float, float]], angle_rad: float) -> float:
    values = [x * math.cos(angle_rad) + y * math.sin(angle_rad) for x, y in points]
    return max(values) - min(values)


def width_integral(observations: list[dict], max_gap_deg: float) -> dict:
    """Periodic trapezoidal integration; repeated angles do not gain weight."""
    max_gap = math.radians(number(max_gap_deg, "max_gap_deg", positive=True))
    if max_gap > math.pi / 2:
        raise InputError("Maximum allowed angular gap cannot exceed 90 degrees")
    grouped = defaultdict(list)
    for row in observations:
        angle = number(row["angle_rad"], "angle_rad") % math.pi
        angle = round(angle, 10)
        if abs(angle - math.pi) < 1e-9:
            angle = 0.0
        grouped[angle].append(number(row["width_m"], "width_m", positive=True))
    angles = sorted(grouped)
    if len(angles) < 3:
        raise InputError("At least three distinct directions required")
    widths = [mean(grouped[a]) for a in angles]
    gaps = [b - a for a, b in zip(angles, angles[1:] + [angles[0] + math.pi])]
    if max(gaps) > max_gap + 1e-9:
        raise InputError("Insufficient angular coverage")
    perimeter = sum(
        g * (a + b) / 2 for g, a, b in zip(gaps, widths, widths[1:] + widths[:1])
    )
    return {
        "value_m": perimeter,
        "unique_angles": len(angles),
        "max_gap_deg": math.degrees(max(gaps)),
        "perimeter_definition": "convex_hull",
    }


def ellipse_perimeter(a: float, b: float) -> float:
    a = number(a, "semi_axis_a_m", positive=True)
    b = number(b, "semi_axis_b_m", positive=True)
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def measure_widths(data: dict, max_gap_deg: float) -> dict:
    if data.get("schema_version") != "width_observations.v1":
        raise InputError("Expected width_observations.v1")
    if data.get("source_kind") not in {"synthetic", "real"}:
        raise InputError("source_kind must be synthetic or real")
    if (
        not isinstance(data.get("measurement_definition_id"), str)
        or not data["measurement_definition_id"]
    ):
        raise InputError("A measurement definition is required")
    if data.get("projection") != "metric_orthographic":
        raise InputError(
            "Raw perspective pixel widths cannot be integrated as metric widths"
        )
    scale_id = data.get("scale_evidence_id")
    if scale_id is not None and not isinstance(scale_id, str):
        raise InputError("scale_evidence_id must be a string or null")
    result = {
        "schema_version": "measurement.v1",
        "status": "needs_validation" if scale_id else "unscaled",
        "method": "cauchy_width_integral",
        "source_kind": data["source_kind"],
        "measurement_definition_id": data["measurement_definition_id"],
        "scale_evidence_id": scale_id,
        "physical_accuracy_validated": False,
        "uncertainty_m": None,
        "value_m": None,
    }
    if scale_id:
        result.update(width_integral(data["observations"], max_gap_deg))
    return result
