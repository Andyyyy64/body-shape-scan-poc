"""Canonical correspondence comparison. No date-based suppression or shape alignment."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import mean

from .geometry import number, polygon_perimeter
from .io import InputError


def validate_mesh(data: dict) -> dict:
    if data.get("schema_version") != "canonical_mesh.v1":
        raise InputError("Expected canonical_mesh.v1, not raw posed SAM output")
    if data.get("source_kind") not in {"real", "synthetic"} or data.get(
        "units"
    ) not in {"m", "model_m"}:
        raise InputError("Mesh source and units must be explicit")
    for key in (
        "reference_frame_id",
        "canonical_pose_id",
        "scale_reference_id",
        "producer",
    ):
        if not isinstance(data.get(key), str) or not data[key]:
            raise InputError(
                "Canonical pose, coordinate frame, scale provenance and producer are required"
            )
    vertices, faces = data["vertices"], data["faces"]
    if not 4 <= len(vertices) <= 100_000 or not 4 <= len(faces) <= 200_000:
        raise InputError("Mesh size is outside supported limits")
    for point in vertices:
        if not isinstance(point, list) or len(point) != 3:
            raise InputError("Vertices must contain three coordinates")
        for value in point:
            number(value, "vertex coordinate")
    for face in faces:
        if not isinstance(face, list) or len(face) != 3 or len(set(face)) != 3:
            raise InputError("Faces must be nondegenerate triangles")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(vertices)
            for i in face
        ):
            raise InputError("Invalid face index")
    region = data.get("comparison_vertex_indices")
    if region is not None:
        if (
            not region
            or len(region) != len(set(region))
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < len(vertices)
                for i in region
            )
        ):
            raise InputError("Invalid comparison region")
    definitions = data.get("sections", [])
    if len(definitions) > 20:
        raise InputError("Too many measurement sections")
    ids = set()
    for section in definitions:
        if (
            not isinstance(section.get("id"), str)
            or not section["id"]
            or section["id"] in ids
        ):
            raise InputError("Section identifiers must be unique")
        ids.add(section["id"])
        number(section["y"], "section plane")
        if len(section["center_xz"]) != 2:
            raise InputError("Section needs an interior reference point")
        for value in section["center_xz"]:
            number(value, "section center")
    return data


def topology_hash(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            [len(data["vertices"]), data["faces"]], separators=(",", ":")
        ).encode()
    ).hexdigest()


def section_perimeter(data: dict, section: dict) -> float | None:
    """Choose the closed surface-intersection loop containing the declared torso seed."""
    y = section["y"]
    edges = set()
    for indices in data["faces"]:
        triangle = [data["vertices"][i] for i in indices]
        offsets = [p[1] - y for p in triangle]
        if all(abs(d) < 1e-10 for d in offsets):
            return None  # A coplanar face is ambiguous; do not perturb the requested measurement plane.
        hits = set()
        for p, q in zip(triangle, triangle[1:] + triangle[:1]):
            a, b = p[1] - y, q[1] - y
            if abs(a) < 1e-10:
                hits.add((round(p[0], 10), round(p[2], 10)))
            if a * b < 0:
                t = a / (a - b)
                hits.add(
                    (
                        round(p[0] + t * (q[0] - p[0]), 10),
                        round(p[2] + t * (q[2] - p[2]), 10),
                    )
                )
        if len(hits) == 2:
            edges.add(tuple(sorted(hits)))
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    if not graph or any(len(neighbors) != 2 for neighbors in graph.values()):
        return None
    unseen = set(graph)
    loops = []
    while unseen:
        start = min(unseen)
        loop, previous, current = [], None, start
        while current in unseen:
            unseen.remove(current)
            loop.append(current)
            nxt = next(p for p in sorted(graph[current]) if p != previous)
            previous, current = current, nxt
        if current != start:
            return None
        loops.append(loop)
    sx, sz = section["center_xz"]
    containing = []
    for loop in loops:
        inside = False
        for (ax, az), (bx, bz) in zip(loop, loop[1:] + loop[:1]):
            if (az > sz) != (bz > sz) and sx < (bx - ax) * (sz - az) / (bz - az) + ax:
                inside = not inside
        if inside:
            containing.append(loop)
    if len(containing) != 1:
        return None
    return polygon_perimeter(containing[0])


def compare_meshes(
    before: dict, after: dict, tolerance_mm: float | None = None
) -> dict:
    validate_mesh(before)
    validate_mesh(after)
    if before.get("method") != after.get("method"):
        raise InputError("Compare reconstructions produced by the same method")
    for key in (
        "source_kind",
        "units",
        "reference_frame_id",
        "canonical_pose_id",
        "scale_reference_id",
    ):
        if before[key] != after[key]:
            raise InputError(
                "Mesh units, canonical coordinates, scale and source kind must match"
            )
    if topology_hash(before) != topology_hash(after):
        raise InputError("Meshes need identical vertex correspondence and topology")
    if before.get("comparison_vertex_indices") != after.get(
        "comparison_vertex_indices"
    ):
        raise InputError("Both meshes must use the same comparison region")
    if before.get("sections", []) != after.get("sections", []):
        raise InputError("Both meshes must use exactly the same measurement planes")
    if tolerance_mm is not None:
        tolerance_mm = number(tolerance_mm, "tolerance_mm", positive=True)
    distances = [
        1000 * math.dist(a, b) for a, b in zip(before["vertices"], after["vertices"])
    ]
    selected = [
        distances[i]
        for i in before.get("comparison_vertex_indices", range(len(distances)))
    ]
    ordered = sorted(selected)
    rms = math.sqrt(mean(d * d for d in selected))
    sections = []
    for definition in before.get("sections", []):
        a, b = (
            section_perimeter(before, definition),
            section_perimeter(after, definition),
        )
        sections.append(
            {
                "id": definition["id"],
                "before_mm": a * 1000 if a is not None else None,
                "after_mm": b * 1000 if b is not None else None,
                "delta_mm": (b - a) * 1000 if a is not None and b is not None else None,
            }
        )
    return {
        "schema_version": "mesh_comparison.v1",
        "source_kind": before["source_kind"],
        "units": "mm" if before["units"] == "m" else "model_mm",
        "alignment": "none_shared_canonical_frame",
        "vertex_count": len(distances),
        "comparison_vertex_count": len(selected),
        "metric_region": "torso"
        if before.get("comparison_vertex_indices")
        else "all_vertices",
        "rms_mm": rms,
        "mean_mm": mean(selected),
        "p95_mm": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_mm": max(selected),
        "vertex_distances_mm": distances,
        "sections": sections,
        "identical_geometry": max(distances) == 0,
        "duplicate_capture_input": bool(
            before.get("capture_sha256")
            and before.get("capture_sha256") == after.get("capture_sha256")
        ),
        "source_quality": [before.get("quality"), after.get("quality")],
        "tolerance_mm": tolerance_mm,
        "tolerance_result": "not_set"
        if tolerance_mm is None
        else (
            "within_user_tolerance" if rms <= tolerance_mm else "exceeds_user_tolerance"
        ),
        "biological_change": "not_assessed",
        "physical_accuracy_validated": False,
        "note": "Dates do not affect the calculation. Tolerance is user-supplied, not a validated repeatability limit.",
    }


def synthetic_mesh(local_delta_m: float = 0.0) -> dict:
    """An artificial closed torso, not derived from any person's data."""
    count, rings = 48, 19
    vertices = []
    for j in range(rings):
        y = j / (rings - 1) * 1.5
        radius = 0.15 + 0.03 * math.sin(math.pi * y / 1.5)
        bump = local_delta_m * math.exp(-(((y - 0.7) / 0.18) ** 2))
        for i in range(count):
            theta = i * 2 * math.pi / count
            vertices.append(
                [
                    (radius + bump) * math.cos(theta),
                    y,
                    (0.70 * radius + bump) * math.sin(theta),
                ]
            )
    faces = []
    for j in range(rings - 1):
        for i in range(count):
            a, b = j * count + i, j * count + (i + 1) % count
            faces.extend([[a, b, a + count], [b, b + count, a + count]])
    bottom, top = len(vertices), len(vertices) + 1
    vertices.extend([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    for i in range(count):
        faces.extend(
            [
                [bottom, (i + 1) % count, i],
                [top, (rings - 1) * count + i, (rings - 1) * count + (i + 1) % count],
            ]
        )
    return {
        "schema_version": "canonical_mesh.v1",
        "source_kind": "synthetic",
        "units": "model_m",
        "reference_frame_id": "synthetic_shared_frame.v1",
        "canonical_pose_id": "synthetic_upright.v1",
        "scale_reference_id": "synthetic_fixed_scale.v1",
        "producer": "synthetic_mesh_generator",
        "sections": [
            {"id": "synthetic_abdomen.v1", "y": 0.71, "center_xz": [0.0, 0.0]}
        ],
        "vertices": vertices,
        "faces": faces,
    }
