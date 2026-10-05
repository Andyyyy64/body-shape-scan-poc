"""Estimated mesh silhouette primitives, guided by its own raster boundary."""

from __future__ import annotations

import hashlib

import numpy as np

from .io import InputError

_TOPOLOGY = {}


def adjacency(faces):
    faces = np.asarray(faces, np.int64)
    key = hashlib.sha256(faces.tobytes()).digest()
    if key not in _TOPOLOGY:
        raw = faces[:, np.array([[0, 1], [1, 2], [2, 0]])].reshape(-1, 2)
        edges, inverse = np.unique(np.sort(raw, axis=1), axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        order = np.argsort(inverse, kind="stable")
        start = np.r_[0, np.cumsum(counts[:-1])]
        face_ids = np.repeat(np.arange(len(faces)), 3)
        a = face_ids[order[start]]
        b = face_ids[order[start + np.minimum(counts - 1, 1)]]
        if (counts > 2).any():
            raise InputError("Nonmanifold mesh silhouette topology")
        if len(_TOPOLOGY) > 4:
            _TOPOLOGY.clear()
        _TOPOLOGY[key] = (edges, a, b, counts)
    return _TOPOLOGY[key]


class VisibilityGrid:
    def __init__(self, positions, projected, faces, bounds):
        self.triangles = np.asarray(projected[faces], np.float64)
        self.depths = np.asarray(positions[faces, 2], np.float64)
        self.grid = {}
        self.cell = 16
        maximum = (np.asarray(bounds) - 1) // self.cell
        lo = np.floor(self.triangles.min(1) / self.cell).astype(int).clip(0, maximum)
        hi = np.floor(self.triangles.max(1) / self.cell).astype(int).clip(0, maximum)
        for i, (a, b) in enumerate(zip(lo, hi)):
            for x in range(a[0], b[0] + 1):
                for y in range(a[1], b[1] + 1):
                    self.grid.setdefault((x, y), []).append(i)

    def external_visible(self, pixel, depth):
        ids = self.grid.get(tuple(np.floor(pixel / self.cell).astype(int)), [])
        if not ids:
            return False
        triangle = self.triangles[ids]
        z = self.depths[ids]
        a = triangle[:, 0]
        u = triangle[:, 1] - a
        v = triangle[:, 2] - a
        r = pixel - a
        determinant = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        valid = abs(determinant) > 1e-12
        determinant = np.where(valid, determinant, 1)
        t = (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) / determinant
        q = (u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0]) / determinant
        barycentric = np.stack([1 - t - q, t, q], -1)
        side_lengths = np.stack(
            [
                np.linalg.norm(u - v, axis=1),
                np.linalg.norm(v, axis=1),
                np.linalg.norm(u, axis=1),
            ],
            -1,
        )
        tolerance = 1e-7 * side_lengths / abs(determinant)[:, None]
        # An occlusion contour can be visible yet inside the person's union.
        if (valid & (barycentric > tolerance).all(-1)).any():
            return False
        hit = valid & (barycentric >= -1e-9).all(-1)
        if not hit.any():
            return False
        minimum_depth = np.min(1 / np.sum(barycentric[hit] / z[hit], axis=-1))
        return depth <= minimum_depth + 2e-5


def guided_primitives(
    points, positions, projected, faces, *, visibility=False, bounds=(640, 640)
):
    edges, a, b, counts = adjacency(faces)
    triangles = positions[np.asarray(faces)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    facing = (normals * triangles.mean(1)).sum(-1) > 0
    edges = edges[(facing[a] != facing[b]) | (counts == 1)]
    if len(edges) < 3:
        raise InputError("No estimated mesh silhouette edges")
    endpoints = projected[edges]
    starts = endpoints[:, 0]
    vectors = endpoints[:, 1] - starts
    length = np.maximum((vectors**2).sum(-1), 1e-12)
    grid = (
        VisibilityGrid(positions, projected, np.asarray(faces), bounds)
        if visibility
        else None
    )
    selected = []
    fractions = []
    errors = []
    for offset in range(0, len(points), 128):
        relative = points[offset : offset + 128, None] - starts[None]
        fraction = np.clip((relative * vectors).sum(-1) / length, 0, 1)
        squared = ((relative - fraction[..., None] * vectors) ** 2).sum(-1)
        if visibility:
            for j in range(len(relative)):
                candidates = np.argsort(squared[j], kind="stable")[
                    : min(128, len(edges))
                ]
                chosen = None
                for k in candidates:
                    pixel = starts[k] + fraction[j, k] * vectors[k]
                    z0, z1 = positions[edges[k], 2]
                    tau = (
                        fraction[j, k]
                        * z0
                        / ((1 - fraction[j, k]) * z1 + fraction[j, k] * z0)
                    )
                    depth = (1 - tau) * z0 + tau * z1
                    if grid.external_visible(pixel, depth):
                        chosen = k
                        break
                if chosen is None:
                    raise InputError(
                        "No visible external mesh edge supports boundary guide"
                    )
                selected.append(chosen)
                fractions.append(fraction[j, chosen])
                errors.append(np.sqrt(squared[j, chosen]))
        else:
            nearest = np.argmin(squared, axis=1)
            selected.extend(nearest)
            fractions.extend(fraction[np.arange(len(nearest)), nearest])
            errors.extend(np.sqrt(squared[np.arange(len(nearest)), nearest]))
    selected = np.asarray(selected)
    edge_ids = edges[selected]
    fraction = np.asarray(fractions)
    depth = positions[edge_ids, 2]
    denominator = (1 - fraction) * depth[:, 1] + fraction * depth[:, 0]
    t = fraction * depth[:, 0] / denominator
    weights = np.stack([1 - t, t], -1)
    if max(errors) > 3:
        raise InputError("Raster boundary unsupported by mesh silhouette edges")
    return edge_ids, weights, np.asarray(errors)


def interpolate(values, edge_ids, weights):
    return np.einsum("ne,ne...->n...", weights, np.asarray(values)[edge_ids])


def continuous_outer_contour(projected, faces, width, height, samples=256):
    """Continuous triangle union, before any integer raster rounding."""
    import shapely

    triangle = np.asarray(projected[faces], np.float64)
    area = np.abs(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    )
    polygons = shapely.polygons(triangle[area > 1e-12])
    try:
        union = shapely.union_all(polygons, grid_size=0)
    except shapely.errors.GEOSException:
        # GEOS double-precision noding can depend on triangle union order even
        # for valid triangles. Bound the retry precision well below input pixel
        # precision; retain the exact default path whenever it succeeds.
        import warnings

        warnings.warn(
            "Estimated silhouette required a fixed 1e-8 pixel precision grid",
            RuntimeWarning,
        )
        union = shapely.union_all(polygons, grid_size=1e-8)
    parts = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    polygons = [x for x in parts if x.geom_type == "Polygon"]
    if not polygons:
        raise InputError("No continuous estimated silhouette")
    ring = max(polygons, key=lambda x: x.exterior.length).exterior
    positions = np.arange(samples) * ring.length / samples
    points = shapely.get_coordinates(shapely.line_interpolate_point(ring, positions))
    inside = (points > 2).all(-1) & (points < [width - 3, height - 3]).all(-1)
    if not inside.all():
        # Match the existing observed-boundary viewport restriction. A predicted
        # mesh outside the camera does not make the actual input truncated.
        dense_count = max(4 * samples, int(np.ceil(ring.length * 2)))
        dense = shapely.get_coordinates(
            shapely.line_interpolate_point(
                ring, np.arange(dense_count) * ring.length / dense_count
            )
        )
        dense = dense[(dense > 2).all(-1) & (dense < [width - 3, height - 3]).all(-1)]
        if len(dense) < 30:
            raise InputError("Insufficient visible estimated silhouette boundary")
        points = dense[np.linspace(0, len(dense) - 1, samples).astype(int)]
    return points
