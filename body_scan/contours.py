"""Per-frame silhouette observations for the shared-shape contour fit.

For one frame this projects the initial posed mesh, extracts its outer
boundary, keeps the mesh vertices that form that boundary, and pairs them
with the observed person-mask boundary. The vertex set and per-frame pose
are then held fixed by the optimizer (a local approximation).
"""

from __future__ import annotations

import numpy as np

from .io import InputError

CONTOUR_SAMPLES = 256
MIN_CONTOUR_POINTS = 30
LINEARITY_TOLERANCE = 1e-4
BORDER_PX = 2
NEAR_PLANE = 0.1


def _outer_contour(binary, width: int, height: int):
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise InputError("No silhouette boundary")
    points = max(contours, key=len)[:, 0, :]
    points = points[
        (points[:, 0] > BORDER_PX)
        & (points[:, 0] < width - 1 - BORDER_PX)
        & (points[:, 1] > BORDER_PX)
        & (points[:, 1] < height - 1 - BORDER_PX)
    ]
    if len(points) < MIN_CONTOUR_POINTS:
        raise InputError("Insufficient untruncated silhouette boundary")
    return points


def _nearest_vertex(points, projected, method="dense"):
    if method == "spatial":
        from scipy.spatial import cKDTree

        tree = cKDTree(projected)
        distance, candidates = tree.query(points, k=2)
        ids = candidates[:, 0].copy()
        # Preserve dense argmin's deterministic lowest-index tie rule.
        for order in np.flatnonzero(abs(distance[:, 0] - distance[:, 1]) < 1e-10):
            tied = np.sort(
                tree.query_ball_point(points[order], distance[order, 0] + 1e-10)
            )
            squared = ((projected[tied] - points[order]) ** 2).sum(-1)
            ids[order] = tied[squared.argmin()]
        return ids
    if method != "dense":
        raise InputError("Invalid boundary vertex search")
    ids = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 256):
        block = points[start : start + 256].astype(np.float64)
        d = ((block[:, None, :] - projected[None, :, :]) ** 2).sum(-1)
        ids[start : start + 256] = d.argmin(1)
    return ids


def contour_observation(
    mask,
    faces,
    generate,
    initial_shape,
    camera,
    focal: float,
    *,
    region_vertex_indices=None,
    region_weight=1.0,
    landmark_generate=None,
    landmark_target=None,
    selection_shape=None,
    selection_vertices=None,
    selection_camera=None,
    observed_contour=None,
    observed_height=None,
    contour_order="vertex_id",
    primitive_mode="vertex",
    samples=CONTOUR_SAMPLES,
) -> dict:
    """Build one frame's fixed-correspondence contour observation.

    ``mask`` is the observed person mask (H×W, nonzero = person); ``generate``
    maps a shape vector to posed vertices [N,3] in the frame's camera
    convention (x right, y down, z forward, model metres).
    """
    import cv2

    if samples not in (256, 512, 1024):
        raise InputError("Invalid contour sample count")
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise InputError("Mask must be a single-channel image")
    height, width = mask.shape
    initial = np.asarray(initial_shape, dtype=np.float32)
    camera = np.asarray(camera, dtype=np.float32).reshape(3)
    posed = np.asarray(generate(initial), dtype=np.float32)
    selection = (
        posed
        if selection_shape is None
        else np.asarray(generate(selection_shape), dtype=np.float32)
    )
    if selection_vertices is not None:
        selection = np.asarray(selection_vertices, np.float32)
        if selection.shape != posed.shape or not np.isfinite(selection).all():
            raise InputError("Invalid refreshed contour geometry")
    position = selection + (
        camera if selection_camera is None else np.asarray(selection_camera, np.float32)
    )
    if primitive_mode == "continuous_visible_edge64_viewport":
        position = selection.astype(np.float64) + (
            camera if selection_camera is None else np.asarray(selection_camera)
        ).astype(np.float64)
    if (position[:, 2] <= NEAR_PLANE).any():
        raise InputError("Invalid initial camera geometry")
    projected = position[:, :2] / position[:, 2:] * focal + [width / 2, height / 2]
    raster = np.zeros((height, width), np.uint8)
    for face in faces:
        cv2.fillConvexPoly(
            raster, np.clip(projected[face], -10000, 10000).astype("int32"), 1
        )
    predicted = _outer_contour(raster, width, height)
    if observed_contour is None:
        observed = _outer_contour((mask != 0).astype("uint8"), width, height)
    else:
        observed = np.asarray(observed_contour, np.float32)
        if (
            observed.ndim != 2
            or observed.shape[1] != 2
            or len(observed) < MIN_CONTOUR_POINTS
            or not np.isfinite(observed).all()
            or (observed < 0).any()
            or (observed >= [width, height]).any()
            or observed_height is None
            or not np.isfinite(observed_height)
            or not 1 <= observed_height <= height
        ):
            raise InputError("Invalid frozen observed contour")
    endpoint_basis = None
    if primitive_mode in ("edge", "continuous_visible_edge64_viewport"):
        if contour_order != "arc":
            raise InputError("Edge silhouette requires ordered segment contours")
        from .silhouette_edges import (
            continuous_outer_contour,
            guided_primitives,
            interpolate,
        )

        guide = (
            continuous_outer_contour(projected, faces, width, height, samples)
            if primitive_mode == "continuous_visible_edge64_viewport"
            else predicted[np.linspace(0, len(predicted) - 1, samples).astype(int)]
        )
        edge_ids, edge_weights, edge_errors = guided_primitives(
            guide,
            position,
            projected,
            faces,
            visibility=primitive_mode == "continuous_visible_edge64_viewport",
            bounds=(width, height),
        )
        ids = edge_ids[:, 0]
        base = interpolate(posed, edge_ids, edge_weights)
        endpoint_basis = []
        for dim in range(len(initial)):
            changed = initial.copy()
            changed[dim] += 1
            endpoint_basis.append(
                np.asarray(generate(changed), np.float32)[edge_ids] - posed[edge_ids]
            )
        endpoint_basis = np.stack(endpoint_basis, -1)
        basis = np.einsum("ne,necd->ncd", edge_weights, endpoint_basis)
    elif primitive_mode == "vertex":
        nearest = _nearest_vertex(
            predicted, projected, "spatial" if contour_order == "arc" else "dense"
        )
        if contour_order == "arc":
            # Retain traversal order, removing consecutive repeats only. A vertex
            # can legitimately occur again on a different boundary branch.
            ids = nearest[np.r_[True, nearest[1:] != nearest[:-1]]]
        elif contour_order == "vertex_id":
            ids = np.unique(nearest)
        else:
            raise InputError("Invalid contour order")
        ids = ids[np.linspace(0, len(ids) - 1, samples).astype(int)]
        base = posed[ids]
        basis = []
        for dim in range(len(initial)):
            changed = initial.copy()
            changed[dim] += 1
            basis.append(np.asarray(generate(changed), dtype=np.float32)[ids] - base)
        basis = np.stack(basis, axis=-1)
    else:
        raise InputError("Invalid contour primitive")
    observed = observed[np.linspace(0, len(observed) - 1, samples).astype(int)]
    perturbation = np.linspace(-0.1, 0.1, len(initial)).astype("float32")
    residual = np.max(
        np.abs(
            (
                interpolate(
                    np.asarray(generate(initial + perturbation), np.float32),
                    edge_ids,
                    edge_weights,
                )
                if primitive_mode in ("edge", "continuous_visible_edge64_viewport")
                else np.asarray(generate(initial + perturbation), dtype=np.float32)[ids]
            )
            - (base + basis @ perturbation)
        )
    )
    if residual > LINEARITY_TOLERANCE:
        raise InputError("MHR local shape basis failed the linearity check")
    observation = {
        "base": base,
        "vertex_indices": ids,
        "basis": basis,
        "target": observed.astype("float32"),
        "camera": camera,
        "focal": float(focal),
        "center": np.array([width / 2, height / 2], np.float32),
        "mask_height": float(np.ptp(np.where(mask)[0]) + 1)
        if observed_contour is None
        else float(observed_height),
    }
    if primitive_mode in ("edge", "continuous_visible_edge64_viewport"):
        observation.update(
            edge_vertex_indices=edge_ids,
            edge_weights=edge_weights
            if primitive_mode == "continuous_visible_edge64_viewport"
            else edge_weights.astype("float32"),
            endpoint_basis=endpoint_basis,
            edge_guide_distance=edge_errors.astype("float32"),
        )
    if landmark_generate is not None:
        if landmark_target is None:
            raise InputError("Landmark target is required")
        anchor = np.asarray(landmark_generate(initial), dtype=np.float32)
        anchor_basis = []
        for dim in range(len(initial)):
            changed = initial.copy()
            changed[dim] += 1
            anchor_basis.append(
                np.asarray(landmark_generate(changed), dtype=np.float32) - anchor
            )
        observation.update(
            anchor_base=anchor,
            anchor_basis=np.stack(anchor_basis, -1),
            anchor_target=np.asarray(landmark_target, dtype=np.float32),
        )
    if region_vertex_indices is not None:
        selected = np.asarray(region_vertex_indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or len(selected) < 1
            or (selected < 0).any()
            or (selected >= len(posed)).any()
        ):
            raise InputError("Invalid contour region indices")
        if not np.isfinite(region_weight) or not 1 <= region_weight <= 32:
            raise InputError("Invalid contour region weight")
        # Both Chamfer directions are weighted. The observed region is selected
        # using the initial projection only, never the renderer's truth/mask.
        box = projected[selected]
        low, high = box.min(0), box.max(0)
        margin = np.maximum((high - low) * 0.05, 2)
        inside = ((observed >= low - margin) & (observed <= high + margin)).all(1)
        observation["base_weight"] = np.where(
            np.isin(ids, selected), region_weight, 1
        ).astype("float32")
        if primitive_mode in ("edge", "continuous_visible_edge64_viewport"):
            observation["base_weight"] = (
                1
                + (region_weight - 1)
                * (edge_weights * np.isin(edge_ids, selected)).sum(1)
            ).astype("float32")
        observation["target_weight"] = np.where(inside, region_weight, 1).astype(
            "float32"
        )
    return observation
