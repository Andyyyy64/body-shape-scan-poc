"""Research-only nuisance constraints, using estimated anatomy and observed masks."""

from __future__ import annotations

import numpy as np

from .contours import contour_observation
from .io import InputError
from .sampling import angular_indices
from .shared_shape import fit_observations

ANCHOR_IDS = [5, 6, 9, 10]


def build_observations(
    estimates,
    initial,
    scales,
    faces,
    generate,
    indices,
    research,
    selection_shape=None,
    selection_fit=None,
    pose_reference=None,
    root_reference=None,
    observed_overrides=None,
):
    observations = []
    strategy = research.get("fit", {})
    if strategy.get("root_mode") == "profile_tilt":
        raise InputError(
            "Historical root convention; regenerate using profile_tilt_xyz"
        )
    canonical = None
    if strategy.get("local_transport") == "canonical":
        from .local_shape import torso_basis

        params = np.zeros(204, np.float32)
        params[136:] = scales
        rest, joints = generate(initial, params)
        rest_basis = np.stack(
            [
                generate(initial + row, params)[0] - rest
                for row in np.eye(len(initial), dtype=np.float32)
            ],
            -1,
        )
        canonical = (
            rest_basis,
            torso_basis(rest, joints[ANCHOR_IDS], strategy["local_mode"]),
        )
    if strategy.get("root_mode", "frame") != "frame" and root_reference is None:
        from .pose_constraints import root_tilt_reference

        root_reference = root_tilt_reference(
            [e[1]["mhr_model_params"] for e in estimates]
        )
    if strategy.get("pose_mode", "frame") != "frame" and pose_reference is None:
        pose_reference = np.median(
            [e[1]["mhr_model_params"][6:136] for e in estimates], axis=0
        )
    selected = angular_indices(
        [e[0] for e in estimates],
        [float(e[1].get("image_detail_score", 1)) for e in estimates],
        strategy.get("max_frames", len(estimates)),
    )
    root_sequence = None
    if (
        strategy.get("heading_smoothness", 0)
        or strategy.get("heading_model", "acceleration") != "acceleration"
    ):
        from .pose_constraints import constrain_root_sequence

        root_sequence = constrain_root_sequence(
            [e[1]["mhr_model_params"] for e in estimates],
            root_reference,
            strategy.get("heading_smoothness", 0),
            [float(e[1].get("frame_index", i)) for i, e in enumerate(estimates)],
            model=strategy.get("heading_model", "acceleration"),
        )
    for fit_order, source_order in enumerate(selected):
        _, value, mask = estimates[source_order]
        if strategy.get("anchor_source") == "yolo17" and (
            "image_anchor_target" not in value or "image_anchor_weights" not in value
        ):
            raise InputError("Independent anatomy observations are missing")
        params = value["mhr_model_params"].copy()
        if strategy.get("shared_skeleton"):
            params[136:] = scales
        if strategy.get("pose_mode", "frame") != "frame":
            params[6:136] = pose_reference
        if root_sequence is not None:
            params[:6] = root_sequence[source_order, :6]
        elif root_reference is not None:
            from .pose_constraints import constrain_root_tilt

            params = constrain_root_tilt(params, root_reference)
        cache = {}
        if hasattr(generate, "batch"):
            requested = [
                initial,
                *(initial + np.eye(len(initial), dtype=np.float32)),
                initial + np.linspace(-0.1, 0.1, len(initial)).astype("float32"),
            ]
            if selection_shape is not None:
                requested.append(selection_shape)
            requested = np.asarray(requested, np.float32)
            vertices, joints = generate.batch(requested, params)
            for shape, v, j in zip(requested, vertices, joints):
                cache[shape.tobytes()] = v * [1, -1, -1], j[ANCHOR_IDS] * [1, -1, -1]

        def anatomy(shape):
            key = np.asarray(shape, dtype=np.float32).tobytes()
            if key not in cache:
                vertices, joints = generate(shape, params)
                vertices = vertices.copy() * [1, -1, -1]
                joints = joints.copy() * [1, -1, -1]
                cache[key] = vertices, joints[ANCHOR_IDS]
            return cache[key]

        anchored = bool(strategy.get("anchor_weight", 0))
        refreshed_vertices, refreshed_camera = None, None
        if selection_fit is not None:
            refreshed_vertices = anatomy(selection_shape)[0].copy()
            coefficients = np.asarray(selection_fit["shape_delta"])[len(initial) :]
            if len(coefficients):
                from .local_shape import torso_basis, transport_basis

                if canonical is not None:
                    full_basis = np.stack(
                        [
                            anatomy(initial + row)[0] - anatomy(initial)[0]
                            for row in np.eye(len(initial), dtype=np.float32)
                        ],
                        -1,
                    )
                    local_field = transport_basis(
                        canonical[0], full_basis, canonical[1]
                    )
                else:
                    local_field = torso_basis(
                        anatomy(initial)[0], anatomy(initial)[1], strategy["local_mode"]
                    )
                refreshed_vertices += local_field @ coefficients
            angle = float(
                np.deg2rad(
                    selection_fit.get("yaw_delta_deg", np.zeros(len(selected)))[
                        fit_order
                    ]
                )
            )
            if angle:
                pivot = anatomy(initial)[1].mean(0)
                v = refreshed_vertices - pivot
                rotation = np.array(
                    [
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)],
                    ]
                )
                refreshed_vertices = v @ rotation.T + pivot
            refreshed_camera = (
                value["pred_cam_t"] + selection_fit["camera_delta"][fit_order]
            )
        observations.append(
            contour_observation(
                mask,
                faces,
                lambda shape: anatomy(shape)[0],
                initial,
                value["pred_cam_t"],
                float(value["focal_length"]),
                region_vertex_indices=indices,
                region_weight=research.get("region_weight", 1),
                landmark_generate=(lambda shape: anatomy(shape)[1])
                if anchored
                else None,
                landmark_target=value["pred_keypoints_2d"][ANCHOR_IDS]
                if anchored
                and strategy.get("anchor_source", "sam_projection") == "sam_projection"
                else value["image_anchor_target"]
                if anchored
                else None,
                selection_shape=selection_shape,
                selection_vertices=refreshed_vertices,
                selection_camera=refreshed_camera,
                primitive_mode=strategy.get("contour_primitive", "vertex"),
                samples=strategy.get("contour_samples", 256),
                contour_order="arc"
                if strategy.get("distance_mode") == "segments"
                else "vertex_id",
                **(
                    observed_overrides[source_order]
                    if observed_overrides is not None
                    else {}
                ),
            )
        )
        observations[-1]["source_order"] = int(source_order)
        if strategy.get("anchor_source") == "yolo17":
            observations[-1]["anchor_weights"] = value["image_anchor_weights"]
        if canonical is not None:
            from .local_shape import transport_basis

            o = observations[-1]
            ids = o["vertex_indices"]
            if "edge_vertex_indices" in o:
                edge = o["edge_vertex_indices"]
                fields = transport_basis(
                    canonical[0][edge].reshape(-1, 3, 45),
                    o.pop("endpoint_basis").reshape(-1, 3, 45),
                    canonical[1][edge].reshape(-1, 3, canonical[1].shape[-1]),
                ).reshape(len(edge), 2, 3, -1)
                o["canonical_local_" + strategy["local_mode"]] = np.einsum(
                    "ne,necd->ncd", o["edge_weights"], fields
                )
            else:
                o["canonical_local_" + strategy["local_mode"]] = transport_basis(
                    canonical[0][ids], o["basis"], canonical[1][ids]
                )
    return observations


def solve(observations, shape_scale, research):
    strategy = research.get("fit", {})
    anchored = bool(strategy.get("anchor_weight", 0))
    local_mode = strategy.get("local_mode", "none")
    local_count = 0 if local_mode == "none" else (1 if local_mode == "radial" else 2)
    if local_count:
        from .local_shape import torso_basis

        observations = [dict(o) for o in observations]
        for o in observations:
            if strategy.get("local_transport") == "canonical":
                key = "canonical_local_" + local_mode
                if key not in o:
                    raise InputError("Missing canonical local shape transport")
                local_basis = o[key]
            else:
                local_basis = torso_basis(o["base"], o["anchor_base"], local_mode)
            o["basis"] = np.concatenate([o["basis"], local_basis], -1)
            o["anchor_basis"] = np.pad(
                o["anchor_basis"], ((0, 0), (0, 0), (0, local_count))
            )
        shape_scale = np.concatenate([shape_scale, np.full(local_count, 0.1)])

    def stack(key):
        return np.array([o[key] for o in observations])

    return fit_observations(
        stack("base"),
        stack("basis"),
        stack("target"),
        stack("camera"),
        stack("focal"),
        stack("center"),
        np.full_like(shape_scale, 0.1)
        if strategy.get("shape_scale_mode") == "fixed"
        else shape_scale,
        shape_weight=research.get("shape_weight", 0.1),
        base_weights=stack("base_weight") if "base_weight" in observations[0] else None,
        target_weights=stack("target_weight")
        if "target_weight" in observations[0]
        else None,
        observation_scale=stack("mask_height") / 256
        if strategy.get("normalize_observation")
        else None,
        anchor_base=stack("anchor_base") if anchored else None,
        anchor_basis=stack("anchor_basis") if anchored else None,
        anchor_target=stack("anchor_target") if anchored else None,
        anchor_weights=stack("anchor_weights")
        if anchored and "anchor_weights" in observations[0]
        else None,
        anchor_weight=strategy.get("anchor_weight", 0),
        pose_limit_deg=strategy.get("pose_limit_deg", 0),
        pose_weight=strategy.get("pose_weight", 1),
        diagnostics=strategy.get("diagnostics", False),
        contour_softening=strategy.get("contour_softening", 2),
        anchor_softening=strategy.get("anchor_softening", 2),
        camera_mode=strategy.get("camera_mode"),
        steps=strategy.get("steps", 300),
        optimizer_mode=strategy.get("optimizer_mode", "adam"),
        distance_mode=strategy.get("distance_mode", "legacy"),
        local_count=local_count,
        local_limit=strategy.get("local_limit", 0.1),
        local_weight=strategy.get("local_weight", 0.01),
        global_shape=strategy.get("global_shape", True),
    )
