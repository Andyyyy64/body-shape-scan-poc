import tempfile
import unittest
from pathlib import Path

from body_scan.capture_manifest import build, verify
from body_scan.io import InputError, read_json, replace_json, sha256, write_json


class CaptureManifestTests(unittest.TestCase):
    def fixture(self, root, kind="synthetic"):
        store = root / "store"
        store.mkdir()
        config = root / "runtime.json"
        write_json(config, {"engine_id": "engine"})
        write_json(
            store / "store.json",
            {
                "schema_version": "private_scan_store.v1",
                **({"synthetic_experiment": True} if kind == "synthetic" else {}),
            },
        )
        ids = ["a" * 32, "b" * 32]
        profile = {
            "id": "profile",
            "source_kind": kind,
            "engine_id": "engine",
            "method": "method",
            "sections": [{"y": 1.0, "center_xz": [0, 0]}],
        }
        for index, identifier in enumerate(ids):
            session = store / identifier
            session.mkdir()
            (session / "capture.bin").write_bytes(b"observed video" + bytes([index]))
            capture_hash = sha256(session / "capture.bin")
            if index == 0:
                profile["source_capture_sha256"] = capture_hash
            write_json(
                session / "session.json",
                {"source_kind": kind, "video_sha256": capture_hash, "label": "unused"},
            )
            write_json(
                session / "reconstruction.json",
                {
                    "state": "complete",
                    "engine_id": "engine",
                    "method": "method",
                    "total_frames": 2,
                    "usable_frames": 2,
                },
            )
            write_json(
                session / "mesh.json",
                {
                    "source_kind": kind,
                    "method": "method",
                    "capture_sha256": capture_hash,
                    "reference_frame_id": "mhr_rest_profile",
                    "scale_reference_id": "model_skeleton_profile",
                    "sections": profile["sections"],
                    "quality": {"usable_frames": 2},
                },
            )
            attempt = session / "attempt_one"
            attempt.mkdir()
            frames = []
            for frame in range(2):
                file = attempt / f"frame_{frame}.npz"
                # The manifest hashes immutable estimates; numerical decoding is
                # the observer's responsibility and is tested separately.
                file.write_bytes(b"native estimate" + bytes([index, frame]))
                frames.append(
                    {
                        "frame": frame,
                        "state": "estimated",
                        "output": file.name,
                        "yaw_rad": float(frame),
                    }
                )
            write_json(attempt / "frames.json", {"frames": frames})
            write_json(
                attempt / "fit.json",
                {
                    "source_kind": kind,
                    "engine_id": "engine",
                    "method": "method",
                    "research_options": {},
                    "fit_source_frame_indices": [0, 1],
                    "implementation_sha256": {
                        name: "0" * 64
                        for name in [
                            "pipeline.py",
                            "fit_strategy.py",
                            "shared_shape.py",
                            "contours.py",
                            "silhouette_edges.py",
                        ]
                    },
                },
            )
            write_json(
                attempt / "qc.json",
                {
                    "source_kind": kind,
                    "method": "method",
                    "research_options": {},
                    "usable_frames": 2,
                    "required_frames": 2,
                },
            )
        write_json(store / "profile.json", profile)
        # Deliberately invalid labels: this adapter must never open a lab report.
        (root / "report.json").write_text("invalid renderer labels")
        return store, ids, config

    def test_direct_pipeline_inputs_preserve_source_kind_without_renderer_report(self):
        for kind in ["real", "synthetic"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                store, ids, config = self.fixture(Path(directory), kind)
                manifest = build(store, ids[0], ids[1:], config)
                self.assertEqual(manifest["source_kind"], kind)
                self.assertEqual(verify(manifest, config), manifest)
                self.assertFalse(manifest["measurement_ready"])
                self.assertFalse(manifest["physical_accuracy_validated"])
                self.assertEqual(len(manifest["captures"][1]["frames"]), 2)
                self.assertNotIn("truth", manifest)

    def test_changes_to_actual_video_or_estimate_are_detected(self):
        for filename in ["capture.bin", "attempt_one/frame_1.npz"]:
            with (
                self.subTest(file=filename),
                tempfile.TemporaryDirectory() as directory,
            ):
                store, ids, config = self.fixture(Path(directory))
                manifest = build(store, ids[0], ids[1:], config)
                (store / ids[1] / filename).write_bytes(b"changed actual observation")
                with self.assertRaises(InputError):
                    verify(manifest, config)

    def test_native_method_engine_source_options_and_implementation_must_match(self):
        changes = [
            ("reconstruction.json", "engine_id", "foreign"),
            ("reconstruction.json", "method", "other method"),
            ("session.json", "source_kind", "real"),
            ("mesh.json", "reference_frame_id", "mhr_rest_other_profile"),
            (
                "attempt_one/fit.json",
                "implementation_sha256",
                {"pipeline.py": "1" * 64},
            ),
            ("attempt_one/fit.json", "research_options", {"shape_weight": 9}),
            (
                "attempt_one/fit.json",
                "implementation_sha256",
                {
                    name: "0" * 64
                    for name in [
                        "pipeline.py",
                        "fit_strategy.py",
                        "shared_shape.py",
                        "contours.py",
                    ]
                },
            ),
        ]
        for filename, key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                store, ids, config = self.fixture(Path(directory))
                path = store / ids[1] / filename
                data = read_json(path)
                data[key] = value
                replace_json(path, data)
                with self.assertRaises(InputError):
                    build(store, ids[0], ids[1:], config)

    def test_duplicate_and_inconsistent_frame_correspondence_is_rejected(self):
        for corrupt in ["frames", "fit", "count"]:
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as directory,
            ):
                store, ids, config = self.fixture(Path(directory))
                attempt = store / ids[1] / "attempt_one"
                if corrupt == "frames":
                    file = attempt / "frames.json"
                    data = read_json(file)
                    data["frames"][1]["frame"] = 0
                elif corrupt == "fit":
                    file = attempt / "fit.json"
                    data = read_json(file)
                    data["fit_source_frame_indices"] = [0, 7]
                else:
                    file = attempt / "qc.json"
                    data = read_json(file)
                    data["usable_frames"] = 1
                replace_json(file, data)
                with self.assertRaises(InputError):
                    build(store, ids[0], ids[1:], config)

    def test_ambiguous_attempts_and_estimate_paths_outside_attempt_are_rejected(self):
        for corrupt in ["attempt", "escape"]:
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as directory,
            ):
                store, ids, config = self.fixture(Path(directory))
                session = store / ids[1]
                if corrupt == "attempt":
                    other = session / "attempt_two"
                    other.mkdir()
                    write_json(other / "frames.json", {"frames": []})
                else:
                    path = session / "attempt_one/frames.json"
                    data = read_json(path)
                    data["frames"][0]["output"] = "../../profile.json"
                    replace_json(path, data)
                with self.assertRaises(InputError):
                    build(store, ids[0], ids[1:], config)

    def test_reference_identity_and_current_ids_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            store, ids, config = self.fixture(Path(directory))
            for reference, current in [
                (ids[0], []),
                (ids[0], [ids[0]]),
                (ids[0], [ids[1], ids[1]]),
                (ids[1], [ids[0]]),
            ]:
                with (
                    self.subTest(reference=reference, current=current),
                    self.assertRaises(InputError),
                ):
                    build(store, reference, current, config)
