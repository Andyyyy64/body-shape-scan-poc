import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from body_scan.io import InputError
from body_scan.pipeline import validate_video_source


class ExperimentIsolationTests(unittest.TestCase):
    def test_all_source_store_flag_combinations(self):
        # This is a provenance boundary: neither a flag alone nor a lab marker
        # alone may make generated videos appear to be actual recordings.
        for kind, lab, flag in itertools.product(
            ("real", "synthetic", "unknown"), (False, True), (False, True)
        ):
            accepted = (kind == "real" and not lab and not flag) or (
                kind == "synthetic" and lab and flag
            )
            with self.subTest(kind=kind, lab=lab, flag=flag):
                if accepted:
                    self.assertEqual(
                        validate_video_source(
                            {"source_kind": kind}, {"synthetic_experiment": lab}, flag
                        ),
                        kind,
                    )
                else:
                    with self.assertRaises(InputError):
                        validate_video_source(
                            {"source_kind": kind}, {"synthetic_experiment": lab}, flag
                        )

    def test_known_local_changes_preserve_remote_vertices(self):
        import numpy as np

        from body_scan.mesh import synthetic_mesh
        from body_scan.synthetic_lab import deform

        mesh = synthetic_mesh()
        vertices = np.asarray(mesh["vertices"], dtype=float)
        body = {
            "faces": np.asarray(mesh["faces"]),
            "section_y": mesh["sections"][0]["y"],
            "center_xz": [0, 0],
        }
        for delta in (-0.02, 0, 0.02):
            with self.subTest(delta=delta):
                changed, before, after = deform(vertices, body, delta)
                self.assertAlmostEqual(after - before, delta, places=6)
                remote = np.abs(vertices[:, 1] - body["section_y"]) > 0.5
                self.assertLess(
                    float(np.max(abs(changed[remote] - vertices[remote]))), 1e-9
                )


class SensitivityEvaluationTests(unittest.TestCase):
    def test_constant_mesh_cannot_pass_known_change(self):
        from body_scan.synthetic_evidence import delta_result

        for truth in (-0.02, -0.01, 0.01, 0.02):
            for null in (0, 0.001):
                result = delta_result(0, truth, null)
                self.assertFalse(result["correct_direction"])
                self.assertFalse(result["clears_null_margin"])

    def test_direction_and_margin_are_independent_requirements(self):
        from body_scan.synthetic_evidence import delta_result

        wrong = delta_result(-0.02, 0.02, 0.001)
        small = delta_result(0.001, 0.02, 0.005)
        self.assertFalse(wrong["correct_direction"])
        self.assertTrue(wrong["clears_null_margin"])
        self.assertTrue(small["correct_direction"])
        self.assertFalse(small["clears_null_margin"])


class ResumeTests(unittest.TestCase):
    def prepare(self, root, state):
        from body_scan.io import write_json

        (root / "store" / ("a" * 32)).mkdir(parents=True)
        write_json(root / "config.json", {"engine_id": "test_engine"})
        write_json(
            root / "matrix.json",
            {
                "engine_id": "test_engine",
                "cases": [
                    {
                        "name": "one",
                        "state": "running",
                        "session_id": "a" * 32,
                        "spec": {"name": "one"},
                    }
                ],
            },
        )
        write_json(
            root / "store" / ("a" * 32) / "reconstruction.json", {"state": state}
        )

    def test_parent_resume_preserves_terminal_worker_without_reprocessing(self):
        from body_scan.io import read_json
        from body_scan.synthetic_lab import run_matrix

        for state in ("complete", "failed"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.prepare(root, state)
                with (
                    patch("body_scan.synthetic_lab.launch") as launch,
                    patch("body_scan.synthetic_lab.report", return_value={}),
                ):
                    run_matrix(str(root / "config.json"), str(root))
                launch.assert_not_called()
                self.assertEqual(
                    read_json(root / "matrix.json")["cases"][0]["state"], state
                )

    def test_worker_crash_is_retained_as_terminal_failure(self):
        from body_scan.io import read_json
        from body_scan.synthetic_lab import run_matrix

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root, "running")
            with (
                patch("body_scan.synthetic_lab.launch", return_value=-9),
                patch("body_scan.synthetic_lab.report", return_value={}),
            ):
                run_matrix(str(root / "config.json"), str(root))
            row = read_json(root / "matrix.json")["cases"][0]
            self.assertEqual(row["state"], "failed")
            self.assertEqual(row["exit_code"], -9)
            self.assertEqual(row["reconstruction"]["stage"], "interrupted")

    def test_second_orchestrator_cannot_mutate_the_same_experiment(self):
        import fcntl

        from body_scan.synthetic_lab import run_matrix

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root, "running")
            with (root / "matrix.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(InputError, "Another orchestrator"):
                    run_matrix(str(root / "config.json"), str(root))

    def test_modified_research_options_are_rejected_before_worker_launch(self):
        from body_scan.io import read_json, replace_json, write_json
        from body_scan.synthetic_lab import run_matrix

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root, "running")
            manifest = read_json(root / "matrix.json")
            manifest["research_options"] = {}
            replace_json(root / "matrix.json", manifest)
            write_json(
                root / "research-options.json", {"qc": {"require_rotation": True}}
            )
            with patch("body_scan.synthetic_lab.launch") as launch:
                with self.assertRaisesRegex(InputError, "research options changed"):
                    run_matrix(str(root / "config.json"), str(root))
                launch.assert_not_called()

    def test_modified_implementation_cannot_mix_pending_cases(self):
        from body_scan.io import write_json
        from body_scan.synthetic_lab import run_matrix

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root, "running")
            (root / "implementation").mkdir()
            write_json(
                root / "implementation" / "hashes.json",
                {"pipeline.py": "different-source"},
            )
            with patch("body_scan.synthetic_lab.launch") as launch:
                with self.assertRaisesRegex(InputError, "implementation changed"):
                    run_matrix(str(root / "config.json"), str(root))
                launch.assert_not_called()


class RefusalEvidenceTests(unittest.TestCase):
    def test_rotation_qc_refusal_needs_diagnostics_and_model_crash_does_not_count(self):
        from body_scan.io import write_json
        from body_scan.synthetic_evidence import evidence
        from body_scan.synthetic_lab import protocol

        for diagnostic, error, expected in (
            (False, "回転範囲が不足しています。", False),
            (True, "回転範囲が不足しています。", True),
            (True, "SAMの処理に失敗しました。", False),
        ):
            with (
                self.subTest(diagnostic=diagnostic, error=error),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                attempt = root / "store" / ("a" * 32) / "attempt_test"
                attempt.mkdir(parents=True)
                write_json(root / "protocol.json", protocol())
                write_json(root / "body.json", {"limitations": []})
                write_json(
                    root / "report.json",
                    {
                        "case_count": 1,
                        "completed": 0,
                        "failed": 1,
                        "pending": 0,
                        "distance_pairs": [],
                        "limitations": [],
                        "results": [
                            {
                                "name": "stress_partial_rotation",
                                "spec": {"group": "stress"},
                                "state": "failed",
                                "session_id": "a" * 32,
                                "reconstruction": {"stage": "failed", "error": error},
                            }
                        ],
                    },
                )
                if diagnostic:
                    write_json(attempt / "qc.json", {"rotation_confirmed": False})
                result = evidence(root)
                self.assertEqual(
                    result["refusal_expectations"][0]["met_expectation"], expected
                )


class SamplingMetadataTests(unittest.TestCase):
    def test_legacy_sampling_does_not_invent_a_config_or_hide_mixed_counts(self):
        from body_scan.io import write_json
        from body_scan.synthetic_comparison import sampling_metadata

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, count in (("first", 8), ("second", 16)):
                path = root / "store" / name / "attempt_test"
                path.mkdir(parents=True)
                write_json(path / "frames.json", {"frames": [{}] * count})
            result = sampling_metadata(root, {})
            self.assertEqual(result["sample_frames"], [8, 16])
            self.assertIn(
                "legacy_config_snapshot_absent", result["sampling_metadata_source"]
            )

    def test_sampling_source_is_frozen_alongside_fitting_sources(self):
        from body_scan.io import read_json, write_json
        from body_scan.synthetic_lab import snapshot_sources

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "config.json", {"sample_frames": 48})
            snapshot_sources(root, str(root / "config.json"))
            self.assertIn(
                "vision.py", read_json(root / "implementation" / "hashes.json")
            )


if __name__ == "__main__":
    unittest.main()
