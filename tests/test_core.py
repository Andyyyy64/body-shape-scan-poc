import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from body_scan.audit import inspect_blob
from body_scan.geometry import (
    convex_hull,
    ellipse_perimeter,
    measure_widths,
    polygon_perimeter,
    support_width,
    width_integral,
)
from body_scan.io import new_run, read_json, relative_asset, write_json
from body_scan.statistics import analyze_pairs, detection_power


class GeometryTests(unittest.TestCase):
    def test_circle_nonuniform_and_repeated_angles(self):
        obs = [
            {"angle_rad": a, "width_m": 0.30}
            for a in [0.0, 0.3, 0.7, 1.0, 1.6, 2.0, 2.5, 3.0]
        ]
        self.assertAlmostEqual(width_integral(obs, 40)["value_m"], math.pi * 0.30)
        self.assertAlmostEqual(
            width_integral(obs + [obs[0]] * 20, 40)["value_m"], math.pi * 0.30
        )

    def test_ellipse_and_nonellipse(self):
        for polygon in [
            [
                (
                    math.cos(i * math.pi / 1024) * 0.18,
                    math.sin(i * math.pi / 1024) * 0.12,
                )
                for i in range(2048)
            ],
            [(-0.2, -0.1), (0.2, -0.1), (0.1, 0.2), (-0.1, 0.18)],
        ]:
            rows = [
                {
                    "angle_rad": i * math.pi / 720,
                    "width_m": support_width(polygon, i * math.pi / 720),
                }
                for i in range(720)
            ]
            self.assertAlmostEqual(
                width_integral(rows, 2)["value_m"],
                polygon_perimeter(polygon),
                delta=0.0001,
            )

    def test_concavity_is_not_observable_in_widths(self):
        polygon = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)]
        hull = convex_hull(polygon)
        for angle in [0.0, 0.3, 0.9, 1.7, 2.8]:
            self.assertAlmostEqual(
                support_width(polygon, angle), support_width(hull, angle)
            )
        self.assertGreater(polygon_perimeter(polygon), polygon_perimeter(hull))

    def test_transform_invariance_and_scale(self):
        p = [(-0.2, -0.1), (0.2, -0.1), (0.1, 0.2)]
        self.assertAlmostEqual(
            polygon_perimeter([(x + 3, z - 5) for x, z in p]), polygon_perimeter(p)
        )
        self.assertAlmostEqual(
            polygon_perimeter([(x * 1.01, z * 1.01) for x, z in p]),
            polygon_perimeter(p) * 1.01,
        )
        self.assertAlmostEqual(ellipse_perimeter(0.15, 0.15), math.pi * 0.30)

    def test_bad_width_inputs(self):
        for rows in [
            [{"angle_rad": 0.0, "width_m": 0.3}] * 3,
            [{"angle_rad": x, "width_m": 0.3} for x in [0.0, 0.01, 0.02]],
            [{"angle_rad": x, "width_m": float("nan")} for x in [0.0, 1.0, 2.0]],
        ]:
            with self.assertRaises(ValueError):
                width_integral(rows, 60)
        for value in [True, 0.0, -1.0, float("inf")]:
            with self.assertRaises(ValueError):
                ellipse_perimeter(value, 0.1)

    def test_units_and_scale_contract(self):
        data = {
            "schema_version": "width_observations.v1",
            "source_kind": "real",
            "measurement_definition_id": "test",
            "projection": "metric_orthographic",
            "scale_evidence_id": None,
            "observations": [],
        }
        self.assertEqual(measure_widths(data, 30)["status"], "unscaled")
        data["projection"] = "perspective_pixels"
        with self.assertRaises(ValueError):
            measure_widths(data, 30)


class StatisticsTests(unittest.TestCase):
    def pair(self, **changes):
        return {
            "pair_id": "synthetic_pair",
            "subject_id": "synthetic_subject",
            "baseline_session_id": "synthetic_before",
            "followup_session_id": "synthetic_after",
            "device_class": "synthetic",
            "part": "test",
            "method": "constant",
            "measurement_definition_id": "test.v1",
            "status": "ok",
            "split": "evaluation",
            "reference_delta_m": -0.01,
            "reference_uncertainty_m": 0.0,
            "predicted_delta_m": 0.0,
            **changes,
        }

    def report(self, rows):
        return analyze_pairs(
            {
                "schema_version": "paired_validation.v1",
                "source_kind": "synthetic",
                "pairs": rows,
            },
            0.003,
            0.01,
        )

    def test_power_not_just_threshold(self):
        for sigma, expected in [(0.005, 0.293), (0.0036, 0.502), (0.0025, 0.807)]:
            self.assertAlmostEqual(
                detection_power(sigma, 0.01)["power"], expected, delta=0.001
            )
        self.assertAlmostEqual(detection_power(0.0025, 0.0)["power"], 0.05)
        self.assertAlmostEqual(
            detection_power(0.0025, -0.01)["power"],
            detection_power(0.0025, 0.01)["power"],
        )
        for sigma in [0.0, -1.0, True, float("nan")]:
            with self.assertRaises(ValueError):
                detection_power(sigma, 0.01)

    def test_constant_shape_fails_sensitivity(self):
        result = self.report([self.pair()])["groups"][0]
        self.assertEqual(result["sensitivity_all_attempts"]["rate"], 0.0)
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_failures_remain_in_denominator(self):
        result = self.report(
            [
                self.pair(predicted_delta_m=-0.01),
                self.pair(
                    pair_id="synthetic_failed", status="failed", predicted_delta_m=None
                ),
            ]
        )["groups"][0]
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["usable_rate"], 0.5)
        self.assertEqual(result["sensitivity_all_attempts"]["rate"], 0.5)
        self.assertTrue(result["correlated_pairs"])

    def test_wrong_direction_not_detected(self):
        self.assertEqual(
            self.report([self.pair(predicted_delta_m=0.01)])["groups"][0][
                "sensitivity_all_attempts"
            ]["rate"],
            0.0,
        )

    def test_invalid_pairs(self):
        for rows in [
            [self.pair(), self.pair()],
            [self.pair(followup_session_id="synthetic_before")],
            [self.pair(status="failed", predicted_delta_m=0.1)],
            [self.pair(split="training")],
            [self.pair(predicted_delta_m=float("inf"))],
        ]:
            with self.assertRaises(ValueError):
                self.report(rows)

    def test_reference_uncertainty_not_counted_as_exact_truth(self):
        result = self.report([self.pair(reference_uncertainty_m=0.002)])["groups"][0]
        self.assertEqual(result["sensitivity_all_attempts"]["n"], 0)
        self.assertIsNone(result["false_positive_all_attempts"]["rate"])


class PrivateIOTests(unittest.TestCase):
    def test_checkout_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "repo").mkdir()
            (root / "repo" / ".git").write_text("gitdir: placeholder")
            (root / "alias").symlink_to(root / "repo")
            for p in [root / "repo" / "data", root / "alias" / "data"]:
                with self.assertRaises(ValueError):
                    new_run(p)

    def test_no_overwrite_or_nonfinite_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = new_run(Path(temp) / "private")
            write_json(root / "value.json", {"a": 1})
            with self.assertRaises(FileExistsError):
                write_json(root / "value.json", {"a": 2})
            self.assertEqual(read_json(root / "value.json"), {"a": 1})
            with self.assertRaises(ValueError):
                write_json(root / "invalid.json", {"a": float("nan")})
            self.assertFalse((root / "invalid.json").exists())
            self.assertEqual((root.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((root / "value.json").stat().st_mode & 0o777), 0o600)

    def test_asset_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "outside").write_text("x")
            for name in ["../outside", str(root / "outside")]:
                with self.assertRaises(ValueError):
                    relative_asset(root / "data", name)

    def test_publication_allowlist(self):
        self.assertTrue(inspect_blob("private/scan.json", b"{}"))
        self.assertTrue(
            inspect_blob(
                "docs/note.md",
                ("/" + "Users" + "/" + "synthetic_user" + "/data").encode(),
            )
        )
        self.assertFalse(
            inspect_blob("body_scan/example.py", b'"""Synthetic geometry only."""')
        )

    def test_cli_end_to_end_and_no_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "body_scan",
                    "demo",
                    "--out",
                    str(Path(temp) / "demo"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(Path(temp) / "demo" / "report.json")
            self.assertFalse(report["physical_accuracy_validated"])
            self.assertTrue((Path(temp) / "demo" / "report.html").is_file())
            self.assertGreater(report["known_scale_false_delta_m"], 0.009)
            again = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "body_scan",
                    "demo",
                    "--out",
                    str(Path(temp) / "demo"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(again.returncode, 2)
            self.assertNotIn(temp, again.stderr)
            self.assertNotIn("Traceback", again.stderr)


if __name__ == "__main__":
    unittest.main()
