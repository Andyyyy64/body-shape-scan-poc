import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from body_scan.io import InputError, new_run, read_json, sha256, write_json
from body_scan.issue52 import CANDIDATE, SECTION, SLOTS, bind, prepare, process, score
from body_scan.mesh import synthetic_mesh
from body_scan.pipeline import run
from body_scan.web import capture_resolution, record_resolution_fallback, store_root


class Issue52Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.config = self.base / "config.json"
        write_json(
            self.config,
            {
                "sample_frames": 16,
                "max_rss_mb": 8192,
                "network_sandbox": "linux",
                "python": "python",
            },
        )
        self.root = prepare(str(self.base / "review"), str(self.config), 120)

    def tearDown(self):
        self.temp.cleanup()

    def captures(self):
        for i, slot in enumerate(SLOTS):
            p = new_run(self.root / "captures" / f"{i + 1:032x}")
            (p / "capture.bin").write_bytes(slot.encode())
            write_json(
                p / "session.json",
                {
                    "source_kind": "real",
                    "created_at": time.time(),
                    "issue52_slot": slot,
                    "video_sha256": sha256(p / "capture.bin"),
                },
            )

    def test_fixed_shape_fails_even_when_null_is_zero(self):
        result = score({s: 1.0 for s in SLOTS}, 20)
        self.assertFalse(result["sensitivity"]["pass"])
        self.assertEqual(result["sensitivity"]["change_min_directed_model_m"], 0)
        self.assertEqual(result["sensitivity"]["null_pair_count"], 6)
        self.assertEqual(result["distance"]["cross_distance_pair_count"], 12)
        self.assertEqual(len(result["all_66_pairs"]), 66)

    def test_all_pairs_direction_and_threefold_margin(self):
        values = dict.fromkeys(SLOTS, 1.0)
        values.update(
            normal_1=1.0,
            normal_2=1.001,
            normal_3=1.002,
            changed_1=1.008,
            changed_2=1.009,
            changed_3=1.01,
        )
        self.assertTrue(score(values, 20)["sensitivity"]["pass"])
        self.assertFalse(score(values, -20)["sensitivity"]["pass"])
        values["changed_3"] = 1.02  # Same changed condition is also null variation.
        self.assertFalse(score(values, 20)["sensitivity"]["pass"])
        values["distance_2.5_2"] = 1.02
        result = score(values, 20)
        self.assertFalse(result["distance"]["within_1_percent"])
        self.assertAlmostEqual(result["distance"]["maximum_percent"], 200 * 0.02 / 2.02)
        del values["normal_2"]
        with self.assertRaises(InputError):
            score(values, 20)

    def test_missing_duplicate_retakes_and_changed_inventory_cannot_be_silent(self):
        with self.assertRaises(InputError):
            bind(self.root, 20, "camera positions marked; change measured", [])
        self.captures()
        extra = new_run(self.root / "captures" / ("f" * 32))
        (extra / "capture.bin").write_bytes(b"retake")
        write_json(
            extra / "session.json",
            {
                "source_kind": "real",
                "created_at": time.time(),
                "issue52_slot": "normal_1",
                "video_sha256": sha256(extra / "capture.bin"),
            },
        )
        with self.assertRaises(InputError):
            bind(self.root, 20, "retake before processing", [])
        bind(
            self.root,
            20,
            "extra accidental recording; select original before processing",
            [f"normal_1={1:032x}"],
        )
        selected = read_json(self.root / "selection.json")
        self.assertEqual(len(selected["all_captures_and_retakes"]), 13)
        self.assertEqual(selected["unselected_ids"], ["f" * 32])
        (extra / "capture.bin").write_bytes(b"changed")
        with self.assertRaises(InputError):
            process(self.root, 120)
        self.assertFalse((self.root / "execution.json").exists())

    def test_second_processing_is_refused_and_constant_mesh_is_reported_failed(self):
        self.captures()
        bind(
            self.root,
            20,
            "physically measured change and all captures within one hour",
            [],
        )
        calls = []

        def worker(root, method, store, identifier, deadline):
            calls.append((method, identifier))
            mesh = synthetic_mesh()
            mesh.update(source_kind="real", units="model_m", method=method)
            mesh["sections"][0]["id"] = SECTION
            write_json(store / identifier / "mesh.json", mesh)
            return {
                "actual_exit_code": 0,
                "timed_out": False,
                "attempts": [],
                "state": {"state": "complete"},
            }

        with (
            patch("body_scan.issue52.verify_network_sandbox"),
            patch("body_scan.issue52._run_worker", side_effect=worker),
        ):
            result = process(self.root, 120)
            self.assertEqual(len(calls), 24)
            for r in result["methods"].values():
                self.assertEqual(r["decision"], "sensitivity_fail_defer")
                self.assertFalse(r["result"]["fixed_shape_control"]["pass"])
            with self.assertRaises(FileExistsError):
                process(self.root, 120)
            self.assertEqual(len(calls), 24)
        self.assertTrue((self.root / "issue52-summary-draft.md").exists())

    def test_failed_recording_is_retained_without_retry_or_score(self):
        self.captures()
        bind(self.root, 20, "capture conditions confirmed", [])
        failed = {
            "actual_exit_code": 75,
            "state": {"state": "failed", "stage": "resource_guard"},
            "attempts": [
                {"frames": {"frames": [{"state": "excluded", "reason": "no_person"}]}}
            ],
        }
        with (
            patch("body_scan.issue52.verify_network_sandbox"),
            patch("body_scan.issue52._run_worker", return_value=failed) as worker,
        ):
            result = process(self.root, 120)
            self.assertEqual(worker.call_count, 1)
            self.assertIsNotNone(result["stop_reason"])
            self.assertNotIn("result", result["methods"]["v1"])
            self.assertEqual(result["methods"]["v1"]["jobs"]["normal_1"], failed)

    def test_research_real_requires_flag_and_rejects_renderer_camera(self):
        store = store_root(str(self.base / "guard-store"))
        identifier = "1" * 32
        session = new_run(store / identifier)
        write_json(session / "session.json", {"source_kind": "real"})
        options = self.base / "options.json"
        write_json(options, CANDIDATE)
        with self.assertRaisesRegex(InputError, "synthetic experiment"):
            run(str(self.config), str(store), identifier, research_options=str(options))
        with self.assertRaisesRegex(InputError, "saved video"):
            run(
                str(self.config),
                str(store),
                identifier,
                research_options=str(options),
                diagnostic_real=True,
            )
        with self.assertRaisesRegex(InputError, "explicit research options"):
            run(str(self.config), str(store), identifier, diagnostic_real=True)
        renderer = copy.deepcopy(CANDIDATE)
        renderer["camera"] = {
            "width": 640,
            "height": 640,
            "fx": 430.0,
            "fy": 430.0,
            "cx": 320.0,
            "cy": 320.0,
            "provenance": "declared_synthetic_renderer_intrinsics",
        }
        other = self.base / "renderer.json"
        write_json(other, renderer)
        with self.assertRaisesRegex(InputError, "renderer intrinsics"):
            run(
                str(self.config),
                str(store),
                identifier,
                research_options=str(other),
                diagnostic_real=True,
            )

    def test_rss_fallback_persists_but_low_available_memory_does_not_trigger(self):
        captures = self.root / "captures"
        self.assertEqual(capture_resolution(captures)["width"], 1920)
        state = {
            "stage": "resource_guard",
            "max_rss_mb": 8192,
            "rss_limit_exceeded": False,
            "peak_rss_mb": 4000,
        }
        self.assertFalse(record_resolution_fallback(captures, "1" * 32, state))
        state.update(rss_limit_exceeded=True, peak_rss_mb=8193)
        self.assertTrue(record_resolution_fallback(captures, "1" * 32, state))
        self.assertEqual(capture_resolution(captures)["height"], 720)
        self.assertTrue(
            read_json(captures / "capture-resolution.json")["original_video_preserved"]
        )


if __name__ == "__main__":
    unittest.main()
