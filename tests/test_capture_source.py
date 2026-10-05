import copy
import tempfile
import unittest
from pathlib import Path

from body_scan.capture_manifest import build
from body_scan.capture_source import (
    from_capture_inputs,
    resolve_source,
    supported_observations,
)
from body_scan.io import InputError, write_json
from tests import test_capture_manifest


class CaptureSourceTests(unittest.TestCase):
    def fixture(self, root, kind="synthetic"):
        store, ids, cfg = test_capture_manifest.CaptureManifestTests().fixture(
            root, kind
        )
        file = root / "capture-inputs.json"
        write_json(file, build(store, ids[0], ids[1:], cfg))
        context = from_capture_inputs(file, cfg)
        observed = {
            **context.metadata,
            "diagnostic_only": True,
            "physical_accuracy_validated": False,
            "cases": [dict(row) for row in context.rows.values()],
        }
        return observed, store, ids, cfg

    def test_explicit_capture_roles_and_source_kind_need_no_renderer_report(self):
        for kind in ["real", "synthetic"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                observed, store, ids, cfg = self.fixture(Path(directory), kind)
                self.assertTrue(supported_observations(observed))
                context = resolve_source(observed, cfg)
                self.assertEqual(context.store, store)
                self.assertEqual(context.metadata["source_kind"], kind)
                self.assertTrue(context.is_reference(context.rows[ids[0]]))
                self.assertFalse(context.is_reference(context.rows[ids[1]]))
                self.assertNotIn("spec", context.rows[ids[0]])
                self.assertNotIn("truth", context.rows[ids[1]])

    def test_altered_cached_capture_or_frame_identity_is_rejected(self):
        for key, value in [
            ("name", "c" * 32),
            ("video_sha256", "1" * 64),
            ("frame_estimate_sha256", {}),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                observed, _, _, cfg = self.fixture(Path(directory))
                observed["cases"][1][key] = value
                with self.assertRaises(InputError):
                    resolve_source(observed, cfg)

    def test_changed_native_file_or_bound_source_metadata_is_rejected(self):
        for corrupt in ["native", "profile", "kind", "source"]:
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as directory,
            ):
                observed, store, ids, cfg = self.fixture(Path(directory), "real")
                if corrupt == "native":
                    (store / ids[1] / "attempt_one/frame_0.npz").write_bytes(b"changed")
                elif corrupt == "profile":
                    observed["profile_sha256"] = "1" * 64
                elif corrupt == "kind":
                    observed["source_kind"] = "synthetic"
                else:
                    observed["source"] = str(Path(directory))
                with self.assertRaises(InputError):
                    resolve_source(observed, cfg)

    def test_changed_input_manifest_and_foreign_runtime_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            observed, _, _, cfg = self.fixture(Path(directory))
            foreign = Path(directory) / "foreign.json"
            write_json(foreign, {"engine_id": "other"})
            with self.assertRaises(InputError):
                resolve_source(observed, foreign)
            Path(observed["capture_inputs_manifest"]).write_text("{}")
            with self.assertRaises(InputError):
                resolve_source(observed, cfg)

    def test_plain_real_cache_cannot_bypass_direct_capture_provenance(self):
        self.assertFalse(
            supported_observations({"source_kind": "real", "diagnostic_only": True})
        )
        with self.assertRaises(InputError):
            resolve_source({"source_kind": "real", "diagnostic_only": True})

    def test_direct_inputs_cannot_enter_renderer_truth_evaluation(self):
        from body_scan.synthetic_refit import evaluate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed, _, _, _ = self.fixture(root)
            cache = root / "cache"
            cache.mkdir()
            write_json(cache / "cache-manifest.json", copy.deepcopy(observed))
            with self.assertRaisesRegex(InputError, "requires a synthetic lab report"):
                evaluate(cache, root / "unused-output")
