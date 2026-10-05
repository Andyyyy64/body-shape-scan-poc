import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from body_scan.io import InputError, read_json, replace_json, sha256, write_json


class CaptureMeasurementTests(unittest.TestCase):
    def fixture(self, root):
        cache = root / "images"
        cache.mkdir()
        file = cache / "capture.npz"
        np.savez_compressed(
            file, frames=np.arange(2), shared_target=np.ones((2, 1024, 2))
        )
        frames = {
            "0": hashlib.sha256(b"0").hexdigest(),
            "1": hashlib.sha256(b"1").hexdigest(),
        }
        row = {
            "name": "capture",
            "cache": file.name,
            "cache_sha256": sha256(file),
            "video_sha256": "0" * 64,
            "frame_estimate_sha256": frames,
        }
        context = SimpleNamespace(
            metadata={"capture_inputs_sha256": "1" * 64}, rows={"capture": row}
        )
        write_json(
            cache / "cache-manifest.json", {**context.metadata, "cases": [dict(row)]}
        )
        return cache, context

    def test_bound_image_rows_preserve_all_dense_frames_and_detect_source_changes(self):
        from body_scan.capture_measurement import image_target

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, context = self.fixture(root)
            data = {"frames": np.arange(2), "shared_source_order": np.array([1, 0])}
            image_target(root, "capture", data, context)
            self.assertEqual(data["width_target"].shape, (2, 1024, 2))
            path = cache / "cache-manifest.json"
            manifest = read_json(path)
            manifest["capture_inputs_sha256"] = "2" * 64
            replace_json(path, manifest)
            with self.assertRaises(InputError):
                image_target(root, "capture", data, context)

    def test_changed_estimates_duplicate_case_or_frame_mismatch_are_rejected(self):
        from body_scan.capture_measurement import image_target, load

        for corrupt in ["file", "duplicate", "frame", "video"]:
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cache, context = self.fixture(root)
                data = {"frames": np.arange(2), "shared_source_order": np.arange(2)}
                file = cache / "cache-manifest.json"
                manifest = read_json(file)
                if corrupt == "file":
                    (cache / "capture.npz").write_bytes(b"changed cached observation")
                elif corrupt == "duplicate":
                    manifest["cases"].append(dict(manifest["cases"][0]))
                    replace_json(file, manifest)
                elif corrupt == "frame":
                    data["frames"] = np.array([0, 3])
                else:
                    manifest["cases"][0]["video_sha256"] = "3" * 64
                    replace_json(file, manifest)
                with self.assertRaises(InputError):
                    image_target(root, "capture", data, context)
                if corrupt in ["file", "duplicate"]:
                    with self.assertRaises(InputError):
                        load(cache, "capture")
