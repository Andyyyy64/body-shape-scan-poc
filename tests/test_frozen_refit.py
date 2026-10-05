import importlib.util
import tempfile
import unittest
from pathlib import Path

from body_scan.io import InputError, replace_json, write_json


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class FrozenRefitTests(unittest.TestCase):
    def test_real_cache_and_changed_synthetic_observations_are_rejected(self):
        from body_scan.synthetic_refit import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            write_json(
                cache / "cache-manifest.json",
                {"source_kind": "real", "diagnostic_only": True, "cases": []},
            )
            with self.assertRaises(InputError):
                run(cache, root / "real-output", {})
            replace_json(
                cache / "cache-manifest.json",
                {
                    "source_kind": "synthetic",
                    "diagnostic_only": True,
                    "cases": [
                        {
                            "name": "baseline",
                            "cache": "data.npz",
                            "cache_sha256": "0" * 64,
                        }
                    ],
                },
            )
            (cache / "data.npz").write_bytes(b"changed")
            with self.assertRaises(InputError):
                run(cache, root / "changed-output", {})
