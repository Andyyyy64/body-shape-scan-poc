import tempfile
import unittest
from pathlib import Path

from body_scan.io import read_json, sha256, write_json
from body_scan.synthetic_replay import replay


class ReplayMethodTests(unittest.TestCase):
    def test_default_replay_does_not_silently_select_an_empty_research_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            session = source / "store" / ("a" * 32)
            session.mkdir(parents=True)
            (session / "capture.bin").write_bytes(b"local synthetic input")
            write_json(
                session / "session.json",
                {
                    "source_kind": "synthetic",
                    "video_sha256": sha256(session / "capture.bin"),
                },
            )
            write_json(session / "truth.json", {"source_kind": "synthetic"})
            for name in ("body.npz", "body.json", "protocol.json"):
                (source / name).write_bytes(b"fixture")
            write_json(
                source / "matrix.json",
                {
                    "cases": [
                        {
                            "name": "baseline",
                            "spec": {"group": "baseline"},
                            "state": "complete",
                            "session_id": "a" * 32,
                        }
                    ]
                },
            )
            write_json(root / "config.json", {"engine_id": "fixture-engine"})
            default = replay(
                str(source), str(root / "default"), str(root / "config.json"), None
            )
            self.assertNotIn("research_options", read_json(default / "matrix.json"))
            self.assertFalse((default / "research-options.json").exists())
            explicit = replay(
                str(source), str(root / "research"), str(root / "config.json"), {}
            )
            self.assertEqual(
                read_json(explicit / "matrix.json")["research_options"], {}
            )
            self.assertTrue((explicit / "research-options.json").exists())
            self.assertEqual(
                sha256(default / "store" / ("a" * 32) / "capture.bin"),
                sha256(session / "capture.bin"),
            )
