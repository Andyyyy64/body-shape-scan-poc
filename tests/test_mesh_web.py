import copy
import http.client
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from body_scan.io import read_json
from body_scan.mesh import (
    compare_meshes,
    section_perimeter,
    synthetic_mesh,
    validate_mesh,
)
from body_scan.web import server, sessions, store_root


class MeshTests(unittest.TestCase):
    def test_identical_and_local_change_on_same_day(self):
        before = synthetic_mesh()
        after = synthetic_mesh(0.003)
        before["capture_date"] = after["capture_date"] = "synthetic_same_day"
        same = compare_meshes(before, copy.deepcopy(before))
        changed = compare_meshes(before, after)
        self.assertEqual(same["rms_mm"], 0.0)
        self.assertEqual(same["sections"][0]["delta_mm"], 0.0)
        self.assertGreater(changed["rms_mm"], 0.5)
        self.assertGreater(changed["sections"][0]["delta_mm"], 10.0)
        self.assertEqual(changed["biological_change"], "not_assessed")
        self.assertEqual(changed["tolerance_result"], "not_set")
        self.assertFalse(changed["physical_accuracy_validated"])

    def test_no_alignment_erases_change(self):
        before = synthetic_mesh()
        shifted = copy.deepcopy(before)
        shifted["vertices"] = [[x + 0.01, y, z] for x, y, z in shifted["vertices"]]
        self.assertAlmostEqual(
            compare_meshes(before, shifted)["rms_mm"], 10.0, places=6
        )

    def test_comparison_contract_and_tolerance(self):
        before = synthetic_mesh()
        after = synthetic_mesh(0.002)
        self.assertEqual(
            compare_meshes(before, after, 100.0)["tolerance_result"],
            "within_user_tolerance",
        )
        self.assertEqual(
            compare_meshes(before, after, 0.001)["tolerance_result"],
            "exceeds_user_tolerance",
        )
        for key, value in [
            ("units", "m"),
            ("source_kind", "real"),
            ("reference_frame_id", "different"),
            ("scale_reference_id", "different"),
        ]:
            mismatch = copy.deepcopy(before)
            mismatch[key] = value
            with self.assertRaises(ValueError):
                compare_meshes(before, mismatch)
        for value in [0.0, -1.0, float("nan")]:
            with self.assertRaises(ValueError):
                compare_meshes(before, after, value)
        mismatch = copy.deepcopy(before)
        mismatch["faces"][0].reverse()
        with self.assertRaises(ValueError):
            compare_meshes(before, mismatch)
        mismatch = copy.deepcopy(before)
        mismatch["sections"][0]["y"] += 0.001
        with self.assertRaises(ValueError):
            compare_meshes(before, mismatch)

    def test_mesh_validation(self):
        for change in [
            lambda m: m["vertices"][0].__setitem__(0, float("nan")),
            lambda m: m["faces"][0].__setitem__(0, -1),
            lambda m: m.update(schema_version="posed_mesh.v1"),
        ]:
            mesh = synthetic_mesh()
            change(mesh)
            with self.assertRaises(ValueError):
                validate_mesh(mesh)

    def test_fixed_slice_ignores_separate_arm_loop(self):
        mesh = synthetic_mesh()
        expected = section_perimeter(mesh, mesh["sections"][0])
        offset = len(mesh["vertices"])
        extra = synthetic_mesh()
        mesh["vertices"] += [[x + 1.0, y, z] for x, y, z in extra["vertices"]]
        mesh["faces"] += [[i + offset for i in f] for f in extra["faces"]]
        self.assertAlmostEqual(section_perimeter(mesh, mesh["sections"][0]), expected)
        self.assertIsNone(section_perimeter(mesh, {"y": 0.0, "center_xz": [0.0, 0.0]}))


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = store_root(str(Path(self.temp.name) / "store"))
        self.http = server(self.root)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.http.server_port}"
        status, body, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.token = (
            re.search(rb'const TOKEN\s*=\s*"([A-Za-z0-9_-]+)"', body).group(1).decode()
        )

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.http.server_port, timeout=5
        )
        merged = {"X-Scan-Session": getattr(self, "token", ""), "Origin": self.origin}
        if headers:
            merged.update(headers)
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        output = (response.status, response.read(), dict(response.getheaders()))
        connection.close()
        return output

    def test_origin_host_token_and_paths(self):
        for headers in [
            {"Origin": "https://example.invalid"},
            {"Host": "example.invalid"},
            {"X-Scan-Session": "wrong"},
        ]:
            self.assertEqual(self.request("POST", "/api/demo", b"", headers)[0], 403)
        self.assertEqual(
            self.request("GET", "/api/sessions", headers={"X-Scan-Session": "wrong"})[
                0
            ],
            403,
        )
        self.assertEqual(self.request("GET", "/api/video/../store.json")[0], 404)
        self.assertEqual(sessions(self.root), [])

    def test_demo_persistence_and_comparison(self):
        status, body, _ = self.request("POST", "/api/demo", b"")
        self.assertEqual(status, 201)
        ids = json.loads(body)["ids"]
        self.assertEqual(len(sessions(store_root(str(self.root)))), 3)
        for after, expected_zero in [(ids[1], True), (ids[2], False)]:
            payload = json.dumps({"before": ids[0], "after": after}).encode()
            status, body, _ = self.request("POST", "/api/compare", payload)
            self.assertEqual(status, 200)
            result = json.loads(body)
            self.assertEqual(result["rms_mm"] == 0, expected_zero)
            self.assertEqual(result["source_kind"], "synthetic")
            self.assertEqual(result["biological_change"], "not_assessed")
        self.assertEqual(len(list(self.root.glob("comparison_*/comparison.json"))), 2)
        self.assertEqual(self.request("GET", "/api/mesh/" + ids[0])[0], 200)
        self.assertEqual(
            self.request(
                "POST", "/api/compare", json.dumps({"before": ids[0], "after": ids[0]})
            )[0],
            400,
        )

    def test_recorded_upload_does_not_create_mesh(self):
        # Container signature tests only; not a claim that this test byte sequence is playable video.
        payload = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32
        status, body, _ = self.request(
            "POST", "/api/captures", payload, {"Content-Type": "video/mp4"}
        )
        self.assertEqual(status, 201)
        identifier = json.loads(body)["id"]
        item = sessions(self.root)[0]
        self.assertTrue(item["has_video"])
        self.assertFalse(item["has_mesh"])
        self.assertEqual(item["source_kind"], "real")
        status, download, _ = self.request("GET", "/api/video/" + identifier)
        self.assertEqual(status, 200)
        self.assertEqual(download, payload)
        self.assertEqual(
            read_json(self.root / identifier / "session.json")["state"],
            "recorded_not_reconstructed",
        )
        self.assertEqual(
            self.request(
                "POST", "/api/captures", b"not a video", {"Content-Type": "video/mp4"}
            )[0],
            400,
        )

    def test_server_does_not_claim_model_ready(self):
        status, body, headers = self.request("GET", "/api/sessions")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["sam_state"], "not_connected")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn(str(self.root).encode(), body)


class AdditionalContracts(unittest.TestCase):
    def test_torso_metrics_exclude_unobserved_vertices(self):
        before = synthetic_mesh()
        before["comparison_vertex_indices"] = [0, 1, 2, 3]
        after = copy.deepcopy(before)
        after["vertices"][-1][0] += 0.01
        result = compare_meshes(before, after)
        self.assertEqual(result["rms_mm"], 0.0)
        self.assertFalse(result["identical_geometry"])
        self.assertEqual(result["metric_region"], "torso")
        before["capture_sha256"] = after["capture_sha256"] = "synthetic_duplicate"
        self.assertTrue(compare_meshes(before, after)["duplicate_capture_input"])

    def test_store_lock_and_session_alias(self):
        from body_scan.web import demo_sessions, session_path

        with tempfile.TemporaryDirectory() as temp:
            root = store_root(str(Path(temp) / "store"))
            first = server(root)
            try:
                with self.assertRaises(ValueError):
                    server(root)
                ids = demo_sessions(root)
                alias = "f" * 32
                (root / alias).symlink_to(root / ids[0])
                with self.assertRaises(ValueError):
                    session_path(root, alias)
            finally:
                first.server_close()

    def test_archive_traversal_is_rejected(self):
        import io
        import tarfile

        from body_scan.runtime_setup import extract_source

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            item = tarfile.TarInfo("prefix/../../outside")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                extract_source(buffer.getvalue(), Path(temp) / "source")
            self.assertFalse((Path(temp) / "outside").exists())


if __name__ == "__main__":
    unittest.main()
