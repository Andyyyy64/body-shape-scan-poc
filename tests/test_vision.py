import importlib.util
import tempfile
import unittest
from pathlib import Path

from body_scan.io import read_json, write_json

VISION = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(VISION, "optional vision extra not installed")
class VisionTests(unittest.TestCase):
    def test_rendered_geometry_and_known_delta(self):
        from body_scan.demo import render_demo

        with tempfile.TemporaryDirectory() as temp:
            before = render_demo(str(Path(temp) / "before"))
            after = render_demo(str(Path(temp) / "after"), -0.01)
            self.assertLess(abs(before["error_m"]), 0.01)
            self.assertLess(
                abs((after["reconstructed_m"] - before["reconstructed_m"]) + 0.01),
                0.004,
            )
            self.assertAlmostEqual(
                after["reference_perimeter_m"] - before["reference_perimeter_m"],
                -0.01,
                places=6,
            )
            self.assertFalse(after["physical_accuracy_validated"])

    def test_bad_scale_rotation_mask_and_coverage(self):
        from body_scan.demo import render_demo
        from body_scan.vision import reconstruct

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "scan"
            render_demo(str(root))
            data = read_json(root / "scan.json")
            data["scale_evidence_id"] = None
            write_json(root / "unscaled.json", data)
            self.assertEqual(
                reconstruct(str(root / "unscaled.json"), str(root / "unscaled"))[
                    "status"
                ],
                "unscaled",
            )
            data["scale_evidence_id"] = "synthetic_scale"
            data["views"][0]["R"][0][0] = 2.0
            write_json(root / "bad_rotation.json", data)
            with self.assertRaises(ValueError):
                reconstruct(str(root / "bad_rotation.json"), str(root / "bad_rotation"))
            data = read_json(root / "scan.json")
            data["views"] = data["views"][:4]
            write_json(root / "bad_coverage.json", data)
            with self.assertRaises(ValueError):
                reconstruct(str(root / "bad_coverage.json"), str(root / "bad_coverage"))
            data = read_json(root / "scan.json")
            data["camera"]["image_size"] = [100, 100]
            write_json(root / "bad_size.json", data)
            with self.assertRaises(ValueError):
                reconstruct(str(root / "bad_size.json"), str(root / "bad_size"))

    def test_video_capture_and_extract(self):
        from body_scan.vision import extract, libraries, record

        cv2, np = libraries()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "synthetic.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
            )
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((120, 160, 3), i * 15, np.uint8))
            writer.release()
            self.assertEqual(
                record(
                    str(source),
                    str(root / "record"),
                    10.0,
                    False,
                    source_kind="synthetic",
                )["frames"],
                12,
            )
            self.assertEqual(
                extract(
                    str(root / "record" / "capture.avi"), str(root / "frames"), 3, 100
                )["frames"],
                4,
            )
            frames = read_json(root / "frames" / "frames.json")["frames"]
            self.assertEqual([x["frame_index"] for x in frames], [0, 3, 6, 9])
            self.assertTrue(all(x["yaw_rad"] is None for x in frames))

    def test_calibration_and_undistortion_round_trip(self):
        from body_scan.vision import calibrate, libraries, save_image, undistort

        cv2, np = libraries()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "boards"
            images.mkdir()
            columns, rows, cell = 7, 5, 80
            board = np.full(((rows + 1) * cell, (columns + 1) * cell), 255, np.uint8)
            for y in range(rows + 1):
                for x in range(columns + 1):
                    if (x + y) % 2 == 0:
                        board[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = 0
            K = np.array([[780.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
            world = np.array(
                [[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.20, 0.15, 0.0], [0.0, 0.15, 0.0]]
            )
            corners = np.array(
                [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]], np.float32
            )
            frames = []
            for i in range(12):
                r = np.array([-0.22 + 0.04 * i, 0.12 * np.sin(i), -0.15 + 0.025 * i])
                t = np.array(
                    [-0.1 + 0.006 * (i % 3), -0.07 + 0.006 * (i % 4), 0.55 + 0.025 * i]
                )
                uv, _ = cv2.projectPoints(world, r, t, K, np.zeros(5))
                transform = cv2.getPerspectiveTransform(
                    corners, uv.reshape(-1, 2).astype("float32")
                )
                image = cv2.warpPerspective(
                    board, transform, (640, 480), borderValue=200
                )
                name = f"synthetic_board_{i}.png"
                save_image(images / name, image)
                frames.append({"image": name, "frame_index": i, "yaw_rad": None})
            write_json(images / "frames.json", {"frames": frames})
            info = calibrate(str(images), str(root / "calib"), columns, rows, 0.025)
            self.assertGreaterEqual(info["accepted"], 8)
            self.assertLess(info["rms_px"], 0.5)
            calibration = read_json(root / "calib" / "calibration.json")
            self.assertLess(abs(calibration["K"][0][0] - 780.0) / 780.0, 0.05)
            outcome = undistort(
                str(images),
                str(root / "calib" / "calibration.json"),
                str(root / "rectified"),
            )
            self.assertEqual(outcome["frames"], 12)
            adjusted = read_json(root / "rectified" / "calibration.json")
            self.assertEqual(adjusted["distortion"], [0.0] * 5)
            self.assertEqual(adjusted["K"], calibration["K"])
            self.assertFalse(adjusted["metric_body_pose_established"])

    def test_network_sources_rejected_before_capture(self):
        from body_scan.vision import extract, record

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with self.assertRaises(ValueError):
                record("https://example.invalid/video", str(out), 1.0, False)
            with self.assertRaises(ValueError):
                extract("https://example.invalid/video", str(out), 1, 1)
            self.assertFalse(out.exists())

    def test_declared_angles_cannot_fake_actual_pose_coverage(self):
        from body_scan.demo import render_demo
        from body_scan.vision import reconstruct

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "scan"
            render_demo(str(root))
            data = read_json(root / "scan.json")
            for view in data["views"]:
                view["R"] = data["views"][0]["R"]
                view["t_m"] = data["views"][0]["t_m"]
            write_json(root / "false_coverage.json", data)
            with self.assertRaisesRegex(ValueError, "coverage"):
                reconstruct(
                    str(root / "false_coverage.json"), str(root / "false_coverage")
                )

    def test_capture_cli_preserves_delay_and_source_provenance(self):
        import contextlib
        import io
        from unittest.mock import patch

        from body_scan.__main__ import main
        from body_scan.vision import libraries

        cv2, np = libraries()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "synthetic.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
            )
            writer.write(np.zeros((120, 160, 3), np.uint8))
            writer.release()
            argv = [
                "body-scan",
                "capture",
                "--source",
                str(video),
                "--out",
                str(root / "record"),
                "--delay-s",
                "0.25",
                "--source-kind",
                "synthetic",
            ]
            with (
                patch("sys.argv", argv),
                patch("body_scan.vision.time.sleep") as sleep,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
            sleep.assert_called_once_with(0.25)
            self.assertEqual(
                read_json(root / "record" / "capture.json")["source_kind"], "synthetic"
            )

    def test_sample_video_without_trusting_index_metadata(self):
        from body_scan.vision import libraries, sampled_video_frames

        cv2, np = libraries()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
            )
            for i in range(11):
                writer.write(np.full((120, 160, 3), i * 20, np.uint8))
            writer.release()
            frames, count = sampled_video_frames(str(path), 3)
            self.assertEqual(count, 11)
            self.assertEqual([i for i, _ in frames], [0, 5, 10])
            with self.assertRaises(ValueError):
                sampled_video_frames(str(path), 3, max_frames=5)

    def test_camera_invalid_matrix(self):
        from body_scan.vision import camera

        for K in [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 2]]]:
            with self.assertRaises(ValueError):
                camera({"K": K, "distortion": [0.0] * 5, "image_size": [640, 480]})


if __name__ == "__main__":
    unittest.main()
