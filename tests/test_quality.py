import importlib.util
import unittest

NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY, "optional vision extra not installed")
class PrimaryPersonTests(unittest.TestCase):
    def test_static_object_is_ignored_but_second_person_is_ambiguous(self):
        from body_scan.quality import FrameRejected, select_primary_person

        self.assertEqual(select_primary_person([96000.0]), 0)
        # A bag-sized false detection next to the operator.
        self.assertEqual(select_primary_person([9000.0, 96000.0]), 1)
        with self.assertRaises(FrameRejected) as ambiguous:
            select_primary_person([60000.0, 96000.0])
        self.assertEqual(ambiguous.exception.reason, "multiple_people")
        for empty in ([], [0.0], [float("nan")]):
            with self.assertRaises(FrameRejected) as missing:
                select_primary_person(empty)
            self.assertEqual(missing.exception.reason, "no_person")


@unittest.skipUnless(NUMPY, "optional vision extra not installed")
class ProtocolTests(unittest.TestCase):
    def relaxed(self):
        import numpy as np

        from body_scan import quality as q

        points = np.full((q.KEYPOINT_COUNT, 2), 320.0)
        points[q.NOSE] = [320, 40]
        points[q.LEFT_SHOULDER], points[q.RIGHT_SHOULDER] = [270, 120], [370, 120]
        points[q.LEFT_ELBOW], points[q.RIGHT_ELBOW] = [250, 220], [390, 220]
        points[q.LEFT_WRIST], points[q.RIGHT_WRIST] = [240, 320], [400, 320]
        points[q.LEFT_HIP], points[q.RIGHT_HIP] = [290, 330], [350, 330]
        return points

    def test_relaxed_arms_down_full_torso_passes(self):
        from body_scan.quality import frame_protocol_violations

        self.assertEqual(frame_protocol_violations(self.relaxed(), 640, 480), [])

    def test_raised_or_crossed_arm_is_excluded(self):
        from body_scan import quality as q

        raised = self.relaxed()
        raised[q.RIGHT_WRIST] = [400, 60]
        self.assertEqual(
            q.frame_protocol_violations(raised, 640, 480), ["arm_not_lowered"]
        )
        crossed = self.relaxed()
        crossed[q.LEFT_WRIST] = [340, 200]  # hand at the chest, above the elbow
        self.assertEqual(
            q.frame_protocol_violations(crossed, 640, 480), ["arm_not_lowered"]
        )

    def test_head_or_hips_outside_the_image_is_excluded(self):
        from body_scan import quality as q

        cut = self.relaxed()
        cut[q.NOSE] = [320, -5]
        self.assertEqual(
            q.frame_protocol_violations(cut, 640, 480), ["core_out_of_frame"]
        )
        close = self.relaxed()
        close[q.LEFT_HIP] = [290, 479]
        self.assertEqual(
            q.frame_protocol_violations(close, 640, 480), ["core_out_of_frame"]
        )
        # Wrists may leave the image; only the head-to-pelvis core is required.
        wide = self.relaxed()
        wide[q.RIGHT_WRIST] = [700, 320]
        self.assertEqual(q.frame_protocol_violations(wide, 640, 480), [])

    def test_invalid_inputs_are_rejected(self):
        import numpy as np

        from body_scan.quality import frame_protocol_violations

        with self.assertRaises(ValueError):
            frame_protocol_violations(np.zeros((17, 2)), 640, 480)
        bad = self.relaxed()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            frame_protocol_violations(bad, 640, 480)

    def test_capture_summary_and_minimum(self):
        from body_scan.quality import minimum_usable_frames, summarize_outcomes

        self.assertEqual(minimum_usable_frames(16), 12)
        self.assertEqual(minimum_usable_frames(1), 1)
        with self.assertRaises(ValueError):
            minimum_usable_frames(0)
        summary = summarize_outcomes(
            [
                {"frame": 0, "state": "estimated"},
                {"frame": 1, "state": "failed", "reason": "no_person"},
                {
                    "frame": 2,
                    "state": "excluded",
                    "reason": "arm_not_lowered",
                    "reasons": ["arm_not_lowered", "core_out_of_frame"],
                },
            ]
        )
        self.assertEqual(summary["excluded_frames"], 2)
        self.assertEqual(
            summary["exclusion_reasons"],
            {"arm_not_lowered": 1, "core_out_of_frame": 1, "no_person": 1},
        )
        self.assertEqual(
            summary["capture_protocol_id"], "relaxed_arms_down_full_torso.v1"
        )
