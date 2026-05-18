import json
import logging
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

import api_server


class ApiServerTests(unittest.TestCase):
    def make_null_logger(self, name: str):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = False
        logger.addHandler(logging.NullHandler())
        return logger

    def test_parse_request_accepts_existing_protocol(self):
        parsed = api_server.parse_request('ONLINE;31415;{"point_id": 123}')

        self.assertEqual(parsed.req_type, "ONLINE")
        self.assertEqual(parsed.param, "31415")
        self.assertEqual(parsed.arg, '{"point_id": 123}')

    def test_convert_provider_data_matches_main_online_shape(self):
        raw = {
            "isLive": True,
            "mode": 2,
            "focus_depth": "7.5",
            "guankuan_a": "10.1",
            "guankuan_b": "20.2",
            "depth": "35",
            "focus_point": "F1",
        }

        self.assertEqual(
            api_server.convert_provider_data(raw),
            {
                "SkinDepth": 7.5,
                "A": 10.1,
                "B": 20.2,
                "Alpha": None,
                "Depth": 35,
                "IsFreeze": False,
                "isHIFU": True,
                "FocusPoint": "F1",
            },
        )

    def test_convert_provider_data_normalizes_online_numbers(self):
        raw = {
            "isLive": None,
            "mode": 1,
            "focus_depth": "7.555",
            "guankuan_a": "10.005",
            "guankuan_b": 20.0,
            "depth": None,
            "focus_point": None,
        }

        self.assertEqual(
            api_server.convert_provider_data(raw),
            {
                "SkinDepth": 7.56,
                "A": 10.01,
                "B": 20,
                "Alpha": None,
                "Depth": None,
                "IsFreeze": None,
                "isHIFU": False,
                "FocusPoint": None,
            },
        )

    def test_convert_provider_data_keeps_non_numeric_values(self):
        raw = {
            "isLive": True,
            "mode": 1,
            "focus_depth": "",
            "guankuan_a": "NaN",
            "guankuan_b": "not-a-number",
            "depth": None,
            "focus_point": "PointF(434.85052, 272.8398)",
        }

        self.assertEqual(
            api_server.convert_provider_data(raw),
            {
                "SkinDepth": "",
                "A": "NaN",
                "B": "not-a-number",
                "Alpha": None,
                "Depth": None,
                "IsFreeze": False,
                "isHIFU": False,
                "FocusPoint": "PointF(434.85052, 272.8398)",
            },
        )

    def test_online_reads_provider_and_returns_json(self):
        response = api_server.handle_request(
            'ONLINE;31415;{"point_id": 123}',
            provider_fetcher=lambda: {"isLive": False, "mode": 1, "depth": "40"},
        )

        payload = json.loads(response)
        self.assertEqual(payload["Depth"], 40)
        self.assertTrue(payload["IsFreeze"])
        self.assertFalse(payload["isHIFU"])

    def test_parse_focus_point_accepts_pointf_text(self):
        self.assertEqual(
            api_server.parse_focus_point("PointF(434.85052, 272.8398)"),
            (434, 272),
        )

    def test_compute_roi_region_uses_extension_params(self):
        self.assertEqual(
            api_server.compute_roi_region(
                (640, 480),
                (100, 120),
                {"left": 10, "right": 20, "top": 30, "bottom": 40},
            ),
            (90, 90, 120, 160),
        )

    def test_compute_roi_region_rejects_out_of_bounds(self):
        self.assertIsNone(
            api_server.compute_roi_region(
                (640, 480),
                (5, 120),
                {"left": 10, "right": 20, "top": 30, "bottom": 40},
            )
        )

    def test_parse_offline_config_reads_roi_and_debug_settings(self):
        config = api_server.parse_offline_config(
            {
                "peak_detect": {
                    "roi2_extension_params": {"left": 11, "right": 12, "top": 13, "bottom": 14},
                    "roi3_extension_params": {"left": 21, "right": 22, "top": 23, "bottom": 24},
                    "difference_threshold": 1.5,
                },
                "offline_tmp_frames": {"enabled": True, "dir": "D:/software_data/tmp"},
            },
            self.make_null_logger("test_parse_offline_config_reads_roi_and_debug_settings"),
        )

        self.assertEqual(config.roi2_extension_params, {"left": 11, "right": 12, "top": 13, "bottom": 14})
        self.assertEqual(config.roi3_extension_params, {"left": 21, "right": 22, "top": 23, "bottom": 24})
        self.assertEqual(config.difference_threshold, 1.5)
        self.assertTrue(config.debug_save_enabled)
        self.assertEqual(config.debug_save_dir, "D:/software_data/tmp")

    def test_offline_start_fails_without_device_frame(self):
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
            frame_fetcher=lambda: None,
            config=api_server.OfflineConfig.default(),
            logger=self.make_null_logger("test_offline_start_fails_without_device_frame"),
        )

        result = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(result, {"success": False, "info": "no_device_frame", "point_id": 123})

    def test_offline_start_fails_without_focus_point(self):
        frame = api_server.FrameSnapshot(np.zeros((20, 20, 3), dtype=np.uint8), 1, 1.0)
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {},
            frame_fetcher=lambda: frame,
            config=api_server.OfflineConfig.default(),
            logger=self.make_null_logger("test_offline_start_fails_without_focus_point"),
        )

        result = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(result, {"success": False, "info": "missing_focus_point", "point_id": 123})

    def test_offline_two_signal_session_returns_green_roi2_result(self):
        frames = [
            api_server.FrameSnapshot(np.full((20, 20, 3), 10, dtype=np.uint8), 1, 1.0),
            api_server.FrameSnapshot(np.full((20, 20, 3), 20, dtype=np.uint8), 2, 2.0),
        ]
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
            frame_fetcher=lambda: frames.pop(0),
            config=api_server.OfflineConfig(
                roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                roi3_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                difference_threshold=5.0,
            ),
            logger=self.make_null_logger("test_offline_two_signal_session_returns_green_roi2_result"),
        )

        start = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')
        stop = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(start, {"success": True, "info": "offline_started", "point_id": 123})
        self.assertEqual(stop["success"], True)
        self.assertEqual(stop["info"], "offline_stop_completed")
        self.assertEqual(stop["roi2_color"], "green")
        self.assertEqual(stop["focus_anchor"], [10, 10])
        self.assertEqual(stop["roi2_rect"], [8, 7, 12, 13])
        self.assertEqual(stop["roi3_rect"], [8, 7, 12, 13])
        self.assertEqual(stop["roi2_before_mean"], 10.0)
        self.assertEqual(stop["roi2_after_mean"], 20.0)
        self.assertEqual(stop["roi2_diff"], 10.0)

    def test_offline_two_signal_session_returns_red_roi2_result(self):
        frames = [
            api_server.FrameSnapshot(np.full((20, 20, 3), 10, dtype=np.uint8), 1, 1.0),
            api_server.FrameSnapshot(np.full((20, 20, 3), 12, dtype=np.uint8), 2, 2.0),
        ]
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
            frame_fetcher=lambda: frames.pop(0),
            config=api_server.OfflineConfig(
                roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                roi3_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                difference_threshold=5.0,
            ),
            logger=self.make_null_logger("test_offline_two_signal_session_returns_red_roi2_result"),
        )

        manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')
        stop = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(stop["roi2_color"], "red")
        self.assertEqual(stop["roi2_diff"], 2.0)

    def test_offline_debug_save_writes_before_after_roi_images_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [
                api_server.FrameSnapshot(np.full((20, 20, 3), 10, dtype=np.uint8), 1, 1.0),
                api_server.FrameSnapshot(np.full((20, 20, 3), 20, dtype=np.uint8), 2, 2.0),
            ]
            manager = api_server.OfflineSessionManager(
                provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
                frame_fetcher=lambda: frames.pop(0),
                config=api_server.OfflineConfig(
                    roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                    roi3_extension_params={"left": 4, "right": 4, "top": 5, "bottom": 5},
                    difference_threshold=5.0,
                    debug_save_enabled=True,
                    debug_save_dir=tmp,
                ),
                logger=self.make_null_logger("test_offline_debug_save_writes_before_after_roi_images_and_meta"),
            )

            start = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')
            stop = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

            debug_dir = Path(start["debug_dir"])
            self.assertEqual(debug_dir, Path(stop["debug_dir"]))
            for name in (
                "before_roi1.png",
                "before_roi2.png",
                "before_roi3.png",
                "after_roi1.png",
                "after_roi2.png",
                "after_roi3.png",
                "meta.json",
            ):
                self.assertTrue((debug_dir / name).exists(), name)
            meta = json.loads((debug_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["point_id"], 123)
            self.assertEqual(meta["focus_anchor"], [10, 10])
            self.assertEqual(meta["roi2_rect"], [8, 7, 12, 13])
            self.assertEqual(meta["roi3_rect"], [6, 5, 14, 15])
            self.assertEqual(meta["result"]["roi2_color"], "green")

    def test_offline_debug_disabled_does_not_create_debug_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [
                api_server.FrameSnapshot(np.full((20, 20, 3), 10, dtype=np.uint8), 1, 1.0),
                api_server.FrameSnapshot(np.full((20, 20, 3), 20, dtype=np.uint8), 2, 2.0),
            ]
            manager = api_server.OfflineSessionManager(
                provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
                frame_fetcher=lambda: frames.pop(0),
                config=api_server.OfflineConfig(
                    roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                    roi3_extension_params={"left": 4, "right": 4, "top": 5, "bottom": 5},
                    difference_threshold=5.0,
                    debug_save_enabled=False,
                    debug_save_dir=tmp,
                ),
                logger=self.make_null_logger("test_offline_debug_disabled_does_not_create_debug_dir"),
            )

            start = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')
            stop = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

            self.assertNotIn("debug_dir", start)
            self.assertNotIn("debug_dir", stop)
            self.assertFalse((Path(tmp) / "pywrapper_offline").exists())

    def test_offline_start_fails_when_roi3_is_out_of_bounds(self):
        frame = api_server.FrameSnapshot(np.zeros((20, 20, 3), dtype=np.uint8), 1, 1.0)
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
            frame_fetcher=lambda: frame,
            config=api_server.OfflineConfig(
                roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                roi3_extension_params={"left": 30, "right": 4, "top": 5, "bottom": 5},
                difference_threshold=5.0,
                debug_save_enabled=False,
                debug_save_dir="D:/software_data/tmp",
            ),
            logger=self.make_null_logger("test_offline_start_fails_when_roi3_is_out_of_bounds"),
        )

        result = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(result, {"success": False, "info": "invalid_roi3_rect", "point_id": 123})

    def test_offline_debug_save_failure_returns_error(self):
        class FailingSaver(api_server.DebugFrameSaver):
            def save_stage(self, *args, **kwargs):
                raise OSError("disk blocked")

        frame = api_server.FrameSnapshot(np.zeros((20, 20, 3), dtype=np.uint8), 1, 1.0)
        manager = api_server.OfflineSessionManager(
            provider_fetcher=lambda: {"focus_point": "PointF(10, 10)"},
            frame_fetcher=lambda: frame,
            config=api_server.OfflineConfig(
                roi2_extension_params={"left": 2, "right": 2, "top": 3, "bottom": 3},
                roi3_extension_params={"left": 4, "right": 4, "top": 5, "bottom": 5},
                difference_threshold=5.0,
                debug_save_enabled=True,
                debug_save_dir="D:/software_data/tmp",
            ),
            logger=self.make_null_logger("test_offline_debug_save_failure_returns_error"),
            debug_saver=FailingSaver(),
        )

        result = manager.handle('{"point_id": 123, "time_out": 100, "is_save": true}')

        self.assertEqual(result["success"], False)
        self.assertEqual(result["info"], "debug_save_failed")
        self.assertEqual(result["point_id"], 123)

    def test_unknown_request_returns_failure(self):
        response = api_server.handle_request("OCR;31415;{}", provider_fetcher=lambda: {})

        self.assertEqual(
            json.loads(response),
            {"success": False, "info": "unknown_request_type", "req_type": "OCR"},
        )

    def test_wrong_password_returns_failure(self):
        response = api_server.handle_request("ONLINE;bad;{}", provider_fetcher=lambda: {})

        self.assertEqual(json.loads(response), {"success": False, "info": "invalid_password"})

    def test_online_logs_raw_provider_data_and_missing_fields(self):
        stream = StringIO()
        logger = logging.getLogger("test_online_logs_raw_provider_data_and_missing_fields")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.addHandler(handler)

        response = api_server.handle_request(
            'ONLINE;31415;{}',
            provider_fetcher=lambda: {"isLive": True},
            logger=logger,
        )

        self.assertEqual(json.loads(response)["IsFreeze"], False)
        log_text = stream.getvalue()
        self.assertIn('ONLINE raw provider data: {"isLive": true}', log_text)
        self.assertIn("ONLINE missing provider fields:", log_text)
        self.assertIn("focus_depth", log_text)
        self.assertIn("guankuan_a", log_text)
        self.assertIn("guankuan_b", log_text)
        self.assertIn("depth", log_text)
        self.assertIn("focus_point", log_text)

    def test_online_logs_timepoints(self):
        stream = StringIO()
        logger = logging.getLogger("test_online_logs_timepoints")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.addHandler(handler)

        response = api_server.handle_request(
            'ONLINE;31415;{}',
            provider_fetcher=lambda: {"isLive": True, "mode": 2, "focus_depth": "6.5"},
            logger=logger,
            trace_id="unit-trace",
        )

        self.assertEqual(json.loads(response)["SkinDepth"], 6.5)
        log_text = stream.getvalue()
        self.assertIn("ONLINE timepoint trace_id=unit-trace | step=handle_request_entered | wall_time=", log_text)
        self.assertIn("step=provider_fetch_start", log_text)
        self.assertIn("step=provider_fetch_completed", log_text)
        self.assertIn("step=convert_provider_completed", log_text)
        self.assertIn("step=json_encode_completed", log_text)
        self.assertIn("perf_counter_ns=", log_text)

    def test_mobile_comm_engine_configures_callbacks_and_d3d_window(self):
        comm = Mock()
        logger = logging.getLogger("test_mobile_comm_engine_configures_callbacks_and_d3d_window")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        engine = api_server.MobileCommEngine(
            comm,
            logger,
            hwnd_factory=lambda: 12345,
            hwnd_destroyer=lambda hwnd: None,
            stream_interval_s=0.01,
        )

        engine.configure()

        comm.SetOnImageInfoOnceMsg.assert_called_once()
        comm.SetOnClientStateInfoOnceMsg.assert_called_once()
        comm.SetD3DRenderHWND.assert_called_once_with(12345)


if __name__ == "__main__":
    unittest.main()
