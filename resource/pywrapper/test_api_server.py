import json
import logging
from io import StringIO
import unittest
from unittest.mock import Mock

import api_server


class ApiServerTests(unittest.TestCase):
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
                "SkinDepth": "7.5",
                "A": "10.1",
                "B": "20.2",
                "Alpha": None,
                "Depth": "35",
                "IsFreeze": False,
                "isHIFU": True,
                "FocusPoint": "F1",
            },
        )

    def test_online_reads_provider_and_returns_json(self):
        response = api_server.handle_request(
            'ONLINE;31415;{"point_id": 123}',
            provider_fetcher=lambda: {"isLive": False, "mode": 1, "depth": "40"},
        )

        payload = json.loads(response)
        self.assertEqual(payload["Depth"], "40")
        self.assertTrue(payload["IsFreeze"])
        self.assertFalse(payload["isHIFU"])

    def test_offline_is_placeholder_success(self):
        response = api_server.handle_request(
            'OFFLINE;31415;{"point_id": 123}',
            provider_fetcher=lambda: {"depth": "40"},
        )

        self.assertEqual(
            json.loads(response),
            {"success": True, "info": "offline_ok"},
        )

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

        comm.SetOnClientOnceMsg.assert_called_once()
        comm.SetOnImageInfoOnceMsg.assert_called_once()
        comm.SetOnClientStateInfoOnceMsg.assert_called_once()
        comm.SetD3DRenderHWND.assert_called_once_with(12345)


if __name__ == "__main__":
    unittest.main()
