"""Smoke tests for the production-style FastAPI routing gateway."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import api_server


class ApiServerSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        api_server.gateway.log_path = Path(self.temp_dir.name) / "routing_requests.jsonl"
        self.client = TestClient(api_server.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_loaded", data)
        self.assertIn("feature_mode", data)

    def test_route_without_api_keys(self) -> None:
        response = self.client.post(
            "/route",
            json={
                "prompt": "What is gravity?",
                "sla_mode": "balanced",
                "session_budget": 2.0,
                "spent_so_far": 0.0,
                "estimated_cloud_cost": 0.05,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["route"], {"local", "cloud"})
        self.assertIsInstance(data["predicted_utility_local"], float)
        self.assertIsInstance(data["predicted_utility_cloud"], float)
        self.assertIsInstance(data["estimated_cost"], float)
        self.assertIsInstance(data["estimated_latency"], float)
        self.assertIn("feature_source", data)
        self.assertIn("explanation", data)

    def test_chat_returns_metadata_when_backend_unavailable(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "What is gravity?",
                "sla_mode": "cost_sensitive",
                "session_budget": 2.0,
                "spent_so_far": 0.0,
                "estimated_cloud_cost": 0.05,
                "call_backend": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["route"], {"local", "cloud"})
        self.assertTrue(data["backend_called"])
        self.assertIn("backend", data)
        self.assertIn("message", data)
        self.assertIn("response_text", data)


if __name__ == "__main__":
    unittest.main()
