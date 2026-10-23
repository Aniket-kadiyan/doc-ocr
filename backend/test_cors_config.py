from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cors_config import install_unrestricted_cors


class UnrestrictedCorsTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()

        @app.get("/probe")
        def probe() -> dict[str, bool]:
            return {"ok": True}

        install_unrestricted_cors(app, expose_headers=("X-Request-ID",))
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_preflight_accepts_an_arbitrary_lan_origin_with_credentials(self) -> None:
        origin = "http://unknown-client-network.example:49152"
        response = self.client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-CSRF-Token",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        self.assertIn(
            "X-CSRF-Token",
            response.headers["access-control-allow-headers"],
        )

    def test_simple_request_echoes_any_origin(self) -> None:
        origin = "https://client-chosen-hostname.internal"
        response = self.client.get(
            "/probe",
            headers={
                "Origin": origin,
                "Cookie": "qtracx_session=test",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertIn("X-Request-ID", response.headers["access-control-expose-headers"])


if __name__ == "__main__":
    unittest.main()
