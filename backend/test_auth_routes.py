from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from auth_models import AuthSession, Base
from auth_repository import create_user
from auth_routes import router
from auth_service import CSRF_COOKIE_NAME, SESSION_COOKIE_NAME
from database import DatabaseManager


class AuthenticationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "auth-routes.db"
        self.database = DatabaseManager(
            f"sqlite:///{path.as_posix()}", allow_test_sqlite=True
        )
        Base.metadata.create_all(self.database.engine)
        with self.database.session() as session:
            create_user(
                session,
                username="operator.one",
                display_name="Operator One",
                password="operator secure password 2026",
            )
        app = FastAPI()
        app.state.auth_database_manager = self.database
        app.state.auth_session_hours = 8
        app.state.auth_cookie_secure = False
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.database.dispose()
        self.temporary.cleanup()

    def test_login_me_and_csrf_protected_logout(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "operator.one",
            "password": "operator secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["user"]["username"], "operator.one")
        self.assertEqual(login.headers["cache-control"], "no-store")
        self.assertTrue(self.client.cookies.get(SESSION_COOKIE_NAME))
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        self.assertTrue(csrf_token)
        self.assertIn("HttpOnly", login.headers.get("set-cookie", ""))

        current = self.client.get("/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["roles"], ["user"])

        rejected = self.client.post("/auth/logout")
        self.assertEqual(rejected.status_code, 403)
        logout = self.client.post(
            "/auth/logout", headers={"X-CSRF-Token": csrf_token}
        )
        self.assertEqual(logout.status_code, 204)
        self.assertIsNone(self.client.cookies.get(SESSION_COOKIE_NAME))
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        with self.database.session() as session:
            stored = session.scalar(select(AuthSession))
            self.assertIsNotNone(stored.revoked_at)

    def test_invalid_credentials_are_generic(self) -> None:
        response = self.client.post("/auth/login", json={
            "username": "operator.one",
            "password": "wrong password",
        })
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password")

    def test_unconfigured_database_returns_service_unavailable(self) -> None:
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            response = client.post("/auth/login", json={
                "username": "operator.one",
                "password": "irrelevant",
            })
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
