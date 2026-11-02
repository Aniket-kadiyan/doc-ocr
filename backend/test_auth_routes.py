from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from auth_models import AuthSession, Base
from auth_repository import create_user
from auth_routes import require_protected_request, router
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
            create_user(
                session,
                username="quality.admin",
                display_name="Quality Administrator",
                password="admin secure password 2026",
                administrator=True,
            )
        app = FastAPI()
        app.state.auth_database_manager = self.database
        app.state.auth_session_hours = 8
        app.state.auth_cookie_secure = False
        app.include_router(router)

        @app.get("/protected", dependencies=[Depends(require_protected_request)])
        def protected_read() -> dict[str, bool]:
            return {"ok": True}

        @app.post("/protected", dependencies=[Depends(require_protected_request)])
        def protected_write() -> dict[str, bool]:
            return {"ok": True}

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

    def test_https_transport_marks_authentication_cookies_secure(self) -> None:
        self.client.app.state.auth_cookie_secure = True
        response = self.client.post(
            "/auth/login",
            json={
                "username": "operator.one",
                "password": "operator secure password 2026",
            },
        )
        self.assertEqual(response.status_code, 200)
        cookie_headers = response.headers.get_list("set-cookie")
        self.assertEqual(len(cookie_headers), 2)
        self.assertTrue(all("Secure" in header for header in cookie_headers))

    def test_protected_routes_require_session_and_csrf(self) -> None:
        self.assertEqual(self.client.get("/protected").status_code, 401)
        login = self.client.post("/auth/login", json={
            "username": "operator.one",
            "password": "operator secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        self.assertEqual(self.client.get("/protected").status_code, 200)
        self.assertEqual(self.client.post("/protected").status_code, 403)
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        accepted = self.client.post(
            "/protected", headers={"X-CSRF-Token": csrf_token}
        )
        self.assertEqual(accepted.status_code, 200)

    def test_unconfigured_database_returns_service_unavailable(self) -> None:
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            response = client.post("/auth/login", json={
                "username": "operator.one",
                "password": "irrelevant",
            })
        self.assertEqual(response.status_code, 503)

    def test_regular_user_cannot_use_administration_endpoints(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "operator.one",
            "password": "operator secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        response = self.client.get("/auth/admin/users")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Administrator access is required")

    def test_administrator_can_list_and_update_existing_users(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "quality.admin",
            "password": "admin secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        listed = self.client.get("/auth/admin/users")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(
            [entry["username"] for entry in listed.json()],
            ["operator.one", "quality.admin"],
        )
        operator = next(
            entry for entry in listed.json() if entry["username"] == "operator.one"
        )

        without_csrf = self.client.patch(
            f"/auth/admin/users/{operator['id']}",
            json={"username": "operator.renamed", "display_name": "Renamed Operator"},
        )
        self.assertEqual(without_csrf.status_code, 403)
        updated = self.client.patch(
            f"/auth/admin/users/{operator['id']}",
            headers={"X-CSRF-Token": csrf_token},
            json={"username": "operator.renamed", "display_name": "Renamed Operator"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["username"], "operator.renamed")
        self.assertEqual(updated.json()["display_name"], "Renamed Operator")

        password = self.client.post(
            f"/auth/admin/users/{operator['id']}/password",
            headers={"X-CSRF-Token": csrf_token},
            json={"password": "new operator password"},
        )
        self.assertEqual(password.status_code, 200)

        other_client = TestClient(self.client.app)
        try:
            old_login = other_client.post("/auth/login", json={
                "username": "operator.one",
                "password": "operator secure password 2026",
            })
            self.assertEqual(old_login.status_code, 401)
            new_login = other_client.post("/auth/login", json={
                "username": "operator.renamed",
                "password": "new operator password",
            })
            self.assertEqual(new_login.status_code, 200)
        finally:
            other_client.close()

    def test_administration_api_has_no_create_user_route(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "quality.admin",
            "password": "admin secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        response = self.client.post(
            "/auth/admin/users",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "username": "must.not.exist",
                "display_name": "Must Not Exist",
                "password": "password",
            },
        )
        self.assertEqual(response.status_code, 405)

    def test_administrator_can_change_existing_user_access_and_role(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "quality.admin",
            "password": "admin secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        listed = self.client.get("/auth/admin/users").json()
        operator = next(entry for entry in listed if entry["username"] == "operator.one")

        promoted = self.client.patch(
            f"/auth/admin/users/{operator['id']}/access",
            headers={"X-CSRF-Token": csrf_token},
            json={"is_active": True, "administrator": True},
        )
        self.assertEqual(promoted.status_code, 200)
        self.assertEqual(set(promoted.json()["roles"]), {"admin", "user"})

        disabled = self.client.patch(
            f"/auth/admin/users/{operator['id']}/access",
            headers={"X-CSRF-Token": csrf_token},
            json={"is_active": False, "administrator": False},
        )
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json()["is_active"])
        self.assertEqual(disabled.json()["roles"], ["user"])

    def test_final_active_administrator_cannot_be_disabled_or_demoted(self) -> None:
        login = self.client.post("/auth/login", json={
            "username": "quality.admin",
            "password": "admin secure password 2026",
        })
        self.assertEqual(login.status_code, 200)
        csrf_token = self.client.cookies.get(CSRF_COOKIE_NAME)
        administrator = login.json()["user"]

        demoted = self.client.patch(
            f"/auth/admin/users/{administrator['id']}/access",
            headers={"X-CSRF-Token": csrf_token},
            json={"is_active": True, "administrator": False},
        )
        self.assertEqual(demoted.status_code, 409)
        disabled = self.client.patch(
            f"/auth/admin/users/{administrator['id']}/access",
            headers={"X-CSRF-Token": csrf_token},
            json={"is_active": False, "administrator": True},
        )
        self.assertEqual(disabled.status_code, 409)


if __name__ == "__main__":
    unittest.main()
