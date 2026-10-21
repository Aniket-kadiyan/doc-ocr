from __future__ import annotations

import unittest
from datetime import timedelta

from sqlalchemy import select

from auth_models import AuthSession, Base, User, utc_now
from auth_repository import create_user, get_user, set_user_active
from auth_service import (
    MAX_FAILED_LOGINS,
    AuthenticationFailedError,
    InvalidCsrfTokenError,
    InvalidSessionError,
    create_login_session,
    get_authenticated_session,
    revoke_session,
    validate_csrf_token,
)
from database import DatabaseManager


class AuthenticationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = DatabaseManager(
            "sqlite:///:memory:", allow_test_sqlite=True
        )
        Base.metadata.create_all(self.database.engine)
        with self.database.session() as session:
            create_user(
                session,
                username="quality.admin",
                display_name="Quality Administrator",
                password="correct secure password 2026",
                administrator=True,
            )

    def tearDown(self) -> None:
        self.database.dispose()

    def _login(self):
        with self.database.session() as session:
            return create_login_session(
                session,
                username="QUALITY.ADMIN",
                password="correct secure password 2026",
                session_hours=8,
            )

    def _reject_login(self, password: str = "wrong password") -> None:
        rejected = False
        with self.database.session() as session:
            try:
                create_login_session(
                    session,
                    username="quality.admin",
                    password=password,
                    session_hours=8,
                )
            except AuthenticationFailedError:
                rejected = True
        self.assertTrue(rejected)

    def test_login_stores_only_token_hashes(self) -> None:
        created = self._login()
        with self.database.session() as session:
            stored = session.scalar(select(AuthSession))
            user = get_user(session, "quality.admin")
            self.assertIsNotNone(stored)
            self.assertNotEqual(stored.token_hash, created.session_token)
            self.assertNotEqual(stored.csrf_token_hash, created.csrf_token)
            self.assertNotIn(created.session_token, repr(stored.__dict__))
            self.assertEqual(user.failed_login_count, 0)
            self.assertIsNotNone(user.last_login_at)

    def test_failed_logins_apply_timed_lockout(self) -> None:
        for _ in range(MAX_FAILED_LOGINS):
            self._reject_login()
        with self.database.session() as session:
            user = get_user(session, "quality.admin")
            self.assertEqual(user.failed_login_count, MAX_FAILED_LOGINS)
            self.assertIsNotNone(user.locked_until)
        self._reject_login("correct secure password 2026")

    def test_unknown_and_inactive_users_use_generic_rejection(self) -> None:
        with self.database.session() as session:
            with self.assertRaisesRegex(
                AuthenticationFailedError, "Invalid username or password"
            ):
                create_login_session(
                    session,
                    username="not.present",
                    password="irrelevant",
                    session_hours=8,
                )
        with self.database.session() as session:
            # A second administrator permits the target account to be disabled.
            create_user(
                session,
                username="backup.admin",
                display_name="Backup Administrator",
                password="backup secure password 2026",
                administrator=True,
            )
            set_user_active(session, username="quality.admin", active=False)
        self._reject_login("correct secure password 2026")

    def test_session_validation_csrf_and_revocation(self) -> None:
        created = self._login()
        with self.database.session() as session:
            authenticated = get_authenticated_session(
                session, session_token=created.session_token
            )
            validate_csrf_token(
                authenticated,
                header_token=created.csrf_token,
                cookie_token=created.csrf_token,
            )
            with self.assertRaises(InvalidCsrfTokenError):
                validate_csrf_token(
                    authenticated,
                    header_token="incorrect",
                    cookie_token=created.csrf_token,
                )
            revoke_session(session, authenticated=authenticated)
        with self.database.session() as session:
            with self.assertRaises(InvalidSessionError):
                get_authenticated_session(
                    session, session_token=created.session_token
                )

    def test_expired_session_is_rejected(self) -> None:
        current = utc_now()
        with self.database.session() as session:
            created = create_login_session(
                session,
                username="quality.admin",
                password="correct secure password 2026",
                session_hours=1,
                now=current,
            )
        with self.database.session() as session:
            with self.assertRaises(InvalidSessionError):
                get_authenticated_session(
                    session,
                    session_token=created.session_token,
                    now=current + timedelta(hours=2),
                )


if __name__ == "__main__":
    unittest.main()
