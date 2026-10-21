from __future__ import annotations

import unittest
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import select

from auth_models import AuthSession, Base, User, utc_now
from auth_repository import (
    DuplicateUserError,
    LastAdministratorError,
    create_user,
    get_user,
    list_users,
    set_user_active,
    set_user_password,
)
from auth_security import verify_password
from database import DatabaseManager


class AuthenticationRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = DatabaseManager(
            "sqlite:///:memory:",
            allow_test_sqlite=True,
        )
        Base.metadata.create_all(self.database.engine)

    def tearDown(self) -> None:
        self.database.dispose()

    def _create_admin(self, username: str = "admin") -> str:
        with self.database.session() as session:
            record = create_user(
                session,
                username=username,
                display_name=f"{username} display",
                password=f"{username} secure password 2026",
                administrator=True,
            )
            return record.id

    def test_create_admin_seeds_roles_and_hashes_password(self) -> None:
        self._create_admin()
        with self.database.session() as session:
            user = get_user(session, "ADMIN")
            self.assertEqual({role.name for role in user.roles}, {"admin", "user"})
            self.assertTrue(verify_password("admin secure password 2026", user.password_hash))
            self.assertNotEqual(user.password_hash, "admin secure password 2026")

    def test_duplicate_username_is_case_insensitive(self) -> None:
        self._create_admin("Quality.Admin")
        with self.database.session() as session:
            with self.assertRaises(DuplicateUserError):
                create_user(
                    session,
                    username="quality.admin",
                    display_name="Duplicate",
                    password="another secure password 2026",
                )

    def test_final_active_administrator_cannot_be_disabled(self) -> None:
        self._create_admin()
        with self.database.session() as session:
            with self.assertRaises(LastAdministratorError):
                set_user_active(session, username="admin", active=False)

    def test_admin_can_be_disabled_when_another_admin_is_active(self) -> None:
        first_id = self._create_admin("admin.one")
        self._create_admin("admin.two")
        with self.database.session() as session:
            session.add(AuthSession(
                id=str(uuid4()),
                token_hash="a" * 64,
                csrf_token_hash="b" * 64,
                user_id=first_id,
                expires_at=utc_now() + timedelta(hours=1),
            ))
        with self.database.session() as session:
            record = set_user_active(session, username="admin.one", active=False)
            self.assertFalse(record.is_active)
        with self.database.session() as session:
            stored_session = session.scalar(select(AuthSession))
            self.assertIsNotNone(stored_session.revoked_at)

    def test_password_change_revokes_sessions(self) -> None:
        user_id = self._create_admin()
        with self.database.session() as session:
            session.add(AuthSession(
                id=str(uuid4()),
                token_hash="c" * 64,
                csrf_token_hash="d" * 64,
                user_id=user_id,
                expires_at=utc_now() + timedelta(hours=1),
            ))
        with self.database.session() as session:
            set_user_password(
                session,
                username="admin",
                password="replacement secure password 2026",
            )
        with self.database.session() as session:
            user = session.get(User, user_id)
            stored_session = session.scalar(select(AuthSession))
            self.assertTrue(verify_password(
                "replacement secure password 2026",
                user.password_hash,
            ))
            self.assertIsNotNone(stored_session.revoked_at)

    def test_users_are_listed_without_password_data(self) -> None:
        self._create_admin("z.admin")
        self._create_admin("a.admin")
        with self.database.session() as session:
            records = list_users(session)
            self.assertEqual(
                [record.username for record in records],
                ["a.admin", "z.admin"],
            )
            payload = records[0].as_public_dict()
            self.assertNotIn("password", repr(payload).lower())


if __name__ == "__main__":
    unittest.main()
