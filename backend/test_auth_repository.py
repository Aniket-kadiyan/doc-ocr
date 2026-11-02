from __future__ import annotations

import unittest
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import select

from auth_models import AuthSession, Base, User, utc_now
from auth_repository import (
    DuplicateUserError,
    LastAdministratorError,
    RoleControlledPermissionError,
    create_user,
    get_user,
    list_users,
    set_user_administrator,
    set_user_access_by_id,
    set_user_active,
    set_user_password,
    set_user_password_by_id,
    set_user_permission_overrides,
    update_user_profile,
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
            record = list_users(session)[0]
            self.assertIn("drawing.use", record.permissions)
            self.assertIn("users.manage", record.permissions)

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

    def test_regular_user_can_be_promoted_and_demoted(self) -> None:
        self._create_admin("primary.admin")
        with self.database.session() as session:
            record = create_user(
                session,
                username="quality.user",
                display_name="Quality User",
                password="quality user secure password 2026",
            )
            user_id = record.id
        with self.database.session() as session:
            session.add(AuthSession(
                id=str(uuid4()),
                token_hash="e" * 64,
                csrf_token_hash="f" * 64,
                user_id=user_id,
                expires_at=utc_now() + timedelta(hours=1),
            ))

        with self.database.session() as session:
            promoted = set_user_administrator(
                session,
                username="quality.user",
                administrator=True,
            )
            self.assertEqual(set(promoted.roles), {"admin", "user"})
        with self.database.session() as session:
            stored_session = session.scalar(
                select(AuthSession).where(AuthSession.user_id == user_id)
            )
            self.assertIsNotNone(stored_session.revoked_at)

        with self.database.session() as session:
            demoted = set_user_administrator(
                session,
                username="quality.user",
                administrator=False,
            )
            self.assertEqual(demoted.roles, ("user",))

    def test_final_active_administrator_role_cannot_be_revoked(self) -> None:
        self._create_admin()
        with self.database.session() as session:
            with self.assertRaises(LastAdministratorError):
                set_user_administrator(
                    session,
                    username="admin",
                    administrator=False,
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
            user = session.get(User, user_id)
            user.failed_login_count = 5
            user.locked_until = utc_now() + timedelta(minutes=15)
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
            self.assertEqual(user.failed_login_count, 0)
            self.assertIsNone(user.locked_until)

    def test_profile_update_uses_stable_id_and_rejects_duplicate_username(self) -> None:
        first_id = self._create_admin("first.admin")
        self._create_admin("second.admin")
        with self.database.session() as session:
            record = update_user_profile(
                session,
                user_id=first_id,
                username="renamed.admin",
                display_name="Renamed Administrator",
            )
            self.assertEqual(record.id, first_id)
            self.assertEqual(record.username, "renamed.admin")
            self.assertEqual(record.display_name, "Renamed Administrator")

        with self.database.session() as session:
            with self.assertRaises(DuplicateUserError):
                update_user_profile(
                    session,
                    user_id=first_id,
                    username="SECOND.ADMIN",
                    display_name="Duplicate",
                )

    def test_password_can_be_changed_by_stable_user_id(self) -> None:
        user_id = self._create_admin("stable.admin")
        with self.database.session() as session:
            record = set_user_password_by_id(
                session,
                user_id=user_id,
                password="new password",
            )
            self.assertEqual(record.id, user_id)
        with self.database.session() as session:
            user = session.get(User, user_id)
            self.assertTrue(verify_password("new password", user.password_hash))

    def test_access_and_administrator_role_can_be_changed_by_stable_id(self) -> None:
        self._create_admin("primary.admin")
        with self.database.session() as session:
            record = create_user(
                session,
                username="operator.one",
                display_name="Operator One",
                password="operator password",
            )
            operator_id = record.id

        with self.database.session() as session:
            promoted = set_user_access_by_id(
                session,
                user_id=operator_id,
                active=True,
                administrator=True,
            )
            self.assertTrue(promoted.is_active)
            self.assertEqual(set(promoted.roles), {"admin", "user"})

        with self.database.session() as session:
            disabled = set_user_access_by_id(
                session,
                user_id=operator_id,
                active=False,
                administrator=False,
            )
            self.assertFalse(disabled.is_active)
            self.assertEqual(disabled.roles, ("user",))

    def test_combined_access_update_protects_final_active_administrator(self) -> None:
        administrator_id = self._create_admin("only.admin")
        with self.database.session() as session:
            with self.assertRaises(LastAdministratorError):
                set_user_access_by_id(
                    session,
                    user_id=administrator_id,
                    active=True,
                    administrator=False,
                )
        with self.database.session() as session:
            with self.assertRaises(LastAdministratorError):
                set_user_access_by_id(
                    session,
                    user_id=administrator_id,
                    active=False,
                    administrator=True,
                )

    def test_user_permission_overrides_allow_deny_and_restore_role_default(self) -> None:
        user_id = self._create_admin("permission.admin")
        with self.database.session() as session:
            updated = set_user_permission_overrides(
                session,
                user_id=user_id,
                overrides={"drawing.export": False},
            )
            self.assertNotIn("drawing.export", updated.permissions)
            self.assertEqual(
                updated.permission_overrides,
                {"drawing.export": False},
            )
            self.assertIn("users.manage", updated.permissions)

        with self.database.session() as session:
            restored = set_user_permission_overrides(
                session,
                user_id=user_id,
                overrides={"drawing.export": None},
            )
            self.assertIn("drawing.export", restored.permissions)
            self.assertEqual(restored.permission_overrides, {})

    def test_role_controlled_permission_cannot_be_overridden(self) -> None:
        user_id = self._create_admin("role.admin")
        with self.database.session() as session:
            with self.assertRaises(RoleControlledPermissionError):
                set_user_permission_overrides(
                    session,
                    user_id=user_id,
                    overrides={"users.manage": False},
                )

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
