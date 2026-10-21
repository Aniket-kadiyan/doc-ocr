"""Database-neutral account and role operations for QTracX authentication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from auth_models import AuthSession, Role, User, utc_now
from auth_security import hash_password, normalize_display_name, normalize_username


DEFAULT_ROLES = {
    "admin": "Manage QTracX users and application configuration",
    "user": "Use QTracX drawing, review, checksheet, and export workflows",
}


class AuthenticationRepositoryError(RuntimeError):
    pass


class DuplicateUserError(AuthenticationRepositoryError):
    pass


class UserNotFoundError(AuthenticationRepositoryError):
    pass


class LastAdministratorError(AuthenticationRepositoryError):
    pass


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    display_name: str
    is_active: bool
    roles: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

    def as_public_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["roles"] = list(self.roles)
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result


def seed_default_roles(session: Session) -> dict[str, Role]:
    existing = {
        role.name: role
        for role in session.scalars(
            select(Role).where(Role.name.in_(tuple(DEFAULT_ROLES)))
        )
    }
    for name, description in DEFAULT_ROLES.items():
        if name not in existing:
            role = Role(id=str(uuid4()), name=name, description=description)
            session.add(role)
            existing[name] = role
    session.flush()
    return existing


def create_user(
    session: Session,
    *,
    username: str,
    display_name: str,
    password: str,
    administrator: bool = False,
) -> UserRecord:
    canonical_username, normalized_username = normalize_username(username)
    canonical_display_name = normalize_display_name(display_name)
    existing = session.scalar(
        select(User.id).where(User.normalized_username == normalized_username)
    )
    if existing is not None:
        raise DuplicateUserError("A user with that username already exists")

    roles = seed_default_roles(session)
    selected_roles = [roles["user"]]
    if administrator:
        selected_roles.append(roles["admin"])
    user = User(
        id=str(uuid4()),
        username=canonical_username,
        normalized_username=normalized_username,
        display_name=canonical_display_name,
        password_hash=hash_password(password, username=canonical_username),
        is_active=True,
        roles=selected_roles,
    )
    session.add(user)
    try:
        session.flush()
    except IntegrityError as error:
        raise DuplicateUserError("A user with that username already exists") from error
    return user_record(user)


def get_user(session: Session, username: str) -> User:
    _, normalized_username = normalize_username(username)
    user = session.scalar(
        select(User)
        .options(selectinload(User.roles))
        .where(User.normalized_username == normalized_username)
    )
    if user is None:
        raise UserNotFoundError("User was not found")
    return user


def list_users(session: Session) -> list[UserRecord]:
    users = session.scalars(
        select(User)
        .options(selectinload(User.roles))
        .order_by(User.normalized_username)
    ).all()
    return [user_record(user) for user in users]


def set_user_active(
    session: Session,
    *,
    username: str,
    active: bool,
) -> UserRecord:
    user = get_user(session, username)
    role_names = {role.name for role in user.roles}
    if not active and user.is_active and "admin" in role_names:
        active_admin_count = session.scalar(
            select(func.count(User.id))
            .select_from(User)
            .join(User.roles)
            .where(User.is_active.is_(True), Role.name == "admin")
        ) or 0
        if active_admin_count <= 1:
            raise LastAdministratorError(
                "The final active administrator cannot be disabled"
            )

    user.is_active = active
    user.updated_at = utc_now()
    if not active:
        _revoke_user_sessions(session, user.id)
    session.flush()
    return user_record(user)


def set_user_administrator(
    session: Session,
    *,
    username: str,
    administrator: bool,
) -> UserRecord:
    """Grant or revoke the administrator role without removing normal access."""

    user = get_user(session, username)
    roles = seed_default_roles(session)
    admin_role = roles["admin"]
    is_administrator = any(role.name == "admin" for role in user.roles)
    if administrator == is_administrator:
        return user_record(user)

    if not administrator and user.is_active:
        active_admin_count = session.scalar(
            select(func.count(User.id))
            .select_from(User)
            .join(User.roles)
            .where(User.is_active.is_(True), Role.name == "admin")
        ) or 0
        if active_admin_count <= 1:
            raise LastAdministratorError(
                "The administrator role cannot be removed from the final active administrator"
            )

    if administrator:
        user.roles.append(admin_role)
    else:
        user.roles.remove(admin_role)
    user.updated_at = utc_now()
    _revoke_user_sessions(session, user.id)
    session.flush()
    return user_record(user)


def set_user_password(
    session: Session,
    *,
    username: str,
    password: str,
) -> UserRecord:
    user = get_user(session, username)
    user.password_hash = hash_password(password, username=user.username)
    user.failed_login_count = 0
    user.locked_until = None
    user.updated_at = utc_now()
    _revoke_user_sessions(session, user.id)
    session.flush()
    return user_record(user)


def delete_expired_sessions(session: Session, *, before: datetime) -> int:
    result = session.execute(delete(AuthSession).where(AuthSession.expires_at < before))
    return int(result.rowcount or 0)


def _revoke_user_sessions(session: Session, user_id: str) -> None:
    session.execute(
        update(AuthSession)
        .where(
            AuthSession.user_id == user_id,
            AuthSession.revoked_at.is_(None),
        )
        .values(revoked_at=utc_now())
    )


def user_record(user: User) -> UserRecord:
    """Return the non-sensitive representation shared by CLI and HTTP APIs."""

    return UserRecord(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_active=user.is_active,
        roles=tuple(sorted(role.name for role in user.roles)),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
