"""Database-neutral account and role operations for QTracX authentication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from auth_models import (
    AuthSession,
    Permission,
    Role,
    User,
    UserPermissionOverride,
    utc_now,
)
from auth_security import hash_password, normalize_display_name, normalize_username


DEFAULT_ROLES = {
    "admin": "Manage QTracX users and application configuration",
    "user": "Use QTracX drawing, review, checksheet, and export workflows",
}

DEFAULT_PERMISSIONS = {
    "drawing.use": (
        "Drawing",
        "Open drawings and use OCR, scanning, ballooning, GD&T and review workflows",
    ),
    "drawing.export": (
        "Drawing",
        "Create drawing, Excel, PDF and checksheet exports",
    ),
    "checksheet.view": (
        "Checksheet",
        "View saved checksheets, inspection runs and source documents",
    ),
    "checksheet.manage": (
        "Checksheet",
        "Create, revise, rename, archive, duplicate and delete checksheets",
    ),
    "checksheet.execute": (
        "Checksheet",
        "Start, fill and complete checksheet inspection runs",
    ),
    "training.manage": (
        "Training",
        "Record GD&T corrections and export training labels",
    ),
    "users.manage": (
        "Administration",
        "Manage existing QTraX users, roles and permissions",
    ),
}

DEFAULT_ROLE_PERMISSIONS = {
    "user": tuple(
        name for name in DEFAULT_PERMISSIONS if name != "users.manage"
    ),
    "admin": ("users.manage",),
}

ROLE_CONTROLLED_PERMISSIONS = frozenset({"users.manage"})


class AuthenticationRepositoryError(RuntimeError):
    pass


class DuplicateUserError(AuthenticationRepositoryError):
    pass


class UserNotFoundError(AuthenticationRepositoryError):
    pass


class LastAdministratorError(AuthenticationRepositoryError):
    pass


class UnknownPermissionError(AuthenticationRepositoryError):
    pass


class RoleControlledPermissionError(AuthenticationRepositoryError):
    pass


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    display_name: str
    is_active: bool
    roles: tuple[str, ...]
    permissions: tuple[str, ...]
    permission_overrides: dict[str, bool]
    created_at: datetime
    updated_at: datetime

    def as_public_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["roles"] = list(self.roles)
        result["permissions"] = list(self.permissions)
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result


@dataclass(frozen=True)
class PermissionRecord:
    name: str
    category: str
    description: str
    role_controlled: bool

    def as_public_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_default_permissions(session: Session) -> dict[str, Permission]:
    existing = {
        permission.name: permission
        for permission in session.scalars(
            select(Permission).where(
                Permission.name.in_(tuple(DEFAULT_PERMISSIONS))
            )
        )
    }
    for name, (category, description) in DEFAULT_PERMISSIONS.items():
        if name not in existing:
            permission = Permission(
                id=str(uuid4()),
                name=name,
                category=category,
                description=description,
            )
            session.add(permission)
            existing[name] = permission
    session.flush()
    return existing


def seed_default_roles(session: Session) -> dict[str, Role]:
    existing = {
        role.name: role
        for role in session.scalars(
            select(Role)
            .options(selectinload(Role.permissions))
            .where(Role.name.in_(tuple(DEFAULT_ROLES)))
        )
    }
    for name, description in DEFAULT_ROLES.items():
        if name not in existing:
            role = Role(id=str(uuid4()), name=name, description=description)
            session.add(role)
            existing[name] = role
    session.flush()
    permissions = seed_default_permissions(session)
    for role_name, permission_names in DEFAULT_ROLE_PERMISSIONS.items():
        role = existing[role_name]
        assigned = {permission.name for permission in role.permissions}
        for permission_name in permission_names:
            if permission_name not in assigned:
                role.permissions.append(permissions[permission_name])
    session.flush()
    return existing


def list_permission_catalog(session: Session) -> list[PermissionRecord]:
    seed_default_roles(session)
    permissions = session.scalars(
        select(Permission).order_by(Permission.category, Permission.name)
    ).all()
    return [
        PermissionRecord(
            name=permission.name,
            category=permission.category,
            description=permission.description,
            role_controlled=permission.name in ROLE_CONTROLLED_PERMISSIONS,
        )
        for permission in permissions
    ]


def _user_load_options():
    return (
        selectinload(User.roles).selectinload(Role.permissions),
        selectinload(User.permission_overrides).selectinload(
            UserPermissionOverride.permission
        ),
    )


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
        .options(*_user_load_options())
        .where(User.normalized_username == normalized_username)
    )
    if user is None:
        raise UserNotFoundError("User was not found")
    return user


def get_user_by_id(session: Session, user_id: str) -> User:
    user = session.scalar(
        select(User)
        .options(*_user_load_options())
        .where(User.id == user_id)
    )
    if user is None:
        raise UserNotFoundError("User was not found")
    return user


def list_users(session: Session) -> list[UserRecord]:
    users = session.scalars(
        select(User)
        .options(*_user_load_options())
        .order_by(User.normalized_username)
    ).all()
    return [user_record(user) for user in users]


def update_user_profile(
    session: Session,
    *,
    user_id: str,
    username: str,
    display_name: str,
) -> UserRecord:
    """Change an existing account's public identity without changing its ID."""

    user = get_user_by_id(session, user_id)
    canonical_username, normalized_username = normalize_username(username)
    canonical_display_name = normalize_display_name(display_name)
    duplicate = session.scalar(
        select(User.id).where(
            User.normalized_username == normalized_username,
            User.id != user.id,
        )
    )
    if duplicate is not None:
        raise DuplicateUserError("A user with that username already exists")

    user.username = canonical_username
    user.normalized_username = normalized_username
    user.display_name = canonical_display_name
    user.updated_at = utc_now()
    try:
        session.flush()
    except IntegrityError as error:
        raise DuplicateUserError("A user with that username already exists") from error
    return user_record(user)


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


def set_user_access_by_id(
    session: Session,
    *,
    user_id: str,
    active: bool,
    administrator: bool,
) -> UserRecord:
    """Update the enforceable account access and role flags in one operation."""

    user = get_user_by_id(session, user_id)
    roles = seed_default_roles(session)
    admin_role = roles["admin"]
    is_administrator = any(role.name == "admin" for role in user.roles)
    removes_active_administrator = (
        user.is_active
        and is_administrator
        and (not active or not administrator)
    )
    if removes_active_administrator:
        active_admin_count = session.scalar(
            select(func.count(User.id))
            .select_from(User)
            .join(User.roles)
            .where(User.is_active.is_(True), Role.name == "admin")
        ) or 0
        if active_admin_count <= 1:
            raise LastAdministratorError(
                "The final active administrator cannot be disabled or demoted"
            )

    changed = user.is_active != active or is_administrator != administrator
    user.is_active = active
    if administrator and not is_administrator:
        user.roles.append(admin_role)
    elif not administrator and is_administrator:
        user.roles.remove(admin_role)

    if changed:
        user.updated_at = utc_now()
        _revoke_user_sessions(session, user.id)
    session.flush()
    return user_record(user)


def set_user_permission_overrides(
    session: Session,
    *,
    user_id: str,
    overrides: dict[str, bool | None],
) -> UserRecord:
    """Set explicit per-user allow/deny values; ``None`` restores role default."""

    permissions = seed_default_permissions(session)
    unknown = sorted(set(overrides) - set(permissions))
    if unknown:
        raise UnknownPermissionError(
            f"Unknown permission: {', '.join(unknown)}"
        )
    role_controlled = sorted(set(overrides) & ROLE_CONTROLLED_PERMISSIONS)
    if role_controlled:
        raise RoleControlledPermissionError(
            f"Permission is controlled by role: {', '.join(role_controlled)}"
        )

    user = get_user_by_id(session, user_id)
    existing = {
        override.permission.name: override
        for override in user.permission_overrides
    }
    changed = False
    for permission_name, allowed in overrides.items():
        stored = existing.get(permission_name)
        if allowed is None:
            if stored is not None:
                user.permission_overrides.remove(stored)
                changed = True
            continue
        if stored is None:
            user.permission_overrides.append(
                UserPermissionOverride(
                    permission=permissions[permission_name],
                    is_allowed=allowed,
                )
            )
            changed = True
        elif stored.is_allowed != allowed:
            stored.is_allowed = allowed
            stored.updated_at = utc_now()
            changed = True

    if changed:
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


def set_user_password_by_id(
    session: Session,
    *,
    user_id: str,
    password: str,
) -> UserRecord:
    """Reset a password by stable account ID and revoke all account sessions."""

    user = get_user_by_id(session, user_id)
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

    roles = tuple(sorted(role.name for role in user.roles))
    permissions = {
        permission.name
        for role in user.roles
        for permission in role.permissions
    }
    overrides = {
        override.permission.name: override.is_allowed
        for override in user.permission_overrides
        if override.permission.name not in ROLE_CONTROLLED_PERMISSIONS
    }
    for permission_name, allowed in overrides.items():
        if allowed:
            permissions.add(permission_name)
        else:
            permissions.discard(permission_name)

    return UserRecord(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_active=user.is_active,
        roles=roles,
        permissions=tuple(sorted(permissions)),
        permission_overrides=dict(sorted(overrides.items())),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
