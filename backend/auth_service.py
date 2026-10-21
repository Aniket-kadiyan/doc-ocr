"""Credential verification and database-backed browser sessions for QTracX."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from auth_models import AuthSession, User, utc_now
from auth_repository import UserRecord, user_record
from auth_security import (
    AuthenticationInputError,
    hash_password,
    normalize_username,
    verify_password,
    verify_password_and_rehash,
)


MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
LAST_SEEN_UPDATE_SECONDS = 60
SESSION_COOKIE_NAME = "qtracx_session"
CSRF_COOKIE_NAME = "qtracx_csrf"


class AuthenticationFailedError(RuntimeError):
    """Credentials are invalid, inactive, or temporarily locked."""


class InvalidSessionError(RuntimeError):
    """A browser session is missing, expired, revoked, or inactive."""


class InvalidCsrfTokenError(RuntimeError):
    """A state-changing request does not carry the session CSRF token."""


@dataclass(frozen=True)
class CreatedSession:
    session_token: str
    csrf_token: str
    expires_at: datetime
    user: UserRecord


@dataclass(frozen=True)
class AuthenticatedSession:
    session_id: str
    token_hash: str
    csrf_token_hash: str
    user: UserRecord


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# A real Argon2id hash ensures unknown usernames take the same expensive
# verification path as known accounts, reducing username-enumeration timing.
_DUMMY_PASSWORD_HASH = hash_password("QTracX dummy verification password 2026")


def create_login_session(
    session: Session,
    *,
    username: str,
    password: str,
    session_hours: int,
    now: datetime | None = None,
) -> CreatedSession:
    """Verify credentials, apply lockout policy, and issue one browser session."""

    current = _as_utc(now or utc_now())
    try:
        _, normalized_username = normalize_username(username)
    except AuthenticationInputError:
        verify_password(password, _DUMMY_PASSWORD_HASH)
        raise AuthenticationFailedError("Invalid username or password") from None

    user = session.scalar(
        select(User)
        .options(selectinload(User.roles))
        .where(User.normalized_username == normalized_username)
        .with_for_update()
    )
    if user is None:
        verify_password(password, _DUMMY_PASSWORD_HASH)
        raise AuthenticationFailedError("Invalid username or password")

    if user.locked_until is not None and _as_utc(user.locked_until) > current:
        verify_password(password, user.password_hash)
        raise AuthenticationFailedError("Invalid username or password")

    verification = verify_password_and_rehash(password, user.password_hash)
    if not verification.valid:
        user.failed_login_count += 1
        if user.failed_login_count >= MAX_FAILED_LOGINS:
            user.failed_login_count = MAX_FAILED_LOGINS
            user.locked_until = current + timedelta(minutes=LOCKOUT_MINUTES)
        user.updated_at = current
        session.flush()
        raise AuthenticationFailedError("Invalid username or password")

    if not user.is_active:
        raise AuthenticationFailedError("Invalid username or password")

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = current
    user.updated_at = current
    if verification.replacement_hash is not None:
        user.password_hash = verification.replacement_hash

    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = current + timedelta(hours=session_hours)
    session.add(
        AuthSession(
            id=str(uuid4()),
            token_hash=_hash_token(session_token),
            csrf_token_hash=_hash_token(csrf_token),
            user_id=user.id,
            created_at=current,
            expires_at=expires_at,
            last_seen_at=current,
        )
    )
    session.flush()
    return CreatedSession(
        session_token=session_token,
        csrf_token=csrf_token,
        expires_at=expires_at,
        user=user_record(user),
    )


def get_authenticated_session(
    session: Session,
    *,
    session_token: str,
    now: datetime | None = None,
) -> AuthenticatedSession:
    """Resolve an active opaque session token and update its last-seen time."""

    if not session_token:
        raise InvalidSessionError("Authentication required")
    current = _as_utc(now or utc_now())
    token_hash = _hash_token(session_token)
    stored = session.scalar(
        select(AuthSession)
        .options(selectinload(AuthSession.user).selectinload(User.roles))
        .where(AuthSession.token_hash == token_hash)
    )
    if (
        stored is None
        or stored.revoked_at is not None
        or _as_utc(stored.expires_at) <= current
        or not stored.user.is_active
    ):
        raise InvalidSessionError("Authentication required")

    # Scan progress polling can call this endpoint several times per second.
    # Keep authorization immediate while limiting audit timestamp writes.
    if _as_utc(stored.last_seen_at) <= current - timedelta(
        seconds=LAST_SEEN_UPDATE_SECONDS
    ):
        stored.last_seen_at = current
        session.flush()
    return AuthenticatedSession(
        session_id=stored.id,
        token_hash=stored.token_hash,
        csrf_token_hash=stored.csrf_token_hash,
        user=user_record(stored.user),
    )


def validate_csrf_token(
    authenticated: AuthenticatedSession,
    *,
    header_token: str | None,
    cookie_token: str | None,
) -> None:
    """Apply a double-submit check tied to the server-side session record."""

    if not header_token or not cookie_token:
        raise InvalidCsrfTokenError("A valid CSRF token is required")
    if not hmac.compare_digest(header_token, cookie_token):
        raise InvalidCsrfTokenError("A valid CSRF token is required")
    if not hmac.compare_digest(
        _hash_token(header_token), authenticated.csrf_token_hash
    ):
        raise InvalidCsrfTokenError("A valid CSRF token is required")


def revoke_session(
    session: Session,
    *,
    authenticated: AuthenticatedSession,
    now: datetime | None = None,
) -> None:
    stored = session.get(AuthSession, authenticated.session_id)
    if stored is None or not hmac.compare_digest(
        stored.token_hash, authenticated.token_hash
    ):
        raise InvalidSessionError("Authentication required")
    if stored.revoked_at is None:
        stored.revoked_at = _as_utc(now or utc_now())
        session.flush()
