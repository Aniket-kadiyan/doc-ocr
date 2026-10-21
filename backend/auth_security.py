"""Input validation and Argon2id password hashing for QTracX accounts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type


MIN_PASSWORD_LENGTH = 4
MAX_PASSWORD_LENGTH = 128
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{2,63}$")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


class AuthenticationInputError(ValueError):
    """An account field is invalid without exposing any sensitive value."""


@dataclass(frozen=True)
class PasswordVerification:
    valid: bool
    replacement_hash: str | None = None


def normalize_username(value: str) -> tuple[str, str]:
    """Return a display-preserving username and case-insensitive lookup key."""

    username = unicodedata.normalize("NFKC", value).strip()
    if not _USERNAME_PATTERN.fullmatch(username):
        raise AuthenticationInputError(
            "Username must be 3-64 characters and use only letters, numbers, "
            "periods, underscores, hyphens, or @"
        )
    return username, username.casefold()


def normalize_display_name(value: str) -> str:
    display_name = " ".join(unicodedata.normalize("NFKC", value).split())
    if not display_name or len(display_name) > 120:
        raise AuthenticationInputError(
            "Display name must contain between 1 and 120 characters"
        )
    if _CONTROL_CHARACTER_PATTERN.search(display_name):
        raise AuthenticationInputError("Display name contains invalid characters")
    return display_name


def validate_password(password: str, *, username: str | None = None) -> None:
    """Apply a length-first password policy without arbitrary composition rules."""

    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthenticationInputError(
            f"Password must contain at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthenticationInputError(
            f"Password must contain no more than {MAX_PASSWORD_LENGTH} characters"
        )
    if not password.strip() or "\x00" in password:
        raise AuthenticationInputError("Password is not valid")
    if username and password.casefold() == username.casefold():
        raise AuthenticationInputError("Password must not equal the username")


def hash_password(password: str, *, username: str | None = None) -> str:
    validate_password(password, username=username)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        return bool(_PASSWORD_HASHER.verify(encoded_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def verify_password_and_rehash(
    password: str,
    encoded_hash: str,
) -> PasswordVerification:
    """Verify a password and return a stronger replacement hash when needed."""

    if not verify_password(password, encoded_hash):
        return PasswordVerification(valid=False)
    replacement = (
        _PASSWORD_HASHER.hash(password)
        if _PASSWORD_HASHER.check_needs_rehash(encoded_hash)
        else None
    )
    return PasswordVerification(valid=True, replacement_hash=replacement)
