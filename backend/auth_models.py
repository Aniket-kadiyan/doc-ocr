"""Portable SQLAlchemy schema for QTracX users, roles, and login sessions."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class User(Base):
    __tablename__ = "qtracx_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_username: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    roles: Mapped[list["Role"]] = relationship(
        secondary="qtracx_user_roles",
        back_populates="users",
        lazy="selectin",
    )
    sessions: Mapped[list["AuthSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "normalized_username",
            name="uq_qtracx_users_normalized_username",
        ),
    )


class Role(Base):
    __tablename__ = "qtracx_roles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    users: Mapped[list[User]] = relationship(
        secondary="qtracx_user_roles",
        back_populates="roles",
        lazy="selectin",
    )

    __table_args__ = (UniqueConstraint("name", name="uq_qtracx_roles_name"),)


class UserRole(Base):
    __tablename__ = "qtracx_user_roles"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("qtracx_users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("qtracx_roles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class AuthSession(Base):
    __tablename__ = "qtracx_auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("qtracx_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="sessions")

    __table_args__ = (
        UniqueConstraint(
            "token_hash",
            name="uq_qtracx_auth_sessions_token_hash",
        ),
        Index("ix_qtracx_auth_sessions_user_id", "user_id"),
        Index("ix_qtracx_auth_sessions_expires_at", "expires_at"),
    )
