"""HTTP login, current-user, and logout endpoints for QTracX."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from auth_repository import (
    AuthenticationRepositoryError,
    DuplicateUserError,
    LastAdministratorError,
    UserNotFoundError,
    list_users,
    set_user_access_by_id,
    set_user_password_by_id,
    update_user_profile,
)
from auth_security import AuthenticationInputError

from auth_service import (
    CSRF_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    AuthenticatedSession,
    AuthenticationFailedError,
    InvalidCsrfTokenError,
    InvalidSessionError,
    create_login_session,
    get_authenticated_session,
    revoke_session,
    validate_csrf_token,
)
from database import DatabaseManager


router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class AuthenticatedUserResponse(BaseModel):
    id: str
    username: str
    display_name: str
    is_active: bool
    roles: list[str]
    created_at: datetime
    updated_at: datetime


class LoginResponse(BaseModel):
    user: AuthenticatedUserResponse
    expires_at: datetime


class UserProfileUpdateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)


class UserPasswordUpdateRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class UserAccessUpdateRequest(BaseModel):
    is_active: bool
    administrator: bool


def _database_manager(request: Request) -> DatabaseManager:
    manager = getattr(request.app.state, "auth_database_manager", None)
    if not isinstance(manager, DatabaseManager):
        raise HTTPException(
            status_code=503,
            detail="Authentication database is not configured",
        )
    return manager


def _cookie_secure(request: Request) -> bool:
    return bool(getattr(request.app.state, "auth_cookie_secure", False))


def _session_hours(request: Request) -> int:
    return int(getattr(request.app.state, "auth_session_hours", 8))


def _public_user(authenticated: AuthenticatedSession) -> AuthenticatedUserResponse:
    return AuthenticatedUserResponse(**authenticated.user.as_public_dict())


def require_authenticated_session(request: Request) -> AuthenticatedSession:
    """Resolve the current database session for protected routes."""

    token = request.cookies.get(SESSION_COOKIE_NAME, "")
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            return get_authenticated_session(session, session_token=token)
    except InvalidSessionError as error:
        raise HTTPException(status_code=401, detail="Authentication required") from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error


def require_protected_request(
    request: Request,
    authenticated: AuthenticatedSession = Depends(require_authenticated_session),
    csrf_header: str | None = Header(None, alias="X-CSRF-Token"),
) -> AuthenticatedSession:
    """Require a session and CSRF proof for state-changing HTTP methods."""

    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        try:
            validate_csrf_token(
                authenticated,
                header_token=csrf_header,
                cookie_token=request.cookies.get(CSRF_COOKIE_NAME),
            )
        except InvalidCsrfTokenError as error:
            raise HTTPException(
                status_code=403,
                detail="A valid CSRF token is required",
            ) from error
    return authenticated


def require_administrator(
    authenticated: AuthenticatedSession = Depends(require_protected_request),
) -> AuthenticatedSession:
    """Authorize management operations independently of frontend visibility."""

    if "admin" not in authenticated.user.roles:
        raise HTTPException(status_code=403, detail="Administrator access is required")
    return authenticated


def _management_error(error: Exception) -> HTTPException:
    if isinstance(error, UserNotFoundError):
        return HTTPException(status_code=404, detail="User was not found")
    if isinstance(error, DuplicateUserError):
        return HTTPException(
            status_code=409,
            detail="A user with that username already exists",
        )
    if isinstance(error, LastAdministratorError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (AuthenticationInputError, AuthenticationRepositoryError)):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail="User management operation failed")


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> LoginResponse:
    manager = _database_manager(request)
    rejected = False
    try:
        # Catch credential rejection inside the transaction scope so failed
        # counters and a resulting lockout are committed rather than rolled back.
        with manager.session() as session:
            try:
                created = create_login_session(
                    session,
                    username=payload.username,
                    password=payload.password,
                    session_hours=_session_hours(request),
                )
            except AuthenticationFailedError:
                rejected = True
                created = None
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error

    if rejected or created is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    max_age = _session_hours(request) * 60 * 60
    secure = _cookie_secure(request)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=created.session_token,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=created.csrf_token,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return LoginResponse(
        user=AuthenticatedUserResponse(**created.user.as_public_dict()),
        expires_at=created.expires_at,
    )


@router.get("/me", response_model=AuthenticatedUserResponse)
def current_user(
    response: Response,
    authenticated: AuthenticatedSession = Depends(require_authenticated_session),
) -> AuthenticatedUserResponse:
    response.headers["Cache-Control"] = "no-store"
    return _public_user(authenticated)


@router.get(
    "/admin/users",
    response_model=list[AuthenticatedUserResponse],
    dependencies=[Depends(require_administrator)],
)
def admin_list_users(request: Request, response: Response) -> list[AuthenticatedUserResponse]:
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            records = list_users(session)
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error
    response.headers["Cache-Control"] = "no-store"
    return [AuthenticatedUserResponse(**record.as_public_dict()) for record in records]


@router.patch(
    "/admin/users/{user_id}",
    response_model=AuthenticatedUserResponse,
)
def admin_update_user_profile(
    user_id: str,
    payload: UserProfileUpdateRequest,
    request: Request,
    response: Response,
    _authenticated: AuthenticatedSession = Depends(require_administrator),
) -> AuthenticatedUserResponse:
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            record = update_user_profile(
                session,
                user_id=user_id,
                username=payload.username,
                display_name=payload.display_name,
            )
    except (AuthenticationInputError, AuthenticationRepositoryError) as error:
        raise _management_error(error) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error
    response.headers["Cache-Control"] = "no-store"
    return AuthenticatedUserResponse(**record.as_public_dict())


@router.post(
    "/admin/users/{user_id}/password",
    response_model=AuthenticatedUserResponse,
)
def admin_update_user_password(
    user_id: str,
    payload: UserPasswordUpdateRequest,
    request: Request,
    response: Response,
    _authenticated: AuthenticatedSession = Depends(require_administrator),
) -> AuthenticatedUserResponse:
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            record = set_user_password_by_id(
                session,
                user_id=user_id,
                password=payload.password,
            )
    except (AuthenticationInputError, AuthenticationRepositoryError) as error:
        raise _management_error(error) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error
    response.headers["Cache-Control"] = "no-store"
    return AuthenticatedUserResponse(**record.as_public_dict())


@router.patch(
    "/admin/users/{user_id}/access",
    response_model=AuthenticatedUserResponse,
)
def admin_update_user_access(
    user_id: str,
    payload: UserAccessUpdateRequest,
    request: Request,
    response: Response,
    _authenticated: AuthenticatedSession = Depends(require_administrator),
) -> AuthenticatedUserResponse:
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            record = set_user_access_by_id(
                session,
                user_id=user_id,
                active=payload.is_active,
                administrator=payload.administrator,
            )
    except (AuthenticationInputError, AuthenticationRepositoryError) as error:
        raise _management_error(error) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error
    response.headers["Cache-Control"] = "no-store"
    return AuthenticatedUserResponse(**record.as_public_dict())


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    authenticated: AuthenticatedSession = Depends(require_protected_request),
) -> None:
    manager = _database_manager(request)
    try:
        with manager.session() as session:
            revoke_session(session, authenticated=authenticated)
    except InvalidSessionError as error:
        raise HTTPException(status_code=401, detail="Authentication required") from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        ) from error

    secure = _cookie_secure(request)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        CSRF_COOKIE_NAME,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store"
