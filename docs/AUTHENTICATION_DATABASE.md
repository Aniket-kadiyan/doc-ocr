# QTracX authentication and session management

The authentication milestone provides the database schema, password hashing,
administrative user tooling, and database-backed browser login sessions.
The frontend restores the current session before showing the application, and
the backend now protects drawing, OCR, GD&T, checksheet, training, and
diagnostic routes.

Production supports PostgreSQL and Microsoft SQL Server. SQLite is used only
by automated schema tests and is deliberately rejected by deployment commands.

## Install into an existing OCR environment

From `doc-ocr/backend` in PowerShell:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-auth.txt
```

For a CPU environment, replace `.venv-gpu` with that environment's folder.
Installing this smaller file avoids reinstalling or replacing PaddlePaddle.

## Configure PostgreSQL

Add a URL like this to the repository's `.env.local`:

```dotenv
DATABASE_URL=postgresql+pg8000://qtracx_user:CHANGE_ME@db-server:5432/qtracx
```

Create a dedicated database and account. The account needs permission to
create and alter QTracX tables while migrations are being applied.

## Configure Microsoft SQL Server

Install Microsoft ODBC Driver 18 for SQL Server on the backend machine, then
add a URL like this to `.env.local`:

```dotenv
DATABASE_URL=mssql+pyodbc://qtracx_user:CHANGE_ME@db-server:1433/qtracx?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no
```

If the SQL Server uses an internal certificate that the backend machine does
not trust yet, install the issuing CA certificate on the backend machine. Do
not put real passwords in `.env.local.example` or commit `.env.local`.

## Create or upgrade the schema

From `doc-ocr/backend`:

```powershell
.\.venv-gpu\Scripts\python.exe manage_auth.py db upgrade
.\.venv-gpu\Scripts\python.exe manage_auth.py db check
```

`db check` returns `status: ok` when the database is reachable and all required
tables exist. Connection strings and credentials are excluded from its output.

The first migration creates:

- `qtracx_users`
- `qtracx_roles`
- `qtracx_user_roles`
- `qtracx_auth_sessions`
- `alembic_version`

Session and CSRF values will be stored as one-way hashes. Plain session tokens
are never stored in the database.

The second migration also adds `failed_login_count`, `locked_until`, and
`last_login_at` to `qtracx_users`. Five consecutive invalid passwords lock the
account for 15 minutes. Login failures always return the same message, whether
the username is absent, inactive, locked, or has a wrong password.

## Configure browser sessions

The defaults are suitable for the current HTTP deployment:

```dotenv
AUTH_SESSION_HOURS=8
AUTH_COOKIE_SECURE=false
```

`AUTH_SESSION_HOURS` accepts 1 through 168. Change `AUTH_COOKIE_SECURE` to
`true` when the optional HTTPS milestone is deployed. A secure cookie is not
sent by browsers over plain HTTP.

For an HTTP LAN deployment, the UI and backend must use the same browser host
name or IP address, although their ports may differ. For example, use
`http://192.168.1.20:3002` for the UI and
`http://192.168.1.20:8000` for the backend. Do not open the UI using an IP while
configuring `NEXT_PUBLIC_OCR_API_URL` with a different alias such as
`localhost`; `SameSite=Lax` cookies are intentionally not sent cross-site.

Login creates two `SameSite=Lax` cookies:

- `qtracx_session` is `HttpOnly` and contains an opaque random session token.
- `qtracx_csrf` is readable by the frontend and must be returned in the
  `X-CSRF-Token` header for logout and future state-changing protected routes.

Only SHA-256 hashes of these values are stored in `qtracx_auth_sessions`.
Passwords remain Argon2id hashes. Authentication responses use
`Cache-Control: no-store`, and request logging does not record request bodies.

## Authentication API

```text
POST /auth/login   JSON: {"username":"...","password":"..."}
GET  /auth/me      Requires qtracx_session cookie
POST /auth/logout  Requires both cookies and X-CSRF-Token header
```

Login returns the public user profile and session expiry. `/auth/me` returns
the currently authenticated public profile. Logout revokes the database
session and clears both cookies. Password hashes and raw session secrets are
never returned.

## Protected application routes

`/health` and `/auth/login` remain public. Application routes require a valid
`qtracx_session` cookie. All protected POST, PUT, PATCH, and DELETE requests
also require the `qtracx_csrf` value in the `X-CSRF-Token` header.

The frontend request helper supplies credentials and the CSRF header for OCR,
scan jobs, document rendering, GD&T, checksheets, training data, and client
diagnostics. A backend `401` returns the user to the sign-in screen. Refreshing
the browser restores an unexpired session through `/auth/me`.

Scan-progress polling still validates every request, but writes
`last_seen_at` at most once per minute to avoid unnecessary database writes.

## Create the first administrator

After `db upgrade` and `db check` succeed, run:

```powershell
.\.venv-gpu\Scripts\python.exe manage_auth.py user create-admin `
  --username admin `
  --display-name "QTracX Administrator"
```

The command prompts twice for the password. The password is not accepted as a
command-line argument, written to the console, stored in shell history, or
included in application logs. Passwords require at least 12 characters and are
stored as Argon2id hashes.

Additional administration commands:

```powershell
.\.venv-gpu\Scripts\python.exe manage_auth.py user list
.\.venv-gpu\Scripts\python.exe manage_auth.py user disable USERNAME
.\.venv-gpu\Scripts\python.exe manage_auth.py user enable USERNAME
.\.venv-gpu\Scripts\python.exe manage_auth.py user set-password USERNAME
```

The final active administrator cannot be disabled. Disabling a user or changing
a password revokes that user's active sessions. Changing a password also clears
the failed-login counter and any current timed lockout.

## Test this stage

From `doc-ocr/backend`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_database.py test_auth_schema.py test_auth_security.py `
  test_auth_repository.py test_auth_service.py test_auth_routes.py `
  test_manage_auth.py test_config_env.py `
  test_app_logging.py test_scan_jobs.py
```

Apply `db upgrade` before starting the backend so migration
`0002_auth_login_security` is present in MSSQL or PostgreSQL.

Frontend verification from the repository root:

```powershell
npm run typecheck
npm run lint
npm run test:frontend
npm run build
```
