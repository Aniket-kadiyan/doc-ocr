# QTracX authentication database foundation

This stage introduces only the database schema and migration tooling required
for login. It does not yet enable login or protect application endpoints.

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

## Test this stage

From `doc-ocr/backend`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_database.py test_auth_schema.py test_config_env.py `
  test_app_logging.py test_scan_jobs.py
```

Expected result: `26 passed` plus the existing subtests.
