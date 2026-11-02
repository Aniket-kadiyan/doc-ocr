# QTraX user management guide

This guide covers account creation and maintenance for a deployed QTraX
installation. Run command-line operations on the server that hosts the QTraX
backend. Browser users cannot create accounts.

## Responsibilities and access

- New accounts are created only from `backend/manage_auth.py` on the server.
- The separate browser management page can change an existing account's
  username, display name, password, active status, or administrator role.
- The browser page requires a signed-in account with the `admin` role.
- The page does not provide account creation or deletion controls.
- Existing passwords cannot be displayed. They are stored as Argon2id hashes;
  an administrator can only set a new password.

## Open the management page

Start the normal QTraX backend and frontend, then open:

```text
http://SERVER_ADDRESS:3002/user-management
```

For a local installation, use:

```text
http://localhost:3002/user-management
```

Use the deployed UI port if it is not `3002`. Sign in with a QTraX
administrator account. The page is intentionally not linked from the normal
QTraX workflow.

Select an existing account and then:

1. Edit **Username** or **Display name**, then choose **Save user details**.
2. To reset a password, enter and confirm the new password, then choose
   **Update password**.
3. Under **Roles and permissions**, enable or disable application access and
   grant or revoke the **Administrator** role.
4. Tell the affected user their new username or temporary password through an
   approved private channel.

A password, access, or role change revokes every active session belonging to
that account. If an administrator changes their own password, access, or role,
they must sign in again. Username and display-name changes retain the account's
stable database ID. The final active administrator cannot be disabled or
demoted.

## Run administrative commands

Open PowerShell and go to the backend folder:

```powershell
cd "C:\QTracX\backend"
```

For a CPU installation, the examples below use `.venv`. For a GPU
installation, replace `.venv` with `.venv-gpu`.

Verify that the database is reachable and up to date:

```powershell
.\.venv\Scripts\python.exe manage_auth.py db check
.\.venv\Scripts\python.exe manage_auth.py db upgrade
```

`db upgrade` is normally needed only after installation or an application
update that contains a database migration.

## Create accounts

Create a regular user:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user create-user `
  --username operator.one `
  --display-name "Operator One"
```

Create an administrator:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user create-admin `
  --username quality.admin `
  --display-name "Quality Administrator"
```

Both commands prompt for the password twice. The password is not accepted on
the command line, shown on screen, or retained in PowerShell history.

Usernames must be 3–64 characters and may contain letters, numbers, periods,
underscores, hyphens, and `@`. Passwords must be 4–128 characters and cannot
equal the username.

## List and edit accounts from PowerShell

List accounts without password data:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user list
```

Change a username:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user set-username `
  operator.one operator.01
```

Change a display name:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user set-display-name `
  operator.01 "Inspection Operator"
```

Reset a password:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user set-password operator.01
```

## Enable, disable, and control administrator access

Disable or enable an account:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user disable operator.01
.\.venv\Scripts\python.exe manage_auth.py user enable operator.01
```

Grant or revoke administrator access:

```powershell
.\.venv\Scripts\python.exe manage_auth.py user grant-admin operator.01
.\.venv\Scripts\python.exe manage_auth.py user revoke-admin operator.01
```

The last active administrator cannot be disabled or have the administrator
role removed. Disabling an account, changing its password, or changing its
administrator role revokes its active sessions.

## Operational notes

- Do not share database credentials with ordinary QTraX users.
- Do not place passwords in command arguments, scripts, tickets, or logs.
- Prefer a temporary password that the administrator changes again when the
  user is present; QTraX does not currently implement forced first-login
  password changes.
- Account deletion is deliberately not supported. Disable an account when a
  person no longer requires access so historical ownership remains traceable.
- If the browser page reports `Administrator access is required`, sign out and
  use an account that has the `admin` role.
