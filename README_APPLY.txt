QTraX user-management overlay v2
Base: origin/deployment-readiness at 539a945

This is a cumulative overlay. It includes the original user-management page
plus role and access management. Apply it directly over the working v1 files;
do not revert v1 first.

WHAT THIS ADDS
- Separate browser UI at /user-management.
- Administrator-only API for listing and changing existing users.
- Existing-user edits: username, display name, and password reset.
- Enable or disable an account's application access.
- Grant or revoke the Administrator role.
- Effective-permission summary in the management UI.
- Last-active-administrator protection and session revocation.
- No account-creation HTTP endpoint or UI control.
- CLI commands and a focused guide in docs/USER_MANAGEMENT.md.

APPLY
1. Extract this ZIP into the repository root.
2. Allow matching files to be overwritten.
3. No database migration is required.

TEST FROM THE REPOSITORY ROOT (POWERSHELL)
Push-Location backend
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_auth_security.py test_auth_repository.py test_auth_service.py `
  test_auth_routes.py test_manage_auth.py test_auth_schema.py
Pop-Location

npm run lint
npm run typecheck
npm run test:frontend
npm run build

Replace .venv-gpu with .venv on a CPU-only development environment.

RUN
Restart the QTraX backend and frontend, then open:
  http://localhost:3002/user-management

For a LAN deployment, replace localhost and 3002 with the deployed server
address and UI port. Sign in with an account that has the admin role.

CURRENT PERMISSION MODEL
- Standard user: required base role for normal QTraX workflows.
- Administrator: access to existing-user management.
- Active status: controls whether the account can sign in.

Granular per-module permissions are not part of the current database or route
authorization model and are not simulated by this UI.

SUGGESTED COMMIT MESSAGE
feat(auth): manage user roles and access from admin console
