QTraX user-management overlay
Base: origin/deployment-readiness at 539a945

WHAT THIS ADDS
- Separate browser UI at /user-management.
- Administrator-only API for listing existing users.
- Existing-user edits: username, display name, and password reset.
- No account-creation HTTP endpoint or UI control.
- CLI commands for username and display-name changes.
- Focused user-management guide in docs/USER_MANAGEMENT.md.

APPLY
1. Start from the latest deployment-readiness branch containing commit 539a945
   or its descendants.
2. Extract this ZIP into the repository root and allow matching files to be
   overwritten.
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
Restart both the QTraX backend and frontend, then open:
  http://localhost:3002/user-management

For a LAN deployment, replace localhost and 3002 with the deployed server
address and UI port. Sign in with a user that has the admin role.

USER CREATION
The management page intentionally cannot create users. See
docs/USER_MANAGEMENT.md for the exact create-user and create-admin commands.

SUGGESTED COMMIT MESSAGE
feat(auth): add restricted existing-user management console
