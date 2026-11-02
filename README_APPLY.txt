QTraX granular permissions — G1 authorization foundation
Base: origin/granular-permissions at 85b3a10

SCOPE
- Permission catalogue and role-default mappings.
- Per-user allow/deny override storage and resolver.
- Effective permissions in login, /auth/me and admin user responses.
- Administrator-readable permission catalogue endpoint.
- Migration 0003_granular_permissions.
- Existing users retain all current non-admin functionality.
- No OCR/checksheet route is denied by granular permissions in G1.

APPLY
1. Confirm you are on the granular-permissions branch at 85b3a10 or a direct
   descendant.
2. Extract this ZIP into the repository root and overwrite matching files.
3. Install no new libraries; this uses the existing SQLAlchemy/Alembic stack.

DATABASE UPGRADE (REQUIRED)
From the backend folder, while the backend is stopped:

  .\.venv-gpu\Scripts\python.exe manage_auth.py db upgrade
  .\.venv-gpu\Scripts\python.exe manage_auth.py db check

Use .venv instead of .venv-gpu for a CPU-only environment. db check must show
status ok before the backend is started. The migration supports PostgreSQL,
MSSQL and the SQLite test environment.

TEST FROM THE REPOSITORY ROOT
Push-Location backend
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_auth_schema.py test_auth_repository.py test_auth_service.py `
  test_auth_routes.py test_manage_auth.py test_config_env.py test_demo_lock.py
Pop-Location

npm run lint
npm run typecheck
npm run test:frontend
npm run build

SMOKE CHECK
1. Start the backend and sign in as an existing normal user.
2. GET /auth/me must include permissions and permission_overrides.
3. Existing drawing, export and checksheet workflows must still work normally.
4. Sign in as an administrator and verify /user-management still loads.

EXPECTED CURRENT BEHAVIOUR
G1 stores and resolves permissions but does not enforce them on application
routes yet. Enforcement is G2, permission editing in the management UI is G3,
and frontend workflow guards are G4.

SUGGESTED COMMIT MESSAGE
feat(auth): add granular permission foundation and migration
