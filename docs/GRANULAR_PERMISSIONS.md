# QTraX granular permissions

## Permission model

QTraX resolves access from two layers:

1. Role defaults provide the normal permissions for `user` and `admin`.
2. A per-user override may explicitly allow or deny an operational permission.

An override can also be removed to return that permission to its role default.
The `users.manage` permission is deliberately role-controlled and cannot be
overridden directly. This keeps administrator access aligned with the existing
last-active-administrator protection.

## Permission catalogue

| Permission | Category | Scope |
| --- | --- | --- |
| `drawing.use` | Drawing | Drawings, OCR, scans, ballooning, GD&T and review |
| `drawing.export` | Drawing | Drawing, Excel, PDF and checksheet export actions |
| `checksheet.view` | Checksheet | Saved checksheets, runs and source documents |
| `checksheet.manage` | Checksheet | Create, revise, rename, archive, duplicate and delete |
| `checksheet.execute` | Checksheet | Start, fill and complete inspection runs |
| `training.manage` | Training | GD&T corrections and training-label operations |
| `users.manage` | Administration | Existing-user, role and permission management |

The `user` role receives every current non-administrative permission. The
`admin` role adds `users.manage`. Administrators also retain the `user` role.
This preserves all existing access immediately after the migration.

## Database objects

Migration `0003_granular_permissions` adds:

- `qtracx_permissions`
- `qtracx_role_permissions`
- `qtracx_user_permission_overrides`

Apply it from `backend` before starting the updated application:

```powershell
.\.venv-gpu\Scripts\python.exe manage_auth.py db upgrade
.\.venv-gpu\Scripts\python.exe manage_auth.py db check
```

Use `.venv` instead of `.venv-gpu` for a CPU-only installation.

## Effective permissions

Authentication responses now include:

- `permissions`: the final sorted list after role defaults and overrides.
- `permission_overrides`: the account's explicit allow/deny values.

Changing an override revokes that account's active sessions so its next login
receives the new authorization state.

## Delivery stages

This foundation stage stores and resolves permissions but deliberately does not
block application routes yet. Subsequent stages will add:

1. Backend route enforcement.
2. Permission editing in `/user-management`.
3. Frontend workflow visibility and action guards.
4. Full authorization and upgrade regression tests.
