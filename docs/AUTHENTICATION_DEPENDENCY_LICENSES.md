# Authentication dependency licenses

QTracX authentication uses only permissively licensed Python packages. This
record covers the direct and runtime-transitive authentication/database
dependency group; it is not yet a complete inventory of every dependency in
QTracX.

| Package | Purpose | License |
| --- | --- | --- |
| SQLAlchemy | Database toolkit and ORM | MIT |
| Alembic | Database schema migrations | MIT |
| pg8000 | PostgreSQL driver | BSD-3-Clause |
| pyodbc | MSSQL/ODBC driver | MIT-0 |
| argon2-cffi | Password hashing API | MIT |
| argon2-cffi-bindings | Argon2 native bindings | MIT |
| Argon2 vendored implementation | Password hashing implementation | CC0 |
| cffi | Native binding support | MIT-0 |
| pycparser | C parser used by CFFI | BSD-3-Clause |
| Mako | Alembic migration templates | MIT |
| MarkupSafe | Mako string handling | BSD-3-Clause |
| typing-extensions | Python typing compatibility | PSF-2.0 |
| scramp | PostgreSQL authentication support used by pg8000 | MIT-0 |
| asn1crypto | SCRAM channel-binding support | MIT |
| python-dateutil | Date handling used by pg8000 | Apache-2.0 OR BSD-3-Clause |
| six | Python compatibility used by python-dateutil | MIT |

MIT-0, CC0, and PSF-2.0 are permissive licenses accepted as equivalent to the
project's MIT/BSD/Apache requirement. Optional extras and development/test-only
packages are not installed by `requirements-auth.txt` and are outside this
runtime list.

Source records:

- https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE
- https://github.com/sqlalchemy/alembic/blob/main/LICENSE
- https://github.com/tlocke/pg8000/blob/main/LICENSE
- https://github.com/mkleehammer/pyodbc/blob/master/LICENSE.txt
- https://github.com/hynek/argon2-cffi/blob/main/LICENSE
- https://github.com/hynek/argon2-cffi-bindings/blob/main/LICENSE
- https://github.com/python-cffi/cffi/blob/main/LICENSE
- https://github.com/eliben/pycparser/blob/main/LICENSE
- https://github.com/sqlalchemy/mako/blob/main/LICENSE
- https://github.com/pallets/markupsafe/blob/main/LICENSE.txt
- https://github.com/python/typing_extensions/blob/main/LICENSE
- https://github.com/tlocke/scramp/blob/main/LICENSE
- https://github.com/wbond/asn1crypto/blob/master/LICENSE
- https://github.com/dateutil/dateutil/blob/master/LICENSE
- https://github.com/benjaminp/six/blob/main/LICENSE
