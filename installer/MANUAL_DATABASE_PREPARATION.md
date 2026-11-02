# Preparing the QTracX database without an `sa` password

Choose the installer's **preconfigured** database mode when `sa`/PostgreSQL
superuser credentials are unavailable or client IT prefers to create database
accounts themselves. The installer then asks only for the prepared QTracX
database account.

The account is used by QTracX after installation and must also be able to run
future schema migrations. Keep its credentials available for the installer,
but do not put them into source control.

This is the recommended path when the `sa` password is unknown. If client IT
instead gives you another server-level account with permission to create a
database, create logins and manage users in `QTracX`, select **installer** mode
and enter that account in place of `sa`. The installer does not require the
administrator login to be named `sa`.

## Microsoft SQL Server

Run the following through SSMS using Windows Authentication or another account
that can create the database and login. Replace the password before execution.
If the database or login already exists, keep the applicable existing object
and apply the database-user and role statements.

```sql
USE [master];
GO

IF DB_ID(N'QTracX') IS NULL
BEGIN
    CREATE DATABASE [QTracX];
END;
GO

IF SUSER_ID(N'qtracx_app') IS NULL
BEGIN
    CREATE LOGIN [qtracx_app]
    WITH PASSWORD = 'REPLACE_WITH_A_SECURE_PASSWORD',
         CHECK_POLICY = ON,
         CHECK_EXPIRATION = OFF;
END;
GO

USE [QTracX];
GO

IF DATABASE_PRINCIPAL_ID(N'qtracx_app') IS NULL
BEGIN
    CREATE USER [qtracx_app]
    FOR LOGIN [qtracx_app]
    WITH DEFAULT_SCHEMA = [dbo];
END;
GO

IF IS_ROLEMEMBER(N'db_datareader', N'qtracx_app') <> 1
    ALTER ROLE [db_datareader] ADD MEMBER [qtracx_app];
IF IS_ROLEMEMBER(N'db_datawriter', N'qtracx_app') <> 1
    ALTER ROLE [db_datawriter] ADD MEMBER [qtracx_app];
IF IS_ROLEMEMBER(N'db_ddladmin', N'qtracx_app') <> 1
    ALTER ROLE [db_ddladmin] ADD MEMBER [qtracx_app];
GO
```

In the installer select:

- Database engine: `mssql`
- Setup mode: `preconfigured`
- Database server/instance:
  - default or fixed-port instance: enter its hostname/IP and enable **Use
    explicit TCP port**;
  - named instance: enter `server\instance` (for example,
    `jains\SQLEXPRESS`) and leave **Use explicit TCP port** unchecked;
- Database name: `QTracX`
- Database user: `qtracx_app`
- The password used above
- An ODBC driver actually installed on the QTracX server

Named-instance discovery requires the SQL Server Browser service and network
access to UDP 1434. If client IT disables SQL Browser, ask for the instance's
fixed TCP port, enable **Use explicit TCP port**, and enter that port instead.
`Test-NetConnection` accepts a numeric TCP port only; `SQLEXPRESS` is an
instance name, not a port number.

Typical installer diagnostics are intentionally credential-safe:

- **ODBC driver unavailable**: install/detect 64-bit Driver 18 or 17;
- **server unreachable**: check the server name, TCP/firewall settings, and
  SQL Browser or the fixed instance port;
- **login rejected**: check the account/password and SQL Server authentication
  mode; and
- **permission denied**: grant the account the provisioning or migration
  permissions described above.

## PostgreSQL

Run the following with a PostgreSQL administrator. Replace the password before
execution. PostgreSQL does not allow `CREATE DATABASE` inside a transaction, so
run the statements through an administration tool that commits each command.

```sql
CREATE ROLE qtracx_app
WITH LOGIN PASSWORD 'REPLACE_WITH_A_SECURE_PASSWORD';

CREATE DATABASE "QTracX"
WITH OWNER = qtracx_app;
```

Then connect to `QTracX` and run:

```sql
GRANT CONNECT ON DATABASE "QTracX" TO qtracx_app;
GRANT USAGE, CREATE ON SCHEMA public TO qtracx_app;
```

In the installer select:

- Database engine: `postgresql`
- Setup mode: `preconfigured`
- Database name: `QTracX`
- Database user: `qtracx_app`
- The password used above

The installer tests the connection and required DDL permissions before it
applies migrations. It does not store or request the server administrator's
credentials in this mode.
