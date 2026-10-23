[CmdletBinding()]
param(
    [ValidateSet("Plan", "Install", "Verify")]
    [string]$Mode = "Plan",

    [ValidateSet("cpu", "gpu")]
    [string]$Runtime = "cpu",

    [ValidateSet("mssql", "postgresql")]
    [string]$Database = "mssql",

    [string]$ApplicationRoot = "",

    [string]$EnvironmentDir = "",

    [switch]$HostOnly,
    [switch]$IncludeGit,
    [switch]$BuildFrontend,
    [switch]$WarmModels
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ApplicationRoot) {
    $ApplicationRoot = Split-Path -Parent $PSScriptRoot
}
$ApplicationRoot = [System.IO.Path]::GetFullPath($ApplicationRoot)

$installCommands = [ordered]@{
    python = "winget install --exact --id Python.Python.3.13 --source winget --scope machine --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity"
    node = "winget install --exact --id OpenJS.NodeJS.LTS --source winget --scope machine --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity"
    git = "winget install --exact --id Git.Git --source winget --scope machine --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity"
    mssqlOdbc = "winget install --exact --id Microsoft.msodbcsql.18 --source winget --scope machine --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity"
}

function Update-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = @($machinePath, $userPath) -join ";"
}

function Test-QTracXAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    $probe = "import json,platform,struct,sys; print(json.dumps({'version':platform.python_version(),'major':sys.version_info.major,'minor':sys.version_info.minor,'bits':struct.calcsize('P')*8,'executable':sys.executable}))"
    try {
        $output = & $Executable @PrefixArguments -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        $details = ($output | Out-String).Trim() | ConvertFrom-Json
        if ($details.major -ne 3 -or $details.minor -ne 13 -or $details.bits -ne 64) {
            return $null
        }
        return [pscustomobject]@{
            available = $true
            supported = $true
            version = [string]$details.version
            architectureBits = [int]$details.bits
            executable = $Executable
            resolvedExecutable = [string]$details.executable
            prefixArguments = @($PrefixArguments)
        }
    }
    catch {
        return $null
    }
}

function Get-QTracXPython {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = Test-PythonCandidate -Executable $launcher.Source `
            -PrefixArguments @("-3.13")
        if ($candidate) {
            return $candidate
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $candidate = Test-PythonCandidate -Executable $python.Source
        if ($candidate) {
            return $candidate
        }
    }

    $knownPaths = @(
        (Join-Path $env:ProgramFiles "Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe")
    )
    foreach ($path in $knownPaths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $candidate = Test-PythonCandidate -Executable $path
            if ($candidate) {
                return $candidate
            }
        }
    }

    return [pscustomobject]@{
        available = $false
        supported = $false
        version = $null
        architectureBits = $null
        executable = $null
        resolvedExecutable = $null
        prefixArguments = @()
    }
}

function Get-QTracXNode {
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $node -or -not $npm) {
        $nodePath = $null
        $npmPath = $null
        if ($node) { $nodePath = $node.Source }
        if ($npm) { $npmPath = $npm.Source }
        return [pscustomobject]@{
            available = $false
            supported = $false
            version = $null
            executable = $nodePath
            npm = $npmPath
        }
    }
    try {
        $rawVersion = (& $node.Source --version 2>$null | Out-String).Trim()
        $version = [version]($rawVersion.TrimStart([char]"v"))
        return [pscustomobject]@{
            available = $true
            supported = $version -ge [version]"20.19.0"
            version = $version.ToString()
            executable = $node.Source
            npm = $npm.Source
        }
    }
    catch {
        return [pscustomobject]@{
            available = $false
            supported = $false
            version = $null
            executable = $node.Source
            npm = $npm.Source
        }
    }
}

function Get-QTracXGit {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    $gitPath = $null
    if ($git) { $gitPath = $git.Source }
    return [pscustomobject]@{
        available = [bool]$git
        executable = $gitPath
    }
}

function Get-QTracXMssqlOdbc {
    $names = @()
    try {
        $names = @(
            Get-OdbcDriver -ErrorAction Stop |
                Where-Object {
                    $_.Platform -eq "64-bit" -and
                    $_.Name -match "^ODBC Driver ([0-9]+) for SQL Server$"
                } |
                Select-Object -ExpandProperty Name -Unique
        )
    }
    catch {
        $names = @()
    }

    $supported = @(
        @(
            foreach ($name in $names) {
                if ($name -match "^ODBC Driver ([0-9]+) for SQL Server$") {
                    $major = [int]$Matches[1]
                    if ($major -in @(17, 18)) {
                        [pscustomobject]@{ name = $name; major = $major }
                    }
                }
            }
        ) | Sort-Object -Property major -Descending
    )

    $selected = if ($supported.Count -gt 0) { $supported[0].name } else { $null }
    return [pscustomobject]@{
        available = [bool]$selected
        installed64BitDrivers = @($names)
        selectedDriver = $selected
        supportedMajorVersions = @(18, 17)
    }
}

function Get-QTracXGpu {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return [pscustomobject]@{ available = $false; devices = @() }
    }
    try {
        $devices = @(
            & $nvidiaSmi.Source `
                --query-gpu=index,name,memory.total `
                --format=csv,noheader,nounits 2>$null |
                Where-Object { $_.Trim() }
        )
        return [pscustomobject]@{
            available = $devices.Count -gt 0
            devices = $devices
        }
    }
    catch {
        return [pscustomobject]@{ available = $false; devices = @() }
    }
}

function Get-QTracXHostState {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        windows64Bit = [Environment]::Is64BitOperatingSystem
        process64Bit = [Environment]::Is64BitProcess
        administrator = Test-QTracXAdministrator
        wingetAvailable = [bool]$winget
        python = Get-QTracXPython
        node = Get-QTracXNode
        git = Get-QTracXGit
        mssqlOdbc = Get-QTracXMssqlOdbc
        gpu = Get-QTracXGpu
    }
}

function Get-QTracXActions {
    param([Parameter(Mandatory = $true)]$State)

    $actions = @()
    if (-not $State.python.supported) {
        $actions += [pscustomobject]@{
            prerequisite = "Python 3.13 x64"
            required = $true
            command = $installCommands.python
        }
    }
    if (-not $State.node.supported) {
        $actions += [pscustomobject]@{
            prerequisite = "Node.js LTS and npm"
            required = $true
            command = $installCommands.node
        }
    }
    if ($IncludeGit -and -not $State.git.available) {
        $actions += [pscustomobject]@{
            prerequisite = "Git"
            required = $true
            command = $installCommands.git
        }
    }
    if ($Database -eq "mssql" -and -not $State.mssqlOdbc.available) {
        $actions += [pscustomobject]@{
            prerequisite = "Microsoft ODBC Driver 18 for SQL Server x64"
            required = $true
            command = $installCommands.mssqlOdbc
        }
    }
    if ($Runtime -eq "gpu" -and -not $State.gpu.available) {
        $actions += [pscustomobject]@{
            prerequisite = "Supported NVIDIA GPU and driver"
            required = $true
            command = $null
        }
    }
    return @($actions)
}

function Invoke-QTracXWingetInstall {
    param([Parameter(Mandatory = $true)][string]$PackageId)

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "WinGet is unavailable. Install the prerequisite manually and rerun Verify mode."
    }
    Write-Host "Installing $PackageId ..."
    & $winget.Source install `
        --exact `
        --id $PackageId `
        --source winget `
        --scope machine `
        --architecture x64 `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "WinGet failed to install $PackageId (exit code $LASTEXITCODE)."
    }
}

function Invoke-QTracXRuntimeSetup {
    param(
        [Parameter(Mandatory = $true)]$Python,
        [Parameter(Mandatory = $true)][ValidateSet("plan", "install", "verify")]
        [string]$RuntimeMode
    )

    $script = Join-Path $PSScriptRoot "runtime_setup.py"
    if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
        throw "Runtime setup module was not found: $script"
    }
    $arguments = @($Python.prefixArguments) + @(
        $script,
        "--mode", $RuntimeMode,
        "--application-root", $ApplicationRoot,
        "--runtime", $Runtime
    )
    if ($EnvironmentDir) {
        $arguments += @("--environment-dir", $EnvironmentDir)
    }
    if ($BuildFrontend) {
        $arguments += "--build-frontend"
    }
    if ($WarmModels) {
        $arguments += "--warm-models"
    }
    & $Python.executable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "QTracX runtime setup failed (exit code $LASTEXITCODE)."
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "QTracX requires 64-bit Windows."
}

$state = Get-QTracXHostState
$actions = @(Get-QTracXActions -State $state)
$report = [ordered]@{
    status = if ($actions.Count -eq 0) { "ready" } else { "action-required" }
    mode = $Mode.ToLowerInvariant()
    changesApplied = $false
    applicationRoot = $ApplicationRoot
    selectedRuntime = $Runtime
    selectedDatabase = $Database
    applicationSetup = $(if ($HostOnly) { "skipped" } else { "included" })
    state = $state
    actions = $actions
    manualDownloadPages = [ordered]@{
        python = "https://www.python.org/downloads/windows/"
        node = "https://nodejs.org/en/download"
        mssqlOdbc = "https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server"
        winget = "https://learn.microsoft.com/windows/package-manager/winget/"
    }
}

if ($Mode -eq "Plan") {
    if (-not $HostOnly -and $state.python.supported -and $state.node.supported) {
        Write-Host "`nApplication dependency plan (read-only):"
        Invoke-QTracXRuntimeSetup -Python $state.python -RuntimeMode "plan"
        Write-Host "`nHost prerequisite plan:"
    }
    $report | ConvertTo-Json -Depth 10
    exit 0
}

if ($Mode -eq "Install") {
    if (-not $state.python.supported) {
        Invoke-QTracXWingetInstall -PackageId "Python.Python.3.13"
    }
    if (-not $state.node.supported) {
        Invoke-QTracXWingetInstall -PackageId "OpenJS.NodeJS.LTS"
    }
    if ($IncludeGit -and -not $state.git.available) {
        Invoke-QTracXWingetInstall -PackageId "Git.Git"
    }
    if ($Database -eq "mssql" -and -not $state.mssqlOdbc.available) {
        Invoke-QTracXWingetInstall -PackageId "Microsoft.msodbcsql.18"
    }
    Update-ProcessPath
    $state = Get-QTracXHostState
    $actions = @(Get-QTracXActions -State $state)
    if ($actions.Count -gt 0) {
        $remaining = ($actions | ForEach-Object { $_.prerequisite }) -join ", "
        throw "Required prerequisites remain unavailable: $remaining"
    }
    if (-not $HostOnly) {
        Invoke-QTracXRuntimeSetup -Python $state.python -RuntimeMode "install"
    }
    $report.status = "ok"
    $report.changesApplied = $true
    $report.state = $state
    $report.actions = @()
    Write-Host "`nHost setup result:"
    $report | ConvertTo-Json -Depth 10
    exit 0
}

$actions = @(Get-QTracXActions -State $state)
if ($actions.Count -gt 0) {
    $report | ConvertTo-Json -Depth 10
    exit 1
}
if (-not $HostOnly) {
    Invoke-QTracXRuntimeSetup -Python $state.python -RuntimeMode "verify"
}
$report.status = "ok"
$report.actions = @()
Write-Host "`nHost verification result:"
$report | ConvertTo-Json -Depth 10
