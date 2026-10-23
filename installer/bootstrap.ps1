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

$catalogPath = Join-Path $PSScriptRoot "config\prerequisites.windows-x64.json"
if (-not (Test-Path -LiteralPath $catalogPath -PathType Leaf)) {
    throw "Prerequisite catalog was not found: $catalogPath"
}
$prerequisiteCatalog = Get-Content -LiteralPath $catalogPath -Raw |
    ConvertFrom-Json
$prerequisiteBundleRoot = Join-Path $PSScriptRoot "prerequisites"

function Get-QTracXPackage {
    param([Parameter(Mandatory = $true)][string]$Name)

    $property = $prerequisiteCatalog.packages.PSObject.Properties[$Name]
    if (-not $property) {
        throw "Unknown prerequisite package: $Name"
    }
    return $property.Value
}

function Get-QTracXWingetCommand {
    param([Parameter(Mandatory = $true)]$Package)

    return (
        "winget install --exact --id {0} --version {1} --source winget " +
        "--scope machine --architecture x64 --silent " +
        "--accept-package-agreements --accept-source-agreements " +
        "--disable-interactivity"
    ) -f $Package.wingetId, $Package.wingetVersion
}

function Get-QTracXBundleLock {
    $lockPath = Join-Path $prerequisiteBundleRoot "prerequisites.lock.json"
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Get-QTracXBundledInstaller {
    param([Parameter(Mandatory = $true)][string]$Name)

    $package = Get-QTracXPackage -Name $Name
    $path = Join-Path $prerequisiteBundleRoot ([string]$package.fileName)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return [pscustomobject]@{
            status = "not-bundled"
            usable = $false
            path = $path
            sha256 = $null
            verification = $null
        }
    }

    $expectedHash = ([string]$package.sha256).ToLowerInvariant()
    $verification = "catalog"
    if (-not $expectedHash) {
        $verification = "bundle-lock"
        $lock = Get-QTracXBundleLock
        if ($lock) {
            $entry = @($lock.files | Where-Object {
                $_.key -eq $Name -and $_.fileName -eq $package.fileName
            }) | Select-Object -First 1
            if ($entry) {
                $expectedHash = ([string]$entry.sha256).ToLowerInvariant()
            }
        }
    }
    if (-not $expectedHash) {
        return [pscustomobject]@{
            status = "unverified"
            usable = $false
            path = $path
            sha256 = $null
            verification = $verification
        }
    }

    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $valid = $actualHash -eq $expectedHash
    $publisherProperty = $package.PSObject.Properties["authenticodePublisher"]
    if ($valid -and $publisherProperty -and [string]$publisherProperty.Value) {
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        $publisher = [string]$publisherProperty.Value
        $valid = (
            $signature.Status -eq [System.Management.Automation.SignatureStatus]::Valid -and
            $signature.SignerCertificate -and
            $signature.SignerCertificate.Subject -like "*$publisher*"
        )
        if (-not $valid) {
            return [pscustomobject]@{
                status = "invalid-signature"
                usable = $false
                path = $path
                sha256 = $actualHash
                verification = "authenticode"
            }
        }
    }
    return [pscustomobject]@{
        status = $(if ($valid) { "ready" } else { "invalid-hash" })
        usable = $valid
        path = $path
        sha256 = $actualHash
        verification = $verification
    }
}

function Get-QTracXPrerequisiteSummary {
    $packages = @()
    foreach ($property in $prerequisiteCatalog.packages.PSObject.Properties) {
        $package = $property.Value
        $bundle = Get-QTracXBundledInstaller -Name $property.Name
        $packages += [ordered]@{
            key = $property.Name
            displayName = [string]$package.displayName
            version = [string]$package.version
            officialDownloadUrl = [string]$package.officialDownloadUrl
            fileName = [string]$package.fileName
            wingetCommand = Get-QTracXWingetCommand -Package $package
            bundledStatus = $bundle.status
            bundledPath = $bundle.path
            license = [string]$package.licenseName
        }
    }
    return [ordered]@{
        catalog = $catalogPath
        platform = [string]$prerequisiteCatalog.platform
        bundleDirectory = $prerequisiteBundleRoot
        packages = $packages
    }
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
    if (-not $node) {
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
            npmVersion = $null
            npmCli = $null
            npmSource = $null
            required = "Node.js 20.19 or newer with npm 10 or newer"
        }
    }
    try {
        $rawVersion = (& $node.Source --version 2>$null | Out-String).Trim()
        $version = [version]($rawVersion.TrimStart([char]"v"))

        # npm.cmd can redirect to a stale globally installed npm under
        # %APPDATA%. Prefer the npm-cli.js shipped with the selected Node.js
        # installation so setup is deterministic and leaves global npm alone.
        $nodeDirectory = Split-Path -Parent $node.Source
        $bundledNpmCli = Join-Path $nodeDirectory "node_modules\npm\bin\npm-cli.js"
        $npmVersion = $null
        $npmCli = $null
        $npmSource = $null
        if (Test-Path -LiteralPath $bundledNpmCli -PathType Leaf) {
            $rawNpmVersion = (
                & $node.Source $bundledNpmCli --version 2>$null |
                    Out-String
            ).Trim()
            if ($LASTEXITCODE -eq 0 -and $rawNpmVersion) {
                $npmVersion = [version]$rawNpmVersion
                $npmCli = $bundledNpmCli
                $npmSource = "node-bundled"
            }
        }
        elseif ($npm) {
            $rawNpmVersion = (& $npm.Source --version 2>$null | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and $rawNpmVersion) {
                $npmVersion = [version]$rawNpmVersion
                $npmCli = $npm.Source
                $npmSource = "path"
            }
        }

        $supported = (
            $version -ge [version]"20.19.0" -and
            $npmVersion -and
            $npmVersion -ge [version]"10.0.0"
        )
        return [pscustomobject]@{
            available = $true
            supported = [bool]$supported
            version = $version.ToString()
            executable = $node.Source
            npm = $(if ($npm) { $npm.Source } else { $null })
            npmVersion = $(if ($npmVersion) { $npmVersion.ToString() } else { $null })
            npmCli = $npmCli
            npmSource = $npmSource
            required = "Node.js 20.19 or newer with npm 10 or newer"
        }
    }
    catch {
        return [pscustomobject]@{
            available = $false
            supported = $false
            version = $null
            executable = $node.Source
            npm = $(if ($npm) { $npm.Source } else { $null })
            npmVersion = $null
            npmCli = $null
            npmSource = $null
            required = "Node.js 20.19 or newer with npm 10 or newer"
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
        $package = Get-QTracXPackage -Name "python"
        $bundle = Get-QTracXBundledInstaller -Name "python"
        $actions += [pscustomobject]@{
            prerequisite = [string]$package.displayName
            required = $true
            preferredSource = $(if ($bundle.usable) { "bundled" } else { "winget" })
            bundledStatus = $bundle.status
            command = Get-QTracXWingetCommand -Package $package
            officialDownloadUrl = [string]$package.officialDownloadUrl
        }
    }
    if (-not $State.node.supported) {
        $package = Get-QTracXPackage -Name "node"
        $bundle = Get-QTracXBundledInstaller -Name "node"
        $actions += [pscustomobject]@{
            prerequisite = [string]$package.displayName
            required = $true
            preferredSource = $(if ($bundle.usable) { "bundled" } else { "winget" })
            bundledStatus = $bundle.status
            command = Get-QTracXWingetCommand -Package $package
            officialDownloadUrl = [string]$package.officialDownloadUrl
        }
    }
    if ($IncludeGit -and -not $State.git.available) {
        $package = Get-QTracXPackage -Name "git"
        $bundle = Get-QTracXBundledInstaller -Name "git"
        $actions += [pscustomobject]@{
            prerequisite = [string]$package.displayName
            required = $true
            preferredSource = $(if ($bundle.usable) { "bundled" } else { "winget" })
            bundledStatus = $bundle.status
            command = Get-QTracXWingetCommand -Package $package
            officialDownloadUrl = [string]$package.officialDownloadUrl
        }
    }
    if ($Database -eq "mssql" -and -not $State.mssqlOdbc.available) {
        $package = Get-QTracXPackage -Name "mssqlOdbc"
        $bundle = Get-QTracXBundledInstaller -Name "mssqlOdbc"
        $actions += [pscustomobject]@{
            prerequisite = [string]$package.displayName
            required = $true
            preferredSource = $(if ($bundle.usable) { "bundled" } else { "winget" })
            bundledStatus = $bundle.status
            command = Get-QTracXWingetCommand -Package $package
            officialDownloadUrl = [string]$package.officialDownloadUrl
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
    param([Parameter(Mandatory = $true)]$Package)

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw (
            "WinGet is unavailable. Install {0} from {1} and rerun Verify mode." -f
            $Package.displayName, $Package.officialDownloadUrl
        )
    }
    Write-Host "Installing $($Package.displayName) with WinGet ..."
    & $winget.Source install `
        --exact `
        --id $Package.wingetId `
        --version $Package.wingetVersion `
        --source winget `
        --scope machine `
        --architecture x64 `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw (
            "WinGet failed to install {0} (exit code {1}). Direct official download: {2}" -f
            $Package.displayName, $LASTEXITCODE, $Package.officialDownloadUrl
        )
    }
}

function Invoke-QTracXBundledInstall {
    param(
        [Parameter(Mandatory = $true)]$Package,
        [Parameter(Mandatory = $true)]$Bundle
    )

    Write-Host "Installing bundled $($Package.displayName) ..."
    $arguments = @($Package.silentArguments | ForEach-Object { [string]$_ })
    if ($Package.installerType -eq "msi") {
        & msiexec.exe /i $Bundle.path @arguments
    }
    elseif ($Package.installerType -eq "exe") {
        & $Bundle.path @arguments
    }
    else {
        throw "Unsupported bundled installer type: $($Package.installerType)"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Bundled installation of $($Package.displayName) failed (exit code $LASTEXITCODE)."
    }
}

function Install-QTracXPrerequisite {
    param([Parameter(Mandatory = $true)][string]$Name)

    $package = Get-QTracXPackage -Name $Name
    $bundle = Get-QTracXBundledInstaller -Name $Name
    if ($bundle.usable) {
        Invoke-QTracXBundledInstall -Package $package -Bundle $bundle
        return
    }
    Invoke-QTracXWingetInstall -Package $package
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
    prerequisiteCatalog = Get-QTracXPrerequisiteSummary
    manualDownloadPages = [ordered]@{
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
        Install-QTracXPrerequisite -Name "python"
    }
    if (-not $state.node.supported) {
        Install-QTracXPrerequisite -Name "node"
    }
    if ($IncludeGit -and -not $state.git.available) {
        Install-QTracXPrerequisite -Name "git"
    }
    if ($Database -eq "mssql" -and -not $state.mssqlOdbc.available) {
        Install-QTracXPrerequisite -Name "mssqlOdbc"
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
