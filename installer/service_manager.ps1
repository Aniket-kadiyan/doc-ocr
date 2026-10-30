[CmdletBinding()]
param(
    [ValidateSet(
        "Install", "Verify", "Start", "Stop", "Status", "Repair",
        "Unregister", "Uninstall"
    )]
    [string]$Mode = "Status",
    [Parameter(Mandatory = $true)][string]$ApplicationRoot,
    [ValidateSet("cpu", "gpu")][string]$Runtime = "cpu",
    [ValidateRange(15, 1800)][int]$HealthTimeoutSeconds = 600
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:BackendTask = "QTracX Backend"
$script:FrontendTask = "QTracX Frontend"
$script:FirewallGroup = "QTracX"
$script:Root = [IO.Path]::GetFullPath($ApplicationRoot)

function Test-QTracXAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function ConvertTo-QTracXArgument {
    param([string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Get-QTracXEnvironment {
    $path = Join-Path $script:Root ".env.local"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "QTracX configuration was not found: $path"
    }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $name, $value = $line -split '=', 2
        $values[$name.Trim()] = $value.Trim()
    }
    return $values
}

function Assert-QTracXLayout {
    $environmentName = if ($Runtime -eq "gpu") { ".venv-gpu" } else { ".venv" }
    $required = @(
        (Join-Path $script:Root ".env.local"),
        (Join-Path $script:Root ".next\BUILD_ID"),
        (Join-Path $script:Root "backend\$environmentName\Scripts\python.exe"),
        (Join-Path $script:Root "backend\main.py"),
        (Join-Path $script:Root "scripts\run-next.mjs"),
        (Join-Path $script:Root "installer\service_host.ps1"),
        (Join-Path $script:Root "tools\dwg2dxf\dist\win-x64\dwg2dxf-acadsharp.exe")
    )
    $missing = @($required | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    })
    if ($missing.Count -gt 0) {
        throw "The QTracX installation is incomplete: $($missing -join ', ')"
    }
}

function Protect-QTracXApplicationRoot {
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sidText in @("S-1-5-18", "S-1-5-32-544")) {
        $sid = [Security.Principal.SecurityIdentifier]::new($sidText)
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            $sid,
            [Security.AccessControl.FileSystemRights]::FullControl,
            (
                [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
                [Security.AccessControl.InheritanceFlags]::ObjectInherit
            ),
            [Security.AccessControl.PropagationFlags]::None,
            [Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $script:Root -AclObject $acl

    # Replace child ACLs with inheritance from the now-protected root. The
    # wildcard deliberately excludes the root itself, whose ACL was just set.
    $icacls = Join-Path $env:SystemRoot "System32\icacls.exe"
    & $icacls (Join-Path $script:Root "*") /reset /T /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not protect the QTracX installation directory ACL."
    }
}

function Get-QTracXNode {
    $command = Get-Command node.exe -ErrorAction Stop
    return $command.Source
}

function Register-QTracXStartupTasks {
    Assert-QTracXLayout
    Protect-QTracXApplicationRoot
    $powershell = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    $hostScript = Join-Path $script:Root "installer\service_host.ps1"
    $node = Get-QTracXNode
    $common = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", (ConvertTo-QTracXArgument $hostScript),
        "-ApplicationRoot", (ConvertTo-QTracXArgument $script:Root),
        "-Runtime", $Runtime
    )
    $backendAction = New-ScheduledTaskAction `
        -Execute $powershell `
        -Argument (($common + @("-Component", "backend")) -join " ") `
        -WorkingDirectory $script:Root
    $frontendAction = New-ScheduledTaskAction `
        -Execute $powershell `
        -Argument (($common + @(
            "-Component", "frontend",
            "-NodeExecutable", (ConvertTo-QTracXArgument $node)
        )) -join " ") `
        -WorkingDirectory $script:Root
    $principal = New-ScheduledTaskPrincipal `
        -UserId "SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -RestartCount 10 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew

    Register-ScheduledTask `
        -TaskName $script:BackendTask `
        -Action $backendAction `
        -Principal $principal `
        -Trigger $trigger `
        -Settings $settings `
        -Description "QTracX OCR/API backend with restart recovery." `
        -Force | Out-Null
    Register-ScheduledTask `
        -TaskName $script:FrontendTask `
        -Action $frontendAction `
        -Principal $principal `
        -Trigger $trigger `
        -Settings $settings `
        -Description "QTracX production web UI with restart recovery." `
        -Force | Out-Null
}

function Set-QTracXFirewallRules {
    $environment = Get-QTracXEnvironment
    Get-NetFirewallRule -Group $script:FirewallGroup -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule -ErrorAction SilentlyContinue
    if ($environment["UI_HOST"] -ne "0.0.0.0") { return }
    foreach ($definition in @(
        @{ Name = "QTracX UI"; Port = [int]$environment["PORT"] },
        @{ Name = "QTracX Backend"; Port = [int]$environment["OCR_PORT"] }
    )) {
        New-NetFirewallRule `
            -DisplayName $definition.Name `
            -Group $script:FirewallGroup `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort $definition.Port `
            -Profile Domain,Private | Out-Null
    }
}

function Stop-QTracXTasks {
    foreach ($name in @($script:FrontendTask, $script:BackendTask)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($task -and $task.State -ne "Disabled") {
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        }
    }
}

function Start-QTracXTasks {
    foreach ($name in @($script:BackendTask, $script:FrontendTask)) {
        if (-not (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)) {
            throw "Scheduled task is not registered: $name"
        }
        Start-ScheduledTask -TaskName $name
    }
}

function Unregister-QTracXTasks {
    Stop-QTracXTasks
    foreach ($name in @($script:BackendTask, $script:FrontendTask)) {
        Unregister-ScheduledTask `
            -TaskName $name `
            -Confirm:$false `
            -ErrorAction SilentlyContinue
    }
    Get-NetFirewallRule -Group $script:FirewallGroup -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule -ErrorAction SilentlyContinue
}

function Test-QTracXUrl {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $request = [Net.HttpWebRequest]::Create($Url)
        $request.Method = "GET"
        $request.Timeout = 5000
        $request.AllowAutoRedirect = $true
        $response = $request.GetResponse()
        try {
            return [int]$response.StatusCode -ge 200 -and
                [int]$response.StatusCode -lt 400
        }
        finally {
            $response.Close()
        }
    }
    catch {
        return $false
    }
}

function Wait-QTracXHealth {
    $environment = Get-QTracXEnvironment
    $backendBase = [string]$environment["NEXT_PUBLIC_OCR_API_URL"]
    if (-not $backendBase) { throw "NEXT_PUBLIC_OCR_API_URL is not configured." }
    $backendUri = [Uri]$backendBase
    $uiPort = [int]$environment["PORT"]
    $uiUrl = "{0}://{1}:{2}/" -f `
        $backendUri.Scheme, $backendUri.Host, $uiPort
    $backendUrl = $backendBase.TrimEnd('/') + "/health"
    $deadline = [DateTime]::UtcNow.AddSeconds($HealthTimeoutSeconds)
    do {
        $backendHealthy = Test-QTracXUrl -Url $backendUrl
        $frontendHealthy = Test-QTracXUrl -Url $uiUrl
        if ($backendHealthy -and $frontendHealthy) {
            return [pscustomobject]@{
                backend = $backendUrl
                frontend = $uiUrl
                healthy = $true
            }
        }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "QTracX health checks did not pass within $HealthTimeoutSeconds seconds."
}

function Assert-QTracXTasksRegistered {
    foreach ($name in @($script:BackendTask, $script:FrontendTask)) {
        if (-not (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)) {
            throw "Scheduled task is not registered: $name"
        }
    }
}

function Get-QTracXTaskStatus {
    $rows = @()
    foreach ($name in @($script:BackendTask, $script:FrontendTask)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        $info = Get-ScheduledTaskInfo -TaskName $name -ErrorAction SilentlyContinue
        $rows += [pscustomobject]@{
            name = $name
            registered = [bool]$task
            state = $(if ($task) { [string]$task.State } else { "Missing" })
            last_result = $(if ($info) { $info.LastTaskResult } else { $null })
        }
    }
    return $rows
}

try {
    if ($Mode -notin @("Status", "Verify") -and -not (Test-QTracXAdministrator)) {
        throw "QTracX startup-task changes require an elevated PowerShell session."
    }
    $health = $null
    switch ($Mode) {
        "Install" {
            Register-QTracXStartupTasks
            Set-QTracXFirewallRules
            Start-QTracXTasks
            $health = Wait-QTracXHealth
        }
        "Repair" {
            Stop-QTracXTasks
            Register-QTracXStartupTasks
            Set-QTracXFirewallRules
            Start-QTracXTasks
            $health = Wait-QTracXHealth
        }
        "Start" {
            Start-QTracXTasks
            $health = Wait-QTracXHealth
        }
        "Stop" { Stop-QTracXTasks }
        "Verify" {
            Assert-QTracXLayout
            Assert-QTracXTasksRegistered
            $health = Wait-QTracXHealth
        }
        "Unregister" { Unregister-QTracXTasks }
        "Uninstall" { Unregister-QTracXTasks }
        "Status" { }
    }
    [pscustomobject]@{
        status = "ok"
        mode = $Mode
        application_root = $script:Root
        tasks = @(Get-QTracXTaskStatus)
        health = $health
        application_data_removed = $false
    } | ConvertTo-Json -Depth 6
    exit 0
}
catch {
    [pscustomobject]@{
        status = "failed"
        mode = $Mode
        message = $_.Exception.Message
        application_data_removed = $false
    } | ConvertTo-Json -Depth 4
    exit 1
}
