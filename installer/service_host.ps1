[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet("backend", "frontend")]
    [string]$Component,
    [Parameter(Mandatory = $true)][string]$ApplicationRoot,
    [ValidateSet("cpu", "gpu")][string]$Runtime = "cpu",
    [string]$NodeExecutable = "node.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [IO.Path]::GetFullPath($ApplicationRoot)
$logDirectory = Join-Path $root "backend\logs\service"
[IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$logPath = Join-Path $logDirectory "$Component.log"

function Rotate-QTracXServiceLog {
    param([string]$Path, [long]$MaximumBytes = 26214400, [int]$Keep = 5)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    if ((Get-Item -LiteralPath $Path).Length -lt $MaximumBytes) { return }
    for ($index = $Keep - 1; $index -ge 1; $index--) {
        $source = "$Path.$index"
        $destination = "$Path.$($index + 1)"
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Move-Item -LiteralPath $source -Destination $destination -Force
        }
    }
    Move-Item -LiteralPath $Path -Destination "$Path.1" -Force
}

Rotate-QTracXServiceLog -Path $logPath
Set-Location -LiteralPath $root

if ($Component -eq "backend") {
    $environmentName = if ($Runtime -eq "gpu") { ".venv-gpu" } else { ".venv" }
    $executable = Join-Path $root "backend\$environmentName\Scripts\python.exe"
    $arguments = @(Join-Path $root "backend\main.py")
}
else {
    $executable = $NodeExecutable
    $arguments = @(
        (Join-Path $root "scripts\run-next.mjs"),
        "start"
    )
}

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "QTracX $Component executable was not found: $executable"
}

"[$([DateTime]::UtcNow.ToString('o'))] Starting QTracX $Component." |
    Out-File -LiteralPath $logPath -Append -Encoding utf8
& $executable @arguments 2>&1 |
    Out-File -LiteralPath $logPath -Append -Encoding utf8
$exitCode = $LASTEXITCODE
"[$([DateTime]::UtcNow.ToString('o'))] QTracX $Component exited with code $exitCode." |
    Out-File -LiteralPath $logPath -Append -Encoding utf8
exit $exitCode
