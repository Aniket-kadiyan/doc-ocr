[CmdletBinding()]
param(
    [ValidateSet("Plan", "Download", "Verify")]
    [string]$Mode = "Plan",

    [string]$Destination = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$catalogPath = Join-Path $PSScriptRoot "config\prerequisites.windows-x64.json"
if (-not (Test-Path -LiteralPath $catalogPath -PathType Leaf)) {
    throw "Prerequisite catalog was not found: $catalogPath"
}
$catalog = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json
if (-not $Destination) {
    $Destination = Join-Path $PSScriptRoot "prerequisites"
}
$Destination = [IO.Path]::GetFullPath($Destination)
$lockPath = Join-Path $Destination "prerequisites.lock.json"

$results = @()
foreach ($property in $catalog.packages.PSObject.Properties) {
    $key = $property.Name
    $package = $property.Value
    $path = Join-Path $Destination ([string]$package.fileName)

    if ($Mode -eq "Download") {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        Write-Host "Downloading $($package.displayName) ..."
        Invoke-WebRequest `
            -Uri ([string]$package.officialDownloadUrl) `
            -OutFile $path `
            -UseBasicParsing
    }

    $status = "missing"
    $actualHash = $null
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        $publishedHash = ([string]$package.sha256).ToLowerInvariant()
        if ($publishedHash -and $actualHash -ne $publishedHash) {
            throw "SHA-256 validation failed for $($package.fileName)."
        }
        $publisherProperty = $package.PSObject.Properties["authenticodePublisher"]
        if ($publisherProperty -and [string]$publisherProperty.Value) {
            $signature = Get-AuthenticodeSignature -LiteralPath $path
            $publisher = [string]$publisherProperty.Value
            if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
                -not $signature.SignerCertificate -or
                $signature.SignerCertificate.Subject -notlike "*$publisher*") {
                throw "Authenticode publisher validation failed for $($package.fileName)."
            }
        }
        $status = "ready"
    }
    elseif ($Mode -eq "Verify") {
        throw "Bundled prerequisite is missing: $path"
    }

    $results += [ordered]@{
        key = $key
        displayName = [string]$package.displayName
        version = [string]$package.version
        fileName = [string]$package.fileName
        officialDownloadUrl = [string]$package.officialDownloadUrl
        status = $status
        sha256 = $actualHash
    }
}

if ($Mode -eq "Download" -or $Mode -eq "Verify") {
    $lock = [ordered]@{
        schemaVersion = 1
        platform = [string]$catalog.platform
        generatedAtUtc = [DateTime]::UtcNow.ToString("o")
        files = @($results | Where-Object { $_.status -eq "ready" })
    }
    $lock | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath $lockPath `
        -Encoding UTF8
}

[ordered]@{
    status = $(if (@($results | Where-Object { $_.status -ne "ready" }).Count -eq 0) { "ready" } else { "action-required" })
    mode = $Mode.ToLowerInvariant()
    destination = $Destination
    lockFile = $lockPath
    packages = $results
} | ConvertTo-Json -Depth 8
