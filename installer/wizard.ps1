[CmdletBinding()]
param(
    [switch]$Elevated
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-QTracXAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

if (-not (Test-QTracXAdministrator)) {
    if ($Elevated) {
        throw "The QTracX installer requires an elevated PowerShell session."
    }
    $arguments = (
        '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Elevated' -f 
        $PSCommandPath.Replace('"', '""')
    )
    try {
        Start-Process -FilePath "powershell.exe" -Verb RunAs `
            -ArgumentList $arguments | Out-Null
    }
    catch {
        Write-Error "Administrator approval was cancelled."
    }
    exit
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:InstallerRoot = $PSScriptRoot
$script:PackageRoot = Split-Path -Parent $PSScriptRoot
$script:BootstrapScript = Join-Path $PSScriptRoot "bootstrap.ps1"
$script:SourceScript = Join-Path $PSScriptRoot "source_acquisition.py"
$script:ConfigurationScript = Join-Path $PSScriptRoot "setup_config.py"
$script:DatabaseScript = Join-Path $PSScriptRoot "database_bootstrap.py"
$script:CurrentPage = 0
$script:Busy = $false
$script:InstallerLogPath = $null

function ConvertTo-ProcessArguments {
    param([string[]]$Values)

    return (($Values | ForEach-Object {
        $value = [string]$_
        if ($value -notmatch '[\s"]') {
            $value
        }
        else {
            '"' + $value.Replace('"', '\"') + '"'
        }
    }) -join ' ')
}

function Invoke-QTracXProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [AllowNull()][string]$InputText = $null,
        [AllowNull()][scriptblock]$OnOutput = $null
    )

    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $FilePath
    $start.Arguments = ConvertTo-ProcessArguments -Values $Arguments
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.RedirectStandardInput = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) {
        throw "Could not start $FilePath"
    }
    $stdout = [Text.StringBuilder]::new()
    $stderr = [Text.StringBuilder]::new()
    $stdoutTask = $process.StandardOutput.ReadLineAsync()
    $stderrTask = $process.StandardError.ReadLineAsync()
    if ($null -ne $InputText) {
        $process.StandardInput.Write($InputText)
    }
    $process.StandardInput.Close()
    while (-not $process.HasExited -or
        $null -ne $stdoutTask -or
        $null -ne $stderrTask) {
        while ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) {
            $line = $stdoutTask.GetAwaiter().GetResult()
            if ($null -eq $line) {
                $stdoutTask = $null
            }
            else {
                [void]$stdout.AppendLine($line)
                if ($null -ne $OnOutput) {
                    & $OnOutput $line $false
                }
                $stdoutTask = $process.StandardOutput.ReadLineAsync()
            }
        }
        while ($null -ne $stderrTask -and $stderrTask.IsCompleted) {
            $line = $stderrTask.GetAwaiter().GetResult()
            if ($null -eq $line) {
                $stderrTask = $null
            }
            else {
                [void]$stderr.AppendLine($line)
                if ($null -ne $OnOutput) {
                    & $OnOutput $line $true
                }
                $stderrTask = $process.StandardError.ReadLineAsync()
            }
        }
        [System.Windows.Forms.Application]::DoEvents()
        [Threading.Thread]::Sleep(40)
    }
    $process.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        StandardOutput = $stdout.ToString()
        StandardError = $stderr.ToString()
    }
}

function Get-QTracXPythonInvocation {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        return [pscustomobject]@{
            FilePath = $launcher.Source
            PrefixArguments = @("-3.13")
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        return [pscustomobject]@{
            FilePath = $python.Source
            PrefixArguments = @()
        }
    }
    throw "Python 3.13 was not found after prerequisite setup."
}

function Get-QTracXMssqlDriver {
    try {
        $drivers = @(
            Get-OdbcDriver -ErrorAction Stop |
                Where-Object {
                    $_.Platform -eq "64-bit" -and
                    $_.Name -match '^ODBC Driver (17|18) for SQL Server$'
                } |
                Select-Object -ExpandProperty Name -Unique
        )
        foreach ($preferred in @(
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server"
        )) {
            if ($preferred -in $drivers) {
                return $preferred
            }
        }
    }
    catch {
        # Host bootstrap installs Driver 18 when no supported driver is found.
    }
    return "ODBC Driver 18 for SQL Server"
}

function Test-QTracXGpu {
    $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $command) { return $false }
    try {
        $devices = @(
            & $command.Source --query-gpu=index --format=csv,noheader 2>$null |
                Where-Object { $_.Trim() }
        )
        return $LASTEXITCODE -eq 0 -and $devices.Count -gt 0
    }
    catch {
        return $false
    }
}

function Add-QTracXLabel {
    param($Parent, [string]$Text, [int]$X, [int]$Y, [int]$Width = 220)
    $label = [System.Windows.Forms.Label]::new()
    $label.Text = $Text
    $label.Location = [Drawing.Point]::new($X, $Y)
    $label.Size = [Drawing.Size]::new($Width, 24)
    $Parent.Controls.Add($label)
    return $label
}

function Add-QTracXTextBox {
    param(
        $Parent,
        [int]$X,
        [int]$Y,
        [int]$Width = 430,
        [string]$Text = "",
        [switch]$Password,
        [switch]$Multiline
    )
    $box = [System.Windows.Forms.TextBox]::new()
    $box.Location = [Drawing.Point]::new($X, $Y)
    $box.Size = [Drawing.Size]::new($Width, $(if ($Multiline) { 220 } else { 27 }))
    $box.BackColor = [Drawing.SystemColors]::Window
    $box.ForeColor = [Drawing.SystemColors]::WindowText
    $box.Text = $Text
    $box.UseSystemPasswordChar = [bool]$Password
    $box.Multiline = [bool]$Multiline
    if ($Multiline) {
        $box.ScrollBars = "Vertical"
    }
    $Parent.Controls.Add($box)
    return $box
}

function Add-QTracXComboBox {
    param($Parent, [int]$X, [int]$Y, [string[]]$Items, [int]$Width = 260)
    $combo = [System.Windows.Forms.ComboBox]::new()
    $combo.Location = [Drawing.Point]::new($X, $Y)
    $combo.Size = [Drawing.Size]::new($Width, 28)
    $combo.DropDownStyle = "DropDownList"
    $combo.BackColor = [Drawing.SystemColors]::Window
    $combo.ForeColor = [Drawing.SystemColors]::WindowText
    [void]$combo.Items.AddRange($Items)
    $combo.SelectedIndex = 0
    $Parent.Controls.Add($combo)
    return $combo
}

function Add-QTracXNumeric {
    param(
        $Parent,
        [int]$X,
        [int]$Y,
        [int]$Value,
        [int]$Minimum,
        [int]$Maximum,
        [int]$Width = 150
    )
    $number = [System.Windows.Forms.NumericUpDown]::new()
    $number.Location = [Drawing.Point]::new($X, $Y)
    $number.Size = [Drawing.Size]::new($Width, 32)
    $number.MinimumSize = [Drawing.Size]::new($Width, 32)
    $number.BackColor = [Drawing.SystemColors]::Window
    $number.ForeColor = [Drawing.SystemColors]::WindowText
    $number.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    $number.TextAlign = [System.Windows.Forms.HorizontalAlignment]::Left
    $number.UpDownAlign = [System.Windows.Forms.LeftRightAlignment]::Right
    $number.ThousandsSeparator = $false
    $number.Minimum = $Minimum
    $number.Maximum = $Maximum
    $number.Value = [decimal]$Value
    $Parent.Controls.Add($number)
    $number.Refresh()
    return $number
}

function Add-QTracXButton {
    param($Parent, [string]$Text, [int]$X, [int]$Y, [int]$Width = 110)
    $button = [System.Windows.Forms.Button]::new()
    $button.Text = $Text
    $button.Location = [Drawing.Point]::new($X, $Y)
    $button.Size = [Drawing.Size]::new($Width, 34)
    $Parent.Controls.Add($button)
    return $button
}

$form = [System.Windows.Forms.Form]::new()
$form.Text = "QTracX Deployment Installer"
$form.StartPosition = "CenterScreen"
$form.ClientSize = [Drawing.Size]::new(900, 700)
$form.MinimumSize = [Drawing.Size]::new(916, 739)
$form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::Dpi
$form.AutoScaleDimensions = [Drawing.SizeF]::new(96.0, 96.0)
$form.AutoScroll = $true
$form.Font = [Drawing.Font]::new("Segoe UI", 10)

$title = Add-QTracXLabel $form "QTracX Deployment Installer" 24 18 500
$title.Font = [Drawing.Font]::new("Segoe UI Semibold", 18)
$subtitle = Add-QTracXLabel $form `
    "Configure and install the central QTracX LAN application server." `
    26 55 700
$subtitle.ForeColor = [Drawing.Color]::DimGray

$tabs = [System.Windows.Forms.TabControl]::new()
$tabs.Location = [Drawing.Point]::new(20, 92)
$tabs.Size = [Drawing.Size]::new(860, 540)
$tabs.Anchor = (
    [System.Windows.Forms.AnchorStyles]::Top -bor
    [System.Windows.Forms.AnchorStyles]::Bottom -bor
    [System.Windows.Forms.AnchorStyles]::Left -bor
    [System.Windows.Forms.AnchorStyles]::Right
)
$form.Controls.Add($tabs)

$sourceTab = [System.Windows.Forms.TabPage]::new("1. Source")
$serverTab = [System.Windows.Forms.TabPage]::new("2. Server")
$databaseTab = [System.Windows.Forms.TabPage]::new("3. Database")
$adminTab = [System.Windows.Forms.TabPage]::new("4. Administrator")
$reviewTab = [System.Windows.Forms.TabPage]::new("5. Review and install")
[void]$tabs.TabPages.AddRange(@(
    $sourceTab,
    $serverTab,
    $databaseTab,
    $adminTab,
    $reviewTab
))
foreach ($page in $tabs.TabPages) {
    $page.AutoScroll = $true
    $page.BackColor = [Drawing.SystemColors]::Control
}

Add-QTracXLabel $sourceTab "Installation directory" 28 32 | Out-Null
$installDirectory = Add-QTracXTextBox $sourceTab 250 28 480 "C:\QTracX"
$browseInstall = Add-QTracXButton $sourceTab "Browse..." 742 26 90
Add-QTracXLabel $sourceTab "Application source" 28 88 | Out-Null
$sourceMode = Add-QTracXComboBox $sourceTab 250 84 @(
    "Offline source ZIP",
    "Predefined Git branch"
) 300
Add-QTracXLabel $sourceTab "Source ZIP" 28 144 | Out-Null
$sourceArchive = Add-QTracXTextBox $sourceTab 250 140 480
$browseArchive = Add-QTracXButton $sourceTab "Browse..." 742 138 90
$sourceNote = Add-QTracXLabel $sourceTab `
    "The destination must be empty. Source ZIPs must not include node_modules, virtual environments, .next, or .git." `
    28 205 790
$sourceNote.AutoSize = $false
$sourceNote.Height = 55
$sourceNote.ForeColor = [Drawing.Color]::DimGray

Add-QTracXLabel $serverTab "Deployment mode" 28 28 | Out-Null
$deploymentMode = Add-QTracXComboBox $serverTab 250 24 @(
    "Central LAN server",
    "Localhost only"
) 260
Add-QTracXLabel $serverTab "Server hostname or IP" 28 76 | Out-Null
$serverAddress = Add-QTracXTextBox $serverTab 250 72 310 "localhost"
Add-QTracXLabel $serverTab "UI port" 28 124 | Out-Null
$uiPort = Add-QTracXNumeric $serverTab 250 120 3002 1 65535
Add-QTracXLabel $serverTab "Backend port" 430 124 | Out-Null
$backendPort = Add-QTracXNumeric $serverTab 590 120 8000 1 65535
Add-QTracXLabel $serverTab "OCR runtime" 28 172 | Out-Null
$runtimeItems = [System.Collections.Generic.List[string]]::new()
$runtimeItems.Add("CPU")
$gpuAvailable = Test-QTracXGpu
if ($gpuAvailable) { $runtimeItems.Add("GPU 0") }
$runtime = Add-QTracXComboBox $serverTab 250 168 $runtimeItems.ToArray() 180
if ($gpuAvailable) { $runtime.SelectedIndex = 1 }
Add-QTracXLabel $serverTab "Model instances" 28 220 | Out-Null
$modelInstances = Add-QTracXNumeric $serverTab 250 216 1 1 8
Add-QTracXLabel $serverTab "Scan workers" 430 220 | Out-Null
$scanWorkers = Add-QTracXNumeric $serverTab 590 216 2 1 32
$warmModels = [System.Windows.Forms.CheckBox]::new()
$warmModels.Text = "Download and warm OCR models during installation"
$warmModels.Location = [Drawing.Point]::new(28, 264)
$warmModels.Size = [Drawing.Size]::new(790, 28)
$warmModels.Checked = $true
$serverTab.Controls.Add($warmModels)
$httpsEnabled = [System.Windows.Forms.CheckBox]::new()
$httpsEnabled.Text = "Enable HTTPS when certificate and private key are available"
$httpsEnabled.Location = [Drawing.Point]::new(28, 304)
$httpsEnabled.Size = [Drawing.Size]::new(790, 28)
$serverTab.Controls.Add($httpsEnabled)
Add-QTracXLabel $serverTab "Certificate file" 28 352 | Out-Null
$certificateFile = Add-QTracXTextBox $serverTab 250 348 480
$browseCertificate = Add-QTracXButton $serverTab "Browse..." 742 346 90
Add-QTracXLabel $serverTab "Private-key file" 28 400 | Out-Null
$privateKeyFile = Add-QTracXTextBox $serverTab 250 396 480
$browsePrivateKey = Add-QTracXButton $serverTab "Browse..." 742 394 90

Add-QTracXLabel $databaseTab "Database engine" 28 28 | Out-Null
$databaseEngine = Add-QTracXComboBox $databaseTab 250 24 @(
    "Microsoft SQL Server",
    "PostgreSQL"
) 260
Add-QTracXLabel $databaseTab "Provisioning mode" 28 76 | Out-Null
$databaseMode = Add-QTracXComboBox $databaseTab 250 72 @(
    "Installer creates database/user",
    "Preconfigured database/user"
) 320
Add-QTracXLabel $databaseTab "Database server / instance" 28 124 | Out-Null
$databaseHost = Add-QTracXTextBox $databaseTab 250 120 300 "localhost"
$explicitDatabasePort = [System.Windows.Forms.CheckBox]::new()
$explicitDatabasePort.Text = "Use explicit TCP port"
$explicitDatabasePort.Location = [Drawing.Point]::new(570, 116)
$explicitDatabasePort.Size = [Drawing.Size]::new(250, 28)
$explicitDatabasePort.Checked = $true
$databaseTab.Controls.Add($explicitDatabasePort)
Add-QTracXLabel $databaseTab "Port" 570 154 80 | Out-Null
$databasePort = Add-QTracXNumeric $databaseTab 650 150 1433 1 65535 120
Add-QTracXLabel $databaseTab "Database name" 28 172 | Out-Null
$databaseName = Add-QTracXTextBox $databaseTab 250 168 300 "QTracX"
Add-QTracXLabel $databaseTab "Application DB user" 28 220 | Out-Null
$databaseApplicationUser = Add-QTracXTextBox $databaseTab 250 216 300 "qtracx_app"
Add-QTracXLabel $databaseTab "Application DB password" 28 268 | Out-Null
$databaseApplicationPassword = Add-QTracXTextBox $databaseTab 250 264 300 "" -Password
Add-QTracXLabel $databaseTab "Confirm DB password" 28 316 | Out-Null
$databaseApplicationPasswordConfirm = Add-QTracXTextBox $databaseTab 250 312 300 "" -Password
Add-QTracXLabel $databaseTab "Database administrator" 28 364 | Out-Null
$databaseAdministratorUser = Add-QTracXTextBox $databaseTab 250 360 300 "sa"
Add-QTracXLabel $databaseTab "Administrator password" 28 412 | Out-Null
$databaseAdministratorPassword = Add-QTracXTextBox $databaseTab 250 408 300 "" -Password
Add-QTracXLabel $databaseTab "MSSQL ODBC driver" 570 205 200 | Out-Null
$odbcDriver = Add-QTracXTextBox $databaseTab 570 233 240 (Get-QTracXMssqlDriver)
$resetDatabasePassword = [System.Windows.Forms.CheckBox]::new()
$resetDatabasePassword.Text = "Reset existing QTracX DB-user password"
$resetDatabasePassword.Location = [Drawing.Point]::new(570, 280)
$resetDatabasePassword.Size = [Drawing.Size]::new(270, 48)
$databaseTab.Controls.Add($resetDatabasePassword)
$databaseConnectionNote = Add-QTracXLabel $databaseTab `
    "Named MSSQL instance example: jains\SQLEXPRESS. Leave explicit TCP port unchecked to use SQL Server Browser, or check it when IT provides a fixed port." `
    570 340 250
$databaseConnectionNote.AutoSize = $false
$databaseConnectionNote.Height = 100
$databaseConnectionNote.ForeColor = [Drawing.Color]::DimGray

Add-QTracXLabel $adminTab "Initial QTracX administrator" 28 35 310 | Out-Null
Add-QTracXLabel $adminTab "Username" 28 95 | Out-Null
$applicationAdminUser = Add-QTracXTextBox $adminTab 250 91 300 "admin"
Add-QTracXLabel $adminTab "Display name" 28 143 | Out-Null
$applicationAdminDisplayName = Add-QTracXTextBox `
    $adminTab 250 139 300 "QTracX Administrator"
Add-QTracXLabel $adminTab "Password" 28 191 | Out-Null
$applicationAdminPassword = Add-QTracXTextBox $adminTab 250 187 300 "" -Password
Add-QTracXLabel $adminTab "Confirm password" 28 239 | Out-Null
$applicationAdminPasswordConfirm = Add-QTracXTextBox `
    $adminTab 250 235 300 "" -Password
$adminNote = Add-QTracXLabel $adminTab `
    "This creates the first in-application administrator. It is separate from the database administrator account. Passwords must contain at least four characters." `
    28 305 760
$adminNote.Height = 70
$adminNote.AutoSize = $false
$adminNote.ForeColor = [Drawing.Color]::DimGray

$reviewSummary = Add-QTracXTextBox $reviewTab 22 22 800 "" -Multiline
$reviewSummary.ReadOnly = $true
$reviewSummary.Height = 170
$checkPlan = Add-QTracXButton $reviewTab "Check plan" 22 208 130
$startInstall = Add-QTracXButton $reviewTab "Install QTracX" 165 208 150
$progressLabel = Add-QTracXLabel $reviewTab "Ready." 335 204 490
$progressLabel.ForeColor = [Drawing.Color]::DarkSlateBlue
$installProgress = [System.Windows.Forms.ProgressBar]::new()
$installProgress.Location = [Drawing.Point]::new(335, 232)
$installProgress.Size = [Drawing.Size]::new(487, 18)
$installProgress.Minimum = 0
$installProgress.Maximum = 100
$installProgress.Value = 0
$installProgress.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous
$reviewTab.Controls.Add($installProgress)
$installLog = Add-QTracXTextBox $reviewTab 22 270 800 "" -Multiline
$installLog.ReadOnly = $true
$installLog.Height = 208

$backButton = Add-QTracXButton $form "Back" 604 648 90
$nextButton = Add-QTracXButton $form "Next" 704 648 90
$cancelButton = Add-QTracXButton $form "Close" 804 648 76
$backButton.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right
$nextButton.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right
$cancelButton.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

function Set-QTracXSourceState {
    $archive = $sourceMode.SelectedIndex -eq 0
    $sourceArchive.Enabled = $archive
    $browseArchive.Enabled = $archive
}

function Set-QTracXHttpsState {
    $enabled = $httpsEnabled.Checked
    $certificateFile.Enabled = $enabled
    $privateKeyFile.Enabled = $enabled
    $browseCertificate.Enabled = $enabled
    $browsePrivateKey.Enabled = $enabled
}

function Set-QTracXDatabaseState {
    $mssql = $databaseEngine.SelectedIndex -eq 0
    $installerManaged = $databaseMode.SelectedIndex -eq 0
    $odbcDriver.Enabled = $mssql
    $databaseAdministratorUser.Enabled = $installerManaged
    $databaseAdministratorPassword.Enabled = $installerManaged
    $resetDatabasePassword.Enabled = $installerManaged
    $explicitDatabasePort.Enabled = $mssql
    if ($mssql) {
        if ([int]$databasePort.Value -eq 5432) { $databasePort.Value = 1433 }
        if ($databaseHost.Text.Contains("\")) {
            $explicitDatabasePort.Checked = $false
        }
        if ($databaseAdministratorUser.Text -eq "postgres") {
            $databaseAdministratorUser.Text = "sa"
        }
    }
    else {
        $explicitDatabasePort.Checked = $true
        if ([int]$databasePort.Value -eq 1433) { $databasePort.Value = 5432 }
        if ($databaseAdministratorUser.Text -eq "sa") {
            $databaseAdministratorUser.Text = "postgres"
        }
    }
    Set-QTracXDatabasePortState
}

function Set-QTracXDatabasePortState {
    $mssql = $databaseEngine.SelectedIndex -eq 0
    $databasePort.Enabled = (-not $mssql) -or $explicitDatabasePort.Checked
}

function Set-QTracXDeploymentState {
    if ($deploymentMode.SelectedIndex -eq 1) {
        $serverAddress.Text = "localhost"
    }
}

function Test-QTracXSelections {
    param([switch]$ForInstall)

    if (-not $installDirectory.Text.Trim()) {
        throw "Select an installation directory."
    }
    if ($sourceMode.SelectedIndex -eq 0 -and
        -not (Test-Path -LiteralPath $sourceArchive.Text -PathType Leaf)) {
        throw "Select a valid dependency-free QTracX source ZIP."
    }
    if ($uiPort.Value -eq $backendPort.Value) {
        throw "The UI and backend ports must be different."
    }
    if (-not $serverAddress.Text.Trim()) {
        throw "Enter the server hostname or IP address."
    }
    if (-not $databaseHost.Text.Trim() -or
        -not $databaseName.Text.Trim() -or
        -not $databaseApplicationUser.Text.Trim()) {
        throw "Complete the database host, name, and application user."
    }
    if ((Get-QTracXDatabaseEngine) -eq "mssql" -and
        -not $odbcDriver.Text.Trim()) {
        throw "Select an installed 64-bit MSSQL ODBC driver."
    }
    if (-not $applicationAdminUser.Text.Trim() -or
        -not $applicationAdminDisplayName.Text.Trim()) {
        throw "Complete the initial QTracX administrator details."
    }
    if ($httpsEnabled.Checked) {
        if (-not (Test-Path -LiteralPath $certificateFile.Text -PathType Leaf) -or
            -not (Test-Path -LiteralPath $privateKeyFile.Text -PathType Leaf)) {
            throw "HTTPS requires an existing certificate and private-key file."
        }
    }
    if ($ForInstall) {
        if ($databaseApplicationPassword.Text.Length -lt 4) {
            throw "The application database password must contain at least four characters."
        }
        if ($databaseApplicationPassword.Text -cne
            $databaseApplicationPasswordConfirm.Text) {
            throw "The application database passwords do not match."
        }
        if ($databaseMode.SelectedIndex -eq 0) {
            if (-not $databaseAdministratorUser.Text.Trim() -or
                -not $databaseAdministratorPassword.Text) {
                throw "Enter the database administrator credentials."
            }
            if ($databaseAdministratorUser.Text.Trim().Equals(
                $databaseApplicationUser.Text.Trim(),
                [StringComparison]::OrdinalIgnoreCase
            )) {
                throw "The database administrator and application user must differ."
            }
        }
        if ($applicationAdminPassword.Text.Length -lt 4) {
            throw "The QTracX administrator password must contain at least four characters."
        }
        if ($applicationAdminPassword.Text -cne $applicationAdminPasswordConfirm.Text) {
            throw "The QTracX administrator passwords do not match."
        }
        $destination = [IO.Path]::GetFullPath($installDirectory.Text.Trim())
        if (Test-Path -LiteralPath $destination) {
            $existing = @(Get-ChildItem -LiteralPath $destination -Force)
            if ($existing.Count -gt 0) {
                throw "The installation directory must be empty: $destination"
            }
        }
    }
}

function Get-QTracXRuntimeName {
    return $(if ($runtime.SelectedIndex -eq 1) { "gpu" } else { "cpu" })
}

function Get-QTracXDatabaseEngine {
    return $(if ($databaseEngine.SelectedIndex -eq 0) { "mssql" } else { "postgresql" })
}

function Get-QTracXProvisioningMode {
    return $(if ($databaseMode.SelectedIndex -eq 0) { "installer" } else { "preconfigured" })
}

function Get-QTracXPayload {
    $engine = Get-QTracXDatabaseEngine
    $portValue = $(
        if ($engine -eq "mssql" -and -not $explicitDatabasePort.Checked) {
            $null
        }
        else {
            [int]$databasePort.Value
        }
    )
    $database = [ordered]@{
        engine = $engine
        provisioningMode = Get-QTracXProvisioningMode
        host = $databaseHost.Text.Trim()
        port = $portValue
        name = $databaseName.Text.Trim()
        applicationUser = $databaseApplicationUser.Text.Trim()
        applicationPassword = $databaseApplicationPassword.Text
        resetExistingApplicationPassword = [bool]$resetDatabasePassword.Checked
        encrypt = $true
        trustServerCertificate = $true
    }
    if ($engine -eq "mssql") {
        $database["odbcDriver"] = $odbcDriver.Text.Trim()
    }
    if ($database.provisioningMode -eq "installer") {
        $database["administratorUser"] = $databaseAdministratorUser.Text.Trim()
        $database["administratorPassword"] = $databaseAdministratorPassword.Text
    }
    return [ordered]@{
        deploymentMode = $(if ($deploymentMode.SelectedIndex -eq 0) { "lan" } else { "local" })
        serverAddress = $serverAddress.Text.Trim()
        uiPort = [int]$uiPort.Value
        backendPort = [int]$backendPort.Value
        runtime = Get-QTracXRuntimeName
        modelInstances = [int]$modelInstances.Value
        scanWorkers = [int]$scanWorkers.Value
        https = [ordered]@{
            enabled = [bool]$httpsEnabled.Checked
            certificateFile = $certificateFile.Text.Trim()
            privateKeyFile = $privateKeyFile.Text.Trim()
        }
        storage = [ordered]@{
            checksheetDataDirectory = "backend/data/checksheets"
            logDirectory = "backend/logs"
        }
        database = $database
        initialAdministrator = [ordered]@{
            username = $applicationAdminUser.Text.Trim()
            displayName = $applicationAdminDisplayName.Text.Trim()
            password = $applicationAdminPassword.Text
        }
    }
}

function Update-QTracXReview {
    $sourceDescription = $(
        if ($sourceMode.SelectedIndex -eq 0) { $sourceArchive.Text.Trim() }
        else { "Predefined deployment-readiness branch" }
    )
    $protocol = $(if ($httpsEnabled.Checked) { "HTTPS" } else { "HTTP" })
    $databaseTarget = $databaseHost.Text.Trim()
    if ((Get-QTracXDatabaseEngine) -eq "postgresql" -or
        $explicitDatabasePort.Checked) {
        $databaseTarget += ":$([int]$databasePort.Value)"
    }
    $reviewSummary.Text = @"
Source: $sourceDescription
Install to: $($installDirectory.Text.Trim())
Deployment: $($deploymentMode.SelectedItem) at $($serverAddress.Text.Trim())
UI / backend: $protocol ports $([int]$uiPort.Value) / $([int]$backendPort.Value)
OCR: $(Get-QTracXRuntimeName), $([int]$modelInstances.Value) model instance(s), $([int]$scanWorkers.Value) scan worker(s)
Database: $(Get-QTracXDatabaseEngine) on $databaseTarget, mode $(Get-QTracXProvisioningMode)
Database: $($databaseName.Text.Trim()) / application user $($databaseApplicationUser.Text.Trim())
Initial QTracX administrator: $($applicationAdminUser.Text.Trim())

Passwords are held only in this process and are not displayed or written to the installer log.
"@
}

function Add-QTracXLog {
    param(
        [AllowEmptyString()][string]$Message,
        [switch]$StandardError
    )
    $stamp = Get-Date -Format "HH:mm:ss"
    # Installers such as pip and npm use stderr for progress and warnings too;
    # label the stream without incorrectly declaring every line an error.
    $prefix = $(if ($StandardError) { "stderr: " } else { "" })
    $entry = "[$stamp] $prefix$Message`r`n"
    $installLog.AppendText($entry)
    $installLog.SelectionStart = $installLog.TextLength
    $installLog.ScrollToCaret()
    if ($script:InstallerLogPath) {
        try {
            [IO.File]::AppendAllText(
                $script:InstallerLogPath,
                $entry,
                [Text.Encoding]::UTF8
            )
        }
        catch {
            # A log-file write problem must not stop an otherwise valid setup.
        }
    }
    [System.Windows.Forms.Application]::DoEvents()
}

function Initialize-QTracXInstallerLog {
    try {
        $directory = Join-Path $env:ProgramData "QTracX\installer-logs"
        [void](New-Item -ItemType Directory -Path $directory -Force)
        $script:InstallerLogPath = Join-Path $directory (
            "install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss")
        )
        [IO.File]::WriteAllText(
            $script:InstallerLogPath,
            "",
            [Text.Encoding]::UTF8
        )
        Add-QTracXLog "Installer log: $($script:InstallerLogPath)"
    }
    catch {
        $script:InstallerLogPath = $null
        Add-QTracXLog "Persistent installer log could not be created; the on-screen log remains available."
    }
}

function Set-QTracXProgress {
    param(
        [int]$Value,
        [string]$Status
    )
    $bounded = [Math]::Max(
        $installProgress.Minimum,
        [Math]::Min($installProgress.Maximum, $Value)
    )
    $installProgress.Value = $bounded
    $progressLabel.Text = $Status
    [System.Windows.Forms.Application]::DoEvents()
}

function Update-QTracXProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = @($machinePath, $userPath) -join ";"
}

function Invoke-QTracXStage {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [AllowNull()][string]$InputText = $null,
        [int]$StartPercent = 0,
        [int]$EndPercent = 100,
        [int]$StageNumber = 1,
        [int]$StageCount = 1
    )
    $status = "Stage $StageNumber of ${StageCount}: $Name"
    Set-QTracXProgress -Value $StartPercent -Status $status
    Add-QTracXLog $status
    $result = Invoke-QTracXProcess `
        -FilePath $FilePath `
        -Arguments $Arguments `
        -InputText $InputText `
        -OnOutput {
            param($Line, $IsError)
            Add-QTracXLog -Message $Line -StandardError:$IsError
        }
    if ($result.ExitCode -ne 0) {
        throw "$Name failed with exit code $($result.ExitCode)."
    }
    Set-QTracXProgress `
        -Value $EndPercent `
        -Status "Completed stage $StageNumber of ${StageCount}: $Name"
    return $result
}

function Get-QTracXBootstrapArguments {
    param([string]$Mode, [switch]$HostOnly)
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $script:BootstrapScript,
        "-Mode", $Mode,
        "-Runtime", (Get-QTracXRuntimeName),
        "-Database", (Get-QTracXDatabaseEngine)
    )
    if ($HostOnly) { $arguments += "-HostOnly" }
    if ($sourceMode.SelectedIndex -eq 1) { $arguments += "-IncludeGit" }
    return $arguments
}

function Invoke-QTracXPlan {
    Test-QTracXSelections
    Update-QTracXReview
    $script:Busy = $true
    $checkPlan.Enabled = $false
    $startInstall.Enabled = $false
    $backButton.Enabled = $false
    $nextButton.Enabled = $false
    try {
        $script:InstallerLogPath = $null
        $installLog.Clear()
        Set-QTracXProgress -Value 0 -Status "Checking installation plan."
        $powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
        [void](Invoke-QTracXStage `
            -Name "Checking host prerequisite plan" `
            -FilePath $powershell `
            -Arguments (Get-QTracXBootstrapArguments -Mode "Plan" -HostOnly) `
            -StartPercent 0 -EndPercent 100 `
            -StageNumber 1 -StageCount 1)
        Set-QTracXProgress `
            -Value 100 `
            -Status "Plan complete. No changes were made."
    }
    finally {
        $script:Busy = $false
        $checkPlan.Enabled = $true
        $startInstall.Enabled = $true
        $backButton.Enabled = $tabs.SelectedIndex -gt 0
        $nextButton.Enabled = $tabs.SelectedIndex -lt ($tabs.TabPages.Count - 1)
    }
}

function Invoke-QTracXInstallation {
    Test-QTracXSelections -ForInstall
    Update-QTracXReview
    $confirmation = [System.Windows.Forms.MessageBox]::Show(
        "Install QTracX using the reviewed settings?",
        "Confirm QTracX installation",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if ($confirmation -ne [System.Windows.Forms.DialogResult]::Yes) { return }

    $script:Busy = $true
    $startInstall.Enabled = $false
    $checkPlan.Enabled = $false
    $backButton.Enabled = $false
    $nextButton.Enabled = $false
    try {
        $installLog.Clear()
        Set-QTracXProgress -Value 0 -Status "Starting QTracX installation."
        Initialize-QTracXInstallerLog
        $powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
        [void](Invoke-QTracXStage `
            -Name "Installing or verifying host prerequisites" `
            -FilePath $powershell `
            -Arguments (Get-QTracXBootstrapArguments -Mode "Install" -HostOnly) `
            -StartPercent 0 -EndPercent 15 `
            -StageNumber 1 -StageCount 6)

        Update-QTracXProcessPath
        $python = Get-QTracXPythonInvocation
        $destination = [IO.Path]::GetFullPath($installDirectory.Text.Trim())
        $sourceArguments = @($python.PrefixArguments) + @(
            $script:SourceScript,
            "--mode",
            $(if ($sourceMode.SelectedIndex -eq 0) { "archive" } else { "git" }),
            "--destination", $destination
        )
        if ($sourceMode.SelectedIndex -eq 0) {
            $sourceArguments += @("--archive", $sourceArchive.Text.Trim())
        }
        [void](Invoke-QTracXStage `
            -Name "Acquiring QTracX application source" `
            -FilePath $python.FilePath `
            -Arguments $sourceArguments `
            -StartPercent 15 -EndPercent 25 `
            -StageNumber 2 -StageCount 6)

        # From this point forward, use the installer modules from the acquired
        # application source. The selected Git branch or ZIP is the source of
        # truth for runtime, configuration and database behavior; the small
        # launcher package is only responsible for host preparation and source
        # acquisition.
        $installedInstaller = Join-Path $destination "installer"
        $installedBootstrap = Join-Path $installedInstaller "bootstrap.ps1"
        $installedConfiguration = Join-Path $installedInstaller "setup_config.py"
        $installedDatabase = Join-Path $installedInstaller "database_bootstrap.py"
        foreach ($requiredScript in @(
            $installedBootstrap,
            $installedConfiguration,
            $installedDatabase
        )) {
            if (-not (Test-Path -LiteralPath $requiredScript -PathType Leaf)) {
                throw "The acquired source is missing installer module: $requiredScript"
            }
        }

        if ((Get-QTracXDatabaseEngine) -eq "mssql") {
            $odbcDriver.Text = Get-QTracXMssqlDriver
        }
        $payload = Get-QTracXPayload
        $payloadJson = $payload | ConvertTo-Json -Depth 10 -Compress
        $configArguments = @($python.PrefixArguments) + @(
            $installedConfiguration,
            "--destination", (Join-Path $destination ".env.local")
        )
        [void](Invoke-QTracXStage `
            -Name "Writing validated deployment configuration" `
            -FilePath $python.FilePath `
            -Arguments $configArguments `
            -InputText $payloadJson `
            -StartPercent 25 -EndPercent 32 `
            -StageNumber 3 -StageCount 6)

        $applicationArguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $installedBootstrap,
            "-Mode", "Install",
            "-Runtime", (Get-QTracXRuntimeName),
            "-Database", (Get-QTracXDatabaseEngine),
            "-ApplicationRoot", $destination,
            "-BuildFrontend"
        )
        if ($warmModels.Checked) { $applicationArguments += "-WarmModels" }
        [void](Invoke-QTracXStage `
            -Name "Creating the application runtime and production build" `
            -FilePath $powershell `
            -Arguments $applicationArguments `
            -StartPercent 32 -EndPercent 72 `
            -StageNumber 4 -StageCount 6)

        $environmentName = $(
            if ((Get-QTracXRuntimeName) -eq "gpu") { ".venv-gpu" }
            else { ".venv" }
        )
        $environmentPython = Join-Path $destination `
            "backend\$environmentName\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
            throw "The installed Python environment could not be located."
        }
        [void](Invoke-QTracXStage `
            -Name "Creating and migrating the QTracX database" `
            -FilePath $environmentPython `
            -Arguments @(
                $installedDatabase,
                "--repo-root", $destination
            ) `
            -InputText $payloadJson `
            -StartPercent 72 -EndPercent 90 `
            -StageNumber 5 -StageCount 6)

        $verifyArguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $installedBootstrap,
            "-Mode", "Verify",
            "-Runtime", (Get-QTracXRuntimeName),
            "-Database", (Get-QTracXDatabaseEngine),
            "-ApplicationRoot", $destination,
            "-BuildFrontend"
        )
        [void](Invoke-QTracXStage `
            -Name "Verifying the completed QTracX installation" `
            -FilePath $powershell `
            -Arguments $verifyArguments `
            -StartPercent 90 -EndPercent 100 `
            -StageNumber 6 -StageCount 6)

        Set-QTracXProgress `
            -Value 100 `
            -Status "Installation completed successfully."
        Add-QTracXLog "QTracX installation completed successfully."
        $logMessage = $(
            if ($script:InstallerLogPath) {
                "`r`n`r`nInstaller log: $($script:InstallerLogPath)"
            }
            else {
                ""
            }
        )
        [System.Windows.Forms.MessageBox]::Show(
            "QTracX was installed and verified successfully.`r`n`r`n" +
            "Windows service registration and automatic startup will be added " +
            "in the next installer slice." + $logMessage,
            "QTracX installation complete",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }
    finally {
        $script:Busy = $false
        $startInstall.Enabled = $true
        $checkPlan.Enabled = $true
        $backButton.Enabled = $tabs.SelectedIndex -gt 0
        $nextButton.Enabled = $tabs.SelectedIndex -lt ($tabs.TabPages.Count - 1)
    }
}

$browseInstall.Add_Click({
    $dialog = [System.Windows.Forms.FolderBrowserDialog]::new()
    $dialog.Description = "Select the empty QTracX installation directory"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $installDirectory.Text = $dialog.SelectedPath
    }
})
$browseArchive.Add_Click({
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    $dialog.Filter = "ZIP archives (*.zip)|*.zip|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $sourceArchive.Text = $dialog.FileName
    }
})
$browseCertificate.Add_Click({
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    $dialog.Filter = "Certificate files (*.crt;*.pem)|*.crt;*.pem|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $certificateFile.Text = $dialog.FileName
    }
})
$browsePrivateKey.Add_Click({
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    $dialog.Filter = "Private-key files (*.key;*.pem)|*.key;*.pem|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $privateKeyFile.Text = $dialog.FileName
    }
})
$sourceMode.Add_SelectedIndexChanged({ Set-QTracXSourceState })
$httpsEnabled.Add_CheckedChanged({ Set-QTracXHttpsState })
$databaseEngine.Add_SelectedIndexChanged({ Set-QTracXDatabaseState })
$databaseMode.Add_SelectedIndexChanged({ Set-QTracXDatabaseState })
$explicitDatabasePort.Add_CheckedChanged({ Set-QTracXDatabasePortState })
$databaseHost.Add_TextChanged({
    if ($databaseEngine.SelectedIndex -eq 0 -and
        $databaseHost.Text.Contains("\")) {
        $explicitDatabasePort.Checked = $false
    }
})
$deploymentMode.Add_SelectedIndexChanged({ Set-QTracXDeploymentState })
$tabs.Add_Selecting({
    param($Sender, $EventArguments)
    if ($script:Busy -and
        $EventArguments.TabPageIndex -ne $tabs.SelectedIndex) {
        $EventArguments.Cancel = $true
    }
})
$tabs.Add_SelectedIndexChanged({
    $script:CurrentPage = $tabs.SelectedIndex
    $backButton.Enabled = $tabs.SelectedIndex -gt 0 -and -not $script:Busy
    $nextButton.Enabled = $tabs.SelectedIndex -lt ($tabs.TabPages.Count - 1) -and -not $script:Busy
    if ($tabs.SelectedIndex -eq ($tabs.TabPages.Count - 1)) {
        Update-QTracXReview
    }
})
$backButton.Add_Click({
    if ($tabs.SelectedIndex -gt 0) {
        $tabs.SelectedIndex = $tabs.SelectedIndex - 1
    }
})
$nextButton.Add_Click({
    if ($tabs.SelectedIndex -lt ($tabs.TabPages.Count - 1)) {
        $tabs.SelectedIndex = $tabs.SelectedIndex + 1
    }
})
$cancelButton.Add_Click({
    if (-not $script:Busy) { $form.Close() }
})
$form.Add_FormClosing({
    param($Sender, $EventArguments)
    if ($script:Busy) {
        $EventArguments.Cancel = $true
        [System.Windows.Forms.MessageBox]::Show(
            "Wait for the current installer stage to finish before closing the wizard.",
            "QTracX installation in progress",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }
})
$checkPlan.Add_Click({
    try {
        Invoke-QTracXPlan
    }
    catch {
        $progressLabel.Text = "Plan check failed."
        Add-QTracXLog $_.Exception.Message
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            "QTracX plan check",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})
$startInstall.Add_Click({
    try {
        Invoke-QTracXInstallation
    }
    catch {
        $progressLabel.Text = "Installation stopped."
        Add-QTracXLog $_.Exception.Message
        $logMessage = $(
            if ($script:InstallerLogPath) {
                "`r`n`r`nInstaller log: $($script:InstallerLogPath)"
            }
            else {
                ""
            }
        )
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message +
            "`r`n`r`nNo existing destination data was deleted." +
            $logMessage,
            "QTracX installation stopped",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

Set-QTracXSourceState
Set-QTracXHttpsState
Set-QTracXDatabaseState
Set-QTracXDeploymentState
$backButton.Enabled = $false
Update-QTracXReview
[void]$form.ShowDialog()
