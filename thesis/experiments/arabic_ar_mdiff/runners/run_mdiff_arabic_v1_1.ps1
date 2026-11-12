[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Setup', 'Check', 'Smoke', 'Overfit', 'Train', 'Validation', 'ValidationStats', 'Test')][string]$Action,
    [string]$Python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\python.exe',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [ValidateSet('mdiff', 'ar')][string]$Arm = 'mdiff',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$')][string]$RunName,
    [ValidatePattern('^$|^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$')][string]$PeerRunName = '',
    [ValidateRange(1, 200)][int]$Epochs = 200,
    [ValidateRange(1, 30)][int]$Patience = 30,
    [ValidateRange(1, 1200)][int]$OverfitUpdates = 1200,
    [ValidateRange(1, 900)][int]$OverfitSeconds = 900,
    [switch]$Resume,
    [switch]$DryRun
)

$savedPreference = $ErrorActionPreference
$savedEncoding = $OutputEncoding
$savedUtf8 = $env:PYTHONUTF8
$savedUnbuffered = $env:PYTHONUNBUFFERED
$transcribing = $false
try {
    $ErrorActionPreference = 'Stop'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $project = Split-Path $PSScriptRoot -Parent
    $repo = Join-Path $project '.codex-pretrained-selection'
    $backend = Join-Path $repo 'thesis\scripts\run_mdiff_arabic_v1.py'
    foreach ($path in @($Python, $backend)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
    }
    $arguments = @($backend, '--action', $Action, '--project', $project,
        '--run-name', $RunName, '--arm', $Arm, '--device', $Device,
        '--readiness-protocol', 'geometry_matched_v1',
        '--epochs', "$Epochs", '--patience', "$Patience",
        '--overfit-updates', "$OverfitUpdates", '--overfit-seconds', "$OverfitSeconds")
    if ($PeerRunName) { $arguments += @('--peer-run-name', $PeerRunName) }
    if ($Resume) { $arguments += '--resume' }
    if ($DryRun) { $arguments += '--dry-run' }
    if (-not $DryRun) {
        $logRoot = Join-Path $PSScriptRoot ("drive_all\logs\" + $RunName)
        New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
        $logPath = Join-Path $logRoot ($Action + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
        Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
        $transcribing = $true
        Write-Host "Log: $logPath"
    }
    # Native libraries may emit harmless stderr. Only the captured native exit
    # code decides success; immediately capture it before any unrelated output.
    $ErrorActionPreference = 'Continue'
    & $Python @arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "Arabic MDiff experiment failed with Python exit code $code" }
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    $OutputEncoding = $savedEncoding
    $env:PYTHONUTF8 = $savedUtf8
    $env:PYTHONUNBUFFERED = $savedUnbuffered
    $ErrorActionPreference = $savedPreference
}
