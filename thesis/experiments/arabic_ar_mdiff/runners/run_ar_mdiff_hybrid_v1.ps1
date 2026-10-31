[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Check', 'Fit', 'Validation', 'ValidationStats', 'Test')][string]$Action,
    [string]$Python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\python.exe',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$')][string]$RunName,
    [string]$ARunName = 'ar1b_s54101',
    [string]$MDiffRunName = 'ma1b_s54101',
    [string]$TestCacheName = 'mdar_20261031_x01',
    [switch]$AllowExploratory,
    [switch]$DryRun
)
$savedPreference = $ErrorActionPreference
$savedUtf8 = $env:PYTHONUTF8
$savedUnbuffered = $env:PYTHONUNBUFFERED
$savedEncoding = $OutputEncoding
$transcribing = $false
try {
    $ErrorActionPreference = 'Stop'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $project = Split-Path $PSScriptRoot -Parent
    $backend = Join-Path $project '.codex-pretrained-selection\thesis\scripts\run_ar_mdiff_hybrid_v1.py'
    foreach ($path in @($Python, $backend)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
    }
    $arguments = @($backend, '--action', $Action, '--project', $project,
        '--run-name', $RunName, '--ar-run-name', $ARunName, '--mdiff-run-name', $MDiffRunName,
        '--device', $Device, '--test-cache-name', $TestCacheName)
    if ($AllowExploratory) { $arguments += '--allow-exploratory' }
    if ($DryRun) { $arguments += '--dry-run' }
    if (-not $DryRun) {
        $logRoot = Join-Path $PSScriptRoot ("drive_all\logs\" + $RunName)
        New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
        $logPath = Join-Path $logRoot ($Action + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
        Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
        $transcribing = $true
        Write-Host "Log: $logPath"
    }
    $ErrorActionPreference = 'Continue'
    & $Python @arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "AR/MDiff hybrid failed with Python exit code $code" }
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    $OutputEncoding = $savedEncoding
    $env:PYTHONUTF8 = $savedUtf8
    $env:PYTHONUNBUFFERED = $savedUnbuffered
    $ErrorActionPreference = $savedPreference
}
