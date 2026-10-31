[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Check', 'Prepare', 'Calibrate', 'TrainShared', 'PrepareShared', 'CalibrateShared', 'Test', 'Stats', 'All')][string]$Action,
    [string]$Python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\python.exe',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$')][string]$RunName = 'ar_md_fusion2_s54101',
    [string]$ARunName = 'ar1b_s54101',
    [string]$MDiffRunName = 'ma1b_s54101',
    [string]$TestCacheName = 'mdar_20261031_x01',
    [string]$PreviousHybridName = 'ar_md_h1_s54101',
    [ValidateRange(1, 200)][int]$Epochs = 60,
    [ValidateRange(1, 200)][int]$Patience = 10,
    [switch]$Resume,
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
    $backend = Join-Path $project '.codex-pretrained-selection\thesis\scripts\run_ar_mdiff_fusion_v2.py'
    foreach ($path in @($Python, $backend)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
    }
    $arguments = @($backend, '--action', $Action, '--project', $project,
        '--run-name', $RunName, '--ar-run-name', $ARunName, '--mdiff-run-name', $MDiffRunName,
        '--device', $Device, '--test-cache-name', $TestCacheName, '--previous-hybrid-name', $PreviousHybridName,
        '--epochs', $Epochs, '--patience', $Patience)
    if ($Resume) { $arguments += '--resume' }
    if ($DryRun) { $arguments += '--dry-run' }
    if (-not $DryRun) {
        $logRoot = Join-Path $PSScriptRoot ('drive_all\logs\' + $RunName)
        New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
        $logPath = Join-Path $logRoot ($Action + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
        Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
        $transcribing = $true
        Write-Host "Log: $logPath"
    }
    # Windows PowerShell treats native stderr as ErrorRecord. The Python exit
    # code determines success; do not let harmless stderr abort a successful run.
    $ErrorActionPreference = 'Continue'
    & $Python @arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "AR/MDiff fusion suite failed with Python exit code $code" }
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    $OutputEncoding = $savedEncoding
    $env:PYTHONUTF8 = $savedUtf8
    $env:PYTHONUNBUFFERED = $savedUnbuffered
    $ErrorActionPreference = $savedPreference
}
