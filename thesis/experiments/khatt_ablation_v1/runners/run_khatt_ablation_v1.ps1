[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'Check', 'Train', 'TrainAll', 'TestAll', 'Stats')][string]$Action,
    [string]$Python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\python.exe',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$')][string]$SuiteName = 'khatt_ar_v1fix1_s54101',
    [ValidateSet('khatt100_only', 'khatt100_ours', 'khatt50_ours', 'khatt30_ours', 'khatt10_ours')][string]$Trial,
    [ValidateRange(100, 100)][int]$Epochs = 100,
    [ValidateRange(10, 10)][int]$Patience = 10,
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
    $backend = Join-Path $project '.codex-pretrained-selection\thesis\scripts\run_khatt_ablation_v1.py'
    foreach ($path in @($Python, $backend)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
    }
    $arguments = @($backend, '--action', $Action, '--project', $project,
        '--suite-name', $SuiteName, '--device', $Device, '--epochs', $Epochs, '--patience', $Patience)
    if ($Trial) { $arguments += @('--trial', $Trial) }
    if ($Resume) { $arguments += '--resume' }
    if ($DryRun) { $arguments += '--dry-run' }
    if (-not $DryRun) {
        $logRoot = Join-Path $PSScriptRoot ('drive_all\logs\' + $SuiteName)
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
    if ($code -ne 0) { throw "KHATT AR suite failed with Python exit code $code" }
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    $OutputEncoding = $savedEncoding
    $env:PYTHONUTF8 = $savedUtf8
    $env:PYTHONUNBUFFERED = $savedUnbuffered
    $ErrorActionPreference = $savedPreference
}
