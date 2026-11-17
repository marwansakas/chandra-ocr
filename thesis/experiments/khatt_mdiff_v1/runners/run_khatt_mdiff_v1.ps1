[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Check', 'Train', 'Test', 'Stats')][string]$Action,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$Python = 'python.exe',
    [switch]$Resume
)

if ($Resume -and $Action -ne 'Train') { throw '-Resume requires -Action Train' }
$project = Split-Path $PSScriptRoot -Parent
$backend = Join-Path $project '.codex-pretrained-selection\thesis\scripts\run_khatt_mdiff_v1.py'
$arguments = @('-X', 'utf8', '-u', $backend, '--action', $Action, '--project', $project, '--device', $Device)
if ($Resume) { $arguments += '--resume' }
$logRoot = Join-Path $PSScriptRoot 'drive_all\logs\khatt_mdiff_v1_s54101'
New-Item -ItemType Directory -Path $logRoot -Force -ErrorAction Stop | Out-Null
$log = Join-Path $logRoot ($Action + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
Start-Transcript -LiteralPath $log -NoClobber -ErrorAction Stop | Out-Null
try {
    Write-Host "Log = $log"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "KHATT MDiff failed with Python exit code $LASTEXITCODE" }
} finally {
    Stop-Transcript | Out-Null
}
