[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Check', 'Calibrate', 'Test')][string]$Action,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$Python = 'python.exe'
)

$project = Split-Path $PSScriptRoot -Parent
$backend = Join-Path $project '.codex-pretrained-selection\thesis\scripts\run_khatt_span_hybrids_v1.py'
$arguments = @('-X', 'utf8', '-u', $backend, '--project', $project, '--action', $Action, '--device', $Device)
$logRoot = Join-Path $PSScriptRoot 'drive_all\logs\khatt_span_hybrids_v1'
New-Item -ItemType Directory -Path $logRoot -Force -ErrorAction Stop | Out-Null
$log = Join-Path $logRoot ($Action + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
Start-Transcript -LiteralPath $log -NoClobber -ErrorAction Stop | Out-Null
try {
    Write-Host "Log = $log"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "KHATT span hybrids failed with Python exit code $LASTEXITCODE" }
} finally {
    Stop-Transcript | Out-Null
}
