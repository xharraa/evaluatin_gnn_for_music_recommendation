[CmdletBinding()]
param([switch]$Production)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot '.tools\environment.json'
$PythonPath = if (Test-Path -LiteralPath $EnvironmentPath) { (Get-Content -LiteralPath $EnvironmentPath -Raw | ConvertFrom-Json).python } else { Join-Path $env:USERPROFILE '.venvs\music\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run scripts/setup_environment.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'models\manifest.json'))) { throw 'Run notebook 03 to train the three models first.' }
$NodeRoot = Join-Path $ProjectRoot '.tools\node-v22.23.2-win-x64'
$env:PATH = $NodeRoot + ';' + $env:PATH
$LogRoot = Join-Path $ProjectRoot 'reports\current'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$ApiScript = Join-Path $PSScriptRoot 'recommender_api.py'
$ApiProcess = Start-Process -FilePath $PythonPath -ArgumentList @('-X','utf8',('"'+$ApiScript+'"')) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogRoot 'api.log') -RedirectStandardError (Join-Path $LogRoot 'api-error.log')
try {
    $Ready = $false
    for ($Attempt=0; $Attempt -lt 60; $Attempt++) {
        if ($ApiProcess.HasExited) { throw 'API failed. Read reports/current/api-error.log.' }
        try { $Health = Invoke-RestMethod 'http://127.0.0.1:8000/health'; $Ready = $Health.status -eq 'ready' } catch { $Ready = $false }
        if ($Ready) { break }
        Start-Sleep -Seconds 1
    }
    if (-not $Ready) { throw 'API did not become ready within 60 seconds.' }
    Push-Location (Join-Path $ProjectRoot 'web')
    Write-Host 'Open http://localhost:3000. Ctrl+C stops both services.'
    $Mode = if ($Production) { 'start' } else { 'dev' }
    & (Join-Path $NodeRoot 'npm.cmd') run $Mode
    if ($LASTEXITCODE -ne 0) { throw 'The web server exited with an error.' }
}
finally {
    if ((Get-Location).Path -eq (Join-Path $ProjectRoot 'web')) { Pop-Location }
    if (-not $ApiProcess.HasExited) { Stop-Process -Id $ApiProcess.Id }
}
