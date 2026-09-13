[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$DockerExe = if ($DockerCommand) { $DockerCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
}
$NodeExe = Join-Path $ProjectRoot '.tools/node-v22.23.2-win-x64/node.exe'
$VercelCli = Join-Path $ProjectRoot '.tools/vercel-cli/node_modules/vercel/dist/vc.js'
$ComposeArgs = @('compose', '-f', 'compose.yaml', '-f', 'compose.tunnel.yaml')

if (-not (Test-Path -LiteralPath $DockerExe)) { throw 'Docker CLI not found. Open Docker Desktop first.' }
if (-not $CheckOnly) {
    foreach ($Path in @($NodeExe, $VercelCli, (Join-Path $ProjectRoot 'web/.vercel/project.json'))) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Required deployment tool or Vercel project link missing: $Path" }
    }
}

Push-Location $ProjectRoot
try {
    if (-not $CheckOnly) {
        & $DockerExe @ComposeArgs up -d api web tunnel
        if ($LASTEXITCODE -ne 0) { throw 'Docker startup failed. Check Docker Desktop and try again.' }
    }
    $TunnelId = & $DockerExe @ComposeArgs ps -q tunnel
    if ($LASTEXITCODE -ne 0 -or -not $TunnelId) { throw 'The public tunnel is not running.' }
    $StartedAt = & $DockerExe inspect --format '{{.State.StartedAt}}' $TunnelId
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the tunnel container.' }
    $ApiUrl = $null
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        $Logs = (& $DockerExe @ComposeArgs logs --no-color --since $StartedAt tunnel 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read the tunnel logs.' }
        $UrlMatches = [regex]::Matches($Logs, 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($UrlMatches.Count) { $ApiUrl = $UrlMatches[$UrlMatches.Count - 1].Value; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ApiUrl) { throw 'No public API URL appeared. Check the tunnel logs.' }
    $Health = Invoke-RestMethod "$ApiUrl/health" -TimeoutSec 30
    $ModelIds = @($Health.models.id)
    if ($Health.status -ne 'ready' -or @('lightgcn', 'sign', 'residual_sign' | Where-Object { $_ -notin $ModelIds }).Count) {
        throw 'The public API did not report all three models ready.'
    }
    Write-Host "Public API ready: $ApiUrl"
    if ($CheckOnly) { return }

    Push-Location (Join-Path $ProjectRoot 'web')
    try {
        $PreviousPath = $env:PATH
        $env:PATH = (Split-Path -Parent $NodeExe) + ';' + $env:PATH
        & $NodeExe $VercelCli env update RECOMMENDER_API_URL production --value $ApiUrl --yes --scope xharra
        if ($LASTEXITCODE -ne 0) { throw 'Vercel environment update failed. Check the CLI login.' }
        & $NodeExe $VercelCli deploy --prod --yes --scope xharra
        if ($LASTEXITCODE -ne 0) { throw 'Vercel deployment failed; inspect the build output.' }
    } finally {
        $env:PATH = $PreviousPath
        Pop-Location
    }
    Write-Host 'Deployment finished. Keep the laptop awake, Docker running and the internet connected.'
} finally {
    Pop-Location
}
