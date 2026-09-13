[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Destination) {
    $Destination = Join-Path $ProjectRoot 'deployment-bundle'
}
$RequiredModels = @(
    'manifest.json',
    'lightgcn_playlist_recommender.pt',
    'sign_playlist_recommender.pt',
    'residual_sign_playlist_recommender.pt',
    'track_catalog.csv.gz',
    'artist_catalog.parquet',
    'content_features.npy',
    'artist_content_features.npy',
    'artist_metadata_index.npz',
    'track_artist_index.npz',
    'metadata_tokens.json'
)

foreach ($Name in $RequiredModels) {
    $Path = Join-Path $ProjectRoot "models/$Name"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required model artifact: $Path"
    }
}

$Destination = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists: $Destination. Choose a new empty path to avoid overwriting files."
}

New-Item -ItemType Directory -Path $Destination | Out-Null
foreach ($Directory in @('docker', 'scripts', 'models', 'web')) {
    New-Item -ItemType Directory -Path (Join-Path $Destination $Directory) | Out-Null
}

foreach ($Name in @('compose.yaml', 'compose.public.yaml', '.dockerignore')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination (Join-Path $Destination $Name)
}
foreach ($Name in @('api.Dockerfile', 'requirements-api.txt', 'check_models.py', 'Caddyfile')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docker/$Name") -Destination (Join-Path $Destination "docker/$Name")
}
foreach ($Name in @('recommender_api.py', 'recommender_inference.py', 'discovery.py')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts/$Name") -Destination (Join-Path $Destination "scripts/$Name")
}
foreach ($Name in @('Dockerfile', '.dockerignore', 'package.json', 'package-lock.json', 'next.config.ts', 'next-env.d.ts', 'tsconfig.json', 'eslint.config.mjs')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "web/$Name") -Destination (Join-Path $Destination "web/$Name")
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'web/app') -Destination (Join-Path $Destination 'web/app') -Recurse
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'web/public')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot 'web/public') -Destination (Join-Path $Destination 'web/public') -Recurse
}
foreach ($Name in $RequiredModels) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "models/$Name") -Destination (Join-Path $Destination "models/$Name")
}

Write-Host "Deployment bundle ready: $Destination"
Write-Host 'Transfer this folder privately to the server. Model files remain outside Git.'
