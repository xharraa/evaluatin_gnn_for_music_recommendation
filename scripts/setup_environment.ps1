[CmdletBinding()]
param([string]$VenvPath = (Join-Path $env:USERPROFILE '.venvs\music'), [ValidateSet('cpu','cu126')][string]$TorchBuild = 'cpu', [switch]$SkipWeb)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'
function Invoke-Checked {
    param([string]$Program,[string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $Arguments" }
}
if (-not (Test-Path -LiteralPath $PythonPath)) { Invoke-Checked 'py' @('-3.13','-m','venv',$VenvPath) }
Invoke-Checked $PythonPath @('-m','pip','install','--timeout','120','-r',(Join-Path $ProjectRoot 'requirements.txt'))
Invoke-Checked $PythonPath @('-m','pip','install','--timeout','120',"torch==2.13.0+$TorchBuild",'--index-url',"https://download.pytorch.org/whl/$TorchBuild")
Invoke-Checked $PythonPath @('-m','ipykernel','install','--prefix',$VenvPath,'--name','music-gnn','--display-name','Python (Music GNN)')
$ToolsRoot = Join-Path $ProjectRoot '.tools'
New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
@{python=$PythonPath;torchBuild=$TorchBuild} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ToolsRoot 'environment.json') -Encoding utf8
if (-not $SkipWeb) {
    $NodeVersion = 'v22.23.2'
    $NodeRoot = Join-Path $ToolsRoot "node-$NodeVersion-win-x64"
    $NodeName = "node-$NodeVersion-win-x64.zip"
    $NodeArchive = Join-Path $ToolsRoot $NodeName
    if (-not (Test-Path -LiteralPath (Join-Path $NodeRoot 'node.exe'))) {
        Invoke-Checked 'curl.exe' @('-fL','--retry','5','--retry-all-errors','-o',$NodeArchive,"https://nodejs.org/dist/$NodeVersion/$NodeName")
        $Checksums = (Invoke-WebRequest "https://nodejs.org/dist/$NodeVersion/SHASUMS256.txt").Content
        $Line = ($Checksums -split "`n" | Where-Object { $_.Trim().EndsWith($NodeName) })
        $Expected = ($Line.Trim() -split '\s+')[0]
        if ((Get-FileHash -LiteralPath $NodeArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) { throw 'Node archive checksum mismatch.' }
        Expand-Archive -LiteralPath $NodeArchive -DestinationPath $ToolsRoot -Force
    }
    $env:PATH = $NodeRoot + ';' + $env:PATH
    Invoke-Checked (Join-Path $NodeRoot 'npm.cmd') @('ci','--prefix',(Join-Path $ProjectRoot 'web'),'--no-audit','--no-fund')
}
Push-Location $ProjectRoot
try { Invoke-Checked $PythonPath @('-X','utf8','-c','from scripts.project_analysis import hardware; hardware()') } finally { Pop-Location }
Write-Host "Ready. Python: $PythonPath"
Write-Host "Next: & '$PythonPath' scripts/download_data.py"
Write-Host "Then: & '$PythonPath' scripts/run_notebooks.py"
