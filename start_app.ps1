$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error 'Create the virtual environment and install requirements.txt first. See README.md.'
}
& $pythonPath -m app.server
