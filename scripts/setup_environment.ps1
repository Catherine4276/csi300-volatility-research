$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw "Python 3.11+ is required. Install Python, then run this script again."
    }
}

& $VenvPython -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'"
& $VenvPython -m pip install -r requirements.txt
Write-Host "Environment ready: $VenvPython"

