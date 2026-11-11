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
        throw "Python 3.11+ is required but was not found."
    }
}

& $VenvPython -m pip install -r requirements.txt
& $VenvPython scripts\prepare_data.py
& $VenvPython -m pytest
& $VenvPython scripts\run_research.py
& $VenvPython scripts\build_static_site.py
& $VenvPython scripts\verify_project.py

Write-Host ""
Write-Host "Research complete."
Write-Host "Report: $(Join-Path $ProjectRoot 'reports\research_report.md')"
Write-Host "Dashboard: .\scripts\start_dashboard.ps1"

