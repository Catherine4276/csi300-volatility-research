$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Streamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
$DashboardUrl = "http://127.0.0.1:8501"
if (-not (Test-Path -LiteralPath $Streamlit)) {
    Write-Host "First start: creating Python environment and installing dependencies..."
    & (Join-Path $PSScriptRoot "setup_environment.ps1")
}

try {
    $health = Invoke-WebRequest -Uri "$DashboardUrl/_stcore/health" -UseBasicParsing -TimeoutSec 2
    if ($health.StatusCode -eq 200) {
        Write-Host "Dashboard is already running. Opening $DashboardUrl"
        Start-Process $DashboardUrl
        exit 0
    }
} catch {
    # No existing server: start one below.
}

Set-Location -LiteralPath $ProjectRoot
$browserJob = Start-Job -ScriptBlock {
    param($url)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $response = Invoke-WebRequest -Uri "$url/_stcore/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Start-Process $url
                return
            }
        } catch {
            # Server is still starting.
        }
    }
    Write-Warning "Server started, but the browser did not open automatically. Visit $url"
} -ArgumentList $DashboardUrl

try {
    Write-Host "Starting dashboard at $DashboardUrl"
    Write-Host "Keep this window open. Press Ctrl+C to stop."
    & $Streamlit run .\streamlit_app.py --server.port 8501
} finally {
    if ($browserJob) {
        Receive-Job -Job $browserJob -ErrorAction SilentlyContinue | Out-Host
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}


