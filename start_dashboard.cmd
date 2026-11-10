@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_dashboard.ps1"
if errorlevel 1 (
  echo.
  echo Dashboard failed to start. Review the error above.
  pause
  exit /b 1
)
endlocal

