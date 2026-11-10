@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_full_research.ps1"
if errorlevel 1 (
  echo.
  echo Research run failed. Review the error above.
  pause
  exit /b 1
)
endlocal

