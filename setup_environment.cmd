@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_environment.ps1"
if errorlevel 1 (
  echo.
  echo Environment setup failed. Review the error above.
  pause
  exit /b 1
)
endlocal

