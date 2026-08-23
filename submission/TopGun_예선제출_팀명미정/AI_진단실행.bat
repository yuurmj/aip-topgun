@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_diagnostic_match.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Diagnostic match failed.
  pause
)
endlocal
