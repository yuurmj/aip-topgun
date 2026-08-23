@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_diagnostic_match_v2_balanced.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] V2_BALANCED diagnostic match failed.
  pause
)
endlocal
