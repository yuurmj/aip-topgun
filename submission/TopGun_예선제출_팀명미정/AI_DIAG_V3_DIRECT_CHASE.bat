@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_diagnostic_match_v3_direct_chase.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] V3_DIRECT_CHASE diagnostic match failed.
  pause
)
endlocal
