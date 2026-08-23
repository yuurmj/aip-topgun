@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_diagnostic_match_v1_reacquire.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] V1 reacquisition diagnostic match failed.
  pause
)
endlocal
