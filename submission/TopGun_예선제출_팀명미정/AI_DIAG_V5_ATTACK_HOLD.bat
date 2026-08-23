@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
set "PYTHON_EXE=python"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\aip\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\aip\python.exe"
if not exist diagnostic_logs mkdir diagnostic_logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
echo Variant: V5_ATTACK_HOLD
"%PYTHON_EXE%" run_unreal_inference.py --mode rl --bundle-dir model_variants\V5_ATTACK_HOLD --team-name-file team_name.txt --server-ip 127.0.0.1 --server-port 9999 --action-repeat 1 --observation-repeat 6 --rebase-pair-positions --initial-pair-range-m 762.0 --plane-info-velocity-frame body --packet-monitor --packet-log "diagnostic_logs\V5_ATTACK_HOLD_%STAMP%.jsonl"
set RESULT=%ERRORLEVEL%
echo AI client exited with code %RESULT%.
pause
exit /b %RESULT%
