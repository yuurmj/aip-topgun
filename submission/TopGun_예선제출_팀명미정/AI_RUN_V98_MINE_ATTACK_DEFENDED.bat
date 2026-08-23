@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
set "PYTHON_EXE=python"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\aip\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\aip\python.exe"
echo Variant: V98_MINE_ATTACK_DEFENDED
"%PYTHON_EXE%" run_unreal_inference.py --mode rl --bundle-dir model_variants\V98_MINE_ATTACK_DEFENDED --team-name-file team_name.txt --server-ip 127.0.0.1 --server-port 9999 --action-repeat 1 --observation-repeat 6 --rebase-pair-positions --initial-pair-range-m 762.0 --plane-info-velocity-frame body
set RESULT=%ERRORLEVEL%
echo AI client exited with code %RESULT%.
pause
exit /b %RESULT%
