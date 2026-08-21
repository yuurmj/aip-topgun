@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
set "PYTHON_EXE=python"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\aip\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\aip\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\aip\python.exe"
"%PYTHON_EXE%" -c "import numpy, torch" >nul 2>&1
if errorlevel 1 goto :missing_python
echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" run_unreal_inference.py --mode rl --bundle-dir model --team-name-file team_name.txt --server-ip 127.0.0.1 --server-port 9999 --action-repeat 1 --observation-repeat 6
set RESULT=%ERRORLEVEL%
echo.
echo AI client exited with code %RESULT%.
pause
exit /b %RESULT%

:missing_python
echo [FAIL] NumPy and PyTorch are not available in: %PYTHON_EXE%
echo Activate the provided AIP conda environment and try again.
pause
exit /b 2
