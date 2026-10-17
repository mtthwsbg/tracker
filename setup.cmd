@echo off
rem One-time setup (Windows): creates .venv next to this script and installs requirements.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
%PY% -m venv .venv || (echo Python 3.9+ is required. Install it from python.org & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)
echo.
echo Setup done. Double-click run.cmd to start.
pause
