@echo off
rem Uses .venv from setup.cmd if present, otherwise the original local venv.
if exist "%~dp0.venv\Scripts\python.exe" (set PYX="%~dp0.venv\Scripts\python.exe") else if exist "%LOCALAPPDATA%\tracker-venv\Scripts\python.exe" (set PYX="%LOCALAPPDATA%\tracker-venv\Scripts\python.exe") else (echo Run setup.cmd first. & pause & exit /b 1)
%PYX% "%~dp0tracker.py" %*
if errorlevel 1 pause
