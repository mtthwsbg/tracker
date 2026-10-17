@echo off
"%LOCALAPPDATA%\vision-tracker-venv\Scripts\python.exe" "%~dp0tracker.py" %*
if errorlevel 1 pause
