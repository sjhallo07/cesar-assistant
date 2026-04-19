@echo off
REM Run the app using the repo .venv and capture stdout/stderr to app_start.log
cd /d "%~dp0\cesar_assistant"
set PORT=7861
"%~dp0\.venv\Scripts\python.exe" -u app.py > app_start.log 2>&1
type app_start.log