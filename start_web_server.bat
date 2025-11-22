@echo off
REM Start the plant disease detection web server

echo ===============================================================
echo PLANT DISEASE DETECTOR - WEB SERVER (IMPROVED)
echo ===============================================================

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting Flask server...
echo.
echo The web interface will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ===============================================================
echo.

python web_interface.py

pause
