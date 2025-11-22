@echo off
REM Plant Disease Detector - Startup Script (Windows)

cls
echo.
echo ==========================================
echo. Plant Disease Detector
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo. Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo. Python found: 
python --version
echo.

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo. Installing Flask dependencies...
    pip install -r web_requirements.txt
    echo.
)

REM Check if model exists
if not exist "PlantSeg\work_dirs\segnext_mscan-l_test\iter_1000.pth" (
    echo. Model checkpoint not found!
    echo. Location: PlantSeg\work_dirs\segnext_mscan-l_test\iter_1000.pth
    echo.
    echo. To train the model, run:
    echo.   cd PlantSeg
    echo.   bash run.sh
    echo.
    set /p response="Continue anyway? (y/n): "
    if /i not "%response%"=="y" exit /b 1
)

echo.
echo. Starting Plant Disease Detector...
echo.
echo. ==========================================
echo. The web interface will open shortly!
echo. If not, visit: http://localhost:5000
echo. ==========================================
echo.

REM Set environment variables for better GPU memory management
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Start the Flask server
python web_interface.py

REM If server exits, show message
echo.
echo. Server stopped
echo. To restart, run this script again
pause
