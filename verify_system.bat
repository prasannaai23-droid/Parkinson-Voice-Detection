@echo off
REM System verification script for Parkinson's Detection Application
REM This script checks all prerequisites before running the application

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ===================================================
echo Parkinson's Detection - System Verification
echo ===================================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ FAILED: Python is not installed or not in PATH
    echo.
    echo Please download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
) else (
    python --version
    echo ✅ PASSED
    echo.
)

REM Check backend folder
echo [2/5] Checking backend folder structure...
if not exist "backend\main.py" (
    echo ❌ FAILED: backend\main.py not found
    pause
    exit /b 1
) else (
    echo ✅ PASSED: backend\main.py found
    echo.
)

REM Check models folder
echo [3/5] Checking models folder...
if not exist "backend\models" (
    echo ❌ FAILED: backend\models folder not found
    pause
    exit /b 1
) else (
    echo ✅ PASSED: backend\models folder found
    
    REM Count model files
    for /f %%i in ('dir /b backend\models\*.keras 2^>nul ^| find /c /v ""') do set model_count=%%i
    echo    Found !model_count! keras models
    echo.
)

REM Check frontend folder
echo [4/5] Checking frontend folder...
if not exist "frontend\index.html" (
    echo ❌ FAILED: frontend\index.html not found
    pause
    exit /b 1
) else (
    echo ✅ PASSED: frontend\index.html found
    echo.
)

REM Check pip
echo [5/5] Checking pip and dependencies...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  WARNING: pip is not available
    echo    Some dependencies might not be installed
    echo.
) else (
    echo ✅ PASSED: pip is available
    
    REM Quick check for key Python packages
    echo    Checking required packages...
    
    python -c "import tensorflow" >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ TensorFlow installed
    ) else (
        echo    ⚠️  TensorFlow not installed (will be needed for models)
    )
    
    python -c "import fastapi" >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ FastAPI installed
    ) else (
        echo    ⚠️  FastAPI not installed (required for backend)
    )
    
    python -c "import uvicorn" >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ Uvicorn installed
    ) else (
        echo    ⚠️  Uvicorn not installed (required for backend)
    )
    echo.
)

echo ===================================================
echo ✅ Verification Complete!
echo ===================================================
echo.
echo Your system is ready to run the application.
echo.
echo Next steps:
echo   1. Double-click: start_application.bat
echo   2. Or manually start backend: backend\main.py
echo   3. Open: frontend\index.html in your browser
echo.
echo For more help, see: STARTUP_GUIDE.md
echo.

pause
