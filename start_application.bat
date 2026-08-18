@echo off
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and add it to your system PATH
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Parkinson's Detection - Auto-Start
echo ===================================================
echo.
echo Starting backend server...
echo.

REM Start backend in a new window
start "Backend Server" cmd /k "cd backend && python main.py"

REM Wait for backend to start (5 seconds)
timeout /t 5 /nobreak

echo.
echo Backend started. Opening frontend...
echo.

REM Open the frontend in default browser
start "" "frontend\index.html"

echo.
echo ✓ Application started successfully!
echo ✓ Backend: http://localhost:8001
echo ✓ Frontend: Opening in your browser...
echo.
echo Note: Keep the backend window open while using the application
echo.

pause
