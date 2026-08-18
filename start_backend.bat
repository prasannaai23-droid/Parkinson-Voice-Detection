@echo off
REM Parkinson's Disease Detection - Backend Startup Script v4.0
REM This script starts the production backend server

cd /d "%~dp0"
cd backend

echo.
echo ===================================================
echo Starting Parkinson's Detection Backend v4.0...
echo ===================================================
echo.
echo The server will run on: http://localhost:8000
echo.
echo Once you see "Model System Ready!" message, you can:
echo 1. Open frontend/predict.html in your browser
echo 2. Test connection and upload audio files
echo.
echo Press Ctrl+C to stop the server
echo.

python main_production.py

pause
