@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Parkinson ML Models (Voice + Gait)
echo ===================================================

echo Stopping any previously running backends...
taskkill /IM python.exe /F >nul 2>&1

:: Dependencies are already installed, skipping pip install to speed up startup

:: Start the voice model backend
start "Voice Backend" cmd /k "cd backend && python main_advanced.py"

:: Start the gait backend
start "Gait Backend" cmd /k "cd backend && python gait_backend.py"

:: Wait 4 seconds for servers to start
timeout /t 4 /nobreak >nul

:: Open the frontends
start "" "frontend1\gait.html"
start "" "frontend\predict_advanced.html"

echo All services are running!
exit
