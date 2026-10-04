@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting the fused Parkinson voice + gait model...
echo Backend: http://127.0.0.1:8008
echo Phone:   http://%COMPUTERNAME%:8008/fused (use the PC LAN IP if needed)

start "Fused Backend" cmd /k "cd /d "%~dp0backend" && "%PY%" -m uvicorn main_advanced:app --host 0.0.0.0 --port 8008"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8008/fused"

echo.
echo Fused model started. Keep the backend window open.
echo Open the same /fused URL on the phone after connecting both devices to Wi-Fi.
endlocal
