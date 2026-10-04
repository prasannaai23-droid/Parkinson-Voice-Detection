@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address"') do set "IP=%%A"
set "IP=%IP: =%"
echo.
echo Fused backend: http://localhost:8008/fused
echo Phone URL:     http://%IP%:8008/fused
echo.
echo 1. Connect the phone and PC to the same Wi-Fi.
echo 2. Open the Phone URL in the phone browser.
echo 3. Allow camera and microphone access.
echo 4. Tap Use phone camera, record gait, then record or upload voice.
echo.
start "Fused backend" cmd /k "cd /d "%~dp0backend" && "%PY%" -m uvicorn main_advanced:app --host 0.0.0.0 --port 8008"
timeout /t 3 /nobreak >nul
start "" "http://localhost:8008/fused"
pause
endlocal
