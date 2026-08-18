@echo off
REM One-click start: backend on port 8000 + frontend page in the browser.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Could not find "%PY%".
    echo Create the venv first:  python -m venv "%ROOT%\.venv"
    pause
    exit /b 1
)

echo Checking dependencies...
"%PY%" -c "import fastapi, uvicorn, librosa, sklearn, joblib, multipart" 2>nul
if errorlevel 1 (
    echo Installing missing packages, this may take a few minutes...
    "%PY%" -m pip install --quiet fastapi uvicorn python-multipart librosa numpy pandas scikit-learn joblib scipy soundfile xgboost
)

echo Starting backend on http://localhost:8000 ...
start "Parkinson backend" "%PY%" "%ROOT%\backend\main_advanced.py"

REM Give uvicorn a moment to bind the port before the page runs its health check.
timeout /t 8 /nobreak >nul
start "" "%ROOT%\frontend\predict_advanced.html"

echo.
echo Backend runs in the other window - close it to stop the server.
endlocal
