# 🚀 Parkinson's Detection Application - Startup Guide

## Quick Start (Recommended)

### Windows Users - Double-click one of these files:

1. **`start_application.bat`** (Easiest - Use this!)
   - Automatically starts the backend server
   - Opens the frontend in your browser
   - Just double-click and you're done!

2. **`start_backend.bat`** (Backend only)
   - Only starts the backend server
   - Use if you want to open the frontend manually

## How It Works

The application consists of two components:

```
┌─────────────────────┐
│   Frontend (UI)     │ ← Opens in your browser
│  index.html         │ ← Runs on your local machine
└──────────┬──────────┘
           │ HTTP requests
           ↓
┌─────────────────────┐
│  Backend (Server)   │ ← Python + FastAPI
│  main.py            │ ← Port 8001
└─────────────────────┘
```

**Both must be running for the application to work!**

## Three Ways to Start the Application

### Method 1: Automatic (Recommended)
```
Double-click: start_application.bat
```
- ✅ Automatically starts backend
- ✅ Opens frontend in browser  
- ✅ Shows clear status messages
- ✅ Most user-friendly

### Method 2: Manual (Backend Only)
```
Double-click: start_backend.bat
```
Then open `frontend/index.html` in your browser

### Method 3: Command Line (Advanced)
Open PowerShell/Command Prompt in the `backend` folder:
```powershell
python main.py
```

Then open `frontend/index.html` in your browser

## Troubleshooting

### ❌ "Backend server may not be running"
- **Solution**: Run `start_application.bat` or `start_backend.bat`
- The warning will disappear when the backend is running

### ❌ Backend won't start
- Make sure Python is installed: `python --version`
- Check that you're in the backend folder
- Verify all models are present in `backend/models/` folder
- Try: `python -m pip install -r requirements.txt`

### ✅ How to verify everything is working
1. Run `start_application.bat`
2. Backend window should show: `INFO: Uvicorn running on http://0.0.0.0:8001`
3. Frontend should load without the warning message
4. "Test Connection" button should show ✅

### ⚠️ Keep windows open
- Keep the backend window open while using the application
- Closing the backend window will stop the server

## System Requirements

- Python 3.8 or higher
- TensorFlow dependencies
- 2GB RAM minimum
- All model files in `backend/models/` folder

## Backend API Endpoints

Once the backend is running, you can:

- 📊 **View API Documentation**: http://localhost:8001/docs
- 💓 **Health Check**: http://localhost:8001/health
- 🎙️ **Make Predictions**: Send voice files to `http://localhost:8001/predict`

## Features

✨ **Multi-Modal Analysis**
- Voice pattern analysis
- Pitch and amplitude stability detection
- Harmonic-to-noise ratio measurement
- Real-time probability scoring

📈 **Visual Dashboard**
- Risk gauge visualization
- Feature breakdowns
- Model confidence metrics
- Diagnostic biomarkers

🔄 **Auto-Reconnection**
- Automatically detects when backend comes online/offline
- Shows helpful warnings with solutions
- No need to refresh the page

## Need Help?

If you encounter issues:
1. Check that both backend and frontend are running
2. Make sure port 8001 is not blocked by firewall
3. Verify all dependencies are installed
4. Check the browser console for error messages (F12 → Console tab)

---

Created as part of Parkinson's Detection ML-Model Project
