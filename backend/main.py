"""
PARKINSON'S DISEASE VOICE DETECTION BACKEND (MAIN ENTRY POINT)
==============================================================
"""

from main_advanced import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
