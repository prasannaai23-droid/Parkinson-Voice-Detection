#!/usr/bin/env python3
"""
Auto-start script for Parkinson's Detection Application
This script automatically starts the backend server and opens the frontend
"""

import os
import sys
import subprocess
import time
import webbrowser
import platform
from pathlib import Path

def main():
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir / "backend"
    frontend_file = script_dir / "frontend" / "index.html"
    
    print("\n" + "="*60)
    print("Parkinson's Detection - Auto-Start Application")
    print("="*60 + "\n")
    
    # Verify backend exists
    main_py = backend_dir / "main.py"
    if not main_py.exists():
        print(f"❌ Error: Backend not found at {main_py}")
        print("Please ensure the backend folder structure is correct.")
        input("Press Enter to exit...")
        return 1
    
    # Verify frontend exists
    if not frontend_file.exists():
        print(f"❌ Error: Frontend not found at {frontend_file}")
        print("Please ensure the frontend folder structure is correct.")
        input("Press Enter to exit...")
        return 1
    
    # Check if backend is already running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    backend_running = sock.connect_ex(('localhost', 8001)) == 0
    sock.close()
    
    if backend_running:
        print("✓ Backend server is already running on http://localhost:8001")
    else:
        print("📍 Starting backend server...")
        print("   Location:", backend_dir / "main.py")
        print("   Python:", sys.executable)
        print()
        
        try:
            # Start backend as a subprocess
            if platform.system() == "Windows":
                # On Windows, create a new window for the backend
                backend_process = subprocess.Popen(
                    [sys.executable, str(main_py)],
                    cwd=str(backend_dir),
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # On Unix/Linux/Mac
                backend_process = subprocess.Popen(
                    [sys.executable, str(main_py)],
                    cwd=str(backend_dir)
                )
            
            print("✓ Backend process started (PID: {})".format(backend_process.pid))
            print("   Waiting 5 seconds for server to initialize...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            input("Press Enter to exit...")
            return 1
    
    # Open frontend in browser
    print("\n📍 Opening frontend application...")
    frontend_url = frontend_file.as_uri() if platform.system() != "Windows" else f"file:///{frontend_file.as_posix()}"
    
    try:
        webbrowser.open(frontend_url)
        print(f"✓ Frontend opened: {frontend_url}")
    except Exception as e:
        print(f"⚠ Could not open browser automatically: {e}")
        print(f"  Please open manually: {frontend_url}")
    
    print("\n" + "="*60)
    print("✓ Application is ready!")
    print("="*60)
    print(f"\n✓ Backend:  http://localhost:8001")
    print(f"✓ API Docs: http://localhost:8001/docs\n")
    
    print("⚠ Note: The backend is running in a separate process.")
    print("   To stop the application, close this window and the backend window.\n")
    
    input("Press Enter to exit the startup script (backend will continue running)...")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Application startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
