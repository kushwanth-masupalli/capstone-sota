"""
start_app.py - Unified Launcher for Dual-View RAG-VLM Clinical Platform
Launches both the FastAPI backend (port 8000) and the modern Vite/React frontend (port 3000) concurrently.
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("=" * 70)
    print("[*] Starting Dual-View RAG-VLM Clinical Diagnostic Platform")
    print("=" * 70)
    print(f"[*] Workspace Root: {BASE_DIR}")
    print(f"[*] Frontend Path : {FRONTEND_DIR}")
    print()

    # 1. Start FastAPI backend
    print("[1/2] Launching FastAPI backend server on http://127.0.0.1:8000 ...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    backend_proc = subprocess.Popen(backend_cmd, cwd=BASE_DIR)

    time.sleep(1.5)

    # 2. Start Vite frontend
    print("[2/2] Launching Vite React frontend on http://127.0.0.1:3000 ...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "3000"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR)

    print()
    print("=" * 70)
    print("[+] Dual-View RAG-VLM Platform is LIVE!")
    print("    * Modern Web Console:  http://127.0.0.1:3000/")
    print("    * Backend API & Docs:  http://127.0.0.1:8000/docs")
    print("=" * 70)
    print("Press Ctrl+C to terminate both servers.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down servers gracefully...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("[*] All servers stopped.")

if __name__ == "__main__":
    main()
