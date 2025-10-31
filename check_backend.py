#!/usr/bin/env python3
"""
Simple health check script for backend API
"""
import requests
import sys

try:
    response = requests.get("http://127.0.0.1:8000/health", timeout=5)
    if response.status_code == 200 and response.json().get("status") == "ok":
        print("✅ Backend is running and healthy!")
        print(f"📍 URL: http://127.0.0.1:8000")
        print(f"📚 Docs: http://127.0.0.1:8000/docs")
        sys.exit(0)
    else:
        print(f"⚠️ Backend responded but status unclear: {response.json()}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to backend at http://127.0.0.1:8000")
    print("💡 Please run: uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
