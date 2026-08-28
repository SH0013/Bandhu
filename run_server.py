"""Launcher script for Bandhu Agent Platform."""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add src/ to python path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import uvicorn
from bandhu.config import settings

if __name__ == "__main__":
    print(f"🚀 Starting Bandhu on http://localhost:{settings.port} ...")
    uvicorn.run(
        "bandhu.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(SRC_DIR),
    )
