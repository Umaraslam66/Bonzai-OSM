"""pytest conftest at the project root.

Inserts src/ on sys.path so editable-install path issues with newer
pip/setuptools don't block test discovery. Pure dev-time fallback.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
