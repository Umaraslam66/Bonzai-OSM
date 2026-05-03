"""Run prepare_tiles from the repo root with a sane default output path."""
import sys
from pathlib import Path

# Editable-install fallback: add src/ to sys.path so import works even
# when pip's .pth file isn't honoured (see conftest.py for why).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.cli.prepare_tiles import app  # noqa: E402

if __name__ == "__main__":
    app()
