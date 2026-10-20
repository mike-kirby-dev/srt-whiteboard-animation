#!/usr/bin/env python3
"""
Streaming pen-stroke animation - environment bootstrap script

Responsibilities:
  1. Create an isolated Python virtual environment under the skill directory (reuse it if it exists)
  2. Check that the required third-party libraries can be imported
  3. Automatically install any missing libraries
  4. Print ENV_PY=<interpreter path> on the last line for the caller to capture

Usage:
  python prepare_env.py          # create env + install dependencies, print ENV_PY
  python prepare_env.py --check  # probe only; exit non-zero if anything is missing
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

# skill root = two levels above this script
SKILL_ROOT = Path(__file__).resolve().parent.parent
VENV_ROOT = SKILL_ROOT / ".venv"

# import name -> pip package name
DEPS: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "av": "av",  # PyAV: pip-only H.264 encoding, no system ffmpeg needed
    "PIL": "Pillow",  # render_annotation_preview.py draws the numbered-region preview image (including labels)
}


def interpreter_path() -> Path:
    """Location of the python executable inside the virtual environment (cross-platform)."""
    if sys.platform.startswith("win"):
        return VENV_ROOT / "Scripts" / "python.exe"
    return VENV_ROOT / "bin" / "python"


def ensure_venv(check_only: bool) -> Path:
    py = interpreter_path()
    if VENV_ROOT.exists() and py.exists():
        print(f"[ok] Reusing existing virtual environment: {VENV_ROOT}")
        return py

    if check_only:
        print(f"[err] Virtual environment not created yet: {VENV_ROOT}")
        sys.exit(1)

    print(f"[..] Creating virtual environment: {VENV_ROOT}")
    venv.create(str(VENV_ROOT), with_pip=True)
    print("[ok] Virtual environment ready")
    return py


def can_import(py: Path, import_name: str) -> bool:
    probe = subprocess.run(
        [str(py), "-c", f"import {import_name}"],
        capture_output=True,
    )
    return probe.returncode == 0


def install(py: Path, packages: list[str]) -> bool:
    if not packages:
        return True
    print(f"[..] Installing dependencies: {', '.join(packages)}")
    res = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", *packages],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[err] Installation failed:\n{res.stderr}")
        return False
    print("[ok] Dependencies installed")
    return True


def main() -> None:
    check_only = "--check" in sys.argv

    py = ensure_venv(check_only)

    missing: list[str] = []
    for import_name, pip_name in DEPS.items():
        if can_import(py, import_name):
            print(f"[ok] {pip_name}")
        else:
            print(f"[miss] {pip_name}")
            missing.append(pip_name)

    if missing:
        if check_only:
            print(f"\nMissing {len(missing)} dependencies: {', '.join(missing)}")
            sys.exit(1)
        if not install(py, missing):
            sys.exit(1)

    # Last line: agreed output for the caller to capture
    print(f"\nENV_PY={py}")


if __name__ == "__main__":
    main()
