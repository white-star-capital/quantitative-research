"""Locations of shared files outside the package."""

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2]


def contracts_dir() -> Path:
    """`contracts/` root. Override with TERMINAL_CONTRACTS_DIR (e.g. in a container image)."""
    return Path(os.environ.get("TERMINAL_CONTRACTS_DIR", MODELS_DIR.parent / "contracts"))


def library_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "library"


def registry_path() -> Path:
    return MODELS_DIR / "registry.yaml"
