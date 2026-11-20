"""Location of the shipped YAML assets in a wheel or source checkout."""

from pathlib import Path

_package_dir = Path(__file__).resolve().parent
CONFIG_DIR = _package_dir / "configs"
if not CONFIG_DIR.is_dir():
    CONFIG_DIR = _package_dir.parent / "configs"
