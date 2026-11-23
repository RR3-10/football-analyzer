"""Load the YAML configuration file."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config file into a nested dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)