"""Load the portable, human-editable robot client configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "client.yaml"


def _resolve_path(value: Any, *, repo_root: Path) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = repo_root / expanded
    return str(expanded.resolve())


def load_client_config(path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"client config must contain a YAML mapping: {config_path}")

    required_sections = ("policy", "runtime", "startup", "action", "tunnel")
    missing = [name for name in required_sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"client config is missing mapping sections: {missing}")

    runtime = config["runtime"]
    for key, width in (("history_offsets", None), ("crop_percent", 3)):
        value = runtime.get(key)
        if not isinstance(value, list) or (width is not None and len(value) != width):
            suffix = "" if width is None else f" with {width} entries"
            raise ValueError(f"runtime.{key} must be a YAML list{suffix}")

    config["_config_path"] = str(config_path)
    config["runtime"]["record_head"] = _resolve_path(
        runtime.get("record_head"), repo_root=REPO_ROOT
    )
    config["startup"]["state_path"] = _resolve_path(
        config["startup"].get("state_path"), repo_root=REPO_ROOT
    )
    config["startup"]["reset_state_path"] = _resolve_path(
        config["startup"].get("reset_state_path"), repo_root=REPO_ROOT
    )
    return config


def _get(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one value from config/client.yaml")
    parser.add_argument("key", help="Dotted key, for example tunnel.local_port")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    value = _get(load_client_config(args.config), args.key)
    if isinstance(value, bool):
        print(str(value).lower())
    elif value is not None:
        print(value)


if __name__ == "__main__":
    main()
