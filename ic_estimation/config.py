from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import json5


_ENV_PATTERN = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?$")
_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


class ConfigNode(dict):
    """Dictionary with attribute-style access for loaded configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({str(key): cls.from_mapping(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.from_mapping(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(self)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        current = json5.load(handle)
    base_ref = current.get("base_config")
    if not base_ref:
        return current

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    if base_path.name == "paths.example.jsonc" and config_path.name != "paths.jsonc":
        local_path = base_path.with_name("paths.jsonc")
        if local_path.exists():
            base_path = local_path
    base_cfg = _load_raw_config(base_path)
    return _merge_dicts(base_cfg, current)


def _lookup(config: dict[str, Any], path: str) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def _resolve_token(token: str, root: dict[str, Any]) -> Any:
    env_match = _ENV_PATTERN.match(token)
    if env_match:
        env_name, default = env_match.groups()
        value = os.environ.get(env_name)
        if value is None:
            if default is None:
                raise KeyError(f"Environment variable {env_name} is not set")
            return default
        return value
    return _lookup(root, token)


def _interpolate_value(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _interpolate_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(item, root) for item in value]
    if not isinstance(value, str):
        return value

    matches = list(_REF_PATTERN.finditer(value))
    if not matches:
        return value
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return _interpolate_value(_resolve_token(matches[0].group(1), root), root)

    result = value
    for match in matches:
        token_value = _resolve_token(match.group(1), root)
        result = result.replace(match.group(0), str(token_value))
    return result


def _interpolate_config(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(raw)
    for _ in range(12):
        updated = _interpolate_value(resolved, resolved)
        if updated == resolved:
            return updated
        resolved = updated
    raise ValueError("Config interpolation did not converge")


def load_config(config_path: str | os.PathLike[str]) -> ConfigNode:
    path = Path(config_path).expanduser().resolve()
    raw = _load_raw_config(path)
    raw.setdefault("config", {})
    raw["config"]["config_path"] = str(path)
    raw["config"]["config_dir"] = str(path.parent)
    interpolated = _interpolate_config(raw)
    return ConfigNode.from_mapping(interpolated)
