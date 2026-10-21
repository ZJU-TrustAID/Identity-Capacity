from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_plain(row), sort_keys=True) + "\n")


def load_bits(path: str | Path) -> np.ndarray:
    bits = np.load(Path(path))
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError(f"Expected binary bits in {path}")
    return flat


def save_bits(path: str | Path, bits: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(bits, dtype=np.uint8).reshape(-1))


def normalize_binary_vector(values: Any, length: int | None = None) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected a binary vector")
    if length is not None:
        if bits.size < int(length):
            padded = np.zeros(int(length), dtype=np.uint8)
            padded[: bits.size] = bits
            bits = padded
        elif bits.size > int(length):
            bits = bits[: int(length)]
    return bits.astype(np.uint8, copy=False)
