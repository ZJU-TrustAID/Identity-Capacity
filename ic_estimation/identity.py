from __future__ import annotations

import hashlib

import numpy as np


def bits_from_key(key: str, nbits: int) -> np.ndarray:
    needed = (int(nbits) + 7) // 8
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).digest()
    while len(digest) < needed:
        digest += hashlib.sha1(digest).digest()

    bits: list[int] = []
    for byte in digest[:needed]:
        for bit_idx in range(8):
            bits.append((byte >> bit_idx) & 1)
    return np.asarray(bits[: int(nbits)], dtype=np.uint8)


def identity_bits(uid: str, nbits: int, seed: int, trial_index: int, mode: str = "uid_trial") -> np.ndarray:
    normalized_mode = str(mode or "uid_trial").strip().lower()
    if normalized_mode == "uid":
        key = f"{int(seed)}|{uid}"
    elif normalized_mode in {"uid_trial", "identity_per_trial"}:
        key = f"{int(seed)}|{uid}|trial={int(trial_index)}"
    elif normalized_mode == "random":
        key = f"{int(seed)}|random|{uid}|trial={int(trial_index)}"
    else:
        raise ValueError(f"Unsupported identity bit mode: {mode!r}")
    return bits_from_key(key, int(nbits))


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for idx, bit in enumerate(np.asarray(bits, dtype=np.uint8).reshape(-1)):
        value |= int(bit) << idx
    return int(value)


def int_to_bits(value: int, nbits: int) -> np.ndarray:
    return np.asarray([(int(value) >> idx) & 1 for idx in range(int(nbits))], dtype=np.uint8)
