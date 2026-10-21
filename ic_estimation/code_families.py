from __future__ import annotations

from typing import Any

from ic_estimation.l2 import shortened_candidates
from ic_estimation.rm import shortened_rm_candidates
from ic_estimation.uncoded import uncoded_candidates


CODE_FAMILY_CHOICES = ("bch", "uncoded_bch", "rm", "uncoded_rm")


def normalize_code_family(value: str | None) -> str:
    normalized = str(value or "bch").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "bch": "bch",
        "uncode_bch": "uncoded_bch",
        "uncoded_bch": "uncoded_bch",
        "raw_bch": "uncoded_bch",
        "raw_uncoded_bch": "uncoded_bch",
        "rm": "rm",
        "reed_muller": "rm",
        "shortened_rm": "rm",
        "uncode_rm": "uncoded_rm",
        "uncoded_rm": "uncoded_rm",
        "raw_rm": "uncoded_rm",
        "raw_uncoded_rm": "uncoded_rm",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown code_family {value!r}; choices: {', '.join(CODE_FAMILY_CHOICES)}")
    return aliases[normalized]


def build_code_candidates(
    *,
    payload_length: int,
    code_family: str | None = "bch",
    min_k: int = 1,
    max_k: int | None = None,
    rm_options: dict[str, Any] | None = None,
) -> list[Any]:
    family = normalize_code_family(code_family)
    candidates: list[Any] = []
    if family in {"uncoded_bch", "uncoded_rm"}:
        candidates.extend(uncoded_candidates(payload_length, min_k=min_k, max_k=max_k))
    if family in {"bch", "uncoded_bch"}:
        candidates.extend(shortened_candidates(payload_length, min_k=min_k, max_k=max_k))
    elif family in {"rm", "uncoded_rm"}:
        candidates.extend(
            shortened_rm_candidates(
                payload_length,
                min_k=min_k,
                max_k=max_k,
                **dict(rm_options or {}),
            )
        )
    candidates.sort(key=lambda code: (-int(code.k), int(code.n), int(code.t), str(code.family)))
    return candidates
