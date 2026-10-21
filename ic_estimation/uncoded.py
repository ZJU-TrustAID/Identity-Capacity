from __future__ import annotations

from dataclasses import dataclass
from math import pow
from typing import Any

import numpy as np

from ic_estimation.bch import DecodeOutcome
from ic_estimation.metrics import wilson_score_interval


@dataclass(frozen=True)
class UncodedL2Selection:
    k: int
    failure_probability: float

    @property
    def n(self) -> int:
        return int(self.k)

    @property
    def t(self) -> int:
        return 0

    @property
    def redundancy(self) -> int:
        return 0

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "uncoded",
            "n": self.n,
            "k": int(self.k),
            "t": self.t,
            "redundancy": self.redundancy,
            "active_positions": list(range(self.n)),
            "failure_probability": float(self.failure_probability),
            "identity_count": self.identity_count,
            "log2_identity_count": int(self.k),
        }


@dataclass(frozen=True)
class UncodedL3Selection:
    k: int
    failure_rate: float
    num_trials: int
    num_failures: int
    identity_failure_ci95: dict[str, float | str]

    @property
    def n(self) -> int:
        return int(self.k)

    @property
    def t(self) -> int:
        return 0

    @property
    def redundancy(self) -> int:
        return 0

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "uncoded",
            "n": self.n,
            "k": int(self.k),
            "t": self.t,
            "redundancy": self.redundancy,
            "active_positions": list(range(self.n)),
            "failure_rate": float(self.failure_rate),
            "identity_failure_rate": float(self.failure_rate),
            "num_trials": int(self.num_trials),
            "num_failures": int(self.num_failures),
            "identity_failure_ci95": self.identity_failure_ci95,
            "identity_count": self.identity_count,
            "log2_identity_count": int(self.k),
        }


def uncoded_l2_failure_probability(k: int, bit_error_rate: float) -> float:
    k_int = max(0, int(k))
    p = min(1.0, max(0.0, float(bit_error_rate)))
    return float(max(0.0, min(1.0, 1.0 - pow(1.0 - p, k_int))))


def select_uncoded_l2_capacity(
    payload_length: int,
    bit_error_rate: float,
    alpha: float,
) -> UncodedL2Selection | None:
    for k in range(int(payload_length), 0, -1):
        failure = uncoded_l2_failure_probability(k, bit_error_rate)
        if failure <= float(alpha):
            return UncodedL2Selection(k=k, failure_probability=failure)
    return None


@dataclass(frozen=True)
class UncodedCode:
    k: int
    active_positions: tuple[int, ...]
    family: str = "uncoded"

    @classmethod
    def for_k(cls, k: int, active_positions: list[int] | tuple[int, ...] | None = None) -> "UncodedCode":
        k_int = int(k)
        if k_int < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        positions = tuple(range(k_int)) if active_positions is None else tuple(int(pos) for pos in active_positions)
        if len(positions) != k_int:
            raise ValueError(f"active_positions must have length {k_int}, got {len(positions)}")
        if len(set(positions)) != len(positions):
            raise ValueError("active_positions must be unique")
        return cls(k=k_int, active_positions=positions)

    @property
    def n(self) -> int:
        return int(self.k)

    @property
    def t(self) -> int:
        return 0

    @property
    def redundancy(self) -> int:
        return 0

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n": self.n,
            "k": int(self.k),
            "t": self.t,
            "redundancy": self.redundancy,
            "active_positions": list(self.active_positions),
            "identity_count": self.identity_count,
            "log2_identity_count": int(self.k),
        }

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if info.size != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {info.size}")
        if np.any((info != 0) & (info != 1)):
            raise ValueError("info bits must be binary")
        return info.copy()

    def decode(self, codeword_bits: np.ndarray) -> DecodeOutcome:
        received = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if received.size != self.k:
            raise ValueError(f"Expected {self.k} codeword bits, got {received.size}")
        if np.any((received != 0) & (received != 1)):
            raise ValueError("codeword bits must be binary")
        return DecodeOutcome(
            status="decoded",
            info_bits=received.copy(),
            codeword_bits=received.copy(),
            num_corrected_errors=0,
            error_positions=(),
        )


def uncoded_candidates(
    payload_length: int,
    *,
    min_k: int = 1,
    max_k: int | None = None,
) -> list[UncodedCode]:
    payload = int(payload_length)
    min_k_int = int(min_k)
    max_k_int = payload if max_k is None else min(payload, int(max_k))
    if max_k_int < min_k_int:
        return []
    return [UncodedCode.for_k(k) for k in range(max_k_int, min_k_int - 1, -1)]


def uncoded_l3_failure_rate(truth: np.ndarray, decoded: np.ndarray, k: int) -> tuple[float, int, int]:
    truth_bits = np.asarray(truth, dtype=np.uint8)
    decoded_bits = np.asarray(decoded, dtype=np.uint8)
    if truth_bits.shape != decoded_bits.shape:
        raise ValueError("truth and decoded matrices must have the same shape")
    if truth_bits.ndim != 2:
        raise ValueError("Expected matrices with shape [num_trials, nbits]")
    k_int = int(k)
    if k_int < 1 or k_int > truth_bits.shape[1]:
        raise ValueError(f"k must be in [1, {truth_bits.shape[1]}], got {k}")
    failures = np.any(truth_bits[:, :k_int] != decoded_bits[:, :k_int], axis=1)
    num_trials = int(failures.size)
    num_failures = int(np.count_nonzero(failures))
    failure_rate = float(num_failures / num_trials) if num_trials else 0.0
    return failure_rate, num_failures, num_trials


def select_uncoded_l3_capacity(
    truth: np.ndarray,
    decoded: np.ndarray,
    alpha: float,
) -> UncodedL3Selection | None:
    truth_bits = np.asarray(truth, dtype=np.uint8)
    decoded_bits = np.asarray(decoded, dtype=np.uint8)
    if truth_bits.shape != decoded_bits.shape:
        raise ValueError("truth and decoded matrices must have the same shape")
    if truth_bits.ndim != 2:
        raise ValueError("Expected matrices with shape [num_trials, nbits]")
    payload_length = int(truth_bits.shape[1])
    for k in range(payload_length, 0, -1):
        failure_rate, num_failures, num_trials = uncoded_l3_failure_rate(truth_bits, decoded_bits, k)
        if failure_rate <= float(alpha):
            return UncodedL3Selection(
                k=k,
                failure_rate=failure_rate,
                num_trials=num_trials,
                num_failures=num_failures,
                identity_failure_ci95=wilson_score_interval(num_failures, num_trials),
            )
    return None
