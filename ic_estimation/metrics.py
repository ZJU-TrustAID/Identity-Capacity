from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateMetrics:
    num_trials: int
    raw_bit_accuracy: float
    decoded_bit_accuracy: float
    identity_accuracy: float
    identity_failure_rate: float
    reject_rate: float
    misattribution_rate: float
    pass_alpha: bool
    bit_diagnostics: dict[str, Any]
    identity_failure_ci95: dict[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_trials": self.num_trials,
            "raw_bit_accuracy": self.raw_bit_accuracy,
            "decoded_bit_accuracy": self.decoded_bit_accuracy,
            "identity_accuracy": self.identity_accuracy,
            "identity_failure_rate": self.identity_failure_rate,
            "reject_rate": self.reject_rate,
            "misattribution_rate": self.misattribution_rate,
            "pass_alpha": self.pass_alpha,
            "bit_diagnostics": self.bit_diagnostics,
            "identity_failure_ci95": self.identity_failure_ci95,
        }


def bit_diagnostics(truth: np.ndarray, detected: np.ndarray) -> dict[str, Any]:
    truth_bits = np.asarray(truth, dtype=np.uint8)
    detected_bits = np.asarray(detected, dtype=np.uint8)
    if truth_bits.shape != detected_bits.shape:
        raise ValueError("truth and detected matrices must have the same shape")
    if truth_bits.ndim != 2:
        raise ValueError("Expected matrices with shape [num_trials, nbits]")
    error = (truth_bits != detected_bits).astype(np.uint8)
    rows, nbits = error.shape
    per_bit_ber = error.mean(axis=0) if rows else np.zeros(nbits, dtype=np.float64)
    error_counts = error.sum(axis=1) if rows else np.zeros(0, dtype=np.int64)
    overall_ber = float(error.mean()) if rows and nbits else 0.0
    return {
        "num_rows": int(rows),
        "num_bits": int(nbits),
        "overall_ber": overall_ber,
        "mean_bit_ber": float(np.mean(per_bit_ber)) if nbits else 0.0,
        "min_bit_ber": float(np.min(per_bit_ber)) if nbits else 0.0,
        "max_bit_ber": float(np.max(per_bit_ber)) if nbits else 0.0,
        "per_bit_ber": per_bit_ber.tolist(),
        "error_count_mean": float(np.mean(error_counts)) if rows else 0.0,
        "error_count_variance": float(np.var(error_counts, ddof=1)) if rows > 1 else 0.0,
        "error_count_histogram": np.bincount(error_counts.astype(np.int64), minlength=nbits + 1).tolist()
        if nbits
        else [],
    }


def compute_candidate_metrics(
    *,
    truth_codeword: np.ndarray,
    decoded_codeword: np.ndarray,
    truth_info: np.ndarray,
    decoded_info: np.ndarray,
    identity_success: np.ndarray,
    rejected: np.ndarray,
    misattributed: np.ndarray,
    alpha: float,
) -> CandidateMetrics:
    success = np.asarray(identity_success, dtype=bool)
    reject = np.asarray(rejected, dtype=bool)
    misattr = np.asarray(misattributed, dtype=bool)
    num_trials = int(success.size)
    raw_acc = float(np.mean(np.asarray(truth_codeword) == np.asarray(decoded_codeword))) if num_trials else 0.0
    decoded_acc = float(np.mean(np.asarray(truth_info) == np.asarray(decoded_info))) if num_trials else 0.0
    identity_acc = float(np.mean(success)) if num_trials else 0.0
    failure = 1.0 - identity_acc
    failures = int(np.count_nonzero(~success)) if num_trials else 0
    return CandidateMetrics(
        num_trials=num_trials,
        raw_bit_accuracy=raw_acc,
        decoded_bit_accuracy=decoded_acc,
        identity_accuracy=identity_acc,
        identity_failure_rate=failure,
        reject_rate=float(np.mean(reject)) if num_trials else 0.0,
        misattribution_rate=float(np.mean(misattr)) if num_trials else 0.0,
        pass_alpha=bool(failure <= float(alpha)),
        bit_diagnostics=bit_diagnostics(truth_codeword, decoded_codeword),
        identity_failure_ci95=wilson_score_interval(failures, num_trials),
    )


def wilson_score_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | str]:
    n = int(total)
    if n <= 0:
        return {"method": "wilson", "confidence": 0.95, "lower": 0.0, "upper": 1.0}
    x = max(0, min(int(successes), n))
    phat = x / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denominator
    half_width = float(z) * sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if lower < 1e-15:
        lower = 0.0
    if 1.0 - upper < 1e-15:
        upper = 1.0
    return {
        "method": "wilson",
        "confidence": 0.95,
        "lower": lower,
        "upper": upper,
    }
