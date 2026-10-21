from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from ic_estimation.bch import BCHMotherCode, ShortenedBCHCode, primitive_bch_mothers


@dataclass(frozen=True)
class L2Selection:
    code: ShortenedBCHCode
    failure_probability: float

    @property
    def n(self) -> int:
        return self.code.n

    @property
    def k(self) -> int:
        return self.code.k

    @property
    def t(self) -> int:
        return self.code.t

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.code.to_dict(),
            "failure_probability": self.failure_probability,
            "identity_count": self.identity_count,
            "log2_identity_count": self.k,
        }


def binomial_failure_probability(n: int, t: int, p: float) -> float:
    n_int = int(n)
    t_int = int(t)
    p_float = min(1.0, max(0.0, float(p)))
    success = 0.0
    for errors in range(0, min(t_int, n_int) + 1):
        success += comb(n_int, errors) * (p_float**errors) * ((1.0 - p_float) ** (n_int - errors))
    return float(max(0.0, min(1.0, 1.0 - success)))


def shortened_candidates(
    payload_length: int,
    mothers: list[BCHMotherCode] | None = None,
    *,
    min_k: int = 1,
    max_k: int | None = None,
) -> list[ShortenedBCHCode]:
    payload = int(payload_length)
    min_k_int = int(min_k)
    max_k_int = None if max_k is None else int(max_k)
    code_mothers = mothers if mothers is not None else primitive_bch_mothers(max_m=10)
    candidates: list[ShortenedBCHCode] = []
    for mother in code_mothers:
        mother_max_k = min(mother.k, payload - mother.redundancy)
        if max_k_int is not None:
            mother_max_k = min(mother_max_k, max_k_int)
        if mother_max_k < min_k_int:
            continue
        for k in range(mother_max_k, min_k_int - 1, -1):
            n = k + mother.redundancy
            if n <= payload:
                candidates.append(
                    ShortenedBCHCode.from_mother(mother, k=k, active_positions=list(range(n)))
                )
    candidates.sort(key=lambda code: (-code.k, code.n, code.t, code.mother.n))
    return candidates


def select_l2_capacity(
    payload_length: int,
    bit_error_rate: float,
    alpha: float,
    mothers: list[BCHMotherCode] | None = None,
) -> L2Selection | None:
    best: L2Selection | None = None
    for code in shortened_candidates(payload_length, mothers):
        failure = binomial_failure_probability(code.n, code.t, bit_error_rate)
        if failure > float(alpha):
            continue
        selection = L2Selection(code=code, failure_probability=failure)
        if best is None:
            best = selection
        elif selection.k > best.k:
            best = selection
        elif selection.k == best.k and selection.failure_probability < best.failure_probability:
            best = selection
    return best


def select_l2_capacity_from_candidates(
    candidates: list[Any],
    bit_error_rate: float,
    alpha: float,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for candidate in candidates:
        failure = binomial_failure_probability(int(candidate.n), int(candidate.t), bit_error_rate)
        if failure > float(alpha):
            continue
        payload = {
            **candidate.to_dict(),
            "failure_probability": failure,
            "identity_count": int(candidate.identity_count),
            "log2_identity_count": int(candidate.k),
        }
        if best is None:
            best = payload
        elif int(payload["k"]) > int(best["k"]):
            best = payload
        elif int(payload["k"]) == int(best["k"]) and float(payload["failure_probability"]) < float(
            best["failure_probability"]
        ):
            best = payload
    return best


def monte_carlo_failure_probability(
    candidate: Any,
    bit_error_rate: float,
    *,
    num_trials: int = 10000,
    seed: int = 1453,
) -> dict[str, object]:
    trial_count = max(1, int(num_trials))
    p = min(1.0, max(0.0, float(bit_error_rate)))
    rng = np.random.default_rng(int(seed))
    num_failures = 0
    num_rejections = 0
    num_misattributions = 0
    for _trial_index in range(trial_count):
        info = rng.integers(0, 2, size=int(candidate.k), dtype=np.uint8)
        codeword = candidate.encode(info)
        errors = (rng.random(int(candidate.n)) < p).astype(np.uint8)
        received = np.bitwise_xor(codeword, errors).astype(np.uint8)
        outcome = candidate.decode(received)
        rejected = outcome.status != "decoded" or outcome.info_bits is None
        misattributed = bool(
            outcome.status == "decoded"
            and outcome.info_bits is not None
            and not np.array_equal(np.asarray(outcome.info_bits, dtype=np.uint8), info)
        )
        if rejected:
            num_rejections += 1
        if misattributed:
            num_misattributions += 1
        if rejected or misattributed:
            num_failures += 1
    return {
        "prediction_method": "bsc_monte_carlo",
        "failure_probability": float(num_failures / trial_count),
        "num_trials": int(trial_count),
        "num_failures": int(num_failures),
        "num_rejections": int(num_rejections),
        "num_misattributions": int(num_misattributions),
        "bit_error_rate": float(p),
        "seed": int(seed),
    }


def select_l2_capacity_monte_carlo_from_candidates(
    candidates: list[Any],
    bit_error_rate: float,
    alpha: float,
    *,
    num_trials: int = 10000,
    seed: int = 1453,
    start_k: int | None = None,
    worker_count: int = 1,
) -> dict[str, object] | None:
    if not candidates:
        return None

    grouped: dict[int, list[tuple[int, Any]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        grouped.setdefault(int(candidate.k), []).append((candidate_index, candidate))
    available_ks = sorted(grouped)

    actual_start_k = start_k
    if actual_start_k is None:
        if any(int(getattr(candidate, "t", 0)) > 0 for candidate in candidates):
            t_selection = select_l2_capacity_from_candidates(candidates, bit_error_rate, alpha)
            if isinstance(t_selection, dict) and t_selection.get("k") is not None:
                actual_start_k = int(t_selection["k"])
    if actual_start_k is None:
        actual_start_k = available_ks[-1]
    if int(actual_start_k) not in grouped:
        actual_start_k = min(available_ks, key=lambda value: (abs(value - int(actual_start_k)), -value))

    max_workers = max(1, int(worker_count))
    searched_ks: list[int] = []
    candidates_evaluated = 0

    def evaluate_assignment(candidate_index: int, candidate: Any) -> tuple[int, Any, dict[str, object]]:
        return (
            int(candidate_index),
            candidate,
            monte_carlo_failure_probability(
                candidate,
                bit_error_rate,
                num_trials=num_trials,
                seed=int(seed) + int(candidate_index),
            ),
        )

    def evaluate_k(k_value: int) -> dict[str, object] | None:
        nonlocal candidates_evaluated
        searched_ks.append(int(k_value))
        assignments = grouped[int(k_value)]
        candidates_evaluated += len(assignments)

        completed: list[tuple[int, Any, dict[str, object]]] = []
        if max_workers == 1 or len(assignments) == 1:
            completed = [evaluate_assignment(candidate_index, candidate) for candidate_index, candidate in assignments]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(assignments))) as executor:
                futures = [
                    executor.submit(evaluate_assignment, candidate_index, candidate)
                    for candidate_index, candidate in assignments
                ]
                for future in as_completed(futures):
                    completed.append(future.result())
        completed.sort(key=lambda item: item[0])

        best_at_k: dict[str, object] | None = None
        for _candidate_index, candidate, result in completed:
            failure = float(result["failure_probability"])
            if failure > float(alpha):
                continue
            payload = {
                **candidate.to_dict(),
                "failure_probability": failure,
                "identity_count": int(candidate.identity_count),
                "log2_identity_count": int(candidate.k),
                "l2_estimator": "monte_carlo",
                "monte_carlo": result,
            }
            if best_at_k is None or float(payload["failure_probability"]) < float(best_at_k["failure_probability"]):
                best_at_k = payload
        return best_at_k

    best = evaluate_k(int(actual_start_k))
    if best is not None:
        for k_value in [value for value in available_ks if value > int(actual_start_k)]:
            candidate = evaluate_k(k_value)
            if candidate is None:
                break
            best = candidate
    else:
        for k_value in [value for value in reversed(available_ks) if value < int(actual_start_k)]:
            candidate = evaluate_k(k_value)
            if candidate is not None:
                best = candidate
                break

    if best is None:
        return None

    return {
        **best,
        "search_start_k": int(actual_start_k),
        "searched_ks": searched_ks,
        "candidates_evaluated": int(candidates_evaluated),
        "worker_count": int(max_workers),
    }
