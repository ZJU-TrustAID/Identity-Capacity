from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import combinations
from typing import Any

import numpy as np

from ic_estimation.bch import DecodeOutcome


DEFAULT_MAX_M = 8
DEFAULT_MAX_EXHAUSTIVE_K = 16
DEFAULT_MAX_SYNDROME_PATTERNS = 250_000
RM_DECODER_TYPE = "mother_majority"


@dataclass(frozen=True)
class RMMotherCode:
    r: int
    m: int
    n: int
    k: int
    distance: int
    generator: np.ndarray
    family: str = "reed_muller"

    @property
    def t(self) -> int:
        return int((self.distance - 1) // 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "r": int(self.r),
            "m": int(self.m),
            "n": int(self.n),
            "k": int(self.k),
            "distance": int(self.distance),
            "t": self.t,
        }

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if info.size != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {info.size}")
        if np.any((info != 0) & (info != 1)):
            raise ValueError("info bits must be binary")
        return np.mod(info @ self.generator, 2).astype(np.uint8)

    def decode(self, codeword_bits: np.ndarray) -> DecodeOutcome:
        received = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if received.size != self.n:
            raise ValueError(f"Expected {self.n} codeword bits, got {received.size}")
        if np.any((received != 0) & (received != 1)):
            raise ValueError("codeword bits must be binary")
        return _decode_rm_majority(received, self)


@dataclass
class ShortenedRMCode:
    mother: RMMotherCode
    k: int
    n: int
    t: int
    generator: np.ndarray
    active_positions: tuple[int, ...]
    shortened_positions: tuple[int, ...]
    decoder_type: str
    family: str = "shortened_reed_muller"

    @property
    def redundancy(self) -> int:
        return int(self.n - self.k)

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    @property
    def mother_n(self) -> int:
        return int(self.mother.n)

    @property
    def mother_k(self) -> int:
        return int(self.mother.k)

    @property
    def mother_r(self) -> int:
        return int(self.mother.r)

    @property
    def mother_m(self) -> int:
        return int(self.mother.m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "mother": self.mother.to_dict(),
            "mother_n": self.mother_n,
            "mother_k": self.mother_k,
            "mother_r": self.mother_r,
            "mother_m": self.mother_m,
            "n": int(self.n),
            "k": int(self.k),
            "t": int(self.t),
            "redundancy": self.redundancy,
            "distance_lower_bound": int(self.mother.distance),
            "active_positions": list(self.active_positions),
            "shortened_positions": list(self.shortened_positions),
            "decoder_type": self.decoder_type,
            "identity_count": self.identity_count,
            "log2_identity_count": int(self.k),
        }

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if info.size != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {info.size}")
        if np.any((info != 0) & (info != 1)):
            raise ValueError("info bits must be binary")
        return np.mod(info @ self.generator, 2).astype(np.uint8)

    def decode(self, codeword_bits: np.ndarray) -> DecodeOutcome:
        received = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if received.size != self.n:
            raise ValueError(f"Expected {self.n} codeword bits, got {received.size}")
        if np.any((received != 0) & (received != 1)):
            raise ValueError("codeword bits must be binary")
        if self.decoder_type != RM_DECODER_TYPE:
            raise ValueError(f"Unsupported RM decoder type: {self.decoder_type}")
        full_received = np.zeros(self.mother_n, dtype=np.uint8)
        full_received[list(self.active_positions)] = received
        mother_outcome = self.mother.decode(full_received)
        if mother_outcome.status != "decoded":
            return DecodeOutcome(
                status="rejected",
                info_bits=None,
                codeword_bits=received.copy(),
                num_corrected_errors=mother_outcome.num_corrected_errors,
                error_positions=(),
            )
        full_corrected = mother_outcome.codeword_bits
        if self.shortened_positions and np.any(full_corrected[list(self.shortened_positions)]):
            return DecodeOutcome(
                status="rejected",
                info_bits=None,
                codeword_bits=full_corrected[list(self.active_positions)].copy(),
                num_corrected_errors=mother_outcome.num_corrected_errors,
                error_positions=(),
            )
        corrected = full_corrected[list(self.active_positions)].copy()
        distance = int(np.count_nonzero(corrected != received))
        if distance > self.t:
            return DecodeOutcome(
                status="rejected",
                info_bits=None,
                codeword_bits=corrected,
                num_corrected_errors=distance,
                error_positions=tuple(np.flatnonzero(corrected != received).astype(int).tolist()),
            )
        pivots, inverse = self._message_solver
        info = np.mod(corrected[list(pivots)] @ inverse, 2).astype(np.uint8)
        if not np.array_equal(self.encode(info), corrected):
            return DecodeOutcome(
                status="rejected",
                info_bits=None,
                codeword_bits=corrected,
                num_corrected_errors=distance,
                error_positions=tuple(np.flatnonzero(corrected != received).astype(int).tolist()),
            )
        return DecodeOutcome(
            status="decoded",
            info_bits=info,
            codeword_bits=corrected,
            num_corrected_errors=distance,
            error_positions=tuple(np.flatnonzero(corrected != received).astype(int).tolist()),
        )

    @cached_property
    def _message_solver(self) -> tuple[tuple[int, ...], np.ndarray]:
        pivots = _independent_columns(self.generator, self.k)
        square = self.generator[:, list(pivots)]
        inverse = _gf2_inverse(square)
        return pivots, inverse


def reed_muller_mothers(max_m: int = DEFAULT_MAX_M) -> list[RMMotherCode]:
    mothers: list[RMMotherCode] = []
    for m in range(1, int(max_m) + 1):
        n = 2**m
        for r in range(0, m + 1):
            distance = 2 ** (m - r)
            t = (distance - 1) // 2
            if t < 1:
                continue
            generator = _rm_generator(r, m)
            mothers.append(
                RMMotherCode(
                    r=r,
                    m=m,
                    n=n,
                    k=int(generator.shape[0]),
                    distance=distance,
                    generator=generator,
                )
            )
    mothers.sort(key=lambda code: (code.n, code.r, code.k))
    return mothers


def shortened_rm_candidates(
    payload_length: int,
    mothers: list[RMMotherCode] | None = None,
    *,
    min_k: int = 1,
    max_k: int | None = None,
    max_m: int = DEFAULT_MAX_M,
    max_exhaustive_k: int = DEFAULT_MAX_EXHAUSTIVE_K,
    max_syndrome_patterns: int = DEFAULT_MAX_SYNDROME_PATTERNS,
) -> list[ShortenedRMCode]:
    _ = (max_exhaustive_k, max_syndrome_patterns)
    payload = int(payload_length)
    min_k_int = int(min_k)
    max_k_int = None if max_k is None else int(max_k)
    code_mothers = mothers if mothers is not None else reed_muller_mothers(max_m=max_m)
    candidates: list[ShortenedRMCode] = []
    for mother in code_mothers:
        if mother.n < 1:
            continue
        max_n = min(payload, mother.n)
        for n in range(max_n, 0, -1):
            shortened_positions = tuple(range(n, mother.n))
            full_shortened_generator = _shortened_full_generator(mother.generator, shortened_positions)
            shortened_generator = full_shortened_generator[:, :n].copy()
            shortened_dim = int(shortened_generator.shape[0])
            if shortened_dim < min_k_int:
                continue
            mother_max_k = shortened_dim if max_k_int is None else min(shortened_dim, max_k_int)
            for k in range(mother_max_k, min_k_int - 1, -1):
                generator = shortened_generator[:k, :].copy()
                candidates.append(
                    ShortenedRMCode(
                        mother=mother,
                        k=k,
                        n=n,
                        t=mother.t,
                        generator=generator,
                        active_positions=tuple(range(n)),
                        shortened_positions=shortened_positions,
                        decoder_type=RM_DECODER_TYPE,
                    )
                )
    candidates.sort(key=lambda code: (-code.k, code.n, code.t, code.mother.n, code.mother.r))
    return candidates


def _decode_rm_majority(received: np.ndarray, mother: RMMotherCode) -> DecodeOutcome:
    residual = np.asarray(received, dtype=np.uint8).copy()
    coefficients_by_monomial: dict[tuple[int, ...], int] = {}
    for degree in range(int(mother.r), -1, -1):
        for variables in combinations(range(int(mother.m)), degree):
            parities = [
                int(np.bitwise_xor.reduce(residual[indices], dtype=np.uint8))
                for indices in _majority_check_indices(variables, int(mother.m))
            ]
            coefficient = 1 if sum(parities) > len(parities) / 2 else 0
            coefficients_by_monomial[variables] = coefficient
            if coefficient:
                residual ^= _monomial_evaluation(variables, int(mother.m))

    info = np.asarray(
        [
            coefficients_by_monomial[variables]
            for degree in range(0, int(mother.r) + 1)
            for variables in combinations(range(int(mother.m)), degree)
        ],
        dtype=np.uint8,
    )
    corrected = mother.encode(info)
    distance = int(np.count_nonzero(corrected != received))
    error_positions = tuple(np.flatnonzero(corrected != received).astype(int).tolist())
    if distance > mother.t:
        return DecodeOutcome(
            status="rejected",
            info_bits=None,
            codeword_bits=corrected,
            num_corrected_errors=distance,
            error_positions=error_positions,
        )
    return DecodeOutcome(
        status="decoded",
        info_bits=info,
        codeword_bits=corrected,
        num_corrected_errors=distance,
        error_positions=error_positions,
    )


def _rm_generator(r: int, m: int) -> np.ndarray:
    points = _evaluation_points(m)
    rows: list[np.ndarray] = []
    for degree in range(0, int(r) + 1):
        for variables in combinations(range(int(m)), degree):
            if not variables:
                rows.append(np.ones(2 ** int(m), dtype=np.uint8))
            else:
                rows.append(np.prod(points[:, variables], axis=1).astype(np.uint8))
    return np.vstack(rows).astype(np.uint8)


def _evaluation_points(m: int) -> np.ndarray:
    values = np.arange(2 ** int(m), dtype=np.uint64)
    return ((values[:, None] >> np.arange(int(m), dtype=np.uint64)) & 1).astype(np.uint8)


def _shortened_full_generator(generator: np.ndarray, shortened_positions: tuple[int, ...]) -> np.ndarray:
    if not shortened_positions:
        return np.asarray(generator, dtype=np.uint8).copy()
    constraints = np.asarray(generator[:, list(shortened_positions)].T, dtype=np.uint8)
    kernel = _gf2_nullspace(constraints)
    if kernel.size == 0:
        return np.zeros((0, int(generator.shape[1])), dtype=np.uint8)
    return np.mod(kernel @ generator, 2).astype(np.uint8)


@lru_cache(maxsize=None)
def _monomial_evaluation(variables: tuple[int, ...], m: int) -> np.ndarray:
    points = _evaluation_points(int(m))
    if not variables:
        return np.ones(2 ** int(m), dtype=np.uint8)
    return np.prod(points[:, list(variables)], axis=1).astype(np.uint8)


@lru_cache(maxsize=None)
def _majority_check_indices(variables: tuple[int, ...], m: int) -> tuple[np.ndarray, ...]:
    variable_set = set(variables)
    complement = tuple(index for index in range(int(m)) if index not in variable_set)
    groups: list[np.ndarray] = []
    for assignment in range(2 ** len(complement)):
        base = 0
        for bit_index, variable in enumerate(complement):
            if (assignment >> bit_index) & 1:
                base |= 1 << variable
        indices = []
        for free_assignment in range(2 ** len(variables)):
            point = base
            for bit_index, variable in enumerate(variables):
                if (free_assignment >> bit_index) & 1:
                    point |= 1 << variable
            indices.append(point)
        groups.append(np.asarray(indices, dtype=np.int64))
    return tuple(groups)


def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = mat.shape
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            mat[[pivot_row, pivot]] = mat[[pivot, pivot_row]]
        for row in range(rows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_cols = [col for col in range(cols) if col not in set(pivot_cols)]
    if not free_cols:
        return np.zeros((0, cols), dtype=np.uint8)
    basis = []
    for free_col in free_cols:
        vector = np.zeros(cols, dtype=np.uint8)
        vector[free_col] = 1
        for row, pivot_col in enumerate(pivot_cols):
            if mat[row, free_col]:
                vector[pivot_col] = 1
        basis.append(vector)
    return np.vstack(basis).astype(np.uint8)


def _independent_columns(matrix: np.ndarray, rank: int) -> tuple[int, ...]:
    mat = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = mat.shape
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            mat[[pivot_row, pivot]] = mat[[pivot, pivot_row]]
        for row in range(rows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rank:
            return tuple(pivot_cols)
    raise ValueError("generator matrix does not have full row rank")


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = mat.shape
    if rows != cols:
        raise ValueError("matrix must be square")
    inverse = np.eye(rows, dtype=np.uint8)
    pivot_row = 0
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            raise ValueError("matrix is singular over GF(2)")
        if pivot != pivot_row:
            mat[[pivot_row, pivot]] = mat[[pivot, pivot_row]]
            inverse[[pivot_row, pivot]] = inverse[[pivot, pivot_row]]
        for row in range(rows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
                inverse[row] ^= inverse[pivot_row]
        pivot_row += 1
    return inverse.astype(np.uint8)
