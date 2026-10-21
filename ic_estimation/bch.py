from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np


PRIMITIVE_POLYS: dict[int, int] = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10000011,
    8: 0b100011101,
    9: 0b1000010001,
    10: 0b10000001001,
}


@dataclass(frozen=True)
class BCHMotherCode:
    m: int
    n: int
    k: int
    t: int
    generator: int
    generator_degree: int
    designed_distance: int
    family: str = "primitive_narrow_sense_bch"

    @property
    def redundancy(self) -> int:
        return int(self.n - self.k)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "family": self.family,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "t": self.t,
            "redundancy": self.redundancy,
            "generator": self.generator,
            "generator_degree": self.generator_degree,
            "designed_distance": self.designed_distance,
        }


@dataclass(frozen=True)
class DecodeOutcome:
    status: str
    info_bits: np.ndarray | None
    codeword_bits: np.ndarray
    num_corrected_errors: int
    error_positions: tuple[int, ...]


@dataclass(frozen=True)
class ShortenedBCHCode:
    mother: BCHMotherCode
    k: int
    n: int
    t: int
    redundancy: int
    active_positions: tuple[int, ...]
    family: str = "shortened_primitive_narrow_sense_bch"

    @classmethod
    def from_mother(
        cls,
        mother: BCHMotherCode,
        k: int,
        active_positions: list[int] | tuple[int, ...] | None = None,
    ) -> "ShortenedBCHCode":
        k_int = int(k)
        if k_int < 1 or k_int > mother.k:
            raise ValueError(f"k must be in [1, {mother.k}], got {k}")
        n_int = k_int + mother.redundancy
        if n_int > mother.n:
            raise ValueError(f"shortened length {n_int} exceeds mother length {mother.n}")
        positions = tuple(range(n_int)) if active_positions is None else tuple(int(pos) for pos in active_positions)
        if len(positions) != n_int:
            raise ValueError(f"active_positions must have length {n_int}, got {len(positions)}")
        if len(set(positions)) != len(positions):
            raise ValueError("active_positions must be unique")
        return cls(
            mother=mother,
            k=k_int,
            n=n_int,
            t=mother.t,
            redundancy=mother.redundancy,
            active_positions=positions,
        )

    @property
    def identity_count(self) -> int:
        return 2 ** int(self.k)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "mother": self.mother.to_dict(),
            "n": self.n,
            "k": self.k,
            "t": self.t,
            "redundancy": self.redundancy,
            "active_positions": list(self.active_positions),
            "identity_count": self.identity_count,
        }

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if info.size != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {info.size}")
        if np.any((info != 0) & (info != 1)):
            raise ValueError("info bits must be binary")
        msg_int = _bits_to_int(info)
        full_code_int = _encode_systematic_int(msg_int, self.mother.generator, self.redundancy)
        return _int_to_bits(full_code_int, self.n)

    def decode(self, codeword_bits: np.ndarray) -> DecodeOutcome:
        received = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if received.size != self.n:
            raise ValueError(f"Expected {self.n} codeword bits, got {received.size}")
        if np.any((received != 0) & (received != 1)):
            raise ValueError("codeword bits must be binary")

        field = _GF2m.for_degree(self.mother.m)
        received_int = _bits_to_int(received)
        corrected_int, error_positions, status = _decode_bch_int(
            received_int=received_int,
            mother=self.mother,
            field=field,
        )
        corrected_bits = _int_to_bits(corrected_int, self.n)
        if status != "decoded":
            return DecodeOutcome(
                status="rejected",
                info_bits=None,
                codeword_bits=corrected_bits,
                num_corrected_errors=len(error_positions),
                error_positions=tuple(error_positions),
            )

        high_bits = corrected_int >> self.redundancy
        info = _int_to_bits(high_bits, self.k)
        return DecodeOutcome(
            status="decoded",
            info_bits=info,
            codeword_bits=corrected_bits,
            num_corrected_errors=len(error_positions),
            error_positions=tuple(error_positions),
        )


class _GF2m:
    def __init__(self, m: int, primitive_poly: int):
        self.m = int(m)
        self.primitive_poly = int(primitive_poly)
        self.size = 1 << self.m
        self.order = self.size - 1
        self.exp = [0] * (self.order * 2)
        self.log = [0] * self.size
        x = 1
        for idx in range(self.order):
            self.exp[idx] = x
            self.log[x] = idx
            x <<= 1
            if x & self.size:
                x ^= self.primitive_poly
        for idx in range(self.order, self.order * 2):
            self.exp[idx] = self.exp[idx - self.order]

    @classmethod
    @lru_cache(maxsize=None)
    def for_degree(cls, m: int) -> "_GF2m":
        if int(m) not in PRIMITIVE_POLYS:
            raise ValueError(f"No primitive polynomial configured for m={m}")
        return cls(int(m), PRIMITIVE_POLYS[int(m)])

    def add(self, a: int, b: int) -> int:
        return int(a) ^ int(b)

    def mul(self, a: int, b: int) -> int:
        if int(a) == 0 or int(b) == 0:
            return 0
        return self.exp[(self.log[int(a)] + self.log[int(b)]) % self.order]

    def div(self, a: int, b: int) -> int:
        if int(b) == 0:
            raise ZeroDivisionError("GF division by zero")
        if int(a) == 0:
            return 0
        return self.exp[(self.log[int(a)] - self.log[int(b)]) % self.order]

    def pow_alpha(self, exponent: int) -> int:
        return self.exp[int(exponent) % self.order]


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for idx, bit in enumerate(np.asarray(bits, dtype=np.uint8).reshape(-1)):
        value |= int(bit) << idx
    return int(value)


def _int_to_bits(value: int, nbits: int) -> np.ndarray:
    return np.asarray([(int(value) >> idx) & 1 for idx in range(int(nbits))], dtype=np.uint8)


def _poly_degree(poly: int) -> int:
    return int(poly.bit_length() - 1)


def _poly_mod(dividend: int, divisor: int) -> int:
    if divisor <= 0:
        raise ZeroDivisionError("polynomial divisor is zero")
    remainder = int(dividend)
    divisor_degree = _poly_degree(divisor)
    while remainder and _poly_degree(remainder) >= divisor_degree:
        shift = _poly_degree(remainder) - divisor_degree
        remainder ^= divisor << shift
    return int(remainder)


def _encode_systematic_int(msg_int: int, generator: int, redundancy: int) -> int:
    shifted = int(msg_int) << int(redundancy)
    remainder = _poly_mod(shifted, int(generator))
    return shifted ^ remainder


def _cyclotomic_coset(start: int, n: int) -> tuple[int, ...]:
    seen: list[int] = []
    value = int(start) % int(n)
    while value not in seen:
        seen.append(value)
        value = (value * 2) % int(n)
    return tuple(seen)


def _gf_poly_mul(left: list[int], right: list[int], field: _GF2m) -> list[int]:
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            if b != 0:
                out[i + j] ^= field.mul(a, b)
    return out


def _minimal_polynomial(coset: tuple[int, ...], field: _GF2m) -> int:
    coeffs = [1]
    for exponent in coset:
        coeffs = _gf_poly_mul(coeffs, [field.pow_alpha(exponent), 1], field)
    poly = 0
    for idx, coeff in enumerate(coeffs):
        if coeff not in (0, 1):
            raise ValueError(f"minimal polynomial coefficient {coeff} is not in GF(2)")
        if coeff:
            poly |= 1 << idx
    return int(poly)


def _poly_mul_binary(left: int, right: int) -> int:
    out = 0
    cursor = int(right)
    shift = 0
    while cursor:
        if cursor & 1:
            out ^= int(left) << shift
        cursor >>= 1
        shift += 1
    return int(out)


@lru_cache(maxsize=None)
def _bch_generator(m: int, t: int) -> tuple[int, int]:
    field = _GF2m.for_degree(int(m))
    n = (1 << int(m)) - 1
    used: set[int] = set()
    generator = 1
    for exponent in range(1, 2 * int(t) + 1):
        if exponent % n in used:
            continue
        coset = _cyclotomic_coset(exponent, n)
        used.update(coset)
        generator = _poly_mul_binary(generator, _minimal_polynomial(coset, field))
    return int(generator), _poly_degree(generator)


def primitive_bch_mothers(max_m: int = 10, max_t: int | None = None) -> list[BCHMotherCode]:
    mothers: list[BCHMotherCode] = []
    for m in range(3, int(max_m) + 1):
        n = (1 << m) - 1
        t_limit = min((n - 1) // 2, int(max_t) if max_t is not None else 16)
        for t in range(1, t_limit + 1):
            generator, degree = _bch_generator(m, t)
            k = n - degree
            if k < 1:
                k = 1
            mothers.append(
                BCHMotherCode(
                    m=m,
                    n=n,
                    k=k,
                    t=t,
                    generator=generator,
                    generator_degree=degree,
                    designed_distance=2 * t + 1,
                )
            )
    return mothers


def _syndromes(received_int: int, mother: BCHMotherCode, field: _GF2m) -> list[int]:
    syndromes: list[int] = []
    set_positions = [idx for idx in range(mother.n) if (int(received_int) >> idx) & 1]
    for j in range(1, 2 * mother.t + 1):
        value = 0
        for pos in set_positions:
            value ^= field.pow_alpha(j * pos)
        syndromes.append(value)
    return syndromes


def _berlekamp_massey(syndromes: list[int], field: _GF2m) -> tuple[list[int], int]:
    c = [1]
    b = [1]
    locator_degree = 0
    shift = 1
    previous_discrepancy = 1

    for idx, syndrome in enumerate(syndromes):
        discrepancy = int(syndrome)
        for degree in range(1, locator_degree + 1):
            if degree < len(c):
                discrepancy ^= field.mul(c[degree], syndromes[idx - degree])

        if discrepancy == 0:
            shift += 1
            continue

        old_c = c[:]
        scale = field.div(discrepancy, previous_discrepancy)
        needed = len(b) + shift
        if len(c) < needed:
            c.extend([0] * (needed - len(c)))
        for j, coeff in enumerate(b):
            if coeff:
                c[j + shift] ^= field.mul(scale, coeff)

        if 2 * locator_degree <= idx:
            locator_degree = idx + 1 - locator_degree
            b = old_c
            previous_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c, locator_degree


def _eval_gf_poly(coeffs: list[int], x: int, field: _GF2m) -> int:
    value = 0
    power = 1
    for coeff in coeffs:
        if coeff:
            value ^= field.mul(coeff, power)
        power = field.mul(power, x)
    return int(value)


def _decode_bch_int(received_int: int, mother: BCHMotherCode, field: _GF2m) -> tuple[int, tuple[int, ...], str]:
    syndromes = _syndromes(received_int, mother, field)
    if not any(syndromes):
        return int(received_int), tuple(), "decoded"

    locator, locator_degree = _berlekamp_massey(syndromes, field)
    if locator_degree > mother.t:
        return int(received_int), tuple(), "rejected"

    error_positions: list[int] = []
    for pos in range(mother.n):
        root = field.pow_alpha(-pos)
        if _eval_gf_poly(locator, root, field) == 0:
            error_positions.append(pos)

    if len(error_positions) != locator_degree or len(error_positions) > mother.t:
        return int(received_int), tuple(error_positions), "rejected"

    corrected = int(received_int)
    for pos in error_positions:
        corrected ^= 1 << pos
    if any(_syndromes(corrected, mother, field)):
        return int(received_int), tuple(error_positions), "rejected"
    return corrected, tuple(error_positions), "decoded"


def brute_force_nearest_decode(received: np.ndarray, codewords: list[np.ndarray], radius: int) -> tuple[int | None, int]:
    distances = [int(np.count_nonzero(np.asarray(received, dtype=np.uint8) != codeword)) for codeword in codewords]
    best = min(distances)
    if best > int(radius) or distances.count(best) != 1:
        return None, best
    return int(distances.index(best)), best


def enumerate_error_patterns(n: int, t: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = [tuple()]
    for weight in range(1, int(t) + 1):
        patterns.extend(tuple(combo) for combo in combinations(range(int(n)), weight))
    return patterns
