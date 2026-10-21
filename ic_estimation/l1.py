from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, log2


@dataclass(frozen=True)
class L1Bound:
    payload_length: int
    bit_error_rate: float
    alpha: float
    radius: int
    max_log2_identities: float
    max_identities: int
    model: str = "hamming_upper_envelope"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "payload_length": self.payload_length,
            "bit_error_rate": self.bit_error_rate,
            "alpha": self.alpha,
            "radius": self.radius,
            "max_log2_identities": self.max_log2_identities,
            "max_identities": self.max_identities,
        }


@lru_cache(maxsize=None)
def hamming_ball_volume(n: int, radius: int) -> int:
    return sum(comb(int(n), i) for i in range(0, min(int(radius), int(n)) + 1))


def binomial_cdf(n: int, radius: int, p: float) -> float:
    return sum(comb(int(n), i) * (float(p) ** i) * ((1.0 - float(p)) ** (int(n) - i)) for i in range(radius + 1))


def required_radius(n: int, p: float, alpha: float) -> int:
    target = 1.0 - float(alpha)
    for radius in range(int(n) + 1):
        if binomial_cdf(n, radius, p) >= target:
            return radius
    return int(n)


def hamming_upper_identity_bound(payload_length: int, bit_error_rate: float, alpha: float) -> L1Bound:
    n = int(payload_length)
    radius = required_radius(n, min(1.0, max(0.0, bit_error_rate)), alpha)
    volume = max(1, hamming_ball_volume(n, radius))
    max_log2 = max(0.0, float(n) - log2(volume))
    return L1Bound(
        payload_length=n,
        bit_error_rate=float(bit_error_rate),
        alpha=float(alpha),
        radius=radius,
        max_log2_identities=max_log2,
        max_identities=2 ** int(max_log2),
    )
