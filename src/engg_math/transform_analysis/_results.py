"""Typed result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Array


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One-sided power spectral density.

    ``power`` is in signal-units^2 / Hz, so integrating over ``frequencies`` gives the
    mean-square value of the signal.
    """

    frequencies: Array
    power: Array
    sample_rate: float

    @property
    def resolution(self) -> float:
        """Frequency bin spacing in Hz."""
        return float(self.frequencies[1] - self.frequencies[0])


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Response ``H(e^{jw})`` of a rational transfer function at angular frequencies ``omega``."""

    omega: Array
    response: Array

    @property
    def magnitude(self) -> Array:
        """``|H|``."""
        magnitude: Array = abs(self.response)
        return magnitude
