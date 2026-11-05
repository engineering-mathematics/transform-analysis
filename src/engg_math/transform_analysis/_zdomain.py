"""z-domain analysis of sampled sequences and rational transfer functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as _signal

from . import _validate
from ._errors import DomainError
from ._results import FrequencyResponse

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

LEADING_COEFFICIENT_RTOL = 1e-12


def z_transform(x: npt.ArrayLike, z: npt.ArrayLike) -> Array:
    """Finite-sequence z-transform ``X(z) = sum_n x[n] z^-n`` evaluated at ``z``.

    Exact for a finite ``x``; ``z = 0`` is rejected (the transform has a pole there).
    """
    seq = _validate.signal(x)
    points = np.atleast_1d(np.asarray(z, dtype=complex))
    if np.any(points == 0):
        raise DomainError("z = 0 is a pole of the z-transform")
    powers = np.arange(seq.size)
    result: Array = (seq[None, :] * points[:, None] ** (-powers[None, :])).sum(axis=1)
    return result


def _coefficients(name: str, values: npt.ArrayLike) -> Array:
    coeffs = _validate.signal(values, allow_complex=False)
    if abs(coeffs[0]) <= LEADING_COEFFICIENT_RTOL * np.max(np.abs(coeffs)):
        raise DomainError(f"{name} must have a nonzero leading coefficient")
    return coeffs.astype(float)


def frequency_response(
    b: npt.ArrayLike, a: npt.ArrayLike = (1.0,), *, n_points: int = 512
) -> FrequencyResponse:
    """``H(e^{jw}) = B / A`` on ``[0, pi)``, coefficients in descending powers of ``z^-1``."""
    num, den = _coefficients("b", b), _coefficients("a", a)
    if n_points < 2:  # noqa: PLR2004
        raise DomainError(f"n_points must be >= 2, got {n_points}")
    omega, response = _signal.freqz(num, den, worN=n_points)
    return FrequencyResponse(np.asarray(omega), np.asarray(response))


def poles(a: npt.ArrayLike) -> Array:
    """Poles of ``1 / A(z)``: roots of the denominator polynomial."""
    return np.roots(_coefficients("a", a))


def is_stable(a: npt.ArrayLike, *, margin: float = 1e-9) -> bool:
    """Whether all poles lie strictly inside the unit circle (by at least ``margin``)."""
    found = poles(a)
    return bool(found.size == 0 or np.max(np.abs(found)) < 1.0 - margin)
