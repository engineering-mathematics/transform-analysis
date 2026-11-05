"""Discrete Fourier transforms with an explicit normalisation convention.

Conventions (``norm``): ``"backward"`` applies no scaling forward and ``1/n`` inverse;
``"ortho"`` applies ``1/sqrt(n)`` both ways (unitary, preserves energy); ``"forward"``
applies ``1/n`` forward and none inverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy import fft as _fft

from . import _validate
from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array
    from ._validate import Norm


def fft(x: npt.ArrayLike, *, norm: Norm = "backward") -> Array:
    """Discrete Fourier transform of a 1-D signal."""
    result: Array = _fft.fft(_validate.signal(x), norm=_validate.norm(norm))
    return result


def ifft(x: npt.ArrayLike, *, norm: Norm = "backward") -> Array:
    """Inverse discrete Fourier transform; ``ifft(fft(x)) == x`` for the same ``norm``."""
    result: Array = _fft.ifft(_validate.signal(x), norm=_validate.norm(norm))
    return result


def rfft(x: npt.ArrayLike, *, norm: Norm = "backward") -> Array:
    """DFT of a real signal: the ``n // 2 + 1`` non-negative-frequency bins."""
    data = _validate.signal(x, allow_complex=False)
    result: Array = _fft.rfft(data, norm=_validate.norm(norm))
    return result


def irfft(x: npt.ArrayLike, n: int | None = None, *, norm: Norm = "backward") -> Array:
    """Inverse of ``rfft``. Pass ``n`` to recover odd-length signals (default ``2 * (m - 1)``)."""
    result: Array = _fft.irfft(_validate.signal(x), n=n, norm=_validate.norm(norm))
    return result


def fftfreq(n: int, sample_rate: float = 1.0) -> Array:
    """Frequencies in Hz of the ``fft`` bins for ``n`` samples at ``sample_rate`` Hz."""
    _validate.positive("sample_rate", sample_rate)
    if n < 1:
        raise DomainError(f"n must be >= 1, got {n}")
    result: Array = _fft.fftfreq(n, d=1.0 / sample_rate)
    return result
