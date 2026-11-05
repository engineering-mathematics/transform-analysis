"""Windowed spectral estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as _signal

from . import _validate
from ._errors import DomainError
from ._results import Spectrum

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

MIN_SEGMENT = 2
MIN_BINS = 3
WINDOWS = ("boxcar", "hann", "hamming", "blackman")


def window(kind: str, n: int) -> Array:
    """Periodic (DFT-even) window of length ``n``: boxcar, hann, hamming or blackman."""
    if kind not in WINDOWS:
        raise DomainError(f"window must be one of {WINDOWS}, got {kind!r}")
    if n < 1:
        raise DomainError(f"n must be >= 1, got {n}")
    return np.asarray(_signal.get_window(kind, n, fftbins=True), dtype=float)


def periodogram(
    x: npt.ArrayLike, sample_rate: float = 1.0, *, window_kind: str = "hann"
) -> Spectrum:
    """One-sided power spectral density from a single windowed FFT."""
    data = _validate.signal(x, minimum=2, allow_complex=False)
    rate = _validate.positive("sample_rate", sample_rate)
    freqs, power = _signal.periodogram(
        data, fs=rate, window=window(window_kind, data.size), detrend="constant", scaling="density"
    )
    return Spectrum(freqs, power, rate)


def welch(
    x: npt.ArrayLike,
    sample_rate: float = 1.0,
    *,
    segment_length: int = 256,
    overlap: float = 0.5,
    window_kind: str = "hann",
) -> Spectrum:
    """Welch PSD: averaged windowed periodograms of overlapping segments.

    Averaging lowers the variance of the estimate at the cost of frequency resolution
    (``sample_rate / segment_length``). ``overlap`` is a fraction in ``[0, 1)``.
    """
    data = _validate.signal(x, minimum=2, allow_complex=False)
    rate = _validate.positive("sample_rate", sample_rate)
    if not 0.0 <= overlap < 1.0:
        raise DomainError(f"overlap must be in [0, 1), got {overlap}")
    if not MIN_SEGMENT <= segment_length <= data.size:
        raise DomainError(
            f"segment_length must be in [{MIN_SEGMENT}, {data.size}] for this signal, "
            f"got {segment_length}"
        )
    freqs, power = _signal.welch(
        data,
        fs=rate,
        window=window(window_kind, segment_length),
        nperseg=segment_length,
        noverlap=int(overlap * segment_length),
        detrend="constant",
        scaling="density",
    )
    return Spectrum(freqs, power, rate)


def dominant_frequency(spectrum: Spectrum) -> float:
    """Frequency in Hz of the largest non-DC bin, refined by parabolic interpolation."""
    power = spectrum.power
    if power.size < MIN_BINS:
        raise DomainError("need at least 3 spectral bins")
    k = int(np.argmax(power[1:])) + 1
    if k == power.size - 1:
        return float(spectrum.frequencies[k])
    left, mid, right = np.log(np.maximum(power[k - 1 : k + 2], np.finfo(float).tiny))
    denom = left - 2.0 * mid + right
    shift = 0.0 if denom == 0.0 else 0.5 * (left - right) / denom
    return float(spectrum.frequencies[k] + shift * spectrum.resolution)
