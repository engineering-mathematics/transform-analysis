"""FFT, spectral estimation and z-domain analysis with explicit conventions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._errors import DomainError
from ._fft import fft, fftfreq, ifft, irfft, rfft
from ._results import FrequencyResponse, Spectrum
from ._spectral import dominant_frequency, periodogram, welch, window
from ._zdomain import frequency_response, is_stable, poles, z_transform

try:
    __version__ = version("engg-math-transform-analysis")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "DomainError",
    "FrequencyResponse",
    "Spectrum",
    "__version__",
    "dominant_frequency",
    "fft",
    "fftfreq",
    "frequency_response",
    "ifft",
    "irfft",
    "is_stable",
    "periodogram",
    "poles",
    "rfft",
    "welch",
    "window",
    "z_transform",
]
