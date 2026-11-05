"""Reference-solution, property and error-path tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engg_math.core import ShapeError
from engg_math.transform_analysis import (
    DomainError,
    dominant_frequency,
    fft,
    fftfreq,
    frequency_response,
    ifft,
    irfft,
    is_stable,
    periodogram,
    poles,
    rfft,
    welch,
    window,
    z_transform,
)

if TYPE_CHECKING:
    from engg_math.transform_analysis._validate import Norm

finite_signals = st.lists(st.floats(-1e3, 1e3), min_size=2, max_size=64)


def tone(freq: float, rate: float, n: int, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / rate)


# --- FFT -------------------------------------------------------------------------------
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@given(finite_signals)
@settings(max_examples=30, deadline=None)
def test_ifft_inverts_fft(norm: str, values: list[float]) -> None:
    """REQ-TA-001: ifft(fft(x)) == x for every normalisation."""
    x = np.array(values)
    np.testing.assert_allclose(
        ifft(fft(x, norm=cast("Norm", norm)), norm=cast("Norm", norm)).real, x, rtol=1e-9, atol=1e-9
    )


@given(finite_signals)
@settings(max_examples=30, deadline=None)
def test_ortho_fft_preserves_energy(values: list[float]) -> None:
    """REQ-TA-002: Parseval holds for the unitary transform."""
    x = np.array(values)
    energy_time = np.sum(x**2)
    energy_freq = np.sum(np.abs(fft(x, norm="ortho")) ** 2)
    assert energy_freq == pytest.approx(energy_time, rel=1e-9, abs=1e-9)


def test_fft_of_pure_tone_has_single_bin() -> None:
    x = tone(5, 64, 64)
    spectrum = np.abs(fft(x))
    assert np.argmax(spectrum[:32]) == 5
    assert spectrum[5] == pytest.approx(32.0)


def test_rfft_roundtrip_odd_length_needs_n() -> None:
    x = np.arange(7, dtype=float)
    np.testing.assert_allclose(irfft(rfft(x), n=7), x, atol=1e-12)


def test_fftfreq_matches_definition() -> None:
    np.testing.assert_allclose(fftfreq(4, 8.0), [0.0, 2.0, -4.0, -2.0])


def test_fft_input_validation() -> None:
    with pytest.raises(DomainError):
        fft([1.0, 2.0], norm="weird")  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        fft([1.0, math.nan])
    with pytest.raises(DomainError):
        fft(["a", "b"])
    with pytest.raises(ShapeError):
        fft(np.ones((2, 2)))
    with pytest.raises(ShapeError):
        fft([])
    with pytest.raises(DomainError):
        fftfreq(0)
    with pytest.raises(DomainError):
        fftfreq(4, 0.0)


# --- spectral --------------------------------------------------------------------------
def test_window_shapes_and_validation() -> None:
    assert window("boxcar", 8).tolist() == [1.0] * 8
    assert window("hann", 8)[0] == pytest.approx(0.0)
    with pytest.raises(DomainError):
        window("tukey", 8)
    with pytest.raises(DomainError):
        window("hann", 0)


def test_periodogram_power_integrates_to_mean_square() -> None:
    """A unit-amplitude sine has mean square 1/2; the PSD integral must match within 2%."""
    rate, n = 256.0, 1024
    spectrum = periodogram(tone(32, rate, n), rate)
    area = np.sum(spectrum.power) * spectrum.resolution
    assert area == pytest.approx(0.5, rel=0.02)


def test_dominant_frequency_recovers_off_bin_tone() -> None:
    """Parabolic refinement must land within a tenth of a bin of a 10.3 Hz tone."""
    rate, n = 128.0, 1024
    spectrum = periodogram(tone(10.3, rate, n), rate)
    assert dominant_frequency(spectrum) == pytest.approx(10.3, abs=0.1 * spectrum.resolution)


def test_welch_finds_tone_in_noise() -> None:
    rng = np.random.default_rng(0)
    rate = 200.0
    x = tone(40, rate, 4096, 2.0) + rng.normal(0, 1.0, 4096)
    spectrum = welch(x, rate, segment_length=512)
    assert dominant_frequency(spectrum) == pytest.approx(40.0, abs=spectrum.resolution)


def test_dominant_frequency_edge_cases() -> None:
    spectrum = periodogram(np.arange(16.0), 1.0)
    assert dominant_frequency(spectrum) >= 0.0
    with pytest.raises(DomainError):
        dominant_frequency(periodogram([1.0, 2.0], 1.0))


def test_spectral_validation() -> None:
    with pytest.raises(DomainError):
        welch(np.ones(64), overlap=1.0)
    with pytest.raises(DomainError):
        welch(np.ones(64), segment_length=128)
    with pytest.raises(DomainError):
        periodogram(np.ones(8), sample_rate=-1.0)
    with pytest.raises(DomainError):
        periodogram(np.ones(8, dtype=complex))


# --- z-domain --------------------------------------------------------------------------
def test_z_transform_reference() -> None:
    """X(z) = 1 + 2 z^-1 + 3 z^-2 at z = 2 is 1 + 1 + 0.75 = 2.75."""
    assert z_transform([1, 2, 3], 2.0)[0] == pytest.approx(2.75)


def test_z_transform_on_unit_circle_equals_dft() -> None:
    x = np.array([1.0, -2.0, 0.5, 3.0])
    points = np.exp(2j * np.pi * np.arange(4) / 4)
    np.testing.assert_allclose(z_transform(x, points), fft(x), atol=1e-12)


def test_z_transform_rejects_origin() -> None:
    with pytest.raises(DomainError):
        z_transform([1.0, 2.0], 0.0)


def test_frequency_response_of_moving_average() -> None:
    """Two-tap average: |H(w)| = |cos(w/2)|, so unity at DC and zero at Nyquist."""
    resp = frequency_response([0.5, 0.5], n_points=64)
    np.testing.assert_allclose(resp.magnitude, np.abs(np.cos(resp.omega / 2)), atol=1e-12)


def test_poles_and_stability() -> None:
    assert is_stable([1.0, -0.5])
    assert not is_stable([1.0, -1.5])
    assert not is_stable([1.0, -1.0])  # pole on the circle is marginal, not stable
    assert is_stable([1.0])
    np.testing.assert_allclose(poles([1.0, -0.5]), [0.5])


def test_z_domain_validation() -> None:
    with pytest.raises(DomainError):
        frequency_response([1.0], [0.0, 1.0])
    with pytest.raises(DomainError):
        frequency_response([1.0], n_points=1)
