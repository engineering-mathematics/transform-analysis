# engg-math-transform-analysis

FFT, spectral estimation and z-domain analysis for `engg_math`. Typed layers over SciPy with explicit
conventions, validated inputs and results that carry their own frequency axis.

```bash
pip install engg-math-transform-analysis
```

```python
from engg_math.transform_analysis import periodogram, dominant_frequency, frequency_response

spectrum = periodogram(signal, sample_rate=1000.0)  # PSD in signal^2/Hz
dominant_frequency(spectrum)  # Hz, parabolic-refined
frequency_response([0.5, 0.5]).magnitude  # |H(e^jw)| of a 2-tap average
```

## Conventions

| Item | Convention |
|---|---|
| FFT scaling | `norm="backward"` (default), `"ortho"` (unitary, preserves energy) or `"forward"`; always explicit |
| PSD | One-sided, signal^2/Hz; its integral over frequency is the signal's mean square |
| Windows | Periodic (DFT-even); mean removed before estimation |
| Transfer functions | Coefficients in descending powers of z^-1; leading coefficient of `a` must be nonzero |
| Stability | Strictly inside the unit circle, with a configurable margin |

## API

| Area | Names |
|---|---|
| Transforms | `fft`, `ifft`, `rfft`, `irfft`, `fftfreq` |
| Spectral | `window`, `periodogram`, `welch`, `dominant_frequency`, `Spectrum` |
| z-domain | `z_transform`, `frequency_response`, `poles`, `is_stable`, `FrequencyResponse` |

Laplace transforms, spectrograms and units are not part of 0.1. NaN and infinity in a signal raise `DomainError`.

## Development

```bash
uv sync && uv run pytest && uv run ruff check . && uv run mypy && uv run lint-imports
```
