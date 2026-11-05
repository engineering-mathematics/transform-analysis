"""Input validation shared by the transform routines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from engg_math.core import ShapeError

from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

Norm = Literal["backward", "ortho", "forward"]
NORMS: tuple[Norm, ...] = ("backward", "ortho", "forward")


def signal(x: npt.ArrayLike, *, minimum: int = 1, allow_complex: bool = True) -> Array:
    """Return ``x`` as a finite 1-D array with at least ``minimum`` samples."""
    arr = np.asarray(x)
    if arr.dtype.kind not in "fiub" + ("c" if allow_complex else ""):
        raise DomainError(f"expected a numeric signal, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ShapeError(f"signal must be 1-D, got shape {arr.shape}")
    if arr.size < minimum:
        raise ShapeError(f"need at least {minimum} samples, got {arr.size}")
    if not np.isfinite(arr).all():
        raise DomainError("signal contains NaN or infinity")
    return arr


def norm(value: Norm) -> Norm:
    """Validate an FFT normalisation convention."""
    if value not in NORMS:
        raise DomainError(f"norm must be one of {NORMS}, got {value!r}")
    return value


def positive(name: str, value: float) -> float:
    """Require a positive, finite number."""
    if not np.isfinite(value) or value <= 0:
        raise DomainError(f"{name} must be positive and finite, got {value}")
    return float(value)
