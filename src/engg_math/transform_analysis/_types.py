"""Shared type aliases."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

# Array dtype varies (real or complex) per call site; the alias keeps annotations short.
Array = npt.NDArray[Any]
