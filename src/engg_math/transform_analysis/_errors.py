"""Transform-analysis errors."""

from __future__ import annotations

from engg_math.core import EnggMathError


class DomainError(EnggMathError, ValueError):
    """An argument is outside the routine's valid domain."""
