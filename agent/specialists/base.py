"""Common protocol for specialist tool wrappers."""

from __future__ import annotations

from typing import Optional, Protocol

from graph import OrbitalGraph

from ..state import Finding, OrbitalInvestigationState


class Specialist(Protocol):
    name: str

    def applicable(self, state: OrbitalInvestigationState) -> bool: ...

    def run(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph]) -> Finding: ...


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
