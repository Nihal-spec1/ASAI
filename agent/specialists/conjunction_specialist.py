"""Conjunction specialist: miss geometry, B-plane sigmas, Foster Pc, dilution flag.

Confidence semantics
--------------------
``confidence`` expresses how far the *current* Pc can be trusted as a
statement about the true geometry, not how large Pc is. It decays with the
absolute size of the major B-plane sigma:

    confidence = exp(-sigma_major_km / SIGMA_DEFENSIBLE_KM)

so a 3.8 km sigma (Scenario 1) yields ~0.02 while a 0.3 km sigma (Scenario 4)
yields ~0.74. The ``DILUTION_REGION`` flag is reported verbatim from the
Foster assessment; ``HIGH_COVARIANCE`` is the *policy-relevant* flag (R1/R2)
and fires when sigma_major exceeds ``HIGH_COVARIANCE_KM``.
"""

from __future__ import annotations

import math
from typing import Optional

from astrodynamics import assess_conjunction
from graph import OrbitalGraph

from ..state import Finding, OrbitalInvestigationState
from .base import clamp01

PC_THRESHOLD = 1e-4
HIGH_COVARIANCE_KM = 2.0
SIGMA_DEFENSIBLE_KM = 1.0


class ConjunctionSpecialist:
    name = "conjunction"

    def __init__(self, *, pc_threshold: float = PC_THRESHOLD, high_covariance_km: float = HIGH_COVARIANCE_KM) -> None:
        self.pc_threshold = pc_threshold
        self.high_covariance_km = high_covariance_km

    def applicable(self, state: OrbitalInvestigationState) -> bool:
        return state.has_conjunction

    def run(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph] = None) -> Finding:
        geom = state.geometry()
        a = assess_conjunction(geom, pc_threshold=self.pc_threshold)

        flags = []
        if a.pc >= self.pc_threshold:
            flags.append("PC_ABOVE_THRESHOLD")
        else:
            flags.append("PC_BELOW_THRESHOLD")
        if a.in_dilution_region:
            flags.append("DILUTION_REGION")
        if a.bplane_sigma_major_km > self.high_covariance_km:
            flags.append("HIGH_COVARIANCE")
        if a.encounter_regime == "low-velocity":
            flags.append("LOW_VELOCITY_REGIME")
        if a.sigma_to_miss_ratio > 1.0:
            flags.append("MISS_INSIDE_ONE_SIGMA")

        confidence = clamp01(math.exp(-a.bplane_sigma_major_km / SIGMA_DEFENSIBLE_KM))

        summary = (
            f"{a.primary_id} vs {a.secondary_id}: miss {a.miss_distance_km*1000:.0f} m, Pc {a.pc:.3g} "
            f"({a.encounter_regime}), B-plane sigma {a.bplane_sigma_major_km:.2f} km, "
            f"sigma/miss {a.sigma_to_miss_ratio:.1f}"
            + (", dilution region" if a.in_dilution_region else "")
        )
        return Finding(
            specialist=self.name,
            produced_at=state.now,
            summary=summary,
            confidence=confidence,
            metrics={
                "pc": a.pc,
                "pc_2d": a.pc_2d,
                "pc_3d": a.pc_3d,
                "miss_km": a.miss_distance_km,
                "miss_r_km": a.miss_rtn_km[0],
                "miss_t_km": a.miss_rtn_km[1],
                "miss_n_km": a.miss_rtn_km[2],
                "relative_speed_kms": a.relative_speed_kms,
                "bplane_miss_km": a.bplane_miss_km,
                "bplane_sigma_major_km": a.bplane_sigma_major_km,
                "bplane_sigma_minor_km": a.bplane_sigma_minor_km,
                "mahalanobis_distance": a.mahalanobis_distance,
                "covariance_volume_km3": a.covariance_volume_km3,
                "sigma_to_miss_ratio": a.sigma_to_miss_ratio,
                "hbr_combined_km": a.hbr_combined_km,
                "pc_threshold": self.pc_threshold,
            },
            flags=flags,
            details={"encounter_regime": a.encounter_regime, "tca": a.tca.isoformat()},
            evidence_refs=[f"state:{a.primary_id}", f"state:{a.secondary_id}", "tool:assess_conjunction"],
        )
