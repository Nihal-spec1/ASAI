"""Kinematics specialist: orbital-element residuals, unannounced Δv, drift, decay.

Inputs are taken from ``state.aux_evidence["kinematics"]`` (a dict with the
``KinematicsFixture`` layout: element_history, maneuver_history,
range_history_km, residuals). All Δv figures are *recomputed* from the
element history with two-body relations; fixture residuals are only used as
a cross-check and never trusted blindly.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astrodynamics import MU_EARTH_KM3_S2
from graph import OrbitalGraph

from ..state import Finding, OrbitalInvestigationState
from .base import clamp01

UNANNOUNCED_DV_THRESHOLD_MPS = 1.0
SHADOWING_RANGE_KM = 100.0
DECAY_RATIO_THRESHOLD = 3.0
LOS_HOURS_THRESHOLD = 12.0


def _parse_epoch(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def plane_change_dv_mps(a_km: float, delta_i_deg: float) -> float:
    v = math.sqrt(MU_EARTH_KM3_S2 / a_km)  # circular speed km/s
    return 2.0 * v * math.sin(math.radians(abs(delta_i_deg)) / 2.0) * 1000.0


def altitude_change_dv_mps(a_km: float, delta_a_km: float) -> float:
    """Hohmann-equivalent |Δv| for a small semi-major-axis change (two burns)."""
    v = math.sqrt(MU_EARTH_KM3_S2 / a_km)
    return abs(delta_a_km) / (2.0 * a_km) * v * 1000.0


def _linear_rate(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


class KinematicsSpecialist:
    name = "kinematics"

    def applicable(self, state: OrbitalInvestigationState) -> bool:
        return "kinematics" in state.aux_evidence

    def run(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph] = None) -> Finding:
        kin: Dict[str, Any] = state.aux_evidence["kinematics"]
        object_id = kin.get("object_id", state.secondary_id or state.primary_id)
        history = kin.get("element_history", []) or []
        maneuvers = kin.get("maneuver_history", []) or []
        ranges: List[Tuple[float, float]] = [tuple(x) for x in (kin.get("range_history_km") or [])]
        residuals: Dict[str, float] = kin.get("residuals", {}) or {}

        metrics: Dict[str, float] = {}
        flags: List[str] = []
        parts: List[str] = []
        evidence = [f"elements:{object_id}"]

        # ---------------------------------------------------- element residuals
        if len(history) >= 2:
            first, last = history[0], history[-1]
            t0, t1 = _parse_epoch(first["epoch"]), _parse_epoch(last["epoch"])
            days = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
            a0, a1 = float(first["a_km"]), float(last["a_km"])
            di = float(last["i_deg"]) - float(first["i_deg"])
            da = a1 - a0
            metrics.update(
                delta_i_deg=di,
                delta_a_km=da,
                window_days=days,
                da_dt_km_day=_linear_rate(
                    [(_parse_epoch(h["epoch"]) - t0).total_seconds() / 86400.0 for h in history],
                    [float(h["a_km"]) for h in history],
                ),
            )
            dv_plane = plane_change_dv_mps(a0, di) if abs(di) > 0.01 else 0.0
            # a *drop* in semi-major axis is explained by drag, never inferred as a burn;
            # only a raise (or a plane change) costs propellant
            dv_alt = altitude_change_dv_mps(a0, da) if da > 0.5 else 0.0
            metrics["dv_plane_change_mps"] = dv_plane
            metrics["dv_altitude_mps"] = dv_alt
            metrics["dv_total_estimated_mps"] = dv_plane + dv_alt
            if abs(di) > 0.05:
                flags.append("PLANE_CHANGE")
                parts.append(f"inclination changed {di:+.2f} deg ({dv_plane:.1f} m/s equivalent)")
            if abs(da) > 5.0:
                flags.append("ALTITUDE_CHANGE")
                parts.append(f"semi-major axis changed {da:+.1f} km")

            # decay anomaly: most recent interval rate (the window fit averages
            # pre- and post-anomaly segments and would understate a sudden change)
            prev = history[-2]
            dt_last = max((t1 - _parse_epoch(prev["epoch"])).total_seconds() / 86400.0, 1e-9)
            observed = (a1 - float(prev["a_km"])) / dt_last
            metrics["da_dt_recent_km_day"] = observed
            nominal = residuals.get("decay_rate_nominal_km_day")
            if nominal is not None and nominal < 0 and observed < 0:
                ratio = observed / nominal
                metrics["decay_rate_observed_km_day"] = observed
                metrics["decay_rate_nominal_km_day"] = float(nominal)
                metrics["decay_ratio"] = ratio
                if ratio >= DECAY_RATIO_THRESHOLD:
                    flags.append("ANOMALOUS_DECAY")
                    parts.append(f"decay rate {observed:.2f} km/day is {ratio:.1f}x nominal")

        # ---------------------------------------------------- manoeuvre ledger
        dv_unannounced = float(sum(float(m["dv_mps"]) for m in maneuvers if not m.get("announced", False)))
        dv_announced = float(sum(float(m["dv_mps"]) for m in maneuvers if m.get("announced", False)))
        metrics["dv_unannounced_mps"] = dv_unannounced
        metrics["dv_announced_mps"] = dv_announced
        metrics["maneuver_count"] = float(len(maneuvers))
        if dv_unannounced >= UNANNOUNCED_DV_THRESHOLD_MPS:
            flags.append("UNANNOUNCED_DV")
            parts.append(f"{len(maneuvers)} unannounced burns totalling {dv_unannounced:.1f} m/s")
            evidence.append(f"maneuvers:{object_id}")
        elif "dv_total_estimated_mps" in metrics and metrics["dv_total_estimated_mps"] >= UNANNOUNCED_DV_THRESHOLD_MPS and not maneuvers:
            flags.append("UNANNOUNCED_DV")
            parts.append(f"element residuals imply {metrics['dv_total_estimated_mps']:.1f} m/s with no filed manoeuvre")

        # ---------------------------------------------------- shadowing pattern
        if len(ranges) >= 3:
            days_r = [float(d) for d, _ in ranges]
            rng = [float(r) for _, r in ranges]
            metrics["range_km"] = rng[-1]
            metrics["range_initial_km"] = rng[0]
            metrics["approach_rate_km_day"] = _linear_rate(days_r, rng)
            # interval closing rates: a controlled approach brakes hard once inside the
            # shadowing range, so the recent rate collapses relative to the peak rate
            rates = [abs(rng[i + 1] - rng[i]) / max(days_r[i + 1] - days_r[i], 1e-9) for i in range(len(rng) - 1)]
            peak_rate, recent_rate = max(rates), rates[-1]
            metrics["peak_closing_rate_km_day"] = peak_rate
            metrics["recent_closing_rate_km_day"] = recent_rate
            approaching = rng[-1] < rng[0] * 0.5
            holding = rng[-1] < SHADOWING_RANGE_KM and recent_rate < 0.1 * peak_rate
            if approaching and holding:
                flags.append("SHADOWING_PATTERN")
                parts.append(
                    f"closed from {rng[0]:.0f} km to {rng[-1]:.1f} km, braking from {peak_rate:.0f} to {recent_rate:.0f} km/day: station-keeping"
                )
            elif approaching:
                flags.append("APPROACHING")
                parts.append(f"closing at {metrics['approach_rate_km_day']:.1f} km/day")
            evidence.append(f"ranges:{object_id}")

        # ---------------------------------------------------- loss of signal
        los_h = residuals.get("hours_since_last_telemetry")
        if los_h is not None:
            metrics["hours_since_last_telemetry"] = float(los_h)
            if float(los_h) >= LOS_HOURS_THRESHOLD:
                flags.append("LOSS_OF_SIGNAL")
                parts.append(f"no telemetry for {float(los_h):.0f} h")
        if "ballistic_coefficient_ratio" in residuals:
            metrics["ballistic_coefficient_ratio"] = float(residuals["ballistic_coefficient_ratio"])

        # ---------------------------------------------------- cross-check with reported residuals
        rep = residuals.get("dv_total_estimated_mps")
        if rep is not None and "dv_total_estimated_mps" in metrics:
            metrics["dv_residual_crosscheck_mps"] = abs(float(rep) - metrics["dv_total_estimated_mps"])

        if not parts:
            parts.append("no significant kinematic residuals")

        # confidence: data richness
        conf = 0.35
        if len(history) >= 3:
            conf += 0.25
        if maneuvers:
            conf += 0.15
        if len(ranges) >= 3:
            conf += 0.2
        if "decay_ratio" in metrics:
            conf += 0.1
        return Finding(
            specialist=self.name,
            produced_at=state.now,
            summary=f"{object_id}: " + "; ".join(parts),
            confidence=clamp01(conf),
            metrics=metrics,
            flags=flags,
            details={"object_id": object_id, "notes": kin.get("notes", "")},
            evidence_refs=evidence,
        )
