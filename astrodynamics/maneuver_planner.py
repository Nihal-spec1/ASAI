"""Impulsive minimum-delta-v collision-avoidance manoeuvre (CAM) planner.

Model
-----
A single impulsive burn applied ``lead_time_s`` before TCA in the primary's
RTN frame. The primary's displacement at TCA is obtained from the
Clohessy-Wiltshire (Hill) equations for a near-circular reference orbit with
mean motion n:

    x(t) = (dvx/n) sin nt + (2 dvy/n)(1 - cos nt)              # radial
    y(t) = (2 dvx/n)(cos nt - 1) + dvy (4 sin nt / n - 3t)     # along-track
    z(t) = (dvz/n) sin nt                                      # cross-track

The new miss vector is ``miss_old - displacement`` (secondary minus primary).
Pc is re-evaluated with the unchanged combined covariance. For each canonical
burn direction the smallest magnitude that satisfies both the Pc target and a
minimum stand-off distance is found by a geometric scan followed by bisection.

The CW displacement is cross-checked against numerical two-body propagation in
the Stage-1 test-suite.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .coordinates import MU_EARTH_KM3_S2, eci_to_rtn_matrix, ensure_utc
from .foster_pc import (
    ConjunctionAssessment,
    ConjunctionGeometry,
    assess_conjunction,
    collision_probability_2d,
    collision_probability_3d,
    LOW_VELOCITY_THRESHOLD_KMS,
)
from .propagation import StateVector, TLE, propagate_tle, state_to_classical_elements, two_body_propagate

G0_M_S2 = 9.80665


class BurnDirection(str, Enum):
    PROGRADE = "prograde"  # +T
    RETROGRADE = "retrograde"  # -T
    RADIAL_OUT = "radial_out"  # +R
    RADIAL_IN = "radial_in"  # -R
    NORMAL_PLUS = "normal_plus"  # +N
    NORMAL_MINUS = "normal_minus"  # -N

    @property
    def unit_rtn(self) -> np.ndarray:
        return {
            BurnDirection.PROGRADE: np.array([0.0, 1.0, 0.0]),
            BurnDirection.RETROGRADE: np.array([0.0, -1.0, 0.0]),
            BurnDirection.RADIAL_OUT: np.array([1.0, 0.0, 0.0]),
            BurnDirection.RADIAL_IN: np.array([-1.0, 0.0, 0.0]),
            BurnDirection.NORMAL_PLUS: np.array([0.0, 0.0, 1.0]),
            BurnDirection.NORMAL_MINUS: np.array([0.0, 0.0, -1.0]),
        }[self]


# --------------------------------------------------------------------------- #
# Clohessy-Wiltshire kernel
# --------------------------------------------------------------------------- #


def mean_motion_rad_s(r_eci, v_eci) -> float:
    el = state_to_classical_elements(r_eci, v_eci)
    return math.sqrt(MU_EARTH_KM3_S2 / el.a_km**3)


def cw_displacement_at_tca(dv_rtn_kms, n_rad_s: float, lead_time_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(position km, velocity km/s) change of the primary in RTN at TCA."""
    dvx, dvy, dvz = (float(x) for x in dv_rtn_kms)
    n, t = n_rad_s, lead_time_s
    s, c = math.sin(n * t), math.cos(n * t)
    x = (dvx / n) * s + (2 * dvy / n) * (1 - c)
    y = (2 * dvx / n) * (c - 1) + dvy * (4 * s / n - 3 * t)
    z = (dvz / n) * s
    xd = dvx * c + 2 * dvy * s
    yd = -2 * dvx * s + dvy * (4 * c - 3)
    zd = dvz * c
    return np.array([x, y, z]), np.array([xd, yd, zd])


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #


class ManeuverPlan(BaseModel):
    direction: BurnDirection
    dv_rtn_mps: Tuple[float, float, float]
    dv_mag_mps: float
    burn_epoch: datetime
    lead_time_s: float
    predicted_miss_km: float = Field(description="minimum separation (B-plane miss) after the burn")
    predicted_separation_at_tca_km: float = Field(description="separation at the nominal TCA epoch")
    predicted_miss_rtn_km: Tuple[float, float, float]
    predicted_pc: float
    baseline_pc: float
    feasible: bool
    propellant_kg: Optional[float] = None
    note: str = ""


class ManeuverStudy(BaseModel):
    primary_id: str
    secondary_id: str
    tca: datetime
    target_pc: float
    min_standoff_km: float
    baseline: ConjunctionAssessment
    plans: List[ManeuverPlan]
    recommended: Optional[ManeuverPlan]

    def alternatives(self) -> List[ManeuverPlan]:
        return [p for p in self.plans if self.recommended is None or p.direction != self.recommended.direction]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def _post_burn_pc_and_miss(
    geom: ConjunctionGeometry, dv_rtn_kms: np.ndarray, n: float, lead_s: float, cov_eci: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Return (pc, min_separation_km, separation_at_nominal_tca_km, miss_rtn) after the burn.

    For hypervelocity encounters an along-track displacement also shifts the
    time of closest approach, so the true minimum separation is the component
    of the new miss vector perpendicular to the relative velocity (the
    encounter-plane / B-plane miss), not the separation at the nominal TCA.
    """
    M = eci_to_rtn_matrix(geom.r_primary, geom.v_primary)  # x_rtn = M x_eci
    dr_rtn, dvel_rtn = cw_displacement_at_tca(dv_rtn_kms, n, lead_s)
    miss_eci = geom.miss_vector_eci() - M.T @ dr_rtn
    vrel_eci = geom.relative_velocity_eci() - M.T @ dvel_rtn
    hbr = geom.hbr_combined_km
    sep_tca = float(np.linalg.norm(miss_eci))
    vn = np.linalg.norm(vrel_eci)
    if vn > LOW_VELOCITY_THRESHOLD_KMS:
        pc = collision_probability_2d(miss_eci, vrel_eci, cov_eci, hbr, method="gauss", order=48)
        k = vrel_eci / vn
        perp = miss_eci - np.dot(miss_eci, k) * k
        min_sep = float(np.linalg.norm(perp))
    else:
        pc = collision_probability_3d(miss_eci, cov_eci, hbr, order=24)
        min_sep = sep_tca
    return pc, min_sep, sep_tca, M @ miss_eci


def plan_minimum_dv_cam(
    geom: ConjunctionGeometry,
    *,
    lead_time_s: float,
    target_pc: float = 1e-6,
    min_standoff_km: float = 1.0,
    directions: Sequence[BurnDirection] = tuple(BurnDirection),
    max_dv_mps: float = 5.0,
    spacecraft_mass_kg: Optional[float] = None,
    isp_s: Optional[float] = None,
    prograde_tie_break: float = 0.02,
) -> ManeuverStudy:
    """Find the smallest single impulse per canonical direction that clears the conjunction.

    Feasibility requires ``Pc <= target_pc`` AND ``miss >= min_standoff_km``
    at TCA. ``recommended`` is the feasible plan with minimum |dv|; a prograde
    plan within ``prograde_tie_break`` (fraction) of the minimum is preferred
    because it raises perigee and preserves drag margin.
    """
    if lead_time_s <= 0:
        raise ValueError("lead_time_s must be positive")
    baseline = assess_conjunction(geom)
    n = mean_motion_rad_s(geom.r_primary, geom.v_primary)
    cov = geom.combined_covariance_eci()
    burn_epoch = ensure_utc(geom.tca) - timedelta(seconds=lead_time_s)

    def ok(pc: float, miss: float) -> bool:
        return pc <= target_pc and miss >= min_standoff_km

    plans: List[ManeuverPlan] = []
    for direction in directions:
        u = direction.unit_rtn
        grid = np.geomspace(1e-4, max_dv_mps, 60)  # m/s
        feasible_m = None
        prev_m = 0.0
        for m in grid:
            pc, miss, _, _ = _post_burn_pc_and_miss(geom, u * m / 1000.0, n, lead_time_s, cov)
            if ok(pc, miss):
                feasible_m = float(m)
                break
            prev_m = float(m)
        if feasible_m is None:
            pc, miss, sep, miss_rtn = _post_burn_pc_and_miss(geom, u * max_dv_mps / 1000.0, n, lead_time_s, cov)
            plans.append(
                ManeuverPlan(
                    direction=direction,
                    dv_rtn_mps=tuple(u * max_dv_mps),
                    dv_mag_mps=max_dv_mps,
                    burn_epoch=burn_epoch,
                    lead_time_s=lead_time_s,
                    predicted_miss_km=miss,
                    predicted_separation_at_tca_km=sep,
                    predicted_miss_rtn_km=tuple(map(float, miss_rtn)),
                    predicted_pc=pc,
                    baseline_pc=baseline.pc,
                    feasible=False,
                    note=f"infeasible within {max_dv_mps} m/s",
                )
            )
            continue
        lo, hi = prev_m, feasible_m
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            pc, miss, _, _ = _post_burn_pc_and_miss(geom, u * mid / 1000.0, n, lead_time_s, cov)
            if ok(pc, miss):
                hi = mid
            else:
                lo = mid
        m_star = hi
        pc, miss, sep, miss_rtn = _post_burn_pc_and_miss(geom, u * m_star / 1000.0, n, lead_time_s, cov)
        prop = None
        if spacecraft_mass_kg and isp_s:
            prop = spacecraft_mass_kg * (1.0 - math.exp(-(m_star) / (isp_s * G0_M_S2)))
        plans.append(
            ManeuverPlan(
                direction=direction,
                dv_rtn_mps=tuple(float(x) for x in u * m_star),
                dv_mag_mps=float(m_star),
                burn_epoch=burn_epoch,
                lead_time_s=lead_time_s,
                predicted_miss_km=miss,
                predicted_separation_at_tca_km=sep,
                predicted_miss_rtn_km=tuple(map(float, miss_rtn)),
                predicted_pc=pc,
                baseline_pc=baseline.pc,
                feasible=True,
                propellant_kg=prop,
            )
        )

    feasible = sorted([p for p in plans if p.feasible], key=lambda p: p.dv_mag_mps)
    recommended = None
    if feasible:
        best = feasible[0]
        for p in feasible:
            if p.direction == BurnDirection.PROGRADE and p.dv_mag_mps <= best.dv_mag_mps * (1 + prograde_tie_break):
                best = p
                break
        recommended = best

    return ManeuverStudy(
        primary_id=geom.primary_id,
        secondary_id=geom.secondary_id,
        tca=geom.tca,
        target_pc=target_pc,
        min_standoff_km=min_standoff_km,
        baseline=baseline,
        plans=sorted(plans, key=lambda p: (not p.feasible, p.dv_mag_mps)),
        recommended=recommended,
    )


# --------------------------------------------------------------------------- #
# Post-manoeuvre secondary screening
# --------------------------------------------------------------------------- #


class ScreeningHit(BaseModel):
    object_id: str
    min_distance_km: float
    epoch: datetime
    violates_standoff: bool


class ScreeningResult(BaseModel):
    primary_id: str
    burn_epoch: datetime
    window_s: float
    standoff_km: float
    hits: List[ScreeningHit]

    @property
    def clear(self) -> bool:
        return not any(h.violates_standoff for h in self.hits)


def primary_state_at_burn(geom: ConjunctionGeometry, lead_time_s: float, tle: Optional[TLE] = None) -> StateVector:
    """Primary state at the burn epoch: SGP4 if a TLE is given, else two-body back-propagation from TCA."""
    burn_epoch = ensure_utc(geom.tca) - timedelta(seconds=lead_time_s)
    if tle is not None:
        return propagate_tle(tle, burn_epoch, frame=geom.frame)
    r, v = two_body_propagate(geom.r_primary, geom.v_primary, -lead_time_s)
    return StateVector.from_arrays(burn_epoch, r, v, frame=geom.frame)


def apply_impulse(state: StateVector, dv_rtn_mps) -> StateVector:
    M = eci_to_rtn_matrix(state.r_km, state.v_kms)
    dv_eci = M.T @ (np.asarray(dv_rtn_mps, float) / 1000.0)
    return StateVector.from_arrays(state.epoch, state.r_km, state.v_kms + dv_eci, frame=state.frame)


def screen_post_maneuver(
    primary_post_burn: StateVector,
    secondaries: Dict[str, Union[TLE, StateVector]],
    *,
    window_s: float,
    step_s: float = 30.0,
    standoff_km: float = 1.0,
) -> ScreeningResult:
    """Screen the post-burn primary trajectory against other catalogued objects.

    The primary is propagated two-body from the burn; each secondary is
    propagated via SGP4 (if a TLE) or two-body (if a state). The closest
    approach per object is located on a grid then refined with a bounded
    scalar minimisation.
    """
    from scipy.optimize import minimize_scalar

    t0 = ensure_utc(primary_post_burn.epoch)
    n_steps = int(window_s // step_s) + 1
    times = np.arange(n_steps) * step_s

    def primary_at(t: float) -> np.ndarray:
        return two_body_propagate(primary_post_burn.r_km, primary_post_burn.v_kms, float(t))[0]

    prim = np.array([primary_at(t) for t in times])

    hits: List[ScreeningHit] = []
    for oid, obj in secondaries.items():
        if isinstance(obj, TLE):

            def sec_at(t: float, _obj=obj) -> np.ndarray:
                return propagate_tle(_obj, t0 + timedelta(seconds=float(t)), frame=primary_post_burn.frame).r_km

        else:
            base = obj.to_frame(primary_post_burn.frame)
            offset = (t0 - ensure_utc(base.epoch)).total_seconds()

            def sec_at(t: float, _b=base, _off=offset) -> np.ndarray:
                return two_body_propagate(_b.r_km, _b.v_kms, _off + float(t))[0]

        sec = np.array([sec_at(t) for t in times])
        dist = np.linalg.norm(prim - sec, axis=1)
        k = int(np.argmin(dist))
        lo, hi = times[max(k - 1, 0)], times[min(k + 1, n_steps - 1)]
        if hi > lo:
            res = minimize_scalar(lambda t: float(np.linalg.norm(primary_at(t) - sec_at(t))), bounds=(lo, hi), method="bounded")
            t_min, d_min = float(res.x), float(res.fun)
        else:
            t_min, d_min = float(times[k]), float(dist[k])
        hits.append(
            ScreeningHit(
                object_id=oid,
                min_distance_km=d_min,
                epoch=t0 + timedelta(seconds=t_min),
                violates_standoff=d_min < standoff_km,
            )
        )

    return ScreeningResult(
        primary_id="primary",
        burn_epoch=t0,
        window_s=window_s,
        standoff_km=standoff_km,
        hits=sorted(hits, key=lambda h: h.min_distance_km),
    )
