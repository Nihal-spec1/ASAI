"""Deterministic orbit propagation: SGP4 from TLEs, plus a two-body integrator.

Time convention: SGP4 is fed plain UTC (matching Skyfield's ``EarthSatellite``),
because TLE epochs are UTC per AIAA 2006-6753.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sgp4.api import SGP4_ERRORS, Satrec, jday
from scipy.integrate import solve_ivp

from .coordinates import MU_EARTH_KM3_S2, R_EARTH_KM, ensure_utc, j2000_to_teme, teme_to_j2000

Frame = Literal["TEME", "J2000"]

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class StateVector(BaseModel):
    """Inertial Cartesian state (km, km/s) at a UTC epoch."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epoch: datetime
    r: Tuple[float, float, float] = Field(description="position km")
    v: Tuple[float, float, float] = Field(description="velocity km/s")
    frame: Frame = "J2000"

    @field_validator("epoch")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @property
    def r_km(self) -> np.ndarray:
        return np.array(self.r, dtype=float)

    @property
    def v_kms(self) -> np.ndarray:
        return np.array(self.v, dtype=float)

    @property
    def altitude_km(self) -> float:
        return float(np.linalg.norm(self.r_km) - R_EARTH_KM)

    @property
    def speed_kms(self) -> float:
        return float(np.linalg.norm(self.v_kms))

    def to_frame(self, frame: Frame) -> "StateVector":
        if frame == self.frame:
            return self
        if frame == "J2000":
            r, v = teme_to_j2000(self.r_km, self.v_kms, self.epoch)
        else:
            r, v = j2000_to_teme(self.r_km, self.v_kms, self.epoch)
        return StateVector(epoch=self.epoch, r=tuple(r), v=tuple(v), frame=frame)

    @classmethod
    def from_arrays(cls, epoch: datetime, r, v, frame: Frame = "J2000") -> "StateVector":
        return cls(epoch=epoch, r=tuple(float(x) for x in r), v=tuple(float(x) for x in v), frame=frame)


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum over the first 68 characters ('-' counts as 1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


class TLE(BaseModel):
    """A validated two-line element set."""

    name: str = ""
    line1: str
    line2: str

    @field_validator("line1", "line2")
    @classmethod
    def _validate_line(cls, line: str) -> str:
        line = line.rstrip("\n")
        if len(line) != 69:
            raise ValueError(f"TLE line must be 69 characters, got {len(line)}: {line!r}")
        if int(line[68]) != tle_checksum(line):
            raise ValueError(f"TLE checksum mismatch on line: {line!r}")
        return line

    @property
    def norad_id(self) -> int:
        return int(self.line1[2:7])

    @property
    def intl_designator(self) -> str:
        return self.line1[9:17].strip()

    @property
    def epoch(self) -> datetime:
        yy = int(self.line1[18:20])
        year = 2000 + yy if yy < 57 else 1900 + yy
        doy = float(self.line1[20:32])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)

    @property
    def inclination_deg(self) -> float:
        return float(self.line2[8:16])

    @property
    def raan_deg(self) -> float:
        return float(self.line2[17:25])

    @property
    def eccentricity(self) -> float:
        return float("0." + self.line2[26:33].strip())

    @property
    def argp_deg(self) -> float:
        return float(self.line2[34:42])

    @property
    def mean_anomaly_deg(self) -> float:
        return float(self.line2[43:51])

    @property
    def mean_motion_rev_day(self) -> float:
        return float(self.line2[52:63])

    @property
    def period_minutes(self) -> float:
        return 1440.0 / self.mean_motion_rev_day

    def satrec(self) -> Satrec:
        return Satrec.twoline2rv(self.line1, self.line2)


# --------------------------------------------------------------------------- #
# TLE construction (for synthetic fixtures and state->TLE approximation)
# --------------------------------------------------------------------------- #


def _fmt_assumed_decimal(x: float) -> str:
    """Format like TLE bstar / ndd fields: sign, 5 mantissa digits, signed exponent."""
    if x == 0.0:
        return " 00000-0"
    sign = "-" if x < 0 else " "
    ax = abs(x)
    exp = math.floor(math.log10(ax)) + 1
    mant = ax / 10**exp  # in [0.1, 1)
    digits = int(round(mant * 1e5))
    if digits >= 100000:
        digits //= 10
        exp += 1
    if exp > 9 or exp < -9:
        raise ValueError(f"value {x} out of TLE exponent range")
    return f"{sign}{digits:05d}{exp:+d}"


def _fmt_ndot(x: float) -> str:
    sign = "-" if x < 0 else " "
    frac = f"{abs(x):.8f}"  # 0.00001764
    return f"{sign}.{frac.split('.')[1]}"


def _epoch_to_tle_epoch(epoch: datetime) -> str:
    e = ensure_utc(epoch)
    start = datetime(e.year, 1, 1, tzinfo=timezone.utc)
    doy = (e - start).total_seconds() / 86400.0 + 1.0
    return f"{e.year % 100:02d}{doy:012.8f}"


def build_tle(
    *,
    norad_id: int,
    epoch: datetime,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    argp_deg: float,
    mean_anomaly_deg: float,
    mean_motion_rev_day: float,
    name: str = "",
    intl_designator: str = "26001A",
    classification: str = "U",
    ndot_rev_day2: float = 0.0,
    nddot: float = 0.0,
    bstar: float = 0.0,
    element_set_no: int = 999,
    rev_number: int = 1,
) -> TLE:
    """Assemble a checksum-correct TLE from element values."""
    if not (0 <= eccentricity < 1):
        raise ValueError("eccentricity must be in [0,1)")
    l1 = (
        f"1 {norad_id:05d}{classification} {intl_designator:<8s} {_epoch_to_tle_epoch(epoch)} "
        f"{_fmt_ndot(ndot_rev_day2)} {_fmt_assumed_decimal(nddot)} {_fmt_assumed_decimal(bstar)} 0 "
        f"{element_set_no % 10000:4d}"
    )
    l2 = (
        f"2 {norad_id:05d} {inclination_deg % 360:8.4f} {raan_deg % 360:8.4f} "
        f"{int(round(eccentricity * 1e7)):07d} {argp_deg % 360:8.4f} {mean_anomaly_deg % 360:8.4f} "
        f"{mean_motion_rev_day:11.8f}{rev_number % 100000:5d}"
    )
    assert len(l1) == 68 and len(l2) == 68, (len(l1), len(l2))
    l1 += str(tle_checksum(l1))
    l2 += str(tle_checksum(l2))
    return TLE(name=name, line1=l1, line2=l2)


# --------------------------------------------------------------------------- #
# SGP4 propagation
# --------------------------------------------------------------------------- #


def _jd_utc(epoch: datetime) -> Tuple[float, float]:
    e = ensure_utc(epoch)
    return jday(e.year, e.month, e.day, e.hour, e.minute, e.second + e.microsecond * 1e-6)


def propagate_tle(tle: TLE, epoch: datetime, frame: Frame = "J2000") -> StateVector:
    """SGP4 state at ``epoch`` (UTC) in the requested inertial frame."""
    jd, fr = _jd_utc(epoch)
    err, r, v = tle.satrec().sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error {err} for NORAD {tle.norad_id}: {SGP4_ERRORS[err]}")
    state = StateVector.from_arrays(ensure_utc(epoch), r, v, frame="TEME")
    return state.to_frame(frame)


def propagate_range(
    tle: TLE, start: datetime, end: datetime, step_s: float, frame: Frame = "J2000"
) -> List[StateVector]:
    """SGP4 states from ``start`` to ``end`` inclusive at ``step_s`` seconds."""
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    start, end = ensure_utc(start), ensure_utc(end)
    n = int(math.floor((end - start).total_seconds() / step_s)) + 1
    return [propagate_tle(tle, start + timedelta(seconds=i * step_s), frame) for i in range(n)]


# --------------------------------------------------------------------------- #
# Two-body integration (used for post-burn screening and CW cross-checks)
# --------------------------------------------------------------------------- #


def _two_body_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    rn = np.linalg.norm(r)
    return np.concatenate([y[3:], -MU_EARTH_KM3_S2 * r / rn**3])


def two_body_propagate(r0, v0, dt_s: float, *, rtol: float = 1e-11, atol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a Cartesian state under point-mass gravity for ``dt_s`` seconds."""
    y0 = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    if dt_s == 0.0:
        return y0[:3].copy(), y0[3:].copy()
    sol = solve_ivp(_two_body_rhs, (0.0, dt_s), y0, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"two-body integration failed: {sol.message}")
    return sol.y[:3, -1], sol.y[3:, -1]


# --------------------------------------------------------------------------- #
# Classical elements
# --------------------------------------------------------------------------- #


class ClassicalElements(BaseModel):
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    nu_deg: float
    M_deg: float
    period_s: float
    mean_motion_rev_day: float
    h_km2_s: float


def state_to_classical_elements(r, v, mu: float = MU_EARTH_KM3_S2) -> ClassicalElements:
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    nn = np.linalg.norm(n)
    e_vec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    energy = vn**2 / 2 - mu / rn
    if energy >= 0:
        raise ValueError("state is not a bound orbit")
    a = -mu / (2 * energy)
    i = math.acos(np.clip(h[2] / hn, -1, 1))
    tol = 1e-10
    if nn < tol:  # equatorial
        raan = 0.0
        if e < tol:
            argp = 0.0
            nu = math.atan2(r[1], r[0])
        else:
            argp = math.atan2(e_vec[1], e_vec[0])
            nu = math.acos(np.clip(np.dot(e_vec, r) / (e * rn), -1, 1))
            if np.dot(r, v) < 0:
                nu = 2 * math.pi - nu
    else:
        raan = math.atan2(n[1], n[0])
        if e < tol:
            argp = 0.0
            nu = math.acos(np.clip(np.dot(n, r) / (nn * rn), -1, 1))
            if r[2] < 0:
                nu = 2 * math.pi - nu
        else:
            argp = math.acos(np.clip(np.dot(n, e_vec) / (nn * e), -1, 1))
            if e_vec[2] < 0:
                argp = 2 * math.pi - argp
            nu = math.acos(np.clip(np.dot(e_vec, r) / (e * rn), -1, 1))
            if np.dot(r, v) < 0:
                nu = 2 * math.pi - nu
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = (E - e * math.sin(E)) % (2 * math.pi)
    period = 2 * math.pi * math.sqrt(a**3 / mu)
    return ClassicalElements(
        a_km=float(a),
        e=e,
        i_deg=math.degrees(i),
        raan_deg=math.degrees(raan) % 360,
        argp_deg=math.degrees(argp) % 360,
        nu_deg=math.degrees(nu) % 360,
        M_deg=math.degrees(M),
        period_s=float(period),
        mean_motion_rev_day=86400.0 / period,
        h_km2_s=float(hn),
    )


def state_to_tle(
    state: StateVector,
    *,
    norad_id: int,
    name: str = "",
    intl_designator: str = "26001A",
    bstar: float = 0.0,
    iterations: int = 60,
    tol_km: float = 1e-4,
) -> TLE:
    """Fit a TLE to a Cartesian state at the state's epoch.

    Starting from the osculating elements, the TLE mean elements are refined
    by fixed-point iteration so that SGP4 evaluated at the epoch reproduces
    the input state (typically to a few metres after ~10 iterations). The fit
    is exact only at the epoch; conjunction math in ASAI always uses the
    stored state vectors, never a regenerated TLE.
    """
    teme = state.to_frame("TEME")
    target = state_to_classical_elements(teme.r_km, teme.v_kms)
    mean = dict(
        inclination_deg=target.i_deg,
        raan_deg=target.raan_deg,
        eccentricity=target.e,
        argp_deg=target.argp_deg,
        mean_anomaly_deg=target.M_deg,
        mean_motion_rev_day=target.mean_motion_rev_day,
    )

    def wrap(delta_deg: float) -> float:
        return (delta_deg + 180.0) % 360.0 - 180.0

    def make(m) -> TLE:
        return build_tle(
            norad_id=norad_id,
            name=name,
            intl_designator=intl_designator,
            epoch=teme.epoch,
            bstar=bstar,
            **m,
        )

    tle = make(mean)
    for _ in range(iterations):
        got = propagate_tle(tle, teme.epoch, frame="TEME")
        osc = state_to_classical_elements(got.r_km, got.v_kms)
        mean = dict(
            inclination_deg=mean["inclination_deg"] + (target.i_deg - osc.i_deg),
            raan_deg=mean["raan_deg"] + wrap(target.raan_deg - osc.raan_deg),
            eccentricity=min(max(mean["eccentricity"] + (target.e - osc.e), 1e-7), 0.99),
            argp_deg=mean["argp_deg"] + wrap(target.argp_deg - osc.argp_deg),
            mean_anomaly_deg=mean["mean_anomaly_deg"] + wrap(target.M_deg - osc.M_deg),
            mean_motion_rev_day=mean["mean_motion_rev_day"] + (target.mean_motion_rev_day - osc.mean_motion_rev_day),
        )
        tle = make(mean)
        if np.linalg.norm(got.r_km - teme.r_km) < tol_km:
            break
    return tle
