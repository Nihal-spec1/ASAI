"""Collision probability (Pc) from miss geometry and covariance.

Two deterministic estimators are provided:

``collision_probability_2d``
    The Foster / Alfano encounter-plane integral used for hypervelocity
    (short-duration, rectilinear) encounters. The combined covariance is
    projected onto the plane perpendicular to the relative velocity and the
    bivariate Gaussian is integrated over the disk of radius HBR (combined
    hard-body radius) centred on the projected miss vector.

``collision_probability_3d``
    The instantaneous 3-D probability that the relative position at TCA lies
    inside the combined hard-body sphere, integrating the trivariate Gaussian
    over the sphere. This is the appropriate variant when the relative
    velocity is low (RPO / co-orbital cases) and the encounter-plane
    approximation breaks down.

All covariances are position covariances in km^2. Combined covariance is the
sum of the two objects' covariances expressed in the same inertial frame,
which assumes the two state errors are independent.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy import integrate

from .coordinates import eci_to_rtn_matrix, ensure_utc, relative_state_rtn

Matrix3 = List[List[float]]
PcMethod = Literal["adaptive", "gauss"]

# --------------------------------------------------------------------------- #
# Covariance helpers
# --------------------------------------------------------------------------- #


def _as_cov(cov) -> np.ndarray:
    c = np.asarray(cov, float)
    if c.shape != (3, 3):
        raise ValueError("covariance must be 3x3")
    if not np.allclose(c, c.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(c) < -1e-15):
        raise ValueError("covariance must be positive semi-definite")
    return c


def rtn_covariance_to_eci(cov_rtn, r_eci, v_eci) -> np.ndarray:
    """Rotate a covariance expressed in the RTN frame of (r, v) into ECI."""
    M = eci_to_rtn_matrix(r_eci, v_eci)  # x_rtn = M x_eci
    return M.T @ _as_cov(cov_rtn) @ M


def diagonal_rtn_covariance(sigma_r_km: float, sigma_t_km: float, sigma_n_km: float) -> np.ndarray:
    return np.diag([sigma_r_km**2, sigma_t_km**2, sigma_n_km**2])


# --------------------------------------------------------------------------- #
# Encounter plane geometry
# --------------------------------------------------------------------------- #


def encounter_plane_basis(miss_eci, v_rel_eci) -> np.ndarray:
    """Rows: i_hat (in-plane, along projected miss), j_hat, k_hat (along v_rel).

    If the projected miss is zero, i_hat is chosen arbitrarily perpendicular
    to k_hat.
    """
    v = np.asarray(v_rel_eci, float)
    vn = np.linalg.norm(v)
    if vn == 0:
        raise ValueError("relative velocity is zero; use collision_probability_3d")
    k = v / vn
    d = np.asarray(miss_eci, float)
    d_perp = d - np.dot(d, k) * k
    if np.linalg.norm(d_perp) < 1e-12:
        trial = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        d_perp = trial - np.dot(trial, k) * k
    i = d_perp / np.linalg.norm(d_perp)
    j = np.cross(k, i)
    return np.vstack([i, j, k])


def project_to_encounter_plane(miss_eci, v_rel_eci, cov_eci) -> Tuple[np.ndarray, np.ndarray]:
    """Return (miss_2d, cov_2d) in the encounter plane. miss_2d = (d, 0)."""
    B = encounter_plane_basis(miss_eci, v_rel_eci)
    P = B[:2]
    miss_2d = P @ np.asarray(miss_eci, float)
    cov_2d = P @ _as_cov(cov_eci) @ P.T
    return miss_2d, cov_2d


# --------------------------------------------------------------------------- #
# 2-D Foster / Alfano integral
# --------------------------------------------------------------------------- #


def _pc_2d_from_plane(miss_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float, method: PcMethod, order: int) -> float:
    det = np.linalg.det(cov_2d)
    if det <= 0:
        raise ValueError("projected covariance is singular")
    inv = np.linalg.inv(cov_2d)
    norm = 1.0 / (2 * math.pi * math.sqrt(det))
    d = float(np.hypot(*miss_2d))
    if hbr_km <= 0:
        return 0.0

    # The relative-position distribution is centred on the ORIGIN of the
    # encounter plane; the hard-body disk is centred on the miss point (d, 0).
    def pdf_xy(x, y):
        return norm * math.exp(-0.5 * (inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y))

    if method == "adaptive":
        # polar coordinates about the miss point: (d + rho cos th, rho sin th)
        val, _err = integrate.dblquad(
            lambda th, rho: pdf_xy(d + rho * math.cos(th), rho * math.sin(th)) * rho,
            0.0,
            hbr_km,
            0.0,
            2 * math.pi,
            epsabs=1e-14,
            epsrel=1e-8,
        )
        return float(min(max(val, 0.0), 1.0))

    # fixed-order Gauss-Legendre tensor product (fast, deterministic)
    xr, wr = np.polynomial.legendre.leggauss(order)
    rho = 0.5 * hbr_km * (xr + 1.0)
    w_rho = 0.5 * hbr_km * wr
    th = math.pi * (xr + 1.0)
    w_th = math.pi * wr
    RHO, TH = np.meshgrid(rho, th, indexing="ij")
    X = d + RHO * np.cos(TH)  # absolute encounter-plane coordinates
    Y = RHO * np.sin(TH)
    quad = inv[0, 0] * X * X + 2 * inv[0, 1] * X * Y + inv[1, 1] * Y * Y
    vals = norm * np.exp(-0.5 * quad) * RHO
    total = float(np.einsum("i,j,ij->", w_rho, w_th, vals))
    return min(max(total, 0.0), 1.0)


def collision_probability_2d(
    miss_eci,
    v_rel_eci,
    cov_combined_eci,
    hbr_km: float,
    *,
    method: PcMethod = "adaptive",
    order: int = 64,
) -> float:
    """Foster/Alfano encounter-plane collision probability."""
    miss_2d, cov_2d = project_to_encounter_plane(miss_eci, v_rel_eci, cov_combined_eci)
    return _pc_2d_from_plane(miss_2d, cov_2d, hbr_km, method, order)


# --------------------------------------------------------------------------- #
# 3-D instantaneous integral
# --------------------------------------------------------------------------- #


def collision_probability_3d(miss_eci, cov_combined_eci, hbr_km: float, *, order: int = 32) -> float:
    """Probability that the relative position lies within the HBR sphere at TCA.

    Gauss-Legendre tensor quadrature in spherical coordinates about the miss
    point. ``order`` nodes per dimension (default 32^3 = 32768 evaluations).
    """
    cov = _as_cov(cov_combined_eci)
    det = np.linalg.det(cov)
    if det <= 0:
        raise ValueError("covariance is singular")
    if hbr_km <= 0:
        return 0.0
    inv = np.linalg.inv(cov)
    norm = 1.0 / ((2 * math.pi) ** 1.5 * math.sqrt(det))
    d = np.asarray(miss_eci, float)

    x, w = np.polynomial.legendre.leggauss(order)
    rho = 0.5 * hbr_km * (x + 1.0)
    w_rho = 0.5 * hbr_km * w
    th = 0.5 * math.pi * (x + 1.0)
    w_th = 0.5 * math.pi * w
    ph = math.pi * (x + 1.0)
    w_ph = math.pi * w
    RHO, TH, PH = np.meshgrid(rho, th, ph, indexing="ij")
    X = RHO * np.sin(TH) * np.cos(PH)
    Y = RHO * np.sin(TH) * np.sin(PH)
    Z = RHO * np.cos(TH)
    # offset from origin of the distribution (mean 0) to point d + (X,Y,Z)
    px, py, pz = d[0] + X, d[1] + Y, d[2] + Z
    quad = (
        inv[0, 0] * px * px
        + inv[1, 1] * py * py
        + inv[2, 2] * pz * pz
        + 2 * inv[0, 1] * px * py
        + 2 * inv[0, 2] * px * pz
        + 2 * inv[1, 2] * py * pz
    )
    vals = norm * np.exp(-0.5 * quad) * RHO * RHO * np.sin(TH)
    total = float(np.einsum("i,j,k,ijk->", w_rho, w_th, w_ph, vals))
    return min(max(total, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# End-to-end conjunction assessment
# --------------------------------------------------------------------------- #


class ConjunctionGeometry(BaseModel):
    """Everything needed to compute Pc for one conjunction at TCA."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tca: datetime
    primary_id: str
    secondary_id: str
    r_primary: Tuple[float, float, float]
    v_primary: Tuple[float, float, float]
    r_secondary: Tuple[float, float, float]
    v_secondary: Tuple[float, float, float]
    cov_primary_rtn: Matrix3 = Field(description="3x3 position covariance km^2 in primary RTN")
    cov_secondary_rtn: Matrix3 = Field(description="3x3 position covariance km^2 in secondary RTN")
    hbr_primary_m: float = Field(gt=0)
    hbr_secondary_m: float = Field(gt=0)
    frame: Literal["J2000", "TEME"] = "J2000"

    @field_validator("tca")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @property
    def hbr_combined_km(self) -> float:
        return (self.hbr_primary_m + self.hbr_secondary_m) / 1000.0

    def combined_covariance_eci(self) -> np.ndarray:
        cp = rtn_covariance_to_eci(self.cov_primary_rtn, self.r_primary, self.v_primary)
        cs = rtn_covariance_to_eci(self.cov_secondary_rtn, self.r_secondary, self.v_secondary)
        return cp + cs

    def miss_vector_eci(self) -> np.ndarray:
        return np.asarray(self.r_secondary, float) - np.asarray(self.r_primary, float)

    def relative_velocity_eci(self) -> np.ndarray:
        return np.asarray(self.v_secondary, float) - np.asarray(self.v_primary, float)

    def with_covariances(self, cov_primary_rtn, cov_secondary_rtn) -> "ConjunctionGeometry":
        return self.model_copy(
            update={
                "cov_primary_rtn": np.asarray(cov_primary_rtn, float).tolist(),
                "cov_secondary_rtn": np.asarray(cov_secondary_rtn, float).tolist(),
            }
        )

    def with_secondary_state(self, r_secondary, v_secondary) -> "ConjunctionGeometry":
        return self.model_copy(
            update={"r_secondary": tuple(map(float, r_secondary)), "v_secondary": tuple(map(float, v_secondary))}
        )


class ConjunctionAssessment(BaseModel):
    """Deterministic summary of a conjunction's risk state."""

    primary_id: str
    secondary_id: str
    tca: datetime
    miss_distance_km: float
    miss_rtn_km: Tuple[float, float, float] = Field(description="secondary - primary in primary RTN")
    relative_speed_kms: float
    encounter_regime: Literal["hypervelocity", "low-velocity"]
    pc_2d: float
    pc_3d: float
    pc: float = Field(description="Pc selected for the regime")
    hbr_combined_km: float
    bplane_miss_km: float
    bplane_sigma_major_km: float
    bplane_sigma_minor_km: float
    mahalanobis_distance: float
    covariance_volume_km3: float
    in_dilution_region: bool = Field(
        description="True when shrinking covariance would RAISE Pc, i.e. Pc is uncertainty-driven"
    )
    sigma_to_miss_ratio: float = Field(description="major B-plane sigma / B-plane miss; >1 means miss is inside 1-sigma")

    @property
    def exceeds_threshold(self) -> bool:
        return self.pc >= 1e-4


LOW_VELOCITY_THRESHOLD_KMS = 0.1


def assess_conjunction(geom: ConjunctionGeometry, *, pc_threshold: float = 1e-4, method: PcMethod = "adaptive") -> ConjunctionAssessment:
    miss = geom.miss_vector_eci()
    v_rel = geom.relative_velocity_eci()
    cov = geom.combined_covariance_eci()
    hbr = geom.hbr_combined_km
    miss_km = float(np.linalg.norm(miss))
    v_rel_n = float(np.linalg.norm(v_rel))

    miss_rtn, _ = relative_state_rtn(geom.r_primary, geom.v_primary, geom.r_secondary, geom.v_secondary)

    pc3 = collision_probability_3d(miss, cov, hbr)
    if v_rel_n > LOW_VELOCITY_THRESHOLD_KMS:
        regime = "hypervelocity"
        miss_2d, cov_2d = project_to_encounter_plane(miss, v_rel, cov)
        pc2 = _pc_2d_from_plane(miss_2d, cov_2d, hbr, method, 64)
        eig = np.sort(np.linalg.eigvalsh(cov_2d))[::-1]
        b_miss = float(np.hypot(*miss_2d))
        maha = float(math.sqrt(miss_2d @ np.linalg.inv(cov_2d) @ miss_2d))
        sig_major, sig_minor = math.sqrt(max(eig[0], 0)), math.sqrt(max(eig[1], 0))
        pc_sel = pc2
        # dilution test: does halving the covariance raise Pc?
        pc_half = _pc_2d_from_plane(miss_2d, 0.25 * cov_2d, hbr, "gauss", 64)
    else:
        regime = "low-velocity"
        pc2 = collision_probability_2d(miss, v_rel, cov, hbr, method=method) if v_rel_n > 0 else 0.0
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        b_miss = miss_km
        maha = float(math.sqrt(miss @ np.linalg.inv(cov) @ miss))
        sig_major, sig_minor = math.sqrt(eig[0]), math.sqrt(eig[-1])
        pc_sel = pc3
        pc_half = collision_probability_3d(miss, 0.25 * cov, hbr)

    vol = float(4.0 / 3.0 * math.pi * math.sqrt(max(np.linalg.det(cov), 0.0)))
    return ConjunctionAssessment(
        primary_id=geom.primary_id,
        secondary_id=geom.secondary_id,
        tca=geom.tca,
        miss_distance_km=miss_km,
        miss_rtn_km=tuple(float(x) for x in miss_rtn),
        relative_speed_kms=v_rel_n,
        encounter_regime=regime,
        pc_2d=pc2,
        pc_3d=pc3,
        pc=pc_sel,
        hbr_combined_km=hbr,
        bplane_miss_km=b_miss,
        bplane_sigma_major_km=sig_major,
        bplane_sigma_minor_km=sig_minor,
        mahalanobis_distance=maha,
        covariance_volume_km3=vol,
        in_dilution_region=bool(pc_half > pc_sel),
        sigma_to_miss_ratio=float(sig_major / b_miss) if b_miss > 0 else float("inf"),
    )
