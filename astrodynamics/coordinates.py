"""Reference-frame conversions: TEME <-> J2000 (GCRS) and ECI <-> RTN.

Conventions
-----------
* All positions in km, velocities in km/s, unless a name says otherwise.
* TEME (True Equator, Mean Equinox) is the native output frame of SGP4.
* "J2000" here means the GCRS/ICRS-aligned inertial frame that Skyfield
  returns from ``EarthSatellite.at``; the difference to the strict J2000
  mean equator frame is a fixed ~20 mas frame bias, irrelevant at km scale.
* RTN is the primary-centred local orbital frame:
    R  = radial  (along +r)
    T  = transverse / along-track (in the orbital plane, roughly along +v)
    N  = normal (along r x v, angular-momentum direction)
  RTN is identical to what many CDMs call "RIC" (Radial, In-track, Cross-track).

Skyfield's frame convention (verified empirically in this repo) is that
``TEME.rotation_at(t)`` returns the matrix R such that ``x_TEME = R @ x_GCRS``.
Hence ``x_GCRS = R.T @ x_TEME`` for both position and velocity; the rotation
rate of TEME relative to GCRS (precession + nutation, ~50 arcsec/yr) is
negligible for velocity at the mm/s level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Tuple

import numpy as np

MU_EARTH_KM3_S2: float = 398600.4418  # WGS-84 gravitational parameter
R_EARTH_KM: float = 6378.137  # WGS-84 equatorial radius

Vector3 = np.ndarray


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _timescale():
    from skyfield.api import load

    return load.timescale()


def ensure_utc(epoch: datetime) -> datetime:
    """Return a timezone-aware UTC datetime (naive input is assumed UTC)."""
    if epoch.tzinfo is None:
        return epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc)


def skyfield_time(epoch: datetime):
    """Convert a datetime to a Skyfield ``Time`` (UTC)."""
    e = ensure_utc(epoch)
    ts = _timescale()
    return ts.utc(e.year, e.month, e.day, e.hour, e.minute, e.second + e.microsecond * 1e-6)


# --------------------------------------------------------------------------- #
# TEME <-> J2000 / GCRS
# --------------------------------------------------------------------------- #
def teme_rotation_matrix(epoch: datetime) -> np.ndarray:
    """Matrix R with ``x_TEME = R @ x_GCRS`` at ``epoch``."""
    from skyfield.sgp4lib import TEME

    return np.asarray(TEME.rotation_at(skyfield_time(epoch)), dtype=float)


def teme_to_j2000(r_teme: Vector3, v_teme: Vector3, epoch: datetime) -> Tuple[Vector3, Vector3]:
    """Rotate a TEME state vector into the J2000/GCRS inertial frame."""
    R = teme_rotation_matrix(epoch)
    return R.T @ np.asarray(r_teme, float), R.T @ np.asarray(v_teme, float)


def j2000_to_teme(r_j2000: Vector3, v_j2000: Vector3, epoch: datetime) -> Tuple[Vector3, Vector3]:
    """Rotate a J2000/GCRS state vector into TEME."""
    R = teme_rotation_matrix(epoch)
    return R @ np.asarray(r_j2000, float), R @ np.asarray(v_j2000, float)


# --------------------------------------------------------------------------- #
# ECI <-> RTN
# --------------------------------------------------------------------------- #
def eci_to_rtn_matrix(r_eci: Vector3, v_eci: Vector3) -> np.ndarray:
    """Rows are the unit vectors R, T, N expressed in ECI.

    ``x_rtn = M @ x_eci`` and ``x_eci = M.T @ x_rtn``.
    """
    r = np.asarray(r_eci, float)
    v = np.asarray(v_eci, float)
    if np.linalg.norm(r) == 0.0:
        raise ValueError("position vector must be non-zero")
    h = np.cross(r, v)
    if np.linalg.norm(h) == 0.0:
        raise ValueError("r and v are parallel; RTN frame undefined")
    R_hat = r / np.linalg.norm(r)
    N_hat = h / np.linalg.norm(h)
    T_hat = np.cross(N_hat, R_hat)
    return np.vstack([R_hat, T_hat, N_hat])


def eci_to_rtn(vec_eci: Vector3, r_ref: Vector3, v_ref: Vector3) -> Vector3:
    """Express an ECI vector in the RTN frame defined by the reference state."""
    return eci_to_rtn_matrix(r_ref, v_ref) @ np.asarray(vec_eci, float)


def rtn_to_eci(vec_rtn: Vector3, r_ref: Vector3, v_ref: Vector3) -> Vector3:
    """Express an RTN vector in ECI using the reference state."""
    return eci_to_rtn_matrix(r_ref, v_ref).T @ np.asarray(vec_rtn, float)


def relative_state_rtn(
    r_primary: Vector3, v_primary: Vector3, r_secondary: Vector3, v_secondary: Vector3
) -> Tuple[Vector3, Vector3]:
    """Secondary-minus-primary relative position and velocity in the primary's RTN frame.

    The RTN frame is treated as instantaneously inertial (no transport term),
    which is the standard CDM convention for reporting relative state at TCA.
    """
    M = eci_to_rtn_matrix(r_primary, v_primary)
    dr = np.asarray(r_secondary, float) - np.asarray(r_primary, float)
    dv = np.asarray(v_secondary, float) - np.asarray(v_primary, float)
    return M @ dr, M @ dv


def rotate_covariance(cov: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Return ``M @ cov @ M.T`` (covariance under the linear map ``x' = M x``)."""
    cov = np.asarray(cov, float)
    return M @ cov @ M.T
