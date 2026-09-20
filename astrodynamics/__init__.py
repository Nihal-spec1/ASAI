"""ASAI deterministic astrodynamics layer.

Every quantity produced here is the output of a closed-form or numerically
integrated computation. Nothing in this package calls a language model.
"""

from .coordinates import (
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    eci_to_rtn,
    eci_to_rtn_matrix,
    relative_state_rtn,
    rtn_to_eci,
    teme_to_j2000,
    j2000_to_teme,
)
from .propagation import (
    TLE,
    StateVector,
    build_tle,
    propagate_tle,
    propagate_range,
    two_body_propagate,
    state_to_classical_elements,
    state_to_tle,
)
from .foster_pc import (
    ConjunctionGeometry,
    ConjunctionAssessment,
    assess_conjunction,
    collision_probability_2d,
    collision_probability_3d,
    rtn_covariance_to_eci,
)
from .maneuver_planner import (
    BurnDirection,
    ManeuverPlan,
    ManeuverStudy,
    cw_displacement_at_tca,
    plan_minimum_dv_cam,
    screen_post_maneuver,
)

__all__ = [name for name in dir() if not name.startswith("_")]
