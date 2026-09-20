"""Pydantic schema for ``data/fixtures.json``."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from graph.models import GroundStation, LaunchEvent, Operator

Matrix3 = List[List[float]]
Vec3 = Tuple[float, float, float]


class TLEFixture(BaseModel):
    line1: str
    line2: str


class StateFixture(BaseModel):
    epoch: datetime
    r: Vec3
    v: Vec3
    frame: Literal["J2000", "TEME"] = "J2000"


class ObjectFixture(BaseModel):
    id: str
    name: str
    norad_id: int
    kind: Literal["satellite", "debris"]
    cospar_id: str = ""
    operator_id: Optional[str] = None
    launch_id: Optional[str] = None
    parent_object_id: Optional[str] = None
    fragmentation_event_id: Optional[str] = None
    tle: Optional[TLEFixture] = None
    hbr_m: float = 5.0
    mass_kg: Optional[float] = None
    isp_s: Optional[float] = None
    bus_type: str = ""
    status: Literal["operational", "degraded", "non-operational", "unknown"] = "operational"
    orbit_regime: Literal["LEO", "MEO", "GEO", "HEO", "unknown"] = "LEO"
    country: str = ""
    rcs_category: Literal["small", "medium", "large"] = "medium"
    registered: bool = True
    launch_date: Optional[date] = None
    role: Literal["primary", "secondary", "neighbor", "catalog", "parent"] = "catalog"


class EdgeFixture(BaseModel):
    type: str
    source: str
    target: str
    attrs: Dict[str, object] = Field(default_factory=dict)


class GraphFixture(BaseModel):
    operators: List[Operator] = Field(default_factory=list)
    launch_events: List[LaunchEvent] = Field(default_factory=list)
    ground_stations: List[GroundStation] = Field(default_factory=list)
    edges: List[EdgeFixture] = Field(default_factory=list)


class CovariancePair(BaseModel):
    primary_rtn: Matrix3
    secondary_rtn: Matrix3
    note: str = ""


class ConjunctionFixture(BaseModel):
    primary_id: str
    secondary_id: str
    tca: datetime
    frame: Literal["J2000", "TEME"] = "J2000"
    states_pre: Dict[str, StateFixture]
    covariance_pre: CovariancePair
    states_post: Optional[Dict[str, StateFixture]] = None
    covariance_post: Optional[CovariancePair] = None
    screening_pc_threshold: float = 1e-4
    cdm_reported_miss_km: Optional[float] = None
    cdm_reported_pc: Optional[float] = None


class ObservationFixture(BaseModel):
    sensor_type: Literal["radar", "optical", "stc", "telemetry"]
    sensor_id: str
    epoch: datetime
    description: str
    governance_level: Literal["L0", "L1", "L2"] = "L0"
    effect: str = Field(description="which fixture block this observation unlocks, e.g. 'conjunction.states_post'")
    payload: Dict[str, object] = Field(default_factory=dict)


class LightCurveFixture(BaseModel):
    sample_rate_hz: float
    duration_s: float
    dominant_frequency_hz: float
    harmonic_frequency_hz: Optional[float] = None
    amplitude_mag: float
    mean_mag: float
    noise_sigma_mag: float
    seed: int
    samples_mag: List[float]


class ElementSnapshot(BaseModel):
    epoch: datetime
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    source: str = "TLE"


class ManeuverRecord(BaseModel):
    epoch: datetime
    dv_mps: float
    direction: str
    announced: bool = False


class KinematicsFixture(BaseModel):
    object_id: str
    element_history: List[ElementSnapshot] = Field(default_factory=list)
    residuals: Dict[str, float] = Field(default_factory=dict)
    maneuver_history: List[ManeuverRecord] = Field(default_factory=list)
    range_history_km: List[Tuple[float, float]] = Field(default_factory=list, description="(day, range_km) to shadowed asset")
    notes: str = ""


class CamPolicyFixture(BaseModel):
    lead_time_s: float
    target_pc: float
    min_standoff_km: float
    spacecraft_mass_kg: float
    isp_s: float
    screening_window_s: float
    screening_catalog_ids: List[str] = Field(default_factory=list)


class ExpectedOutcome(BaseModel):
    classification: str
    severity: Literal["CRITICAL", "WARNING", "ADVISORY", "INFO"]
    governance_level: Literal["L0", "L1", "L2"]
    recommended_action: str
    key_numbers: Dict[str, float] = Field(default_factory=dict)


class ScenarioFixture(BaseModel):
    id: str
    title: str
    trigger_type: Literal["CDM", "KINEMATIC", "SIGNAL", "INQUIRY"]
    summary: str
    objects: List[ObjectFixture]
    graph: GraphFixture
    conjunction: Optional[ConjunctionFixture] = None
    observations: List[ObservationFixture] = Field(default_factory=list)
    light_curve: Optional[LightCurveFixture] = None
    kinematics: Optional[KinematicsFixture] = None
    cam_policy: Optional[CamPolicyFixture] = None
    expected: ExpectedOutcome

    def object(self, object_id: str) -> ObjectFixture:
        for o in self.objects:
            if o.id == object_id:
                return o
        raise KeyError(object_id)

    @property
    def primary(self) -> ObjectFixture:
        return next(o for o in self.objects if o.role == "primary")


class FixtureFile(BaseModel):
    version: str
    generated_by: str
    generated_at: datetime
    scenarios: List[ScenarioFixture]

    def scenario(self, scenario_id: str) -> ScenarioFixture:
        for s in self.scenarios:
            if s.id == scenario_id:
                return s
        raise KeyError(scenario_id)
