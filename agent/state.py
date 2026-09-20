"""Investigation state schema for the supervisor state machine (Pydantic v2).

The ``OrbitalInvestigationState`` is the single mutable belief object that
flows through the supervisor. Every number in it is written by a
deterministic specialist tool; the LLM narrative layer may only read it.

Governance invariant (enforced at the model level, see ``_governance``):
an action with ``delta_v_mps > 0`` can never coexist with an approval route
below ``L2_HUMAN_MANDATORY``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrodynamics import ConjunctionGeometry, StateVector

Vec3 = Tuple[float, float, float]
Matrix3 = List[List[float]]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ApprovalRoute(str, Enum):
    L0_AUTO = "L0_AUTO"
    L1_ADVISORY = "L1_ADVISORY"
    L2_HUMAN_MANDATORY = "L2_HUMAN_MANDATORY"

    @property
    def rank(self) -> int:
        return {"L0_AUTO": 0, "L1_ADVISORY": 1, "L2_HUMAN_MANDATORY": 2}[self.value]

    @property
    def short(self) -> str:
        return self.value[:2]


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    AWAITING_SENSING = "AWAITING_SENSING"
    READY_FOR_HUMAN = "READY_FOR_HUMAN"
    RESOLVED = "RESOLVED"


class TriggerType(str, Enum):
    CDM = "CDM"
    KINEMATIC = "KINEMATIC"
    SIGNAL = "SIGNAL"
    INQUIRY = "INQUIRY"


class ActionType(str, Enum):
    HOLD_MONITOR = "HOLD_MONITOR"  # passive monitoring / insufficient evidence
    CLEARED = "CLEARED"  # conjunction cleared, CAM cancelled or never needed
    EXECUTE_CAM = "EXECUTE_CAM"  # propulsive collision-avoidance manoeuvre package
    ESCALATE_FDO = "ESCALATE_FDO"  # deadline-forced decision, no defensible CAM
    OPERATOR_ADVISORY = "OPERATOR_ADVISORY"  # L1 advisory to an operator
    NEIGHBOR_ADVISORY = "NEIGHBOR_ADVISORY"  # L1 bulletin to orbital neighbours
    RECLASSIFY_UNCONTROLLED = "RECLASSIFY_UNCONTROLLED"  # catalogue status change
    UN_REGISTRY_REPORT = "UN_REGISTRY_REPORT"  # formal registry notification (L2)
    ITU_DISPUTE = "ITU_DISPUTE"  # formal interference/dispute filing (L2)


class SensorTaskType(str, Enum):
    RADAR_REOBSERVATION = "RADAR_REOBSERVATION"
    OPTICAL_PHOTOMETRY = "OPTICAL_PHOTOMETRY"
    STC_OPERATOR_QUERY = "STC_OPERATOR_QUERY"


class TaskStatus(str, Enum):
    DISPATCHED = "DISPATCHED"
    FULFILLED = "FULFILLED"
    UNANSWERED = "UNANSWERED"


class Phase(str, Enum):
    """Internal supervisor phase (finer-grained than ``CaseStatus``)."""

    INGESTED = "INGESTED"
    SPECIALISTS_DONE = "SPECIALISTS_DONE"
    INITIAL_ACTION = "INITIAL_ACTION"
    SENSING_DISPATCHED = "SENSING_DISPATCHED"
    OBSERVATIONS_FOLDED = "OBSERVATIONS_FOLDED"
    FINAL_ACTION = "FINAL_ACTION"
    ROUTED = "ROUTED"
    NARRATED = "NARRATED"
    COMMITTED = "COMMITTED"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


class StateRecord(BaseModel):
    """An inertial state vector for one object at one epoch."""

    epoch: datetime
    r: Vec3
    v: Vec3
    frame: Literal["J2000", "TEME"] = "J2000"
    source: str = "CDM"

    @field_validator("epoch")
    @classmethod
    def _utc_epoch(cls, value: datetime) -> datetime:
        return _utc(value)

    def to_state_vector(self) -> StateVector:
        return StateVector.from_arrays(self.epoch, np.asarray(self.r), np.asarray(self.v), frame=self.frame)


class Finding(BaseModel):
    """Structured output of one specialist tool run."""

    specialist: str
    produced_at: datetime = Field(default_factory=utcnow)
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)

    def has(self, flag: str) -> bool:
        return flag in self.flags


class SensorTaskingRequest(BaseModel):
    task_id: str
    task_type: SensorTaskType
    target_id: str
    sensor_id: str
    requested_at: datetime
    latency_s: float = Field(ge=0)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    rationale: str
    governance_level: ApprovalRoute = ApprovalRoute.L0_AUTO
    status: TaskStatus = TaskStatus.DISPATCHED


class SensorObservation(BaseModel):
    """Fresh evidence returned by a sensor or a coordination channel."""

    task_id: str
    task_type: SensorTaskType
    sensor_id: str
    target_id: str
    observed_at: datetime
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _utc_obs(cls, value: datetime) -> datetime:
        return _utc(value)


class RecommendedAction(BaseModel):
    """A Next-Best-Action. All numeric fields originate from deterministic tools."""

    action_type: ActionType
    secondary_actions: List[ActionType] = Field(default_factory=list)
    delta_v_mps: float = Field(default=0.0, ge=0.0)
    burn_direction: str = "NONE"
    burn_epoch: Optional[datetime] = None
    dv_rtn_mps: Optional[Vec3] = None
    propellant_kg: Optional[float] = None
    predicted_pc: Optional[float] = None
    predicted_miss_km: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    policy_refs: List[str] = Field(default_factory=list)
    outbound_targets: List[str] = Field(default_factory=list, description="operators / registries addressed by outbound actions")
    key_numbers: Dict[str, float] = Field(default_factory=dict)
    formulated_at: datetime = Field(default_factory=utcnow)
    stage: Literal["initial", "final"] = "initial"

    @property
    def all_actions(self) -> List[ActionType]:
        return [self.action_type, *self.secondary_actions]

    @property
    def is_propulsive(self) -> bool:
        return self.delta_v_mps > 0.0 or ActionType.EXECUTE_CAM in self.all_actions

    @model_validator(mode="after")
    def _consistency(self) -> "RecommendedAction":
        if self.delta_v_mps > 0.0 and self.burn_direction == "NONE":
            raise ValueError("delta_v_mps > 0 requires a burn_direction")
        if self.delta_v_mps == 0.0 and self.burn_direction != "NONE":
            raise ValueError("burn_direction must be 'NONE' when delta_v_mps == 0")
        return self


class LogEvent(BaseModel):
    at: datetime
    phase: Phase
    message: str


class HumanDecision(BaseModel):
    decided_at: datetime
    approver: str
    approved: bool
    note: str = ""


# --------------------------------------------------------------------------- #
# The investigation state
# --------------------------------------------------------------------------- #


class OrbitalInvestigationState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    case_id: str
    scenario_id: Optional[str] = None
    opened_epoch: datetime
    now: datetime = Field(description="simulation clock; advanced by sensing latency")
    trigger_type: TriggerType
    primary_id: str
    secondary_id: Optional[str] = None
    tca: Optional[datetime] = None

    # deterministic physical belief -------------------------------------------------
    current_state_vectors: Dict[str, StateRecord] = Field(default_factory=dict)
    covariance_matrix: Dict[str, Matrix3] = Field(default_factory=dict, description="3x3 position covariance km^2 in each object's RTN frame")
    hbr_m: Dict[str, float] = Field(default_factory=dict)
    aux_evidence: Dict[str, Any] = Field(default_factory=dict, description="non-state evidence: element histories, light curves, operator replies")

    # specialist outputs -----------------------------------------------------------
    findings: Dict[str, Finding] = Field(default_factory=dict)
    sensor_tasking_requests: List[SensorTaskingRequest] = Field(default_factory=list)
    observations: List[SensorObservation] = Field(default_factory=list)
    sensing_rounds: int = 0
    max_sensing_rounds: int = 2

    # decision protocol ------------------------------------------------------------
    initial_action: Optional[RecommendedAction] = None
    final_action: Optional[RecommendedAction] = None
    what_changed: str = ""
    approval_route: Optional[ApprovalRoute] = None
    status: CaseStatus = CaseStatus.OPEN
    phase: Phase = Phase.INGESTED
    classification: str = "unclassified"
    severity: Literal["CRITICAL", "WARNING", "ADVISORY", "INFO"] = "INFO"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    deadline_forced: bool = False
    policy_citations: List[str] = Field(default_factory=list)
    narrative: str = ""
    narrative_source: Literal["", "groq", "offline"] = ""
    human_decision: Optional[HumanDecision] = None
    case_node_id: Optional[str] = None
    log: List[LogEvent] = Field(default_factory=list)

    @field_validator("opened_epoch", "now", "tca")
    @classmethod
    def _utc_fields(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _utc(value) if value is not None else None

    @model_validator(mode="after")
    def _governance(self) -> "OrbitalInvestigationState":
        """R6: a propulsive action can only ever be routed L2."""
        for action in (self.initial_action, self.final_action):
            if action is None or self.approval_route is None:
                continue
            if action.is_propulsive and self.approval_route != ApprovalRoute.L2_HUMAN_MANDATORY:
                raise ValueError(
                    f"governance violation: propulsive action ({action.delta_v_mps} m/s) routed {self.approval_route.value}; "
                    "R6 requires L2_HUMAN_MANDATORY"
                )
        return self

    # ------------------------------------------------------------------ helpers
    def record(self, message: str) -> None:
        self.log = [*self.log, LogEvent(at=self.now, phase=self.phase, message=message)]

    @property
    def has_conjunction(self) -> bool:
        return (
            self.secondary_id is not None
            and self.tca is not None
            and self.primary_id in self.current_state_vectors
            and self.secondary_id in self.current_state_vectors
        )

    @property
    def time_to_tca_s(self) -> Optional[float]:
        if self.tca is None:
            return None
        return (self.tca - self.now).total_seconds()

    def geometry(self) -> ConjunctionGeometry:
        if not self.has_conjunction:
            raise ValueError("state has no complete conjunction geometry")
        p = self.current_state_vectors[self.primary_id]
        s = self.current_state_vectors[self.secondary_id]  # type: ignore[index]
        return ConjunctionGeometry(
            tca=self.tca,  # type: ignore[arg-type]
            primary_id=self.primary_id,
            secondary_id=self.secondary_id,  # type: ignore[arg-type]
            r_primary=p.r,
            v_primary=p.v,
            r_secondary=s.r,
            v_secondary=s.v,
            cov_primary_rtn=self.covariance_matrix[self.primary_id],
            cov_secondary_rtn=self.covariance_matrix[self.secondary_id],  # type: ignore[index]
            hbr_primary_m=self.hbr_m.get(self.primary_id, 5.0),
            hbr_secondary_m=self.hbr_m.get(self.secondary_id, 2.0),  # type: ignore[arg-type]
            frame=p.frame,
        )

    def finding(self, name: str) -> Optional[Finding]:
        return self.findings.get(name)

    def snapshot(self) -> Dict[str, Any]:
        """Flat metric snapshot used to compute ``what_changed`` diffs."""
        snap: Dict[str, Any] = {}
        conj = self.findings.get("conjunction")
        if conj:
            for k in ("pc", "miss_km", "bplane_sigma_major_km", "covariance_volume_km3", "sigma_to_miss_ratio"):
                if k in conj.metrics:
                    snap[k] = conj.metrics[k]
            snap["dilution_region"] = bool(conj.has("DILUTION_REGION"))
        kin = self.findings.get("kinematics")
        if kin:
            for k in ("dv_unannounced_mps", "range_km", "approach_rate_km_day", "decay_ratio"):
                if k in kin.metrics:
                    snap[k] = kin.metrics[k]
        photo = self.findings.get("photometric")
        if photo:
            for k in ("dominant_frequency_hz", "spin_period_s", "amplitude_mag"):
                if k in photo.metrics:
                    snap[k] = photo.metrics[k]
            snap["tumbling"] = bool(photo.has("TUMBLING"))
        stc = self.aux_evidence.get("stc_response")
        if stc is not None:
            snap["operator_responded"] = bool(stc.get("responded", False))
        snap["confidence"] = self.confidence
        snap["classification"] = self.classification
        return snap
