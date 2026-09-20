"""Typed node and edge models for the Orbital Knowledge Graph."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    SATELLITE = "Satellite"
    DEBRIS = "Debris"
    OPERATOR = "Operator"
    LAUNCH_EVENT = "LaunchEvent"
    GROUND_STATION = "GroundStation"
    CASE = "Case"


class EdgeType(str, Enum):
    OPERATED_BY = "OPERATED_BY"  # (object) -> (operator)
    CO_LAUNCHED_WITH = "CO_LAUNCHED_WITH"  # (object) <-> (object)   symmetric
    ORBITAL_NEIGHBOR = "ORBITAL_NEIGHBOR"  # (object) <-> (object)   symmetric, attr shell_km
    IN_CONJUNCTION_WITH = "IN_CONJUNCTION_WITH"  # (object) <-> (object) symmetric, attrs tca, miss_km, pc
    LAUNCHED_IN = "LAUNCHED_IN"  # (object) -> (launch event)
    PARENT_OF = "PARENT_OF"  # (parent object) -> (debris)
    TRACKED_BY = "TRACKED_BY"  # (object) -> (ground station)
    INVESTIGATED_IN = "INVESTIGATED_IN"  # (object) -> (case)
    SIMILAR_TO = "SIMILAR_TO"  # (case) <-> (case)  symmetric, attr score

    @property
    def symmetric(self) -> bool:
        return self in {
            EdgeType.CO_LAUNCHED_WITH,
            EdgeType.ORBITAL_NEIGHBOR,
            EdgeType.IN_CONJUNCTION_WITH,
            EdgeType.SIMILAR_TO,
        }


class _Node(BaseModel):
    id: str
    name: str

    @property
    def kind(self) -> NodeKind:  # pragma: no cover - overridden
        raise NotImplementedError


class Satellite(_Node):
    norad_id: int
    cospar_id: str = ""
    bus_type: str = ""
    dry_mass_kg: Optional[float] = None
    operator_id: Optional[str] = None
    country: str = ""
    launch_date: Optional[date] = None
    status: Literal["operational", "degraded", "non-operational", "unknown"] = "operational"
    orbit_regime: Literal["LEO", "MEO", "GEO", "HEO", "unknown"] = "LEO"
    hbr_m: float = 5.0
    registered: bool = True
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    @property
    def kind(self) -> NodeKind:
        return NodeKind.SATELLITE


class Debris(_Node):
    norad_id: int
    parent_object_id: Optional[str] = None
    rcs_category: Literal["small", "medium", "large"] = "medium"
    fragmentation_event_id: Optional[str] = None
    hbr_m: float = 2.0
    orbit_regime: Literal["LEO", "MEO", "GEO", "HEO", "unknown"] = "LEO"
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    @property
    def kind(self) -> NodeKind:
        return NodeKind.DEBRIS


class Operator(_Node):
    country: str
    designation: Literal["commercial", "civil_government", "military", "academic", "unknown"] = "commercial"
    contact_protocol: str = "STC email"
    stc_responsive: bool = True

    @property
    def kind(self) -> NodeKind:
        return NodeKind.OPERATOR


class LaunchEvent(_Node):
    launch_date: date
    site: str
    booster: str
    cospar_launch_id: str
    payload_ids: List[str] = Field(default_factory=list)
    declared_payload_count: Optional[int] = None

    @property
    def kind(self) -> NodeKind:
        return NodeKind.LAUNCH_EVENT


class GroundStation(_Node):
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    operator_id: Optional[str] = None
    sensor_type: Literal["radar", "optical", "rf", "telemetry"] = "telemetry"
    min_elevation_deg: float = 10.0

    @property
    def kind(self) -> NodeKind:
        return NodeKind.GROUND_STATION


class CaseNode(_Node):
    """A resolved (or open) investigation anchored into graph memory."""

    scenario_id: Optional[str] = None
    opened_at: datetime
    closed_at: Optional[datetime] = None
    classification: str = Field(description="event typology, e.g. debris_collision_risk, covert_rpo, adcs_tumble")
    severity: Literal["CRITICAL", "WARNING", "ADVISORY", "INFO"] = "WARNING"
    outcome: str = ""
    involved_object_ids: List[str] = Field(default_factory=list)
    operator_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    policy_refs: List[str] = Field(default_factory=list)
    summary: str = ""
    what_changed: str = ""
    embedding: Optional[List[float]] = None

    @property
    def kind(self) -> NodeKind:
        return NodeKind.CASE


NodeModel = Union[Satellite, Debris, Operator, LaunchEvent, GroundStation, CaseNode]


class LaunchLineage(BaseModel):
    object_id: str
    parent_object_id: Optional[str] = None
    launch_event: Optional[LaunchEvent] = None
    co_manifested_ids: List[str] = Field(default_factory=list)
    hops: List[str] = Field(default_factory=list, description="node ids traversed, in order")


class SimilarCase(BaseModel):
    case: CaseNode
    score: float
    reasons: List[str]
