"""Active sensing engine: decide *whether* to sense, *what* to task, and fold results in.

Decision rule (Pillar 3 / PDF §4 Phases 4-6)
--------------------------------------------
Sensing is requested when the current belief is not defensible:

* the conjunction specialist flags ``HIGH_COVARIANCE`` (major B-plane sigma
  above 2 km, the R1/R2 threshold), or
* the synthesised confidence is below ``confidence_floor`` (0.65),

AND the stopping rule allows it: ``time_to_tca > latency + decision margin +
CAM lead time``. If sensing is needed but no candidate fits inside the
deadline, the decision is marked ``deadline_forced`` (R3) and the supervisor
acts on the best available evidence.

Candidate sensing actions are ranked by expected information gain per hour
of latency. The top non-coordination candidate is dispatched together with
every STC coordination query (which is cheap and runs in parallel).

Note on the dilution flag: ``DILUTION_REGION`` alone is *not* sufficient to
trigger sensing. Any sub-hard-body miss under a covariance larger than the
miss is mathematically dilution-region (Scenario 4: 46 m under 0.3 km), yet
its covariance is already below the policy threshold and radar cannot shrink
it meaningfully, so acting is the correct choice.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from astrodynamics import two_body_propagate
from graph import EdgeType, GroundStation, NodeKind, OrbitalGraph

from .specialists.conjunction_specialist import HIGH_COVARIANCE_KM
from .state import (
    ApprovalRoute,
    CaseStatus,
    Finding,
    OrbitalInvestigationState,
    Phase,
    SensorObservation,
    SensorTaskingRequest,
    SensorTaskType,
    StateRecord,
    TaskStatus,
    TriggerType,
)


class SensorProfile(BaseModel):
    task_type: SensorTaskType
    latency_s: float
    governance_level: ApprovalRoute
    default_sensor_id: str
    sigma_floor_km: Optional[float] = None


SENSOR_PROFILES: Dict[SensorTaskType, SensorProfile] = {
    SensorTaskType.RADAR_REOBSERVATION: SensorProfile(
        task_type=SensorTaskType.RADAR_REOBSERVATION,
        latency_s=4 * 3600,
        governance_level=ApprovalRoute.L0_AUTO,
        default_sensor_id="RADAR-NET",
        sigma_floor_km=0.15,
    ),
    SensorTaskType.OPTICAL_PHOTOMETRY: SensorProfile(
        task_type=SensorTaskType.OPTICAL_PHOTOMETRY,
        latency_s=8 * 3600,
        governance_level=ApprovalRoute.L0_AUTO,
        default_sensor_id="OPTICAL-NET",
    ),
    SensorTaskType.STC_OPERATOR_QUERY: SensorProfile(
        task_type=SensorTaskType.STC_OPERATOR_QUERY,
        latency_s=1 * 3600,
        governance_level=ApprovalRoute.L1_ADVISORY,
        default_sensor_id="STC-M2M",
    ),
}


def covariance_actionable(conj: Finding) -> bool:
    """Wide covariance only matters when it actually drives the risk: Pc above
    threshold, or the nominal miss lying inside ~3 sigma of the ellipsoid. A
    35 km stand-off under a 3 km sigma is wide but irrelevant."""
    return conj.has("HIGH_COVARIANCE") and (conj.has("PC_ABOVE_THRESHOLD") or conj.metrics.get("mahalanobis_distance", 0.0) < 3.0)


class SensingDecision(BaseModel):
    needed: bool
    reasons: List[str] = Field(default_factory=list)
    requests: List[SensorTaskingRequest] = Field(default_factory=list)
    deadline_forced: bool = False
    time_to_tca_s: Optional[float] = None
    candidates_considered: int = 0


class _Candidate(BaseModel):
    task_type: SensorTaskType
    target_id: str
    sensor_id: str
    gain: float
    rationale: str

    @property
    def score(self) -> float:
        return self.gain / (SENSOR_PROFILES[self.task_type].latency_s / 3600.0)


class ActiveTaskingEngine:
    def __init__(
        self,
        *,
        confidence_floor: float = 0.65,
        high_covariance_km: float = HIGH_COVARIANCE_KM,
        decision_margin_s: float = 3600.0,
        cam_lead_time_s: float = 2400.0,
    ) -> None:
        self.confidence_floor = confidence_floor
        self.high_covariance_km = high_covariance_km
        self.decision_margin_s = decision_margin_s
        self.cam_lead_time_s = cam_lead_time_s

    # ------------------------------------------------------------------ decide
    def evaluate(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph], synthesis_confidence: float) -> SensingDecision:
        reasons: List[str] = []
        conj = state.findings.get("conjunction")
        if conj is not None and covariance_actionable(conj):
            reasons.append(
                f"B-plane sigma {conj.metrics['bplane_sigma_major_km']:.2f} km exceeds {self.high_covariance_km:.1f} km policy threshold"
                + (" (dilution region: Pc is uncertainty-driven)" if conj.has("DILUTION_REGION") else "")
            )
        if synthesis_confidence < self.confidence_floor:
            reasons.append(f"confidence {synthesis_confidence:.2f} below floor {self.confidence_floor:.2f}")
        ttc = state.time_to_tca_s
        if not reasons:
            return SensingDecision(needed=False, time_to_tca_s=ttc)
        if state.sensing_rounds >= state.max_sensing_rounds:
            return SensingDecision(needed=True, reasons=reasons + ["sensing budget exhausted"], deadline_forced=True, time_to_tca_s=ttc)

        already = {(t.task_type, t.target_id) for t in state.sensor_tasking_requests}
        candidates = [c for c in self._candidates(state, graph) if (c.task_type, c.target_id) not in already]
        admissible = [c for c in candidates if self._within_deadline(c, ttc)]
        if not admissible:
            return SensingDecision(
                needed=True,
                reasons=reasons + ["no sensing option fits inside the TCA deadline (R3)"],
                deadline_forced=True,
                time_to_tca_s=ttc,
                candidates_considered=len(candidates),
            )
        chosen: List[_Candidate] = []
        physical = sorted([c for c in admissible if c.task_type != SensorTaskType.STC_OPERATOR_QUERY], key=lambda c: c.score, reverse=True)
        if physical:
            chosen.append(physical[0])
        chosen += [c for c in admissible if c.task_type == SensorTaskType.STC_OPERATOR_QUERY]
        requests = [self._to_request(state, c, i) for i, c in enumerate(chosen)]
        return SensingDecision(needed=True, reasons=reasons, requests=requests, time_to_tca_s=ttc, candidates_considered=len(candidates))

    def _within_deadline(self, c: _Candidate, ttc: Optional[float]) -> bool:
        if ttc is None:
            return True
        return ttc > SENSOR_PROFILES[c.task_type].latency_s + self.decision_margin_s + self.cam_lead_time_s

    def _to_request(self, state: OrbitalInvestigationState, c: _Candidate, idx: int) -> SensorTaskingRequest:
        prof = SENSOR_PROFILES[c.task_type]
        return SensorTaskingRequest(
            task_id=f"{state.case_id}-T{state.sensing_rounds + 1}{chr(ord('a') + idx)}",
            task_type=c.task_type,
            target_id=c.target_id,
            sensor_id=c.sensor_id,
            requested_at=state.now,
            latency_s=prof.latency_s,
            expected_information_gain=float(min(1.0, max(0.0, c.gain))),
            rationale=c.rationale,
            governance_level=prof.governance_level,
        )

    # -------------------------------------------------------------- candidates
    def _candidates(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph]) -> List[_Candidate]:
        out: List[_Candidate] = []
        conj = state.findings.get("conjunction")
        kin = state.findings.get("kinematics")
        photo = state.findings.get("photometric")
        gfind = state.findings.get("graph")
        sec = state.secondary_id

        # (a) collapse position covariance with a radar pass on the poorly-known object
        if conj is not None and sec is not None and covariance_actionable(conj):
            sigma = conj.metrics["bplane_sigma_major_km"]
            floor = SENSOR_PROFILES[SensorTaskType.RADAR_REOBSERVATION].sigma_floor_km or 0.15
            gain = max(0.0, 1.0 - floor / sigma)
            out.append(
                _Candidate(
                    task_type=SensorTaskType.RADAR_REOBSERVATION,
                    target_id=sec,
                    sensor_id=self._station_for(graph, sec, "radar"),
                    gain=gain,
                    rationale=f"collapse secondary covariance from {sigma:.2f} km toward the {floor:.2f} km radar floor before any manoeuvre decision",
                )
            )

        # (b) confirm the primary operator has no planned burn that would move the geometry
        if state.trigger_type == TriggerType.CDM and graph is not None and graph.has(state.primary_id):
            op = graph.get_operator_of(state.primary_id)
            if op is not None and op.stc_responsive:
                out.append(
                    _Candidate(
                        task_type=SensorTaskType.STC_OPERATOR_QUERY,
                        target_id=op.id,
                        sensor_id="STC-M2M",
                        gain=0.2,
                        rationale=f"confirm with {op.name} that no planned burn invalidates the CDM geometry",
                    )
                )

        # (c) intent of an object with unannounced Δv: ask its operator, track it optically
        if kin is not None and kin.has("UNANNOUNCED_DV") and sec is not None:
            op = graph.get_operator_of(sec) if graph is not None and graph.has(sec) else None
            if op is not None:
                out.append(
                    _Candidate(
                        task_type=SensorTaskType.STC_OPERATOR_QUERY,
                        target_id=op.id,
                        sensor_id="STC-M2M",
                        gain=0.3 if op.stc_responsive else 0.15,
                        rationale=f"machine-to-machine query to {op.name}: were the {kin.metrics.get('dv_unannounced_mps', 0):.1f} m/s burns planned?",
                    )
                )
            if not kin.has("SHADOWING_PATTERN"):
                out.append(
                    _Candidate(
                        task_type=SensorTaskType.OPTICAL_PHOTOMETRY,
                        target_id=sec,
                        sensor_id=self._station_for(graph, sec, "optical"),
                        gain=0.5,
                        rationale="multi-night optical tracking to establish the range history and detect a station-keeping/shadowing pattern",
                    )
                )

        # (d) attitude diagnosis after loss of signal / anomalous decay
        if kin is not None and (kin.has("LOSS_OF_SIGNAL") or kin.has("ANOMALOUS_DECAY")) and photo is None:
            out.append(
                _Candidate(
                    task_type=SensorTaskType.OPTICAL_PHOTOMETRY,
                    target_id=state.primary_id,
                    sensor_id=self._station_for(graph, state.primary_id, "optical"),
                    gain=0.7,
                    rationale="task optical photometry to obtain a light curve and diagnose attitude stability (tumble vs. controlled)",
                )
            )
        return out

    @staticmethod
    def _station_for(graph: Optional[OrbitalGraph], object_id: str, sensor_type: str) -> str:
        if graph is None or not graph.has(object_id):
            return SENSOR_PROFILES[SensorTaskType.RADAR_REOBSERVATION if sensor_type == "radar" else SensorTaskType.OPTICAL_PHOTOMETRY].default_sensor_id
        for gs_id in graph.neighbors(object_id, EdgeType.TRACKED_BY, "out"):
            gs = graph.get(gs_id)
            if isinstance(gs, GroundStation) and gs.sensor_type == sensor_type:
                return gs.id
        # any station of that type in the graph
        for gs in graph.nodes_of_kind(NodeKind.GROUND_STATION):
            if isinstance(gs, GroundStation) and gs.sensor_type == sensor_type:
                return gs.id
        return SENSOR_PROFILES[SensorTaskType.RADAR_REOBSERVATION if sensor_type == "radar" else SensorTaskType.OPTICAL_PHOTOMETRY].default_sensor_id

    # ------------------------------------------------------------------- fold
    def apply_sensor_observation(self, state: OrbitalInvestigationState, observation: SensorObservation) -> OrbitalInvestigationState:
        return apply_sensor_observation(state, observation)


def _fuse_covariance(prior: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Information-form fusion of two independent RTN position covariances."""
    return np.linalg.inv(np.linalg.inv(prior) + np.linalg.inv(measurement))


def apply_sensor_observation(state: OrbitalInvestigationState, observation: SensorObservation) -> OrbitalInvestigationState:
    """Fold fresh evidence into the belief state.

    RADAR_REOBSERVATION
        payload.state  = {epoch, r, v, frame}  (observed/OD'd state of target)
        payload.covariance_rtn = 3x3 km^2 (OD covariance in the target's RTN)
        payload.mode = "replace" (default; OD supersedes the CDM covariance) | "fuse"
        optional payload.covariance_primary_rtn
        The observed state is re-propagated (two-body) to TCA so the geometry
        stays expressed at the conjunction epoch.
    OPTICAL_PHOTOMETRY
        payload.light_curve = {sample_rate_hz, samples_mag, ...}  -> aux_evidence["light_curve"]
        payload.range_history_km = [[day, km], ...]              -> aux_evidence["kinematics"]
    STC_OPERATOR_QUERY
        payload.responded, payload.planned_burn, payload.message  -> aux_evidence["stc_response"]
    """
    matching = [t for t in state.sensor_tasking_requests if t.task_id == observation.task_id]
    if not matching:
        raise ValueError(f"observation references unknown tasking request {observation.task_id!r}")
    task = matching[0]
    p = observation.payload
    messages: List[str] = []

    if observation.task_type == SensorTaskType.RADAR_REOBSERVATION:
        target = observation.target_id
        st = p["state"]
        epoch = StateRecord(epoch=st["epoch"], r=tuple(st["r"]), v=tuple(st["v"]), frame=st.get("frame", "J2000")).epoch
        r, v = np.asarray(st["r"], float), np.asarray(st["v"], float)
        if state.tca is not None:
            dt = (state.tca - epoch).total_seconds()
            if abs(dt) > 1e-6:
                r, v = two_body_propagate(r, v, dt)
                messages.append(f"re-propagated {target} {dt:+.0f} s to TCA")
            epoch = state.tca
        new_states = dict(state.current_state_vectors)
        new_states[target] = StateRecord(epoch=epoch, r=tuple(map(float, r)), v=tuple(map(float, v)), frame=st.get("frame", "J2000"), source=f"RADAR:{observation.sensor_id}")
        state.current_state_vectors = new_states

        new_cov = dict(state.covariance_matrix)
        if "covariance_rtn" in p:
            meas = np.asarray(p["covariance_rtn"], float)
            prior = np.asarray(state.covariance_matrix.get(target, meas), float)
            fused = _fuse_covariance(prior, meas) if p.get("mode", "replace") == "fuse" else meas
            shrink = float(np.sqrt(np.linalg.det(prior) / max(np.linalg.det(fused), 1e-300)))
            new_cov[target] = fused.tolist()
            messages.append(f"{target} covariance volume shrank {shrink:.0f}x ({p.get('mode', 'replace')})")
        if "covariance_primary_rtn" in p:
            new_cov[state.primary_id] = np.asarray(p["covariance_primary_rtn"], float).tolist()
        state.covariance_matrix = new_cov
        status = TaskStatus.FULFILLED

    elif observation.task_type == SensorTaskType.OPTICAL_PHOTOMETRY:
        aux = dict(state.aux_evidence)
        if "light_curve" in p:
            lc = dict(p["light_curve"])
            lc.setdefault("sensor_id", observation.sensor_id)
            aux["light_curve"] = lc
            messages.append(f"light curve ingested ({len(lc.get('samples_mag', []))} samples @ {lc.get('sample_rate_hz')} Hz)")
        if "range_history_km" in p:
            kin = dict(aux.get("kinematics", {}))
            kin["range_history_km"] = [list(x) for x in p["range_history_km"]]
            aux["kinematics"] = kin
            messages.append(f"range history ingested ({len(kin['range_history_km'])} nights)")
        state.aux_evidence = aux
        status = TaskStatus.FULFILLED

    elif observation.task_type == SensorTaskType.STC_OPERATOR_QUERY:
        aux = dict(state.aux_evidence)
        responded = bool(p.get("responded", True))
        aux["stc_response"] = {
            "operator_id": observation.target_id,
            "responded": responded,
            "planned_burn": bool(p.get("planned_burn", False)),
            "message": p.get("message", ""),
        }
        state.aux_evidence = aux
        status = TaskStatus.FULFILLED if responded else TaskStatus.UNANSWERED
        messages.append(f"{observation.target_id} " + ("responded" if responded else "did not respond") + (" (planned burn confirmed)" if p.get("planned_burn") else ""))
    else:  # pragma: no cover
        raise ValueError(f"unsupported task type {observation.task_type}")

    state.sensor_tasking_requests = [t.model_copy(update={"status": status}) if t.task_id == task.task_id else t for t in state.sensor_tasking_requests]
    state.observations = [*state.observations, observation]
    if observation.observed_at > state.now:
        state.now = observation.observed_at
    state.phase = Phase.OBSERVATIONS_FOLDED
    state.record(f"folded {observation.task_type.value} from {observation.sensor_id}: " + "; ".join(messages))
    if all(t.status != TaskStatus.DISPATCHED for t in state.sensor_tasking_requests):
        state.status = CaseStatus.OPEN
    return state
