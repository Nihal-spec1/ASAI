"""Supervisor state machine: orchestrates one investigation from trigger to graph commit.

Lifecycle (each step is a pure function of the state plus injected tools):

    INGESTED -> SPECIALISTS_DONE -> INITIAL_ACTION
             -> [SENSING_DISPATCHED -> (AWAITING_SENSING) -> OBSERVATIONS_FOLDED -> SPECIALISTS_DONE]*
             -> FINAL_ACTION -> ROUTED -> NARRATED -> COMMITTED

``CaseStatus`` is the externally visible status:
OPEN while the machine is working, AWAITING_SENSING while tasking requests
are outstanding, READY_FOR_HUMAN when an L2 decision package is at the gate,
RESOLVED when the case is closed (L0/L1 outcomes, or after human sign-off).

No LangGraph: transitions are explicit methods, ``step`` advances exactly one
phase, ``run`` loops until the machine blocks or terminates.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from astrodynamics import TLE, plan_minimum_dv_cam, screen_post_maneuver
from astrodynamics.maneuver_planner import apply_impulse, primary_state_at_burn
from data.schema import ScenarioFixture
from graph import CaseNode, EdgeType, OrbitalGraph

from .active_tasking import ActiveTaskingEngine, SensingDecision, apply_sensor_observation
from .narrative import NarrativeGenerator
from .policy_engine import (
    PC_THRESHOLD,
    PolicyContext,
    PolicyViolation,
    cam_permitted,
    cite,
    enforce_cleared,
    route_action,
    validate_action,
)
from .specialists import ConjunctionSpecialist, GraphSpecialist, KinematicsSpecialist, PhotometricSpecialist
from .specialists.base import Specialist
from .state import (
    ActionType,
    ApprovalRoute,
    CaseStatus,
    Finding,
    HumanDecision,
    Matrix3,
    OrbitalInvestigationState,
    Phase,
    RecommendedAction,
    SensorObservation,
    StateRecord,
    TaskStatus,
    TriggerType,
)

# --------------------------------------------------------------------------- #
# Trigger ingestion
# --------------------------------------------------------------------------- #


class CaseTrigger(BaseModel):
    """Everything the supervisor needs to open a case."""

    scenario_id: Optional[str] = None
    title: str = ""
    trigger_type: TriggerType
    primary_id: str
    secondary_id: Optional[str] = None
    tca: Optional[datetime] = None
    opened_epoch: datetime
    states: Dict[str, StateRecord] = Field(default_factory=dict)
    covariances: Dict[str, Matrix3] = Field(default_factory=dict)
    hbr_m: Dict[str, float] = Field(default_factory=dict)
    kinematics: Optional[Dict[str, Any]] = None
    light_curve: Optional[Dict[str, Any]] = None
    cam_policy: Optional[Dict[str, Any]] = None
    tles: Dict[str, Tuple[str, str]] = Field(default_factory=dict, description="object_id -> (line1, line2)")
    max_sensing_rounds: int = 2


def trigger_from_scenario(
    sc: ScenarioFixture,
    *,
    now: Optional[datetime] = None,
    hours_before_tca: float = 36.0,
    include_light_curve: bool = False,
    include_range_history: bool = False,
) -> CaseTrigger:
    """Build the *pre-sensing* trigger from a fixture scenario.

    Evidence that the fixture marks as unlocked by an observation (post-radar
    states, light curve, optical range history) is withheld unless explicitly
    included, so the supervisor has to earn it through active sensing.
    """
    states: Dict[str, StateRecord] = {}
    covs: Dict[str, Matrix3] = {}
    hbr = {o.id: o.hbr_m for o in sc.objects}
    tles = {o.id: (o.tle.line1, o.tle.line2) for o in sc.objects if o.tle is not None}
    tca = None
    secondary = None
    if sc.conjunction is not None:
        c = sc.conjunction
        tca, secondary = c.tca, c.secondary_id
        for oid, st in c.states_pre.items():
            states[oid] = StateRecord(epoch=st.epoch, r=st.r, v=st.v, frame=st.frame, source="CDM")
        covs[c.primary_id] = c.covariance_pre.primary_rtn
        covs[c.secondary_id] = c.covariance_pre.secondary_rtn
    kin = None
    if sc.kinematics is not None:
        kin = sc.kinematics.model_dump(mode="json")
        if not include_range_history:
            kin["range_history_km"] = []
        if secondary is None and sc.kinematics.object_id != sc.primary.id:
            secondary = sc.kinematics.object_id
    lc = None
    if include_light_curve and sc.light_curve is not None:
        lc = {"sample_rate_hz": sc.light_curve.sample_rate_hz, "samples_mag": sc.light_curve.samples_mag, "sensor_id": "fixture"}
    if now is None:
        if tca is not None:
            now = tca - timedelta(hours=hours_before_tca)
        elif sc.kinematics and sc.kinematics.element_history:
            now = sc.kinematics.element_history[-1].epoch
        else:
            now = datetime.now(timezone.utc)
    return CaseTrigger(
        scenario_id=sc.id,
        title=sc.title,
        trigger_type=TriggerType(sc.trigger_type),
        primary_id=sc.primary.id,
        secondary_id=secondary,
        tca=tca,
        opened_epoch=now,
        states=states,
        covariances=covs,
        hbr_m=hbr,
        kinematics=kin,
        light_curve=lc,
        cam_policy=sc.cam_policy.model_dump(mode="json") if sc.cam_policy else None,
        tles=tles,
    )


# --------------------------------------------------------------------------- #
# what_changed diff
# --------------------------------------------------------------------------- #


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3g}"
        return f"{v:.3f}"
    return str(v)


_LABELS = {
    "pc": "Pc",
    "miss_km": "miss distance [km]",
    "bplane_sigma_major_km": "B-plane sigma major [km]",
    "covariance_volume_km3": "covariance volume [km^3]",
    "sigma_to_miss_ratio": "sigma/miss ratio",
    "dilution_region": "dilution region",
    "dv_unannounced_mps": "unannounced dv [m/s]",
    "range_km": "range to asset [km]",
    "approach_rate_km_day": "approach rate [km/day]",
    "decay_ratio": "decay ratio vs nominal",
    "dominant_frequency_hz": "light-curve frequency [Hz]",
    "spin_period_s": "spin period [s]",
    "amplitude_mag": "light-curve amplitude [mag]",
    "tumbling": "tumbling",
    "operator_responded": "operator responded",
    "confidence": "confidence",
    "classification": "classification",
}


def diff_snapshots(before: Dict[str, Any], after: Dict[str, Any], before_action: RecommendedAction, after_action: RecommendedAction) -> str:
    lines: List[str] = []
    keys = list(dict.fromkeys([*before.keys(), *after.keys()]))
    for k in keys:
        b, a = before.get(k), after.get(k)
        if b is None and a is None:
            continue
        if b is None:
            lines.append(f"{_LABELS.get(k, k)}: (new evidence) -> {_fmt(a)}")
        elif a is None:
            continue
        elif isinstance(a, float) and isinstance(b, float):
            if b == a or (b != 0 and abs(a - b) / abs(b) < 1e-6):
                continue
            lines.append(f"{_LABELS.get(k, k)}: {_fmt(b)} -> {_fmt(a)}")
        elif a != b:
            lines.append(f"{_LABELS.get(k, k)}: {_fmt(b)} -> {_fmt(a)}")
    if before_action.action_type != after_action.action_type or before_action.secondary_actions != after_action.secondary_actions:
        lines.append(f"action: {'+'.join(x.value for x in before_action.all_actions)} -> {'+'.join(x.value for x in after_action.all_actions)}")
    if before_action.delta_v_mps != after_action.delta_v_mps:
        lines.append(f"delta-v [m/s]: {_fmt(before_action.delta_v_mps)} -> {_fmt(after_action.delta_v_mps)}")
    if lines:
        return "\n".join(lines)
    return "no material change: sensing confirmed the baseline assessment"


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


class Supervisor:
    def __init__(
        self,
        graph: OrbitalGraph,
        *,
        sensor_simulator: Optional[Any] = None,
        narrative: Optional[NarrativeGenerator] = None,
        tasking: Optional[ActiveTaskingEngine] = None,
        specialists: Optional[Sequence[Specialist]] = None,
        pc_threshold: float = PC_THRESHOLD,
    ) -> None:
        self.graph = graph
        self.sensor_simulator = sensor_simulator
        self.narrative = narrative or NarrativeGenerator()
        self.tasking = tasking or ActiveTaskingEngine()
        self.pc_threshold = pc_threshold
        self.specialists: List[Specialist] = list(specialists) if specialists is not None else [
            ConjunctionSpecialist(pc_threshold=pc_threshold),
            KinematicsSpecialist(),
            GraphSpecialist(),
            PhotometricSpecialist(),
        ]

    # ------------------------------------------------------------------ public
    def open_case(self, trig: CaseTrigger, *, case_id: Optional[str] = None) -> OrbitalInvestigationState:
        aux: Dict[str, Any] = {}
        if trig.kinematics:
            aux["kinematics"] = trig.kinematics
        if trig.light_curve:
            aux["light_curve"] = trig.light_curve
        if trig.cam_policy:
            aux["cam_policy"] = trig.cam_policy
        if trig.tles:
            aux["tles"] = {k: list(v) for k, v in trig.tles.items()}
        state = OrbitalInvestigationState(
            case_id=case_id or f"CASE-{(trig.scenario_id or 'X')}-{uuid.uuid4().hex[:6].upper()}",
            scenario_id=trig.scenario_id,
            opened_epoch=trig.opened_epoch,
            now=trig.opened_epoch,
            trigger_type=trig.trigger_type,
            primary_id=trig.primary_id,
            secondary_id=trig.secondary_id,
            tca=trig.tca,
            current_state_vectors=trig.states,
            covariance_matrix=trig.covariances,
            hbr_m=trig.hbr_m,
            aux_evidence=aux,
            max_sensing_rounds=trig.max_sensing_rounds,
        )
        state.record(f"trigger ingested ({trig.trigger_type.value}): {trig.title or trig.primary_id}")
        return state

    def run(self, state: OrbitalInvestigationState, *, max_steps: int = 40) -> OrbitalInvestigationState:
        """Advance until the machine terminates or blocks on sensing without a simulator."""
        for _ in range(max_steps):
            if state.phase == Phase.COMMITTED:
                return state
            if state.status == CaseStatus.AWAITING_SENSING:
                if self.sensor_simulator is None:
                    return state  # blocked: caller must submit observations
                for task in [t for t in state.sensor_tasking_requests if t.status == TaskStatus.DISPATCHED]:
                    self.submit_observation(state, self.sensor_simulator.observe(task))
                continue
            self.step(state)
        raise RuntimeError("supervisor did not terminate within max_steps")

    def step(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        """Advance exactly one phase."""
        ph = state.phase
        if ph in (Phase.INGESTED, Phase.OBSERVATIONS_FOLDED):
            return self._dispatch_specialists(state)
        if ph == Phase.SPECIALISTS_DONE:
            if state.initial_action is None:
                return self._synthesize_initial(state)
            return self._synthesize_final(state)
        if ph == Phase.INITIAL_ACTION:
            return self._evaluate_uncertainty(state)
        if ph == Phase.SENSING_DISPATCHED:
            # nothing to do until observations arrive
            state.status = CaseStatus.AWAITING_SENSING
            return state
        if ph == Phase.FINAL_ACTION:
            return self._route(state)
        if ph == Phase.ROUTED:
            return self._narrate(state)
        if ph == Phase.NARRATED:
            return self._commit(state)
        return state

    def submit_observation(self, state: OrbitalInvestigationState, obs: SensorObservation) -> OrbitalInvestigationState:
        if state.status != CaseStatus.AWAITING_SENSING:
            raise RuntimeError(f"case {state.case_id} is not awaiting sensing (status {state.status.value})")
        apply_sensor_observation(state, obs)
        return state

    def approve(self, state: OrbitalInvestigationState, approver: str, *, approved: bool = True, note: str = "") -> OrbitalInvestigationState:
        """Record the human L2 decision. Even on approval the agent does not execute anything."""
        if state.status != CaseStatus.READY_FOR_HUMAN:
            raise RuntimeError("no decision package is awaiting human sign-off")
        state.human_decision = HumanDecision(decided_at=state.now, approver=approver, approved=approved, note=note)
        state.status = CaseStatus.RESOLVED
        state.record(f"L2 decision by {approver}: {'APPROVED' if approved else 'REJECTED'} {note}".rstrip())
        self._write_case(state, closed=True)
        return state

    # ------------------------------------------------------------- transitions
    def _dispatch_specialists(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        ran: List[str] = []
        for sp in self.specialists:
            if sp.applicable(state):
                finding = sp.run(state, self.graph)
                state.findings = {**state.findings, sp.name: finding}
                ran.append(f"{sp.name}({finding.confidence:.2f})")
        state.phase = Phase.SPECIALISTS_DONE
        state.record("specialists dispatched: " + ", ".join(ran))
        return state

    def _synthesize_initial(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        action, classification, severity, conf = self._synthesize(state, stage="initial")
        state.classification, state.severity, state.confidence = classification, severity, conf
        state.initial_action = action
        state.aux_evidence = {**state.aux_evidence, "_snapshot_initial": state.snapshot()}
        state.phase = Phase.INITIAL_ACTION
        state.record(f"baseline recommendation: {'+'.join(a.value for a in action.all_actions)} (dv {action.delta_v_mps:.3f} m/s, conf {conf:.2f})")
        return state

    def _evaluate_uncertainty(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        decision: SensingDecision = self.tasking.evaluate(state, self.graph, state.confidence)
        if decision.needed and decision.requests:
            state.sensor_tasking_requests = [*state.sensor_tasking_requests, *decision.requests]
            state.sensing_rounds += 1
            state.phase = Phase.SENSING_DISPATCHED
            state.status = CaseStatus.AWAITING_SENSING
            state.record(
                "sensing needed: " + "; ".join(decision.reasons) + " -> dispatched "
                + ", ".join(f"{r.task_type.value}->{r.target_id} via {r.sensor_id} (gain {r.expected_information_gain:.2f}, {r.governance_level.short})" for r in decision.requests)
            )
            return state
        if decision.needed and decision.deadline_forced:
            state.deadline_forced = True
            state.record("sensing needed but " + "; ".join(decision.reasons))
        else:
            state.record("evidence defensible: no sensing required")
        # skip straight to the final synthesis
        state.phase = Phase.SPECIALISTS_DONE
        return self._synthesize_final(state)

    def _synthesize_final(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        if state.initial_action is None:  # pragma: no cover - defensive
            return self._synthesize_initial(state)
        # after a sensing round we may need another one: re-evaluate before finalising
        if state.phase == Phase.SPECIALISTS_DONE and state.observations and state.sensing_rounds < state.max_sensing_rounds:
            action, classification, severity, conf = self._synthesize(state, stage="final")
            decision = self.tasking.evaluate(state, self.graph, conf)
            if decision.needed and decision.requests:
                state.classification, state.severity, state.confidence = classification, severity, conf
                state.sensor_tasking_requests = [*state.sensor_tasking_requests, *decision.requests]
                state.sensing_rounds += 1
                state.phase = Phase.SENSING_DISPATCHED
                state.status = CaseStatus.AWAITING_SENSING
                state.record("further sensing needed: " + "; ".join(decision.reasons) + " -> " + ", ".join(r.task_type.value for r in decision.requests))
                return state
            if decision.needed and decision.deadline_forced:
                state.deadline_forced = True
        action, classification, severity, conf = self._synthesize(state, stage="final")
        state.classification, state.severity, state.confidence = classification, severity, conf
        state.final_action = action
        before = state.aux_evidence.get("_snapshot_initial", {})
        after = state.snapshot()
        diff = diff_snapshots(before, after, state.initial_action, action)
        folded = [o for o in state.observations]
        if folded:
            evid = "; ".join(f"{o.task_type.value} by {o.sensor_id}" for o in folded)
            diff = f"evidence folded: {evid}\n" + diff
        elif diff.startswith("no material change"):
            diff = "no active sensing required: evidence already defensible; definitive recommendation equals the baseline"
            if state.deadline_forced:
                diff += " (deadline-forced, R3)"
        state.what_changed = diff
        state.phase = Phase.FINAL_ACTION
        state.record(f"definitive recommendation: {'+'.join(a.value for a in action.all_actions)} (dv {action.delta_v_mps:.3f} m/s, conf {conf:.2f})")
        return state

    def _route(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        assert state.final_action is not None
        ctx = self._policy_context(state)
        verdict = route_action(state.final_action, ctx)
        validate_action(state.final_action, verdict.route, ctx)  # raises PolicyViolation
        state.policy_citations = verdict.citations
        state.approval_route = verdict.route  # model validator re-checks R6 here
        state.status = CaseStatus.READY_FOR_HUMAN if verdict.route == ApprovalRoute.L2_HUMAN_MANDATORY else CaseStatus.RESOLVED
        state.phase = Phase.ROUTED
        state.record(f"policy route {verdict.route.value}; " + ("; ".join(verdict.notes) if verdict.notes else "no notes"))
        return state

    def _narrate(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        frozen = state.final_action.model_dump(mode="json") if state.final_action else None
        nar = self.narrative.generate(state)
        state.narrative, state.narrative_source = nar.text, nar.source  # type: ignore[assignment]
        # R10: the narrative layer must not have touched the action
        if state.final_action is not None and state.final_action.model_dump(mode="json") != frozen:  # pragma: no cover
            raise PolicyViolation("R10", "narrative layer mutated the action")
        state.phase = Phase.NARRATED
        state.record(f"narrative generated ({nar.source})")
        return state

    def _commit(self, state: OrbitalInvestigationState) -> OrbitalInvestigationState:
        self._write_case(state, closed=state.status == CaseStatus.RESOLVED)
        state.phase = Phase.COMMITTED
        state.record(f"case anchored in graph as {state.case_node_id}; status {state.status.value}")
        return state

    # ---------------------------------------------------------------- synthesis
    def _synthesize(self, state: OrbitalInvestigationState, *, stage: str) -> Tuple[RecommendedAction, str, str, float]:
        conj = state.findings.get("conjunction")
        kin = state.findings.get("kinematics")
        photo = state.findings.get("photometric")
        gf = state.findings.get("graph")
        stc = state.aux_evidence.get("stc_response")
        gflags = set(gf.flags) if gf else set()

        # ---- 1. unannounced manoeuvres near a protected asset (covert RPO branch)
        if kin is not None and kin.has("UNANNOUNCED_DV") and state.trigger_type == TriggerType.KINEMATIC:
            return self._synth_rpo(state, stage, kin, gflags, stc)

        # ---- 2. attitude / drag anomaly branch
        if photo is not None and photo.has("TUMBLING"):
            return self._synth_tumble(state, stage, photo, kin, gf)
        if kin is not None and (kin.has("ANOMALOUS_DECAY") or kin.has("LOSS_OF_SIGNAL")) and state.trigger_type == TriggerType.SIGNAL:
            if photo is not None:  # attitude stable, decay unexplained
                return (
                    RecommendedAction(
                        action_type=ActionType.OPERATOR_ADVISORY,
                        confidence=0.7,
                        rationale="decay anomaly persists but the light curve shows a stable attitude; advise operator to check propulsion/drag configuration",
                        policy_refs=cite("R7"),
                        outbound_targets=self._operator_ids(state.primary_id),
                        key_numbers={k: kin.metrics[k] for k in ("decay_ratio",) if k in kin.metrics},
                        stage=stage,  # type: ignore[arg-type]
                    ),
                    "drag_anomaly_attitude_stable",
                    "WARNING",
                    0.7,
                )
            return (
                RecommendedAction(
                    action_type=ActionType.HOLD_MONITOR,
                    confidence=0.45,
                    rationale="anomalous decay with loss of signal; attitude state unknown, so no reclassification or advisory is defensible yet",
                    policy_refs=cite("R8"),
                    key_numbers={k: kin.metrics[k] for k in ("decay_ratio", "hours_since_last_telemetry") if k in kin.metrics},
                    stage=stage,  # type: ignore[arg-type]
                ),
                "drag_anomaly_unexplained",
                "WARNING",
                0.45,
            )

        # ---- 3. conjunction branch
        if conj is not None:
            return self._synth_conjunction(state, stage, conj)

        return (
            RecommendedAction(action_type=ActionType.HOLD_MONITOR, confidence=0.3, rationale="insufficient evidence for any classification", policy_refs=cite("R8"), stage=stage),  # type: ignore[arg-type]
            "unclassified",
            "INFO",
            0.3,
        )

    def _synth_conjunction(self, state: OrbitalInvestigationState, stage: str, conj: Finding) -> Tuple[RecommendedAction, str, str, float]:
        pc = conj.metrics["pc"]
        miss = conj.metrics["miss_km"]
        sigma = conj.metrics["bplane_sigma_major_km"]
        ctx = self._policy_context(state)
        numbers = {k: conj.metrics[k] for k in ("pc", "miss_km", "bplane_sigma_major_km", "sigma_to_miss_ratio", "covariance_volume_km3")}
        was_hold = state.initial_action is not None and state.initial_action.action_type == ActionType.HOLD_MONITOR and stage == "final"

        if pc < self.pc_threshold:
            if conj.has("HIGH_COVARIANCE") and miss > 1.5 and not state.observations:
                # R1: clear nominal miss under wide covariance -> passive monitoring
                return (
                    RecommendedAction(
                        action_type=ActionType.HOLD_MONITOR,
                        confidence=conj.confidence,
                        rationale=f"Pc {pc:.3g} below threshold but sigma {sigma:.2f} km > 2 km with nominal miss {miss:.2f} km; passive monitoring, no burn (R1)",
                        policy_refs=cite("R1", "R8"),
                        key_numbers=numbers,
                        stage=stage,  # type: ignore[arg-type]
                    ),
                    "conjunction_low_risk_wide_covariance",
                    "ADVISORY",
                    conj.confidence,
                )
            action = enforce_cleared(
                RecommendedAction(
                    action_type=ActionType.CLEARED,
                    confidence=max(conj.confidence, 0.66),
                    rationale=(
                        f"Pc {pc:.3g} < {self.pc_threshold:g} at miss {miss:.3f} km with B-plane sigma {sigma:.2f} km"
                        + ("; collision-avoidance manoeuvre cancelled, propellant preserved" if was_hold else "; no manoeuvre required")
                    ),
                    policy_refs=cite("R4", "R8"),
                    key_numbers=numbers,
                    stage=stage,  # type: ignore[arg-type]
                )
            )
            return action, ("debris_collision_risk_false_alarm" if was_hold else "conjunction_cleared"), "ADVISORY", action.confidence

        ok, cites, reasons = cam_permitted(ctx)
        if not ok:
            if state.deadline_forced:
                return (
                    RecommendedAction(
                        action_type=ActionType.ESCALATE_FDO,
                        confidence=conj.confidence,
                        rationale="TCA deadline reached with Pc above threshold but covariance still not defensible; no CAM package can be justified - Flight Director decision required. " + "; ".join(reasons),
                        policy_refs=cite("R3", *cites, "R6"),
                        key_numbers=numbers,
                        stage=stage,  # type: ignore[arg-type]
                    ),
                    "conjunction_uncertain_deadline",
                    "CRITICAL",
                    conj.confidence,
                )
            return (
                RecommendedAction(
                    action_type=ActionType.HOLD_MONITOR,
                    confidence=conj.confidence,
                    rationale="Pc is above threshold but uncertainty-driven: " + "; ".join(reasons) + ". Hold, do not burn, collapse covariance first",
                    policy_refs=cite(*cites, "R8"),
                    key_numbers=numbers,
                    stage=stage,  # type: ignore[arg-type]
                ),
                "conjunction_uncertain_high_covariance",
                "WARNING",
                conj.confidence,
            )

        # R5: defensible high Pc -> CAM package
        maneuver = state.findings.get("maneuver")
        if maneuver is None or maneuver.details.get("for_snapshot") != self._geometry_key(state):
            maneuver = self._plan_cam(state)
            state.findings = {**state.findings, "maneuver": maneuver}
        if not maneuver.has("FEASIBLE"):
            return (
                RecommendedAction(
                    action_type=ActionType.ESCALATE_FDO,
                    confidence=conj.confidence,
                    rationale="Pc above threshold but no single-impulse CAM within the delta-v budget clears the target Pc and stand-off; Flight Director decision required",
                    policy_refs=cite("R5", "R6"),
                    key_numbers=numbers,
                    stage=stage,  # type: ignore[arg-type]
                ),
                "debris_collision_risk_no_feasible_cam",
                "CRITICAL",
                conj.confidence,
            )
        m = maneuver.metrics
        action = RecommendedAction(
            action_type=ActionType.EXECUTE_CAM,
            delta_v_mps=m["dv_mag_mps"],
            burn_direction=maneuver.details["direction"],
            burn_epoch=datetime.fromisoformat(maneuver.details["burn_epoch"]),
            dv_rtn_mps=tuple(maneuver.details["dv_rtn_mps"]),
            propellant_kg=m.get("propellant_kg"),
            predicted_pc=m["predicted_pc"],
            predicted_miss_km=m["predicted_miss_km"],
            confidence=conj.confidence,
            rationale=(
                f"Pc {pc:.3g} >= {self.pc_threshold:g} at {miss*1000:.0f} m miss with defensible sigma {sigma:.2f} km. "
                f"Minimum-dv CAM: {m['dv_mag_mps']:.3f} m/s {maneuver.details['direction']} at TCA-{m['lead_time_s']/60:.0f} min "
                f"-> miss {m['predicted_miss_km']:.2f} km, Pc {m['predicted_pc']:.2g}; post-burn screening "
                + ("clear" if maneuver.has("SCREENING_CLEAR") else "FLAGGED")
                + ". Package requires Flight Director sign-off; not executed autonomously"
            ),
            policy_refs=cite("R5", "R6", "R10"),
            key_numbers={**numbers, **{k: m[k] for k in ("dv_mag_mps", "predicted_pc", "predicted_miss_km", "lead_time_s") if k in m}, **({"propellant_kg": m["propellant_kg"]} if "propellant_kg" in m else {})},
            stage=stage,  # type: ignore[arg-type]
        )
        return action, "debris_collision_risk_confirmed", "CRITICAL", conj.confidence

    def _synth_rpo(self, state, stage, kin: Finding, gflags: set, stc: Optional[dict]) -> Tuple[RecommendedAction, str, str, float]:
        dv = kin.metrics.get("dv_unannounced_mps", kin.metrics.get("dv_total_estimated_mps", 0.0))
        numbers = {k: kin.metrics[k] for k in ("dv_unannounced_mps", "delta_i_deg", "delta_a_km", "range_km", "approach_rate_km_day") if k in kin.metrics}
        sec_op = self._operator_ids(state.secondary_id) if state.secondary_id else []
        prim_op = self._operator_ids(state.primary_id)
        if stc is not None and stc.get("responded") and stc.get("planned_burn"):
            return (
                RecommendedAction(
                    action_type=ActionType.HOLD_MONITOR,
                    confidence=0.8,
                    rationale=f"operator confirmed the {dv:.1f} m/s manoeuvres as planned station-keeping; monitor only",
                    policy_refs=cite("R8"),
                    key_numbers=numbers,
                    stage=stage,
                ),
                "announced_station_keeping",
                "INFO",
                0.8,
            )
        conf = 0.5
        if "UNDECLARED_PAYLOAD" in gflags:
            conf += 0.1
        if "MILITARY_OPERATOR" in gflags or "FOREIGN_STATE_OPERATOR" in gflags:
            conf += 0.05
        if kin.has("SHADOWING_PATTERN"):
            conf += 0.2
        if stc is not None and not stc.get("responded"):
            conf += 0.15
        conf = min(conf, 0.98)
        non_coop = "NON_RESPONSIVE_OPERATOR" in gflags or (stc is not None and not stc.get("responded"))
        if kin.has("SHADOWING_PATTERN") and non_coop:
            return (
                RecommendedAction(
                    action_type=ActionType.OPERATOR_ADVISORY,
                    secondary_actions=[ActionType.NEIGHBOR_ADVISORY, ActionType.UN_REGISTRY_REPORT],
                    confidence=conf,
                    rationale=(
                        f"{state.secondary_id} executed {dv:.1f} m/s of unannounced manoeuvres, matched planes with {state.primary_id} and is station-keeping at "
                        f"{kin.metrics.get('range_km', float('nan')):.1f} km; operator non-responsive"
                        + (" and launch manifest under-declared" if "UNDECLARED_PAYLOAD" in gflags else "")
                        + ". Issue L1 advisory to the asset operator and neighbours; draft L2 UN registry report for human release"
                    ),
                    policy_refs=cite("R9", "R7", "R6"),
                    outbound_targets=[*prim_op, *sec_op, "UNOOSA-REGISTRY"],
                    key_numbers=numbers,
                    stage=stage,
                ),
                "covert_rpo_shadowing",
                "CRITICAL",
                conf,
            )
        # intent is unresolved by definition here: confidence stays below the sensing floor
        conf = min(conf, 0.6)
        return (
            RecommendedAction(
                action_type=ActionType.HOLD_MONITOR,
                confidence=conf,
                rationale=f"{dv:.1f} m/s of unannounced manoeuvres by {state.secondary_id}; intent unresolved - query operator and establish range history before attribution",
                policy_refs=cite("R8"),
                key_numbers=numbers,
                stage=stage,
            ),
            "unannounced_maneuver_under_investigation",
            "WARNING",
            conf,
        )

    def _synth_tumble(self, state, stage, photo: Finding, kin: Optional[Finding], gf: Optional[Finding]) -> Tuple[RecommendedAction, str, str, float]:
        conf = min(0.98, 0.55 + 0.4 * photo.confidence)
        numbers = {k: photo.metrics[k] for k in ("dominant_frequency_hz", "spin_period_s", "amplitude_mag")}
        if kin:
            numbers.update({k: kin.metrics[k] for k in ("decay_ratio", "ballistic_coefficient_ratio") if k in kin.metrics})
        targets = self._operator_ids(state.primary_id)
        if gf and isinstance(gf.details.get("primary"), dict):
            targets += [o for o in gf.details["primary"].get("neighbour_operators", []) if o not in targets]
        return (
            RecommendedAction(
                action_type=ActionType.RECLASSIFY_UNCONTROLLED,
                secondary_actions=[ActionType.NEIGHBOR_ADVISORY, ActionType.OPERATOR_ADVISORY],
                confidence=conf,
                rationale=(
                    f"light curve shows a {photo.metrics['dominant_frequency_hz']:.2f} Hz periodic tumble (period {photo.metrics['spin_period_s']:.1f} s) "
                    + (f"consistent with {kin.metrics['decay_ratio']:.1f}x nominal decay; " if kin and 'decay_ratio' in kin.metrics else "; ")
                    + "ADCS failure: reclassify as uncontrolled and advise operator and orbital neighbours"
                ),
                policy_refs=cite("R7"),
                outbound_targets=targets,
                key_numbers=numbers,
                stage=stage,
            ),
            "adcs_failure_tumbling",
            "WARNING",
            conf,
        )

    # ---------------------------------------------------------------- planner
    def _plan_cam(self, state: OrbitalInvestigationState) -> Finding:
        pol = dict(state.aux_evidence.get("cam_policy") or {})
        geom = state.geometry()
        lead = float(pol.get("lead_time_s", 2400.0))
        study = plan_minimum_dv_cam(
            geom,
            lead_time_s=lead,
            target_pc=float(pol.get("target_pc", 1e-6)),
            min_standoff_km=float(pol.get("min_standoff_km", 1.0)),
            spacecraft_mass_kg=pol.get("spacecraft_mass_kg"),
            isp_s=pol.get("isp_s"),
        )
        rec = study.recommended
        flags: List[str] = []
        metrics: Dict[str, float] = {"lead_time_s": lead, "target_pc": study.target_pc, "min_standoff_km": study.min_standoff_km, "baseline_pc": study.baseline.pc}
        details: Dict[str, Any] = {
            "for_snapshot": self._geometry_key(state),
            "alternatives": [
                {"direction": p.direction.value, "dv_mag_mps": p.dv_mag_mps, "predicted_miss_km": p.predicted_miss_km, "predicted_pc": p.predicted_pc, "feasible": p.feasible, "propellant_kg": p.propellant_kg}
                for p in study.plans
            ],
        }
        if rec is None:
            flags.append("INFEASIBLE")
            return Finding(specialist="maneuver", produced_at=state.now, summary="no feasible single-impulse CAM within budget", confidence=0.9, metrics=metrics, flags=flags, details=details, evidence_refs=["tool:plan_minimum_dv_cam"])
        flags.append("FEASIBLE")
        metrics.update(
            dv_mag_mps=rec.dv_mag_mps,
            predicted_pc=rec.predicted_pc,
            predicted_miss_km=rec.predicted_miss_km,
            predicted_separation_at_tca_km=rec.predicted_separation_at_tca_km,
        )
        if rec.propellant_kg is not None:
            metrics["propellant_kg"] = rec.propellant_kg
        details.update(direction=rec.direction.value, burn_epoch=rec.burn_epoch.isoformat(), dv_rtn_mps=list(rec.dv_rtn_mps))

        # post-burn catalogue screening
        tles = state.aux_evidence.get("tles", {})
        catalog_ids = list(pol.get("screening_catalog_ids") or [])
        secondaries = {oid: TLE(name=oid, line1=tles[oid][0], line2=tles[oid][1]) for oid in catalog_ids if oid in tles}
        if secondaries:
            prim_tle = TLE(name=state.primary_id, line1=tles[state.primary_id][0], line2=tles[state.primary_id][1]) if state.primary_id in tles else None
            pre = primary_state_at_burn(geom, lead, prim_tle)
            post = apply_impulse(pre, rec.dv_rtn_mps)
            res = screen_post_maneuver(post, secondaries, window_s=float(pol.get("screening_window_s", 6000.0)), standoff_km=study.min_standoff_km)
            metrics["screening_objects"] = float(len(res.hits))
            metrics["screening_min_distance_km"] = min((h.min_distance_km for h in res.hits), default=float("inf"))
            details["screening"] = [{"object_id": h.object_id, "min_distance_km": h.min_distance_km, "epoch": h.epoch.isoformat(), "violates_standoff": h.violates_standoff} for h in res.hits]
            flags.append("SCREENING_CLEAR" if res.clear else "SCREENING_FLAGGED")
        else:
            flags.append("SCREENING_CLEAR")
            details["screening"] = []
        summary = (
            f"min-dv CAM {rec.dv_mag_mps:.3f} m/s {rec.direction.value} at TCA-{lead/60:.0f} min -> miss {rec.predicted_miss_km:.2f} km, Pc {rec.predicted_pc:.2g}"
            + (f", propellant {rec.propellant_kg:.3f} kg" if rec.propellant_kg is not None else "")
            + f"; {len(study.plans)} directions studied; screening " + ("clear" if "SCREENING_CLEAR" in flags else "flagged")
        )
        return Finding(specialist="maneuver", produced_at=state.now, summary=summary, confidence=0.9, metrics=metrics, flags=flags, details=details, evidence_refs=["tool:plan_minimum_dv_cam", "tool:screen_post_maneuver"])

    # ----------------------------------------------------------------- helpers
    def _policy_context(self, state: OrbitalInvestigationState) -> PolicyContext:
        conj = state.findings.get("conjunction")
        return PolicyContext(
            pc=conj.metrics["pc"] if conj else None,
            miss_km=conj.metrics["miss_km"] if conj else None,
            bplane_sigma_major_km=conj.metrics["bplane_sigma_major_km"] if conj else None,
            in_dilution_region=bool(conj and conj.has("DILUTION_REGION")),
            time_to_tca_s=state.time_to_tca_s,
            sensing_performed=bool(state.observations),
            deadline_forced=state.deadline_forced,
        )

    @staticmethod
    def _geometry_key(state: OrbitalInvestigationState) -> str:
        parts = []
        for oid in sorted(state.current_state_vectors):
            s = state.current_state_vectors[oid]
            parts.append(f"{oid}:{s.r}:{s.v}:{state.covariance_matrix.get(oid)}")
        return "|".join(parts)

    def _operator_ids(self, object_id: Optional[str]) -> List[str]:
        if object_id is None or not self.graph.has(object_id):
            return []
        op = self.graph.get_operator_of(object_id)
        return [op.id] if op else []

    def _embedding(self, state: OrbitalInvestigationState) -> List[float]:
        conj = state.findings.get("conjunction")
        kin = state.findings.get("kinematics")
        photo = state.findings.get("photometric")
        pc = conj.metrics["pc"] if conj else 0.0
        a = state.final_action
        return [
            min(1.0, -math.log10(max(pc, 1e-300)) / 50.0) if conj else 0.0,
            min(1.0, conj.metrics["miss_km"] / 10.0) if conj else 0.0,
            min(1.0, conj.metrics["bplane_sigma_major_km"] / 5.0) if conj else 0.0,
            min(1.0, (a.delta_v_mps if a else 0.0) / 5.0),
            min(1.0, kin.metrics.get("dv_unannounced_mps", 0.0) / 50.0) if kin else 0.0,
            1.0 if photo and photo.has("TUMBLING") else 0.0,
            {"CDM": 0.25, "KINEMATIC": 0.5, "SIGNAL": 0.75, "INQUIRY": 1.0}[state.trigger_type.value],
            state.approval_route.rank / 2.0 if state.approval_route else 0.0,
        ]

    def _write_case(self, state: OrbitalInvestigationState, *, closed: bool) -> None:
        involved = [state.primary_id] + ([state.secondary_id] if state.secondary_id else [])
        operators: List[str] = []
        for oid in involved:
            operators += [o for o in self._operator_ids(oid) if o not in operators]
        a = state.final_action
        outcome = "+".join(x.value for x in a.all_actions) if a else "n/a"
        if state.human_decision:
            outcome += f" [{'APPROVED' if state.human_decision.approved else 'REJECTED'} by {state.human_decision.approver}]"
        elif state.status == CaseStatus.READY_FOR_HUMAN:
            outcome += " [AWAITING L2 SIGN-OFF]"
        case = CaseNode(
            id=state.case_id,
            name=f"{state.classification} :: {state.primary_id}" + (f" vs {state.secondary_id}" if state.secondary_id else ""),
            scenario_id=state.scenario_id,
            opened_at=state.opened_epoch,
            closed_at=state.now if closed else None,
            classification=state.classification,
            severity=state.severity,
            outcome=outcome,
            involved_object_ids=[o for o in involved if self.graph.has(o)],
            operator_ids=operators,
            evidence_refs=sorted({ref for f in state.findings.values() for ref in f.evidence_refs}),
            policy_refs=list(state.policy_citations),
            summary=state.narrative or (a.rationale if a else ""),
            what_changed=state.what_changed,
            embedding=self._embedding(state),
        )
        state.case_node_id = self.graph.write_case_node(case)
        conj = state.findings.get("conjunction")
        if conj and state.secondary_id and self.graph.has(state.primary_id) and self.graph.has(state.secondary_id):
            self.graph.add_edge(
                EdgeType.IN_CONJUNCTION_WITH,
                state.primary_id,
                state.secondary_id,
                tca=state.tca.isoformat() if state.tca else None,
                miss_km=conj.metrics["miss_km"],
                pc=conj.metrics["pc"],
                case_id=state.case_id,
            )
