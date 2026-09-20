"""Deterministic Flight Policy Engine (rules R1-R10).

The engine has three jobs:

``route_action``      decide the *minimum* approval route an action requires
``validate_action``   raise ``PolicyViolation`` if an action/route pair or an
                      action/evidence pair breaks a rule
``cam_permitted``     say whether a propulsive recommendation may even be
                      formulated from the current evidence (R1/R2/R5)

Nothing here consults an LLM. Routes are monotone: a caller may escalate an
action above its minimum route (a human can always be involved) but never
below it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .state import ActionType, ApprovalRoute, RecommendedAction

PC_THRESHOLD = 1e-4
HIGH_COVARIANCE_KM = 2.0
R1_MISS_KM = 1.5


class Rule(BaseModel):
    id: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.id} {self.title}"


RULES: Dict[str, Rule] = {
    r.id: r
    for r in [
        Rule(
            id="R1",
            title="No manoeuvre on wide covariance with a clear nominal miss",
            text=f"If the major B-plane sigma exceeds {HIGH_COVARIANCE_KM:.0f} km and the nominal miss exceeds {R1_MISS_KM:.1f} km, "
            "CAM burns are forbidden; the case is routed to L0 passive monitoring or sensor tasking.",
        ),
        Rule(
            id="R2",
            title="Dilution-region conjunctions require sensing before action",
            text=f"When the major B-plane sigma exceeds {HIGH_COVARIANCE_KM:.0f} km and the Pc is uncertainty-driven (sigma > miss), "
            "the agent must not recommend a manoeuvre; it issues HOLD/MONITOR and tasks the sensor with the highest expected covariance reduction.",
        ),
        Rule(
            id="R3",
            title="Deadline stopping rule",
            text="Sensing may only be requested if time-to-TCA exceeds sensor latency plus decision margin plus CAM lead time; "
            "otherwise the agent decides on the best available Pc and records that the deadline, not the evidence, closed the loop.",
        ),
        Rule(
            id="R4",
            title="Cleared conjunctions carry zero delta-v",
            text=f"If Pc < {PC_THRESHOLD:g} the action is CLEARED with delta_v = 0.0 and burn_direction = NONE. No propulsive fields may be populated.",
        ),
        Rule(
            id="R5",
            title="Defensible high-Pc conjunctions yield a CAM decision package",
            text=f"If Pc >= {PC_THRESHOLD:g} and the covariance is defensible (major sigma <= {HIGH_COVARIANCE_KM:.0f} km), the agent produces a "
            "minimum-delta-v CAM package with post-burn catalogue screening and alternatives.",
        ),
        Rule(
            id="R6",
            title="L2 human sign-off for burns and formal filings",
            text="Any propulsive CAM burn, formal ITU dispute, or UN registry notification is routed L2_HUMAN_MANDATORY. "
            "The agent produces the decision package and stops; it never executes a burn autonomously.",
        ),
        Rule(
            id="R7",
            title="L1 for outbound advisories",
            text="Operator advisories, neighbourhood safety bulletins and STC coordination queries are non-binding outbound actions routed L1_ADVISORY.",
        ),
        Rule(
            id="R8",
            title="L0 is read-only against the world",
            text="L0_AUTO covers computations, catalogue/graph lookups, passive monitoring, simulated sensor tasking requests and case logging only.",
        ),
        Rule(
            id="R9",
            title="Non-cooperative approach escalation",
            text="Unannounced delta-v by an object whose operator is non-responsive, combined with a shadowing pattern near a protected asset, "
            "is CRITICAL: an L1 operator advisory is issued and an L2 UN registry report is drafted for human release.",
        ),
        Rule(
            id="R10",
            title="Numeric provenance",
            text="Every numeric field in an action must originate from a deterministic specialist tool. The narrative layer may cite but never alter action fields.",
        ),
    ]
}


class PolicyViolation(Exception):
    def __init__(self, rule_id: str, message: str) -> None:
        self.rule_id = rule_id
        super().__init__(f"[{rule_id}] {message}")


class PolicyContext(BaseModel):
    pc: Optional[float] = None
    miss_km: Optional[float] = None
    bplane_sigma_major_km: Optional[float] = None
    in_dilution_region: bool = False
    time_to_tca_s: Optional[float] = None
    sensing_performed: bool = False
    deadline_forced: bool = False

    @property
    def high_covariance(self) -> bool:
        return self.bplane_sigma_major_km is not None and self.bplane_sigma_major_km > HIGH_COVARIANCE_KM


class PolicyVerdict(BaseModel):
    route: ApprovalRoute
    citations: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


L2_ACTIONS = {ActionType.EXECUTE_CAM, ActionType.UN_REGISTRY_REPORT, ActionType.ITU_DISPUTE, ActionType.ESCALATE_FDO}
L1_ACTIONS = {ActionType.OPERATOR_ADVISORY, ActionType.NEIGHBOR_ADVISORY, ActionType.RECLASSIFY_UNCONTROLLED}
L0_ACTIONS = {ActionType.HOLD_MONITOR, ActionType.CLEARED}


def required_route(action: RecommendedAction) -> Tuple[ApprovalRoute, List[str]]:
    """Minimum approval route for an action and the rules that force it."""
    cites: List[str] = []
    route = ApprovalRoute.L0_AUTO
    if action.is_propulsive:
        cites.append("R6")
        route = ApprovalRoute.L2_HUMAN_MANDATORY
    for a in action.all_actions:
        if a in L2_ACTIONS:
            route = ApprovalRoute.L2_HUMAN_MANDATORY
            if "R6" not in cites:
                cites.append("R6")
        elif a in L1_ACTIONS and route.rank < ApprovalRoute.L1_ADVISORY.rank:
            route = ApprovalRoute.L1_ADVISORY
            cites.append("R7")
    if route == ApprovalRoute.L0_AUTO:
        cites.append("R8")
    return route, cites


def cam_permitted(ctx: PolicyContext) -> Tuple[bool, List[str], List[str]]:
    """May a propulsive recommendation be formulated from this evidence? -> (ok, citations, reasons)."""
    reasons: List[str] = []
    cites: List[str] = []
    if ctx.pc is None:
        return False, ["R10"], ["no deterministic Pc available"]
    if ctx.pc < PC_THRESHOLD:
        cites.append("R4")
        reasons.append(f"Pc {ctx.pc:.3g} below {PC_THRESHOLD:g}: conjunction is cleared")
    if ctx.high_covariance and ctx.miss_km is not None and ctx.miss_km > R1_MISS_KM:
        cites.append("R1")
        reasons.append(f"sigma {ctx.bplane_sigma_major_km:.2f} km > {HIGH_COVARIANCE_KM:.0f} km with nominal miss {ctx.miss_km:.2f} km > {R1_MISS_KM} km")
    if ctx.high_covariance and ctx.in_dilution_region:
        cites.append("R2")
        reasons.append(f"sigma {ctx.bplane_sigma_major_km:.2f} km > {HIGH_COVARIANCE_KM:.0f} km and Pc is uncertainty-driven")
    if cites:
        return False, cites, reasons
    return True, ["R5"], []


def route_action(action: RecommendedAction, ctx: PolicyContext) -> PolicyVerdict:
    route, cites = required_route(action)
    notes: List[str] = []
    # rules the synthesis itself relied on (stored as full citations "Rn title")
    cites += [ref.split(" ", 1)[0] for ref in action.policy_refs if ref.split(" ", 1)[0] in RULES]
    if action.action_type == ActionType.CLEARED:
        cites.append("R4")
    if action.action_type == ActionType.EXECUTE_CAM:
        cites.append("R5")
    if ctx.deadline_forced:
        cites.append("R3")
        notes.append("deadline-forced decision")
    if ctx.sensing_performed:
        notes.append("decision follows active sensing")
    cites.append("R10")
    # de-duplicate, preserve order
    seen: List[str] = []
    for c in cites:
        if c not in seen:
            seen.append(c)
    return PolicyVerdict(route=route, citations=[RULES[c].citation for c in seen], notes=notes)


def validate_action(action: RecommendedAction, route: ApprovalRoute, ctx: Optional[PolicyContext] = None) -> None:
    """Raise PolicyViolation if the (action, route, evidence) triple breaks a rule."""
    minimum, _ = required_route(action)
    if action.delta_v_mps > 0.0 and route != ApprovalRoute.L2_HUMAN_MANDATORY:
        raise PolicyViolation("R6", f"propulsive action ({action.delta_v_mps:.3f} m/s) routed {route.value}; burns require L2_HUMAN_MANDATORY")
    if route.rank < minimum.rank:
        raise PolicyViolation("R6" if minimum == ApprovalRoute.L2_HUMAN_MANDATORY else "R7", f"{action.action_type.value} requires at least {minimum.value}, got {route.value}")
    if action.action_type == ActionType.CLEARED and (action.delta_v_mps != 0.0 or action.burn_direction != "NONE"):
        raise PolicyViolation("R4", "CLEARED action must carry delta_v = 0.0 and burn_direction = NONE")
    if action.action_type == ActionType.EXECUTE_CAM and action.delta_v_mps <= 0.0:
        raise PolicyViolation("R5", "EXECUTE_CAM must carry a positive delta_v from the planner")
    if ctx is not None:
        if action.action_type == ActionType.CLEARED and ctx.pc is not None and ctx.pc >= PC_THRESHOLD:
            raise PolicyViolation("R4", f"cannot clear a conjunction with Pc {ctx.pc:.3g} >= {PC_THRESHOLD:g}")
        if action.is_propulsive:
            ok, cites, reasons = cam_permitted(ctx)
            if not ok:
                raise PolicyViolation(cites[0], "propulsive action forbidden: " + "; ".join(reasons))


def enforce_cleared(action: RecommendedAction) -> RecommendedAction:
    """R4 helper: strip every propulsive field from a CLEARED action."""
    return action.model_copy(
        update={"delta_v_mps": 0.0, "burn_direction": "NONE", "dv_rtn_mps": None, "propellant_kg": None, "burn_epoch": None}
    )


def cite(*rule_ids: str) -> List[str]:
    return [RULES[r].citation for r in rule_ids]
