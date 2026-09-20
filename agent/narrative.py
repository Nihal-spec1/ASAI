"""Natural-language narrative via Groq (llama-3.3-70b-versatile) with an offline fallback.

The narrative is strictly downstream of the decision: it is generated from a
frozen, read-only rendering of the state and can never modify action fields
(R10). Any failure - missing/placeholder key, network error, SDK error,
empty response - degrades to the deterministic offline template.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from pydantic import BaseModel

from .state import OrbitalInvestigationState, RecommendedAction

DEFAULT_MODEL = "llama-3.3-70b-versatile"
_PLACEHOLDERS = {"", "dummy", "none", "null", "changeme", "your_api_key", "your-api-key", "xxx"}


def key_is_usable(key: Optional[str]) -> bool:
    if key is None:
        return False
    k = key.strip()
    if k.lower() in _PLACEHOLDERS or k.lower().startswith("your"):
        return False
    return len(k) >= 20


class Narrative(BaseModel):
    text: str
    policy_citations: List[str]
    source: str  # "groq" | "offline"
    model: str = ""


class NarrativeGenerator:
    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        enabled: bool = True,
        timeout_s: float = 20.0,
    ) -> None:
        self.model = model
        self.enabled = enabled and os.environ.get("ASAI_OFFLINE", "0") not in ("1", "true", "True")
        self.timeout_s = timeout_s
        self._client = client
        self._api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        if not self.enabled or not key_is_usable(self._api_key):
            return None
        try:
            from groq import Groq  # type: ignore

            self._client = Groq(api_key=self._api_key, timeout=self.timeout_s, max_retries=0)
        except Exception:
            self._client = None
        return self._client

    # ---------------------------------------------------------------- generate
    def generate(self, state: OrbitalInvestigationState) -> Narrative:
        citations = list(state.policy_citations)
        client = self._get_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=700,
                    messages=[
                        {"role": "system", "content": self.system_prompt()},
                        {"role": "user", "content": self.render_state(state)},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return Narrative(text=text, policy_citations=citations, source="groq", model=self.model)
            except Exception:
                pass
        return Narrative(text=self.offline_summary(state), policy_citations=citations, source="offline", model="offline-template")

    # ----------------------------------------------------------------- prompts
    @staticmethod
    def system_prompt() -> str:
        return (
            "You are the narration layer of the Autonomous Space Anomaly Investigator. "
            "Write a concise flight-dynamics case summary (max 220 words) for a Flight Director. "
            "Use ONLY the numbers given; never invent, round differently, or recompute values. "
            "Structure: (1) trigger, (2) evidence and what changed after active sensing, "
            "(3) recommended action with approval level, (4) policy rules cited. Plain prose, no markdown headers."
        )

    @staticmethod
    def render_state(state: OrbitalInvestigationState) -> str:
        lines = [
            f"case_id: {state.case_id}",
            f"trigger: {state.trigger_type.value} primary={state.primary_id} secondary={state.secondary_id}",
            f"classification: {state.classification} severity={state.severity} confidence={state.confidence:.2f}",
            f"status: {state.status.value} route={state.approval_route.value if state.approval_route else 'n/a'}",
        ]
        for name, f in state.findings.items():
            lines.append(f"finding[{name}] (conf {f.confidence:.2f}): {f.summary}; flags={','.join(f.flags)}")
        for t in state.sensor_tasking_requests:
            lines.append(f"tasking: {t.task_type.value} -> {t.target_id} via {t.sensor_id} [{t.status.value}] gain={t.expected_information_gain:.2f}")
        if state.initial_action:
            lines.append("initial_action: " + _render_action(state.initial_action))
        if state.final_action:
            lines.append("final_action: " + _render_action(state.final_action))
        if state.what_changed:
            lines.append("what_changed:\n" + state.what_changed)
        if state.deadline_forced:
            lines.append("note: decision was deadline-forced (R3)")
        lines.append("policy_citations: " + "; ".join(state.policy_citations))
        return "\n".join(lines)

    @staticmethod
    def offline_summary(state: OrbitalInvestigationState) -> str:
        parts: List[str] = []
        parts.append(
            f"Case {state.case_id} opened on a {state.trigger_type.value} trigger for {state.primary_id}"
            + (f" against {state.secondary_id}" if state.secondary_id else "")
            + f"; classified {state.classification.replace('_', ' ')} ({state.severity}, confidence {state.confidence:.2f})."
        )
        for name in ("conjunction", "kinematics", "photometric", "graph", "maneuver"):
            f = state.findings.get(name)
            if f:
                parts.append(f"{name.capitalize()} specialist: {f.summary}.")
        if state.sensor_tasking_requests:
            tasks = ", ".join(f"{t.task_type.value} via {t.sensor_id} ({t.status.value.lower()})" for t in state.sensor_tasking_requests)
            parts.append(f"Active sensing: {tasks}.")
        if state.initial_action and state.final_action and state.initial_action.action_type != state.final_action.action_type:
            parts.append(
                f"Baseline recommendation was {state.initial_action.action_type.value}; after sensing the definitive recommendation is "
                f"{state.final_action.action_type.value}."
            )
        if state.what_changed:
            parts.append("What changed: " + state.what_changed.replace("\n", " | ") + ".")
        if state.final_action:
            a = state.final_action
            act = f"Recommended action: {a.action_type.value}"
            if a.secondary_actions:
                act += " plus " + ", ".join(x.value for x in a.secondary_actions)
            if a.delta_v_mps > 0:
                act += f"; delta-v {a.delta_v_mps:.3f} m/s {a.burn_direction}"
                if a.propellant_kg is not None:
                    act += f", propellant {a.propellant_kg:.3f} kg"
                if a.burn_epoch is not None:
                    act += f", burn at {a.burn_epoch.isoformat()}"
            else:
                act += "; delta-v 0.0 m/s"
            act += f". Approval route {state.approval_route.value if state.approval_route else 'pending'}"
            if state.approval_route and state.approval_route.value.startswith("L2"):
                act += " - decision package prepared, awaiting human sign-off; no burn is executed autonomously"
            parts.append(act + ".")
        if state.deadline_forced:
            parts.append("The TCA deadline, not the evidence, closed the sensing loop (R3).")
        if state.policy_citations:
            parts.append("Policy cited: " + "; ".join(state.policy_citations) + ".")
        return " ".join(parts)


def _render_action(a: RecommendedAction) -> str:
    s = f"{a.action_type.value}"
    if a.secondary_actions:
        s += "+" + "+".join(x.value for x in a.secondary_actions)
    s += f" dv={a.delta_v_mps:.3f} m/s dir={a.burn_direction} conf={a.confidence:.2f}"
    if a.propellant_kg is not None:
        s += f" propellant={a.propellant_kg:.3f} kg"
    if a.predicted_pc is not None:
        s += f" predicted_pc={a.predicted_pc:.3g}"
    return s + f" :: {a.rationale}"
