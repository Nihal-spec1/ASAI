"""Simulated sensors backed by the scenario fixtures.

Turns a ``SensorTaskingRequest`` into the ``SensorObservation`` a real sensor
network would return. Radar passes deliver the fixture's post-sensing orbit
determination (back-propagated a few minutes so the engine must re-propagate
to TCA); optical tasking delivers the light curve or range history; STC
queries replay the operator's fixture response.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np

from astrodynamics import two_body_propagate
from data.schema import ScenarioFixture

from .state import SensorObservation, SensorTaskingRequest, SensorTaskType


class FixtureSensorSimulator:
    def __init__(self, scenario: ScenarioFixture, *, radar_delivery_offset_s: float = 600.0) -> None:
        self.sc = scenario
        self.offset_s = radar_delivery_offset_s

    def observe(self, request: SensorTaskingRequest) -> SensorObservation:
        observed_at = request.requested_at + timedelta(seconds=request.latency_s)
        if request.task_type == SensorTaskType.RADAR_REOBSERVATION:
            payload = self._radar_payload(request)
        elif request.task_type == SensorTaskType.OPTICAL_PHOTOMETRY:
            payload = self._optical_payload(request)
        else:
            payload = self._stc_payload(request)
        return SensorObservation(
            task_id=request.task_id,
            task_type=request.task_type,
            sensor_id=request.sensor_id,
            target_id=request.target_id,
            observed_at=observed_at,
            payload=payload,
        )

    # ------------------------------------------------------------------ radar
    def _radar_payload(self, request: SensorTaskingRequest) -> Dict[str, Any]:
        c = self.sc.conjunction
        if c is None or c.states_post is None or c.covariance_post is None:
            raise ValueError(f"scenario {self.sc.id} has no post-sensing radar solution")
        st = c.states_post[request.target_id]
        r, v = two_body_propagate(np.asarray(st.r), np.asarray(st.v), -self.offset_s)
        fx = next((o for o in self.sc.observations if o.sensor_type == "radar"), None)
        cov = c.covariance_post.secondary_rtn if request.target_id == c.secondary_id else c.covariance_post.primary_rtn
        payload: Dict[str, Any] = {
            "state": {"epoch": (st.epoch - timedelta(seconds=self.offset_s)).isoformat(), "r": [float(x) for x in r], "v": [float(x) for x in v], "frame": st.frame},
            "covariance_rtn": cov,
            "mode": "replace",
        }
        if request.target_id == c.secondary_id:
            payload["covariance_primary_rtn"] = c.covariance_post.primary_rtn
        if fx is not None:
            payload.update(fx.payload)
        return payload

    # ---------------------------------------------------------------- optical
    def _optical_payload(self, request: SensorTaskingRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        fx = next((o for o in self.sc.observations if o.sensor_type == "optical"), None)
        if self.sc.light_curve is not None and request.target_id == self.sc.primary.id:
            lc = self.sc.light_curve
            payload["light_curve"] = {
                "sample_rate_hz": lc.sample_rate_hz,
                "samples_mag": lc.samples_mag,
                "sensor_id": request.sensor_id,
                **({"filter": fx.payload.get("filter", "")} if fx else {}),
            }
        if self.sc.kinematics is not None and self.sc.kinematics.range_history_km and request.target_id == self.sc.kinematics.object_id:
            payload["range_history_km"] = [list(x) for x in self.sc.kinematics.range_history_km]
        if not payload:
            raise ValueError(f"scenario {self.sc.id} has no optical product for {request.target_id}")
        return payload

    # -------------------------------------------------------------------- stc
    def _stc_payload(self, request: SensorTaskingRequest) -> Dict[str, Any]:
        fx = next((o for o in self.sc.observations if o.sensor_type == "stc" and o.sensor_id == request.target_id), None)
        if fx is not None:
            p = dict(fx.payload)
            p.setdefault("responded", True)
            p.setdefault("planned_burn", False)
            p.setdefault("message", fx.description)
            return p
        op = next((o for o in self.sc.graph.operators if o.id == request.target_id), None)
        responsive = bool(op.stc_responsive) if op else False
        return {"responded": responsive, "planned_burn": False, "message": "no fixture response; defaulted from operator responsiveness"}
