"""Stage 2 verification: specialist swarm, active sensing, policy engine, supervisor.

Run with ``pytest tests/test_stage2.py -v`` or ``python -m unittest tests.test_stage2``.
All tests run offline (``ASAI_OFFLINE`` is not required; the Groq key is
forced to a placeholder so the narrative layer falls back deterministically).
"""

from __future__ import annotations

import os
import unittest
from datetime import timedelta
from unittest import mock

import numpy as np
from pydantic import ValidationError

from agent import (
    ActionType,
    ActiveTaskingEngine,
    ApprovalRoute,
    CaseStatus,
    FixtureSensorSimulator,
    NarrativeGenerator,
    OrbitalInvestigationState,
    Phase,
    PolicyContext,
    PolicyViolation,
    PolicyVerdict,
    RecommendedAction,
    SensorObservation,
    SensorTaskType,
    Supervisor,
    TaskStatus,
    TriggerType,
    cam_permitted,
    enforce_cleared,
    key_is_usable,
    required_route,
    trigger_from_scenario,
    validate_action,
)
from agent import supervisor as supervisor_module
from agent.specialists import ConjunctionSpecialist, GraphSpecialist, KinematicsSpecialist, PhotometricSpecialist
from data.loader import build_graph, load_fixtures
from graph import CaseNode, EdgeType, NodeKind

FIXTURES = load_fixtures()
os.environ["GROQ_API_KEY"] = "dummy"


def scenario(sid: str):
    return FIXTURES.scenario(sid)


def fresh_graph():
    return build_graph(FIXTURES.scenarios)


def open_state(sup: Supervisor, sid: str, **kwargs) -> OrbitalInvestigationState:
    return sup.open_case(trigger_from_scenario(scenario(sid), **kwargs), case_id=f"CASE-{sid}")


def s4_now():
    return scenario("S4").conjunction.tca - timedelta(hours=3)


def cam_action(dv: float = 0.355) -> RecommendedAction:
    return RecommendedAction(action_type=ActionType.EXECUTE_CAM, delta_v_mps=dv, burn_direction="prograde", confidence=0.7, rationale="test")


# =========================================================================== #
class TestStateSchema(unittest.TestCase):
    def _state(self, **overrides) -> OrbitalInvestigationState:
        base = dict(
            case_id="C",
            opened_epoch=scenario("S4").conjunction.tca,
            now=scenario("S4").conjunction.tca,
            trigger_type=TriggerType.CDM,
            primary_id="A",
            secondary_id="B",
        )
        base.update(overrides)
        return OrbitalInvestigationState(**base)

    def test_action_consistency_validators(self):
        with self.assertRaises(ValidationError):
            RecommendedAction(action_type=ActionType.EXECUTE_CAM, delta_v_mps=0.3, burn_direction="NONE", confidence=0.5, rationale="x")
        with self.assertRaises(ValidationError):
            RecommendedAction(action_type=ActionType.CLEARED, delta_v_mps=0.0, burn_direction="prograde", confidence=0.5, rationale="x")
        with self.assertRaises(ValidationError):
            RecommendedAction(action_type=ActionType.CLEARED, delta_v_mps=-1.0, confidence=0.5, rationale="x")

    def test_propulsive_action_requires_l2_at_model_level(self):
        for route in (ApprovalRoute.L0_AUTO, ApprovalRoute.L1_ADVISORY):
            with self.assertRaises(ValidationError, msg=route):
                self._state(final_action=cam_action(), approval_route=route)
        st = self._state(final_action=cam_action(), approval_route=ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertEqual(st.approval_route, ApprovalRoute.L2_HUMAN_MANDATORY)
        # assignment is validated too
        with self.assertRaises(ValidationError):
            st.approval_route = ApprovalRoute.L1_ADVISORY

    def test_status_and_route_enums(self):
        self.assertEqual([s.value for s in CaseStatus], ["OPEN", "AWAITING_SENSING", "READY_FOR_HUMAN", "RESOLVED"])
        self.assertEqual([r.value for r in ApprovalRoute], ["L0_AUTO", "L1_ADVISORY", "L2_HUMAN_MANDATORY"])
        self.assertLess(ApprovalRoute.L0_AUTO.rank, ApprovalRoute.L2_HUMAN_MANDATORY.rank)


# =========================================================================== #
class TestPolicyEngine(unittest.TestCase):
    def test_required_routes(self):
        self.assertEqual(required_route(RecommendedAction(action_type=ActionType.CLEARED, confidence=1, rationale="x"))[0], ApprovalRoute.L0_AUTO)
        self.assertEqual(required_route(RecommendedAction(action_type=ActionType.HOLD_MONITOR, confidence=1, rationale="x"))[0], ApprovalRoute.L0_AUTO)
        self.assertEqual(required_route(RecommendedAction(action_type=ActionType.OPERATOR_ADVISORY, confidence=1, rationale="x"))[0], ApprovalRoute.L1_ADVISORY)
        self.assertEqual(
            required_route(RecommendedAction(action_type=ActionType.OPERATOR_ADVISORY, secondary_actions=[ActionType.UN_REGISTRY_REPORT], confidence=1, rationale="x"))[0],
            ApprovalRoute.L2_HUMAN_MANDATORY,
        )
        self.assertEqual(required_route(RecommendedAction(action_type=ActionType.ITU_DISPUTE, confidence=1, rationale="x"))[0], ApprovalRoute.L2_HUMAN_MANDATORY)
        route, cites = required_route(cam_action())
        self.assertEqual(route, ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertIn("R6", cites)

    def test_governance_delta_v_requires_l2(self):
        """Any action with delta_v > 0 must raise if routed L0 or L1."""
        for route in (ApprovalRoute.L0_AUTO, ApprovalRoute.L1_ADVISORY):
            with self.assertRaises(PolicyViolation) as cm:
                validate_action(cam_action(0.001), route)
            self.assertEqual(cm.exception.rule_id, "R6")
        validate_action(cam_action(), ApprovalRoute.L2_HUMAN_MANDATORY)  # must not raise

    def test_r4_cleared_is_zero_dv(self):
        bad = RecommendedAction(action_type=ActionType.CLEARED, delta_v_mps=0.3, burn_direction="prograde", confidence=0.9, rationale="x")
        with self.assertRaises(PolicyViolation) as cm:
            validate_action(bad, ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertEqual(cm.exception.rule_id, "R4")
        fixed = enforce_cleared(bad)
        self.assertEqual(fixed.delta_v_mps, 0.0)
        self.assertEqual(fixed.burn_direction, "NONE")
        self.assertIsNone(fixed.propellant_kg)
        validate_action(fixed, ApprovalRoute.L0_AUTO)
        # cannot clear a live conjunction
        with self.assertRaises(PolicyViolation) as cm:
            validate_action(fixed, ApprovalRoute.L0_AUTO, PolicyContext(pc=5e-3))
        self.assertEqual(cm.exception.rule_id, "R4")

    def test_r1_r2_forbid_cam_on_wide_covariance(self):
        ok, cites, _ = cam_permitted(PolicyContext(pc=2e-4, miss_km=2.0, bplane_sigma_major_km=3.0, in_dilution_region=False))
        self.assertFalse(ok)
        self.assertIn("R1", cites)
        ok, cites, _ = cam_permitted(PolicyContext(pc=2.7e-4, miss_km=0.12, bplane_sigma_major_km=3.8, in_dilution_region=True))
        self.assertFalse(ok)
        self.assertIn("R2", cites)
        ok, cites, _ = cam_permitted(PolicyContext(pc=1e-6, miss_km=4.2, bplane_sigma_major_km=0.2))
        self.assertFalse(ok)
        self.assertIn("R4", cites)
        ok, cites, _ = cam_permitted(PolicyContext(pc=5.2e-3, miss_km=0.046, bplane_sigma_major_km=0.31, in_dilution_region=True))
        self.assertTrue(ok)
        self.assertEqual(cites, ["R5"])
        # even at L2 a burn is rejected when the evidence forbids it
        with self.assertRaises(PolicyViolation) as cm:
            validate_action(cam_action(), ApprovalRoute.L2_HUMAN_MANDATORY, PolicyContext(pc=2.7e-4, miss_km=0.12, bplane_sigma_major_km=3.8, in_dilution_region=True))
        self.assertEqual(cm.exception.rule_id, "R2")

    def test_under_routing_l2_action(self):
        act = RecommendedAction(action_type=ActionType.UN_REGISTRY_REPORT, confidence=0.9, rationale="x")
        with self.assertRaises(PolicyViolation):
            validate_action(act, ApprovalRoute.L1_ADVISORY)
        with self.assertRaises(PolicyViolation):
            validate_action(RecommendedAction(action_type=ActionType.OPERATOR_ADVISORY, confidence=0.9, rationale="x"), ApprovalRoute.L0_AUTO)


# =========================================================================== #
class TestSpecialists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = fresh_graph()
        cls.sup = Supervisor(cls.graph)

    def test_conjunction_specialist_s1_vs_s4(self):
        s1 = open_state(self.sup, "S1")
        f1 = ConjunctionSpecialist().run(s1, self.graph)
        self.assertAlmostEqual(f1.metrics["pc"], 2.688e-4, delta=2e-6)
        self.assertAlmostEqual(f1.metrics["miss_km"], 0.1204, delta=1e-3)
        for flag in ("PC_ABOVE_THRESHOLD", "HIGH_COVARIANCE", "DILUTION_REGION", "MISS_INSIDE_ONE_SIGMA"):
            self.assertIn(flag, f1.flags)
        self.assertLess(f1.confidence, 0.1)

        s4 = open_state(self.sup, "S4", now=s4_now())
        f4 = ConjunctionSpecialist().run(s4, self.graph)
        self.assertAlmostEqual(f4.metrics["pc"], 5.2e-3, delta=5e-5)
        self.assertLess(f4.metrics["miss_km"], 0.05)
        self.assertIn("PC_ABOVE_THRESHOLD", f4.flags)
        self.assertNotIn("HIGH_COVARIANCE", f4.flags)
        self.assertGreater(f4.confidence, 0.65)

    def test_kinematics_specialist_covert_rpo(self):
        st = open_state(self.sup, "S2", include_range_history=True)
        f = KinematicsSpecialist().run(st, self.graph)
        self.assertAlmostEqual(f.metrics["dv_unannounced_mps"], 32.9, delta=0.05)
        self.assertAlmostEqual(f.metrics["delta_i_deg"], -0.57, delta=0.01)
        self.assertAlmostEqual(f.metrics["dv_plane_change_mps"], 30.6, delta=0.5)
        self.assertAlmostEqual(f.metrics["range_km"], 35.0, delta=0.1)
        for flag in ("UNANNOUNCED_DV", "PLANE_CHANGE", "ALTITUDE_CHANGE", "SHADOWING_PATTERN"):
            self.assertIn(flag, f.flags)

    def test_kinematics_specialist_drag_anomaly(self):
        st = open_state(self.sup, "S3")
        f = KinematicsSpecialist().run(st, self.graph)
        self.assertAlmostEqual(f.metrics["decay_ratio"], 8.75, delta=0.1)
        self.assertIn("ANOMALOUS_DECAY", f.flags)
        self.assertIn("LOSS_OF_SIGNAL", f.flags)
        self.assertNotIn("UNANNOUNCED_DV", f.flags, "drag decay must not be misread as a burn")

    def test_photometric_specialist_tumble(self):
        st = open_state(self.sup, "S3", include_light_curve=True)
        f = PhotometricSpecialist().run(st, self.graph)
        self.assertAlmostEqual(f.metrics["dominant_frequency_hz"], 0.4, delta=0.02)
        self.assertAlmostEqual(f.metrics["spin_period_s"], 2.5, delta=0.15)
        self.assertIn("TUMBLING", f.flags)
        self.assertIn("HARMONIC_PRESENT", f.flags)
        self.assertGreater(f.metrics["peak_snr"], 5.0)

    def test_photometric_specialist_stable(self):
        rng = np.random.default_rng(1)
        st = open_state(self.sup, "S3")
        st.aux_evidence = {**st.aux_evidence, "light_curve": {"sample_rate_hz": 10.0, "samples_mag": (8.0 + 0.02 * rng.standard_normal(600)).tolist()}}
        f = PhotometricSpecialist().run(st, self.graph)
        self.assertIn("ATTITUDE_STABLE", f.flags)
        self.assertNotIn("TUMBLING", f.flags)

    def test_graph_specialist_multi_hop(self):
        s2 = open_state(self.sup, "S2")
        f = GraphSpecialist().run(s2, self.graph)
        for flag in ("NON_RESPONSIVE_OPERATOR", "MILITARY_OPERATOR", "UNDECLARED_PAYLOAD"):
            self.assertIn(flag, f.flags)
        self.assertEqual(f.details["secondary"]["lineage"]["launch_id"], "L-2025-088")
        self.assertIn("ZENOBIASAT-9", f.details["secondary"]["lineage"]["co_manifested_ids"])
        s1 = open_state(self.sup, "S1")
        f1 = GraphSpecialist().run(s1, self.graph)
        self.assertIn("SECONDARY_IS_DEBRIS", f1.flags)
        self.assertEqual(f1.details["secondary"]["lineage"]["parent_object_id"], "COSMOS-2251")
        self.assertEqual(f1.details["primary"]["operator"]["fleet_size"], 3)


# =========================================================================== #
class TestActiveTasking(unittest.TestCase):
    def setUp(self):
        self.graph = fresh_graph()
        self.sup = Supervisor(self.graph)
        self.engine = ActiveTaskingEngine()

    def _with_conjunction(self, sid, **kw):
        st = open_state(self.sup, sid, **kw)
        st.findings = {"conjunction": ConjunctionSpecialist().run(st, self.graph)}
        return st

    def test_s1_requests_radar_and_stc(self):
        st = self._with_conjunction("S1")
        d = self.engine.evaluate(st, self.graph, synthesis_confidence=0.02)
        self.assertTrue(d.needed)
        self.assertFalse(d.deadline_forced)
        types = {r.task_type for r in d.requests}
        self.assertEqual(types, {SensorTaskType.RADAR_REOBSERVATION, SensorTaskType.STC_OPERATOR_QUERY})
        radar = next(r for r in d.requests if r.task_type == SensorTaskType.RADAR_REOBSERVATION)
        self.assertEqual(radar.target_id, "COSMOS-2251-DEB-34459")
        self.assertEqual(radar.sensor_id, "GS-LEOLABS-KSR")
        self.assertGreater(radar.expected_information_gain, 0.9)
        self.assertEqual(radar.governance_level, ApprovalRoute.L0_AUTO)
        stc = next(r for r in d.requests if r.task_type == SensorTaskType.STC_OPERATOR_QUERY)
        self.assertEqual(stc.governance_level, ApprovalRoute.L1_ADVISORY)

    def test_s4_needs_no_sensing(self):
        st = self._with_conjunction("S4", now=s4_now())
        d = self.engine.evaluate(st, self.graph, synthesis_confidence=st.findings["conjunction"].confidence)
        self.assertFalse(d.needed)

    def test_deadline_stopping_rule(self):
        st = self._with_conjunction("S1", now=scenario("S1").conjunction.tca - timedelta(hours=2))
        d = self.engine.evaluate(st, self.graph, synthesis_confidence=0.02)
        self.assertTrue(d.needed)
        self.assertTrue(d.deadline_forced)
        self.assertEqual(d.requests, [])
        self.assertGreater(d.candidates_considered, 0)

    def test_no_duplicate_requests(self):
        st = self._with_conjunction("S1")
        d1 = self.engine.evaluate(st, self.graph, 0.02)
        st.sensor_tasking_requests = d1.requests
        d2 = self.engine.evaluate(st, self.graph, 0.02)
        self.assertTrue(d2.needed)
        self.assertEqual(d2.requests, [])

    def test_apply_radar_observation_collapses_and_repropagates(self):
        st = self._with_conjunction("S1")
        d = self.engine.evaluate(st, self.graph, 0.02)
        st.sensor_tasking_requests = d.requests
        st.status = CaseStatus.AWAITING_SENSING
        radar_req = next(r for r in d.requests if r.task_type == SensorTaskType.RADAR_REOBSERVATION)
        obs = FixtureSensorSimulator(scenario("S1")).observe(radar_req)
        # the simulator delivers the state 600 s before TCA; the engine must re-propagate
        self.assertAlmostEqual((st.tca - obs.observed_at).total_seconds() > 0, True)
        vol_before = st.findings["conjunction"].metrics["covariance_volume_km3"]
        self.engine.apply_sensor_observation(st, obs)
        f = ConjunctionSpecialist().run(st, self.graph)
        self.assertAlmostEqual(f.metrics["miss_km"], 4.2024, delta=0.002)
        self.assertLess(f.metrics["covariance_volume_km3"], vol_before / 1000)
        self.assertLess(f.metrics["pc"], 1e-4)
        self.assertEqual(st.current_state_vectors["COSMOS-2251-DEB-34459"].epoch, st.tca)
        self.assertTrue(st.current_state_vectors["COSMOS-2251-DEB-34459"].source.startswith("RADAR:"))
        self.assertEqual(st.sensor_tasking_requests[0].status if st.sensor_tasking_requests[0].task_id == radar_req.task_id else st.sensor_tasking_requests[1].status, TaskStatus.FULFILLED)

    def test_unknown_task_rejected(self):
        st = self._with_conjunction("S1")
        with self.assertRaises(ValueError):
            self.engine.apply_sensor_observation(
                st, SensorObservation(task_id="nope", task_type=SensorTaskType.STC_OPERATOR_QUERY, sensor_id="x", target_id="y", observed_at=st.now)
            )


# =========================================================================== #
class TestSupervisorScenario1(unittest.TestCase):
    """High-covariance CDM -> HOLD/MONITOR -> radar -> 4.2 km -> CLEARED, dv = 0."""

    @classmethod
    def setUpClass(cls):
        cls.graph = fresh_graph()
        cls.sup = Supervisor(cls.graph, sensor_simulator=FixtureSensorSimulator(scenario("S1")))
        cls.state = open_state(cls.sup, "S1")
        cls.sup.run(cls.state)

    def test_initial_action_is_hold(self):
        a = self.state.initial_action
        self.assertEqual(a.action_type, ActionType.HOLD_MONITOR)
        self.assertEqual(a.delta_v_mps, 0.0)
        self.assertEqual(a.burn_direction, "NONE")
        self.assertLess(a.confidence, 0.1)
        self.assertTrue(any("R2" in r for r in a.policy_refs))

    def test_radar_tasked_and_fulfilled(self):
        radar = [t for t in self.state.sensor_tasking_requests if t.task_type == SensorTaskType.RADAR_REOBSERVATION]
        self.assertEqual(len(radar), 1)
        self.assertEqual(radar[0].status, TaskStatus.FULFILLED)
        self.assertTrue(any(e.phase == Phase.SENSING_DISPATCHED for e in self.state.log))
        self.assertGreaterEqual(len(self.state.observations), 1)
        self.assertEqual(self.state.sensing_rounds, 1)

    def test_final_cleared_with_zero_dv(self):
        a = self.state.final_action
        self.assertEqual(a.action_type, ActionType.CLEARED)
        self.assertEqual(a.delta_v_mps, 0.0)
        self.assertEqual(a.burn_direction, "NONE")
        self.assertIsNone(a.propellant_kg)
        conj = self.state.findings["conjunction"]
        self.assertAlmostEqual(conj.metrics["miss_km"], 4.2024, delta=0.002)
        self.assertLess(conj.metrics["pc"], 1e-50)
        self.assertEqual(self.state.classification, "debris_collision_risk_false_alarm")

    def test_what_changed_diff(self):
        wc = self.state.what_changed
        self.assertIn("RADAR_REOBSERVATION", wc)
        self.assertIn("Pc: 0.000269 ->", wc)
        self.assertIn("miss distance [km]: 0.120 -> 4.202", wc)
        self.assertIn("dilution region: yes -> no", wc)
        self.assertIn("action: HOLD_MONITOR -> CLEARED", wc)

    def test_route_and_status(self):
        self.assertEqual(self.state.approval_route, ApprovalRoute.L0_AUTO)
        self.assertEqual(self.state.status, CaseStatus.RESOLVED)
        self.assertEqual(self.state.phase, Phase.COMMITTED)
        self.assertTrue(any(c.startswith("R4") for c in self.state.policy_citations))

    def test_case_written_to_graph(self):
        self.assertTrue(self.graph.has("CASE-S1"))
        node = self.graph.get("CASE-S1")
        self.assertIsInstance(node, CaseNode)
        self.assertEqual(node.classification, "debris_collision_risk_false_alarm")
        self.assertIn("miss distance [km]: 0.120 -> 4.202", node.what_changed)
        investigators = {u for u, _, _ in self.graph.edges_of("CASE-S1", EdgeType.INVESTIGATED_IN, "in")}
        self.assertEqual(investigators, {"ASAI-COM-7", "COSMOS-2251-DEB-34459", "OP-ORBITLINK"})
        # conjunction edge refreshed with the post-sensing numbers
        edge = dict(self.graph.get_conjunctions("ASAI-COM-7"))["COSMOS-2251-DEB-34459"]
        self.assertAlmostEqual(edge["miss_km"], 4.2024, delta=0.002)

    def test_blocked_flow_without_simulator(self):
        graph = fresh_graph()
        sup = Supervisor(graph)  # no simulator: the machine must block
        st = open_state(sup, "S1")
        sup.run(st)
        self.assertEqual(st.status, CaseStatus.AWAITING_SENSING)
        self.assertEqual(st.final_action, None)
        sim = FixtureSensorSimulator(scenario("S1"))
        for t in st.sensor_tasking_requests:
            sup.submit_observation(st, sim.observe(t))
        self.assertEqual(st.status, CaseStatus.OPEN)
        sup.run(st)
        self.assertEqual(st.status, CaseStatus.RESOLVED)
        self.assertEqual(st.final_action.action_type, ActionType.CLEARED)


# =========================================================================== #
class TestSupervisorScenario4(unittest.TestCase):
    """Tight covariance, Pc 5.2e-3 -> 0.355 m/s prograde CAM -> L2 gate, stop."""

    @classmethod
    def setUpClass(cls):
        cls.graph = fresh_graph()
        cls.sup = Supervisor(cls.graph, sensor_simulator=FixtureSensorSimulator(scenario("S4")))
        cls.state = open_state(cls.sup, "S4", now=s4_now())
        cls.sup.run(cls.state)

    def test_no_sensing_needed(self):
        self.assertEqual(self.state.sensor_tasking_requests, [])
        self.assertEqual(self.state.observations, [])
        self.assertFalse(self.state.deadline_forced)
        self.assertTrue(any("no sensing required" in e.message for e in self.state.log))

    def test_cam_package(self):
        a = self.state.final_action
        self.assertEqual(a.action_type, ActionType.EXECUTE_CAM)
        self.assertEqual(a.burn_direction, "prograde")
        self.assertAlmostEqual(a.delta_v_mps, 0.355, delta=0.01)
        self.assertAlmostEqual(a.propellant_kg, 0.197, delta=0.01)
        self.assertGreaterEqual(a.predicted_miss_km, 1.8 - 1e-6)
        self.assertLess(a.predicted_pc, 1e-6)
        self.assertEqual(a.burn_epoch, self.state.tca - timedelta(seconds=2400))
        self.assertEqual(self.state.initial_action.action_type, ActionType.EXECUTE_CAM)
        self.assertAlmostEqual(self.state.initial_action.delta_v_mps, a.delta_v_mps)
        m = self.state.findings["maneuver"]
        self.assertIn("FEASIBLE", m.flags)
        self.assertIn("SCREENING_CLEAR", m.flags)
        self.assertEqual(len(m.details["alternatives"]), 6)
        self.assertEqual({h["object_id"] for h in m.details["screening"]}, {"METEOSTAR-1", "SENTINEL-CLASS-EO"})

    def test_routed_l2_and_stopped_at_gate(self):
        self.assertEqual(self.state.approval_route, ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertEqual(self.state.status, CaseStatus.READY_FOR_HUMAN)
        self.assertEqual(self.state.phase, Phase.COMMITTED)
        self.assertIsNone(self.state.human_decision)
        self.assertTrue(any(c.startswith("R6") for c in self.state.policy_citations))
        self.assertTrue(any(c.startswith("R5") for c in self.state.policy_citations))
        self.assertEqual(self.state.classification, "debris_collision_risk_confirmed")
        self.assertEqual(self.state.severity, "CRITICAL")
        node = self.graph.get("CASE-S4")
        self.assertIn("AWAITING L2 SIGN-OFF", node.outcome)
        self.assertIsNone(node.closed_at)

    def test_human_approval_closes_case_without_execution(self):
        graph = fresh_graph()
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S4")))
        st = open_state(sup, "S4", now=s4_now())
        sup.run(st)
        with self.assertRaises(RuntimeError):
            sup.submit_observation(st, SensorObservation(task_id="x", task_type=SensorTaskType.STC_OPERATOR_QUERY, sensor_id="s", target_id="t", observed_at=st.now))
        sup.approve(st, "FDO-Alpha", note="burn authorised for uplink by ops")
        self.assertEqual(st.status, CaseStatus.RESOLVED)
        self.assertTrue(st.human_decision.approved)
        self.assertIn("APPROVED by FDO-Alpha", graph.get("CASE-S4").outcome)
        # the action itself is untouched: still a package, still routed L2
        self.assertEqual(st.final_action.action_type, ActionType.EXECUTE_CAM)
        self.assertEqual(st.approval_route, ApprovalRoute.L2_HUMAN_MANDATORY)

    def test_deadline_forced_wide_covariance_escalates_without_burn(self):
        """S1 geometry with only 2 h to TCA: sensing impossible, CAM forbidden -> ESCALATE_FDO, dv = 0."""
        graph = fresh_graph()
        sup = Supervisor(graph)
        st = open_state(sup, "S1", now=scenario("S1").conjunction.tca - timedelta(hours=2))
        sup.run(st)
        self.assertTrue(st.deadline_forced)
        self.assertEqual(st.final_action.action_type, ActionType.ESCALATE_FDO)
        self.assertEqual(st.final_action.delta_v_mps, 0.0)
        self.assertEqual(st.approval_route, ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertEqual(st.status, CaseStatus.READY_FOR_HUMAN)
        self.assertTrue(any(c.startswith("R3") for c in st.policy_citations))


# =========================================================================== #
class TestGovernanceGate(unittest.TestCase):
    def test_supervisor_refuses_misrouted_burn(self):
        """Even if the router were wrong, the validator and the state model both stop a burn below L2."""
        graph = fresh_graph()
        sup = Supervisor(graph)
        st = open_state(sup, "S4", now=s4_now())
        bad_verdict = PolicyVerdict(route=ApprovalRoute.L1_ADVISORY, citations=["R7 L1 for outbound advisories"])
        with mock.patch.object(supervisor_module, "route_action", return_value=bad_verdict):
            with self.assertRaises(PolicyViolation) as cm:
                sup.run(st)
        self.assertEqual(cm.exception.rule_id, "R6")
        self.assertIsNone(st.approval_route)
        self.assertNotEqual(st.status, CaseStatus.RESOLVED)

    def test_state_model_rejects_burn_below_l2(self):
        graph = fresh_graph()
        sup = Supervisor(graph)
        st = open_state(sup, "S4", now=s4_now())
        st.final_action = cam_action()
        for route in (ApprovalRoute.L0_AUTO, ApprovalRoute.L1_ADVISORY):
            with self.assertRaises(ValidationError):
                st.approval_route = route

    def test_validate_action_any_positive_dv(self):
        for dv in (1e-6, 0.35, 5.0):
            for route in (ApprovalRoute.L0_AUTO, ApprovalRoute.L1_ADVISORY):
                with self.assertRaises(PolicyViolation):
                    validate_action(cam_action(dv), route)


# =========================================================================== #
class TestOfflineFallback(unittest.TestCase):
    def test_key_heuristics(self):
        for bad in (None, "", "dummy", "DUMMY", "your_api_key", "gsk_short"):
            self.assertFalse(key_is_usable(bad), bad)
        self.assertTrue(key_is_usable("gsk_" + "a" * 40))

    def test_dummy_key_runs_offline(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            graph = fresh_graph()
            sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S1")))
            st = open_state(sup, "S1")
            sup.run(st)
        self.assertEqual(st.narrative_source, "offline")
        self.assertIn("CASE-S1", st.narrative)
        self.assertIn("CLEARED", st.narrative)
        self.assertIn("R4", st.narrative)
        self.assertEqual(st.status, CaseStatus.RESOLVED)

    def test_absent_key_runs_offline(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            gen = NarrativeGenerator()
            graph = fresh_graph()
            sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S4")), narrative=gen)
            st = open_state(sup, "S4", now=s4_now())
            sup.run(st)
        self.assertEqual(st.narrative_source, "offline")
        self.assertIn("0.355 m/s prograde", st.narrative)
        self.assertIn("awaiting human sign-off", st.narrative)

    def test_client_failure_falls_back(self):
        class Boom:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise ConnectionError("groq unreachable")

        gen = NarrativeGenerator(client=Boom(), api_key="gsk_" + "a" * 40)
        graph = fresh_graph()
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S1")), narrative=gen)
        st = open_state(sup, "S1")
        sup.run(st)
        self.assertEqual(st.narrative_source, "offline")

    def test_fake_client_success_does_not_touch_action(self):
        class Msg:
            content = "Narrative from the model."

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        class Fake:
            calls = []

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        Fake.calls.append(kwargs)
                        return Resp()

        gen = NarrativeGenerator(client=Fake(), api_key="gsk_" + "a" * 40)
        graph = fresh_graph()
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S4")), narrative=gen)
        st = open_state(sup, "S4", now=s4_now())
        sup.run(st)
        self.assertEqual(st.narrative_source, "groq")
        self.assertEqual(st.narrative, "Narrative from the model.")
        self.assertEqual(Fake.calls[0]["model"], "llama-3.3-70b-versatile")
        self.assertIn("0.355", Fake.calls[0]["messages"][1]["content"])
        self.assertAlmostEqual(st.final_action.delta_v_mps, 0.355, delta=0.01)  # R10: untouched


# =========================================================================== #
class TestScenarios2And3(unittest.TestCase):
    def test_s2_covert_rpo(self):
        graph = fresh_graph()
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S2")))
        st = open_state(sup, "S2")
        sup.run(st)
        self.assertEqual(st.initial_action.action_type, ActionType.HOLD_MONITOR)
        types = {t.task_type for t in st.sensor_tasking_requests}
        self.assertEqual(types, {SensorTaskType.OPTICAL_PHOTOMETRY, SensorTaskType.STC_OPERATOR_QUERY})
        self.assertNotIn(SensorTaskType.RADAR_REOBSERVATION, types)
        stc = next(t for t in st.sensor_tasking_requests if t.task_type == SensorTaskType.STC_OPERATOR_QUERY)
        self.assertEqual(stc.status, TaskStatus.UNANSWERED)
        self.assertEqual(st.classification, "covert_rpo_shadowing")
        self.assertEqual(st.severity, "CRITICAL")
        a = st.final_action
        self.assertEqual(a.action_type, ActionType.OPERATOR_ADVISORY)
        self.assertIn(ActionType.UN_REGISTRY_REPORT, a.secondary_actions)
        self.assertEqual(a.delta_v_mps, 0.0)
        self.assertEqual(st.approval_route, ApprovalRoute.L2_HUMAN_MANDATORY)
        self.assertEqual(st.status, CaseStatus.READY_FOR_HUMAN)
        self.assertIn("range to asset [km]: (new evidence) -> 35.0", st.what_changed)
        self.assertIn("operator responded: (new evidence) -> no", st.what_changed)
        self.assertTrue(any(c.startswith("R9") for c in st.policy_citations))

    def test_s3_tumble_triage(self):
        graph = fresh_graph()
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario("S3")))
        st = open_state(sup, "S3")
        sup.run(st)
        self.assertEqual(st.initial_action.action_type, ActionType.HOLD_MONITOR)
        self.assertEqual([t.task_type for t in st.sensor_tasking_requests], [SensorTaskType.OPTICAL_PHOTOMETRY])
        self.assertEqual(st.classification, "adcs_failure_tumbling")
        self.assertEqual(st.final_action.action_type, ActionType.RECLASSIFY_UNCONTROLLED)
        self.assertIn(ActionType.NEIGHBOR_ADVISORY, st.final_action.secondary_actions)
        self.assertEqual(st.approval_route, ApprovalRoute.L1_ADVISORY)
        self.assertEqual(st.status, CaseStatus.RESOLVED)
        self.assertAlmostEqual(st.findings["photometric"].metrics["dominant_frequency_hz"], 0.4, delta=0.02)
        self.assertIn("tumbling: (new evidence) -> yes", st.what_changed)
        self.assertIn("OP-POLARVIEW", st.final_action.outbound_targets)


# =========================================================================== #
class TestGraphMemory(unittest.TestCase):
    def test_similar_case_recall_after_commit(self):
        graph = fresh_graph()
        for sid, kw in (("S1", {}), ("S4", {"now": s4_now()})):
            sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(scenario(sid)))
            sup.run(open_state(sup, sid, **kw))
        self.assertEqual(len(graph.nodes_of_kind(NodeKind.CASE)), 2)
        probe = CaseNode(id="probe", name="p", opened_at=s4_now(), classification="debris_collision_risk_confirmed", involved_object_ids=["METEOSTAR-2"], operator_ids=["OP-EUROMET"])
        sims = graph.find_similar_cases(probe)
        self.assertEqual(sims[0].case.id, "CASE-S4")
        self.assertGreater(sims[0].score, 0.9)
        # a second S4 investigation sees the prior case through the graph specialist
        sup = Supervisor(graph)
        st = open_state(sup, "S4", now=s4_now())
        st.classification = "debris_collision_risk_confirmed"
        f = GraphSpecialist().run(st, graph)
        self.assertIn("PRIOR_CASES_FOUND", f.flags)
        self.assertEqual(f.details["similar_cases"][0]["id"], "CASE-S4")


if __name__ == "__main__":
    unittest.main()
