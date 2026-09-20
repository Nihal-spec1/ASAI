"""Stage 1 verification: astrodynamics engine, knowledge graph, and demo fixtures.

Runs under both ``pytest`` and ``python -m unittest``.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrodynamics import (  # noqa: E402
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    BurnDirection,
    ConjunctionGeometry,
    StateVector,
    TLE,
    assess_conjunction,
    build_tle,
    collision_probability_2d,
    collision_probability_3d,
    cw_displacement_at_tca,
    eci_to_rtn,
    eci_to_rtn_matrix,
    j2000_to_teme,
    plan_minimum_dv_cam,
    propagate_tle,
    rtn_to_eci,
    screen_post_maneuver,
    state_to_classical_elements,
    state_to_tle,
    teme_to_j2000,
    two_body_propagate,
)
from astrodynamics.foster_pc import diagonal_rtn_covariance, project_to_encounter_plane  # noqa: E402
from astrodynamics.maneuver_planner import apply_impulse, mean_motion_rad_s, primary_state_at_burn  # noqa: E402
from data import build_graph, conjunction_geometry, light_curve_arrays, load_fixtures, tle_of  # noqa: E402
from graph import CaseNode, EdgeType, NodeKind, OrbitalGraph  # noqa: E402

UTC = timezone.utc
ISS_L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
ISS_EPOCH = datetime(2019, 12, 9, 12, 0, 0, tzinfo=UTC)
# Independent oracle: Skyfield EarthSatellite.at(...) GCRS output for the TLE above.
ISS_R_J2000 = np.array([3518.69480355, -2642.50646644, 5167.67962938])
ISS_V_J2000 = np.array([5.74339431, 4.87675787, -1.40614875])

FIXTURES = load_fixtures()


def _circular_state(alt_km: float, inc_deg: float) -> tuple[np.ndarray, np.ndarray]:
    r = np.array([R_EARTH_KM + alt_km, 0.0, 0.0])
    vmag = math.sqrt(MU_EARTH_KM3_S2 / np.linalg.norm(r))
    v = vmag * np.array([0.0, math.cos(math.radians(inc_deg)), math.sin(math.radians(inc_deg))])
    return r, v


def _geom(r, v, miss_rtn, vrel_rtn, cov_p, cov_s, hbr_m=(12.5, 12.5), tca=ISS_EPOCH) -> ConjunctionGeometry:
    M = eci_to_rtn_matrix(r, v)
    rs = r + M.T @ np.asarray(miss_rtn, float)
    vs = v + M.T @ np.asarray(vrel_rtn, float)
    return ConjunctionGeometry(
        tca=tca, primary_id="P", secondary_id="S", r_primary=tuple(r), v_primary=tuple(v),
        r_secondary=tuple(rs), v_secondary=tuple(vs), cov_primary_rtn=np.asarray(cov_p).tolist(),
        cov_secondary_rtn=np.asarray(cov_s).tolist(), hbr_primary_m=hbr_m[0], hbr_secondary_m=hbr_m[1],
    )


# =========================================================================== #
class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.tle = TLE(name="ISS", line1=ISS_L1, line2=ISS_L2)

    def test_tle_parsing(self):
        self.assertEqual(self.tle.norad_id, 25544)
        self.assertAlmostEqual(self.tle.inclination_deg, 51.6439)
        self.assertAlmostEqual(self.tle.mean_motion_rev_day, 15.50103472)
        self.assertEqual(self.tle.epoch.year, 2019)
        self.assertEqual(self.tle.epoch.timetuple().tm_yday, 343)

    def test_bad_checksum_rejected(self):
        with self.assertRaises(ValueError):
            TLE(line1=ISS_L1[:-1] + "0", line2=ISS_L2)

    def test_sgp4_matches_independent_skyfield_oracle(self):
        st = propagate_tle(self.tle, ISS_EPOCH, frame="J2000")
        self.assertLess(np.linalg.norm(st.r_km - ISS_R_J2000), 1e-3, "position must match Skyfield to < 1 m")
        self.assertLess(np.linalg.norm(st.v_kms - ISS_V_J2000), 1e-6, "velocity must match Skyfield to < 1 mm/s")
        self.assertTrue(400 < st.altitude_km < 430)
        self.assertTrue(7.6 < st.speed_kms < 7.7)

    def test_skyfield_live_cross_check(self):
        from skyfield.api import EarthSatellite, load

        ts = load.timescale()
        sat = EarthSatellite(ISS_L1, ISS_L2, "ISS", ts)
        for minutes in (0, 37, 250):
            epoch = ISS_EPOCH + timedelta(minutes=minutes)
            g = sat.at(ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second))
            st = propagate_tle(self.tle, epoch)
            self.assertLess(np.linalg.norm(st.r_km - g.position.km) * 1000, 1.0)

    def test_teme_j2000_roundtrip_and_frame_shift(self):
        teme = propagate_tle(self.tle, ISS_EPOCH, frame="TEME")
        r_j, v_j = teme_to_j2000(teme.r_km, teme.v_kms, ISS_EPOCH)
        r_t, v_t = j2000_to_teme(r_j, v_j, ISS_EPOCH)
        np.testing.assert_allclose(r_t, teme.r_km, atol=1e-9)
        np.testing.assert_allclose(v_t, teme.v_kms, atol=1e-12)
        # TEME and J2000 differ by tens of km for a 2019 epoch (precession since J2000.0)
        self.assertGreater(np.linalg.norm(r_j - teme.r_km), 10.0)
        self.assertAlmostEqual(np.linalg.norm(r_j), np.linalg.norm(teme.r_km), places=9)

    def test_build_tle_roundtrip(self):
        epoch = datetime(2026, 9, 22, 3, 14, 0, tzinfo=UTC)
        t = build_tle(norad_id=61007, epoch=epoch, inclination_deg=53.05, raan_deg=118.4, eccentricity=0.0001,
                      argp_deg=90.0, mean_anomaly_deg=12.0, mean_motion_rev_day=15.05, bstar=1e-5, intl_designator="25031G")
        self.assertEqual(len(t.line1), 69)
        self.assertEqual(t.norad_id, 61007)
        self.assertAlmostEqual(t.inclination_deg, 53.05)
        self.assertAlmostEqual(t.eccentricity, 0.0001)
        self.assertAlmostEqual(t.mean_motion_rev_day, 15.05)
        self.assertLess(abs((t.epoch - epoch).total_seconds()), 0.01)
        st = propagate_tle(t, epoch)
        self.assertTrue(540 < st.altitude_km < 560)

    def test_state_to_tle_fit(self):
        st = propagate_tle(self.tle, ISS_EPOCH)
        fitted = state_to_tle(st, norad_id=99999)
        again = propagate_tle(fitted, ISS_EPOCH)
        self.assertLess(np.linalg.norm(again.r_km - st.r_km) * 1000, 20.0, "TLE fit must reproduce state to < 20 m")

    def test_classical_elements_of_circular_orbit(self):
        r, v = _circular_state(550.0, 53.0)
        el = state_to_classical_elements(r, v)
        self.assertAlmostEqual(el.a_km, R_EARTH_KM + 550.0, places=6)
        self.assertLess(el.e, 1e-12)
        self.assertAlmostEqual(el.i_deg, 53.0, places=9)
        self.assertAlmostEqual(el.period_s / 60.0, 95.6, delta=0.3)

    def test_two_body_conserves_energy_and_returns_after_one_period(self):
        r, v = _circular_state(820.0, 98.7)
        el = state_to_classical_elements(r, v)
        r1, v1 = two_body_propagate(r, v, el.period_s)
        self.assertLess(np.linalg.norm(r1 - r), 1e-3)
        e0 = np.dot(v, v) / 2 - MU_EARTH_KM3_S2 / np.linalg.norm(r)
        e1 = np.dot(v1, v1) / 2 - MU_EARTH_KM3_S2 / np.linalg.norm(r1)
        self.assertAlmostEqual(e0, e1, places=9)


class TestRTN(unittest.TestCase):
    def test_orthonormal_and_roundtrip(self):
        r, v = _circular_state(550.0, 53.0)
        M = eci_to_rtn_matrix(r, v)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(M), 1.0, places=12)
        x = np.array([1.2, -3.4, 5.6])
        np.testing.assert_allclose(rtn_to_eci(eci_to_rtn(x, r, v), r, v), x, atol=1e-12)
        # radial unit vector along r, transverse along v for a circular orbit
        np.testing.assert_allclose(eci_to_rtn(r, r, v), [np.linalg.norm(r), 0, 0], atol=1e-9)
        np.testing.assert_allclose(eci_to_rtn(v, r, v), [0, np.linalg.norm(v), 0], atol=1e-9)


# =========================================================================== #
class TestFosterPc(unittest.TestCase):
    def setUp(self):
        self.r, self.v = _circular_state(550.0, 53.0)
        self.vrel = [0.0, -7.0, 8.0]

    def test_small_hbr_matches_analytic_area_times_pdf(self):
        g = _geom(self.r, self.v, [0.0, 0.09, 0.08], self.vrel, diagonal_rtn_covariance(0.05, 0.2, 0.05), diagonal_rtn_covariance(0.3, 3.0, 4.5))
        miss, vrel, cov, hbr = g.miss_vector_eci(), g.relative_velocity_eci(), g.combined_covariance_eci(), g.hbr_combined_km
        m2, c2 = project_to_encounter_plane(miss, vrel, cov)
        pdf = math.exp(-0.5 * m2 @ np.linalg.inv(c2) @ m2) / (2 * math.pi * math.sqrt(np.linalg.det(c2)))
        analytic = math.pi * hbr**2 * pdf
        pc_adaptive = collision_probability_2d(miss, vrel, cov, hbr, method="adaptive")
        pc_gauss = collision_probability_2d(miss, vrel, cov, hbr, method="gauss")
        self.assertAlmostEqual(pc_adaptive / analytic, 1.0, delta=0.01)
        self.assertAlmostEqual(pc_gauss / pc_adaptive, 1.0, delta=1e-4)

    def test_pc_dilution_curve(self):
        """Pc vs covariance scale is unimodal: tight -> high, then dilutes toward zero."""
        scales = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.5, 10.0]
        pcs = []
        for s in scales:
            g = _geom(self.r, self.v, [0.02, 0.03, 0.028], self.vrel, diagonal_rtn_covariance(0.01, 0.02, 0.01), diagonal_rtn_covariance(s, 2 * s, s))
            pcs.append(assess_conjunction(g, method="adaptive").pc)
        k = int(np.argmax(pcs))
        self.assertTrue(0 < k < len(pcs) - 1, "peak must be interior (dilution region exists)")
        self.assertTrue(all(pcs[i] >= pcs[i + 1] for i in range(k, len(pcs) - 1)), "monotone decay past the peak")
        self.assertGreater(pcs[k], 1e-2)
        self.assertLess(pcs[-1], 1e-5)

    def test_zero_miss_large_hbr_saturates(self):
        cov = np.diag([1e-4, 1e-4, 1e-4])
        self.assertGreater(collision_probability_2d([0, 0, 0], [0, -7, 8], cov, 0.2), 0.999)
        self.assertGreater(collision_probability_3d([0, 0, 0], cov, 0.2), 0.999)

    def test_3d_small_sphere_matches_volume_times_pdf(self):
        cov = np.diag([0.04, 0.09, 0.01])
        miss = np.array([0.1, -0.05, 0.02])
        hbr = 0.005
        pdf = math.exp(-0.5 * miss @ np.linalg.inv(cov) @ miss) / ((2 * math.pi) ** 1.5 * math.sqrt(np.linalg.det(cov)))
        analytic = 4 / 3 * math.pi * hbr**3 * pdf
        self.assertAlmostEqual(collision_probability_3d(miss, cov, hbr) / analytic, 1.0, delta=0.01)

    def test_scenario1_pre_sensing_is_uncertainty_driven(self):
        a = assess_conjunction(conjunction_geometry(FIXTURES.scenario("S1"), "pre"))
        self.assertAlmostEqual(a.miss_distance_km, 0.120, delta=0.002)
        self.assertGreater(a.pc, 1e-4, "must exceed the screening threshold to open a case")
        self.assertTrue(a.in_dilution_region, "Pc must be driven by covariance, not geometry")
        self.assertGreater(a.sigma_to_miss_ratio, 10.0)
        self.assertGreater(a.bplane_sigma_major_km, 3.0)
        self.assertEqual(a.encounter_regime, "hypervelocity")

    def test_scenario1_post_sensing_clears(self):
        a = assess_conjunction(conjunction_geometry(FIXTURES.scenario("S1"), "post"))
        self.assertAlmostEqual(a.miss_distance_km, 4.2, delta=0.01)
        self.assertLess(a.pc, 1e-9)
        self.assertGreater(a.mahalanobis_distance, 10.0)

    def test_scenario4_tight_conjunction(self):
        a = assess_conjunction(conjunction_geometry(FIXTURES.scenario("S4"), "pre"))
        self.assertLess(a.miss_distance_km, 0.050)
        self.assertAlmostEqual(a.pc, 5.2e-3, delta=5.2e-3 * 0.02)
        self.assertEqual(a.encounter_regime, "hypervelocity")
        # "tight" in the operational sense: sub-km B-plane sigma, miss well inside it, Pc above the action threshold
        self.assertLess(a.bplane_sigma_major_km, 0.5)
        self.assertLess(a.sigma_to_miss_ratio, 10.0)
        self.assertGreater(a.pc, 1e-3)
        s1 = assess_conjunction(conjunction_geometry(FIXTURES.scenario("S1"), "pre"))
        self.assertGreater(s1.sigma_to_miss_ratio, 3 * a.sigma_to_miss_ratio, "S1 is far more uncertainty-dominated than S4")

    def test_scenario2_low_velocity_uses_3d(self):
        a = assess_conjunction(conjunction_geometry(FIXTURES.scenario("S2"), "pre"))
        self.assertEqual(a.encounter_regime, "low-velocity")
        self.assertEqual(a.pc, a.pc_3d)
        self.assertAlmostEqual(a.miss_distance_km, 35.0, delta=0.2)
        self.assertLess(a.pc, 1e-12)


# =========================================================================== #
class TestManeuverPlanner(unittest.TestCase):
    def test_cw_displacement_matches_two_body(self):
        r, v = _circular_state(820.0, 98.7)
        n = mean_motion_rad_s(r, v)
        lead = 2400.0
        for direction in (BurnDirection.PROGRADE, BurnDirection.RADIAL_OUT, BurnDirection.NORMAL_PLUS):
            dv_rtn = direction.unit_rtn * 0.35e-3  # km/s
            M0 = eci_to_rtn_matrix(r, v)
            r_ref, v_ref = two_body_propagate(r, v, lead)
            r_b, v_b = two_body_propagate(r, v + M0.T @ dv_rtn, lead)
            numeric = eci_to_rtn_matrix(r_ref, v_ref) @ (r_b - r_ref)
            cw, _ = cw_displacement_at_tca(dv_rtn, n, lead)
            self.assertLess(np.linalg.norm(cw - numeric), 0.03 * np.linalg.norm(numeric) + 0.005, f"{direction}: CW {cw} vs numeric {numeric}")

    def test_scenario4_minimum_dv_plan(self):
        sc = FIXTURES.scenario("S4")
        pol = sc.cam_policy
        geom = conjunction_geometry(sc, "pre")
        study = plan_minimum_dv_cam(geom, lead_time_s=pol.lead_time_s, target_pc=pol.target_pc, min_standoff_km=pol.min_standoff_km,
                                    spacecraft_mass_kg=pol.spacecraft_mass_kg, isp_s=pol.isp_s)
        self.assertEqual(len(study.plans), 6)
        rec = study.recommended
        self.assertIsNotNone(rec)
        self.assertEqual(rec.direction, BurnDirection.PROGRADE)
        self.assertTrue(0.25 <= rec.dv_mag_mps <= 0.45, f"dv {rec.dv_mag_mps} m/s")
        self.assertLessEqual(rec.predicted_pc, pol.target_pc)
        self.assertGreaterEqual(rec.predicted_miss_km, pol.min_standoff_km - 1e-6)
        self.assertLess(rec.propellant_kg, 0.5)
        self.assertGreater(study.baseline.pc, 1e-3)
        # along-track burns beat radial/normal burns for a 40-minute lead
        by_dir = {p.direction: p for p in study.plans}
        self.assertLess(by_dir[BurnDirection.PROGRADE].dv_mag_mps, by_dir[BurnDirection.RADIAL_OUT].dv_mag_mps)
        self.assertLess(by_dir[BurnDirection.PROGRADE].dv_mag_mps, by_dir[BurnDirection.NORMAL_PLUS].dv_mag_mps)

    def test_post_burn_screening_confirms_predicted_miss_and_clear_catalog(self):
        sc = FIXTURES.scenario("S4")
        pol = sc.cam_policy
        geom = conjunction_geometry(sc, "pre")
        study = plan_minimum_dv_cam(geom, lead_time_s=pol.lead_time_s, target_pc=pol.target_pc, min_standoff_km=pol.min_standoff_km)
        rec = study.recommended
        pre_burn = primary_state_at_burn(geom, pol.lead_time_s, tle=None)
        post_burn = apply_impulse(pre_burn, rec.dv_rtn_mps)
        secondary = StateVector(epoch=geom.tca, r=geom.r_secondary, v=geom.v_secondary)
        catalog = {oid: tle_of(sc.object(oid)) for oid in pol.screening_catalog_ids}
        result = screen_post_maneuver(post_burn, {"SL16-RB-90211": secondary, **catalog}, window_s=pol.lead_time_s + 600, standoff_km=pol.min_standoff_km)
        hit = next(h for h in result.hits if h.object_id == "SL16-RB-90211")
        # independent two-body check of the CW-based minimum-separation prediction
        self.assertAlmostEqual(hit.min_distance_km, rec.predicted_miss_km, delta=0.10 * rec.predicted_miss_km)
        self.assertLess(abs((hit.epoch - geom.tca).total_seconds()), 5.0)
        self.assertGreater(rec.predicted_separation_at_tca_km, rec.predicted_miss_km, "along-track burn shifts the TCA")
        catalog_hits = [h for h in result.hits if h.object_id in catalog]
        self.assertEqual(len(catalog_hits), 2)
        for h in catalog_hits:
            self.assertGreater(h.min_distance_km, 50.0, f"catalog object {h.object_id} too close")
            self.assertFalse(h.violates_standoff)


# =========================================================================== #
class TestOrbitalGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = build_graph(FIXTURES.scenarios)

    def test_population(self):
        s = self.g.stats()
        self.assertGreaterEqual(s["Satellite"], 10)
        self.assertGreaterEqual(s["Debris"], 2)
        self.assertGreaterEqual(s["Operator"], 7)
        self.assertGreaterEqual(s["LaunchEvent"], 6)
        self.assertGreaterEqual(s["GroundStation"], 6)

    def test_symmetric_edges_are_bidirectional(self):
        self.assertIn("ASAI-COM-6", self.g.neighbors("ASAI-COM-7", EdgeType.ORBITAL_NEIGHBOR))
        self.assertIn("ASAI-COM-7", self.g.neighbors("ASAI-COM-6", EdgeType.ORBITAL_NEIGHBOR))
        recs = [e for e in self.g.edge_records() if e["type"] == "CO_LAUNCHED_WITH" and {e["source"], e["target"]} == {"ASAI-COM-7", "ASAI-COM-6"}]
        self.assertEqual(len(recs), 1, "symmetric edge must be de-duplicated in records")

    def test_operator_fleet(self):
        fleet = {n.id for n in self.g.get_operator_fleet("OP-ORBITLINK")}
        self.assertEqual(fleet, {"ASAI-COM-5", "ASAI-COM-6", "ASAI-COM-7"})
        self.assertEqual(self.g.get_operator_of("ASAI-COM-7").id, "OP-ORBITLINK")
        euromet = {n.id for n in self.g.get_operator_fleet("OP-EUROMET")}
        self.assertEqual(euromet, {"METEOSTAR-1", "METEOSTAR-2", "SENTINEL-CLASS-EO"})

    def test_co_orbital_cluster_multi_hop(self):
        one = self.g.get_co_orbital_cluster("ASAI-COM-7", max_hops=1)
        two = self.g.get_co_orbital_cluster("ASAI-COM-7", max_hops=2)
        self.assertEqual(set(one), {"ASAI-COM-6"})
        self.assertEqual(two, {"ASAI-COM-6": 1, "ASAI-COM-5": 2})
        self.assertEqual(self.g.get_co_orbital_cluster("ASAI-COM-7", max_hops=2, max_shell_km=0.1), {})
        with_conj = self.g.get_co_orbital_cluster("METEOSTAR-2", max_hops=1, include_conjunctions=True)
        self.assertEqual(set(with_conj), {"METEOSTAR-1", "SL16-RB-90211"})

    def test_trace_parent_launch_for_debris(self):
        lin = self.g.trace_parent_launch("COSMOS-2251-DEB-34459")
        self.assertEqual(lin.parent_object_id, "COSMOS-2251")
        self.assertEqual(lin.launch_event.id, "L-1993-036")
        self.assertEqual(lin.hops, ["COSMOS-2251-DEB-34459", "COSMOS-2251", "L-1993-036"])
        self.assertEqual(self.g.get_operator_of("COSMOS-2251").designation, "military")

    def test_trace_launch_reveals_unregistered_co_passenger(self):
        lin = self.g.trace_parent_launch("OBJ-2025-088C")
        self.assertIsNone(lin.parent_object_id)
        self.assertEqual(lin.launch_event.id, "L-2025-088")
        self.assertEqual(lin.co_manifested_ids, ["ZENOBIASAT-9"])
        self.assertEqual(lin.launch_event.declared_payload_count, 1)
        self.assertEqual(len(lin.launch_event.payload_ids), 2, "two tracked payloads vs one declared")
        op = self.g.get_operator_of("OBJ-2025-088C")
        self.assertEqual(op.designation, "military")
        self.assertFalse(op.stc_responsive)
        self.assertFalse(self.g.get("OBJ-2025-088C").registered)

    def test_conjunction_edges(self):
        conj = dict(self.g.get_conjunctions("ASAI-COM-7"))
        self.assertIn("COSMOS-2251-DEB-34459", conj)
        self.assertGreater(conj["COSMOS-2251-DEB-34459"]["pc"], 1e-4)

    def test_write_case_node_and_recall(self):
        g = build_graph(FIXTURES.scenarios)  # fresh copy
        c1 = CaseNode(id="CASE-0001", name="S1 false alarm", scenario_id="S1", opened_at=datetime(2026, 9, 21, 21, 0, tzinfo=UTC),
                      closed_at=datetime(2026, 9, 21, 22, 30, tzinfo=UTC), classification="debris_collision_risk_false_alarm", severity="ADVISORY",
                      outcome="CAM cancelled", involved_object_ids=["ASAI-COM-7", "COSMOS-2251-DEB-34459"], operator_ids=["OP-ORBITLINK"],
                      embedding=[1.0, 0.0, 0.2])
        g.write_case_node(c1)
        self.assertEqual(g.kind_of("CASE-0001"), NodeKind.CASE)
        self.assertIn("CASE-0001", g.neighbors("ASAI-COM-7", EdgeType.INVESTIGATED_IN))
        self.assertIn("CASE-0001", g.neighbors("OP-ORBITLINK", EdgeType.INVESTIGATED_IN))
        probe = CaseNode(id="CASE-0002", name="new CDM on ASAI-COM-6", opened_at=datetime(2026, 10, 1, tzinfo=UTC),
                         classification="debris_collision_risk_false_alarm", involved_object_ids=["ASAI-COM-6"], operator_ids=["OP-ORBITLINK"],
                         embedding=[0.9, 0.1, 0.25])
        sims = g.find_similar_cases(probe)
        self.assertEqual(sims[0].case.id, "CASE-0001")
        self.assertGreater(sims[0].score, 0.5)
        g.write_case_node(probe)
        self.assertIn("CASE-0001", g.neighbors("CASE-0002", EdgeType.SIMILAR_TO))
        self.assertIn("CASE-0002", g.neighbors("CASE-0001", EdgeType.SIMILAR_TO))
        unrelated = CaseNode(id="CASE-0003", name="tumble", opened_at=datetime(2026, 10, 2, tzinfo=UTC), classification="adcs_failure_tumbling",
                             involved_object_ids=["TERRA-EYE-4"], operator_ids=["OP-GEOVISTA"])
        self.assertEqual(g.find_similar_cases(unrelated), [])

    def test_subgraph_export(self):
        sub = self.g.subgraph_around(["AEGIS-COM-3"], hops=1)
        self.assertIn("OBJ-2025-088C", sub.nodes)
        self.assertIn("OP-ALLIED-DEFENSE-COMMS", sub.nodes)
        self.assertNotIn("ASAI-COM-7", sub.nodes)


# =========================================================================== #
class TestFixtures(unittest.TestCase):
    def test_four_scenarios_present(self):
        self.assertEqual([s.id for s in FIXTURES.scenarios], ["S1", "S2", "S3", "S4"])
        self.assertEqual({s.trigger_type for s in FIXTURES.scenarios}, {"CDM", "KINEMATIC", "SIGNAL"})

    def test_every_tle_reproduces_its_stored_state(self):
        for sc in FIXTURES.scenarios:
            if sc.conjunction is None:
                continue
            for oid, st in sc.conjunction.states_pre.items():
                tle = tle_of(sc.object(oid))
                prop = propagate_tle(tle, st.epoch, frame=st.frame)
                self.assertLess(np.linalg.norm(prop.r_km - np.array(st.r)), 0.05, f"{sc.id}/{oid} TLE vs stored state (km)")

    def test_scenario3_light_curve_fft_peak_is_0p4_hz(self):
        from scipy.fft import rfft, rfftfreq
        from scipy.signal import detrend

        sc = FIXTURES.scenario("S3")
        t, mag = light_curve_arrays(sc)
        self.assertEqual(len(mag), 600)
        y = detrend(mag)
        spec = np.abs(rfft(y * np.hanning(y.size)))
        freqs = rfftfreq(y.size, d=1.0 / sc.light_curve.sample_rate_hz)
        peak = freqs[int(np.argmax(spec))]
        self.assertAlmostEqual(peak, 0.4, delta=1.0 / sc.light_curve.duration_s)
        # the 0.4 Hz line must dominate everything except its own harmonic
        mask = (np.abs(freqs - 0.4) > 0.05) & (np.abs(freqs - 0.8) > 0.05) & (freqs > 0.05)
        self.assertGreater(spec.max(), 5 * spec[mask].max())

    def test_scenario3_decay_anomaly(self):
        k = FIXTURES.scenario("S3").kinematics
        a = [h.a_km for h in k.element_history]
        observed_rate = a[-1] - a[-2]
        nominal_rate = a[1] - a[0]
        self.assertLess(observed_rate, 4 * nominal_rate)  # both negative: decay accelerated >4x
        self.assertGreater(k.residuals["ballistic_coefficient_ratio"], 2.0)

    def test_scenario2_kinematic_residuals(self):
        k = FIXTURES.scenario("S2").kinematics
        self.assertAlmostEqual(k.element_history[0].i_deg - k.element_history[-1].i_deg, 0.57, places=6)
        self.assertGreater(sum(b.dv_mps for b in k.maneuver_history), 30.0)
        self.assertTrue(all(not b.announced for b in k.maneuver_history))
        self.assertLess(k.range_history_km[-1][1], 40.0)
        self.assertGreater(k.range_history_km[0][1], 1000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
