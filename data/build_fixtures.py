"""Deterministically generate ``data/fixtures.json`` for the four demo scenarios.

Run:  python -m data.build_fixtures

All state vectors are produced by the ASAI astrodynamics code itself (SGP4
from constructed TLEs, RTN offsets for the secondaries, TLE fits for the
secondaries), so fixtures and engine are guaranteed consistent. Scenario 4's
secondary covariance scale is solved numerically so that Pc = 5.2e-3 exactly.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrodynamics import (  # noqa: E402
    ConjunctionGeometry,
    StateVector,
    TLE,
    assess_conjunction,
    build_tle,
    eci_to_rtn_matrix,
    propagate_tle,
    state_to_tle,
)
from astrodynamics.foster_pc import diagonal_rtn_covariance  # noqa: E402
from graph.models import GroundStation, LaunchEvent, Operator  # noqa: E402

from data.schema import (  # noqa: E402
    CamPolicyFixture,
    ConjunctionFixture,
    CovariancePair,
    EdgeFixture,
    ElementSnapshot,
    ExpectedOutcome,
    FixtureFile,
    GraphFixture,
    KinematicsFixture,
    LightCurveFixture,
    ManeuverRecord,
    ObjectFixture,
    ObservationFixture,
    ScenarioFixture,
    StateFixture,
    TLEFixture,
)

UTC = timezone.utc
GEO_MEAN_MOTION = 1.00270  # rev/day


def circular_mean_motion(alt_km: float) -> float:
    from astrodynamics import MU_EARTH_KM3_S2, R_EARTH_KM

    a = R_EARTH_KM + alt_km
    period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH_KM3_S2)
    return 86400.0 / period


def make_primary(
    *, norad: int, name: str, epoch: datetime, alt_km: float | None, inc: float, raan: float, argp: float, ma: float,
    mean_motion: float | None = None, ecc: float = 0.0001, intl: str = "26001A", bstar: float = 1e-5,
) -> Tuple[TLE, StateVector]:
    mm = mean_motion if mean_motion is not None else circular_mean_motion(alt_km)  # type: ignore[arg-type]
    tle = build_tle(
        norad_id=norad, name=name, epoch=epoch, inclination_deg=inc, raan_deg=raan, eccentricity=ecc,
        argp_deg=argp, mean_anomaly_deg=ma, mean_motion_rev_day=mm, intl_designator=intl, bstar=bstar,
    )
    return tle, propagate_tle(tle, epoch, frame="J2000")


def offset_secondary(primary: StateVector, miss_rtn_km, vrel_rtn_kms, *, norad: int, name: str, intl: str) -> Tuple[TLE, StateVector]:
    M = eci_to_rtn_matrix(primary.r_km, primary.v_kms)
    r = primary.r_km + M.T @ np.asarray(miss_rtn_km, float)
    v = primary.v_kms + M.T @ np.asarray(vrel_rtn_kms, float)
    state = StateVector.from_arrays(primary.epoch, r, v, frame="J2000")
    tle = state_to_tle(state, norad_id=norad, name=name, intl_designator=intl)
    return tle, state


def sf(state: StateVector) -> StateFixture:
    return StateFixture(epoch=state.epoch, r=tuple(round(x, 6) for x in state.r), v=tuple(round(x, 9) for x in state.v), frame=state.frame)


def tf(tle: TLE) -> TLEFixture:
    return TLEFixture(line1=tle.line1, line2=tle.line2)


def cov(sr: float, st: float, sn: float) -> List[List[float]]:
    return diagonal_rtn_covariance(sr, st, sn).tolist()


# --------------------------------------------------------------------------- #
# Scenario 1 : high-covariance conjunction -> false alarm cleared
# --------------------------------------------------------------------------- #
def scenario_1() -> ScenarioFixture:
    tca = datetime(2026, 9, 22, 3, 14, 0, tzinfo=UTC)
    p_tle, p_state = make_primary(norad=61007, name="ASAI-COM-7", epoch=tca, alt_km=550.0, inc=53.05, raan=118.4, argp=90.0, ma=12.0, intl="25031G")
    # crossing debris: ~10.6 km/s relative, mostly along-track/normal
    vrel = [0.0, -7.0, 8.0]
    s_tle, s_pre = offset_secondary(p_state, [0.0, 0.09, 0.08], vrel, norad=34459, name="COSMOS 2251 DEB", intl="93036BFF")
    _, s_post = offset_secondary(p_state, [0.5, 3.0, 2.9], vrel, norad=34459, name="COSMOS 2251 DEB", intl="93036BFF")

    cov_pre = CovariancePair(primary_rtn=cov(0.05, 0.20, 0.05), secondary_rtn=cov(0.30, 3.00, 4.50), note="secondary from sparse 2-track radar fit; cross-track sigma 4.5 km")
    cov_post = CovariancePair(primary_rtn=cov(0.05, 0.20, 0.05), secondary_rtn=cov(0.05, 0.15, 0.10), note="after dedicated LeoLabs KSR tracking pass at TCA-6h")

    pre = assess_conjunction(conj_geom(tca, "ASAI-COM-7", "COSMOS-2251-DEB-34459", p_state, s_pre, cov_pre, 12.5, 12.5))
    post = assess_conjunction(conj_geom(tca, "ASAI-COM-7", "COSMOS-2251-DEB-34459", p_state, s_post, cov_post, 12.5, 12.5))

    fleet = []
    for k, (norad, ma) in enumerate([(61005, 4.0), (61006, 8.0)], start=5):
        t, _ = make_primary(norad=norad, name=f"ASAI-COM-{k}", epoch=tca, alt_km=550.0, inc=53.05, raan=118.4, argp=90.0, ma=ma, intl=f"25031{chr(ord('E')+k-5)}")
        fleet.append(ObjectFixture(id=f"ASAI-COM-{k}", name=f"ASAI-COM-{k}", norad_id=norad, kind="satellite", cospar_id=f"2025-031{chr(ord('E')+k-5)}", operator_id="OP-ORBITLINK", launch_id="L-2025-031", tle=tf(t), hbr_m=12.5, mass_kg=260, bus_type="OL-Bus-2", country="USA", launch_date=date(2025, 2, 14), role="neighbor"))

    objects = [
        ObjectFixture(id="ASAI-COM-7", name="ASAI-COM-7", norad_id=61007, kind="satellite", cospar_id="2025-031G", operator_id="OP-ORBITLINK", launch_id="L-2025-031", tle=tf(p_tle), hbr_m=12.5, mass_kg=260, isp_s=1500, bus_type="OL-Bus-2 (Hall thruster)", country="USA", launch_date=date(2025, 2, 14), role="primary"),
        ObjectFixture(id="COSMOS-2251-DEB-34459", name="COSMOS 2251 DEB", norad_id=34459, kind="debris", parent_object_id="COSMOS-2251", fragmentation_event_id="FRAG-2009-02-10-IRIDIUM33-COSMOS2251", tle=tf(s_tle), hbr_m=12.5, rcs_category="small", country="Russia", role="secondary"),
        ObjectFixture(id="COSMOS-2251", name="COSMOS 2251", norad_id=22675, kind="satellite", cospar_id="1993-036A", operator_id="OP-RU-SPACE-FORCES", launch_id="L-1993-036", hbr_m=4.0, mass_kg=900, bus_type="Strela-2M", country="Russia", status="non-operational", launch_date=date(1993, 6, 16), role="parent"),
        *fleet,
    ]
    graph = GraphFixture(
        operators=[
            Operator(id="OP-ORBITLINK", name="OrbitLink Communications", country="USA", designation="commercial", contact_protocol="Space-Track STC + 24/7 NOC email", stc_responsive=True),
            Operator(id="OP-RU-SPACE-FORCES", name="Russian Space Forces (legacy Strela operator)", country="Russia", designation="military", contact_protocol="none (defunct object)", stc_responsive=False),
        ],
        launch_events=[
            LaunchEvent(id="L-2025-031", name="OrbitLink Group 7", launch_date=date(2025, 2, 14), site="Vandenberg SLC-4E", booster="Falcon 9 B5", cospar_launch_id="2025-031", payload_ids=["ASAI-COM-5", "ASAI-COM-6", "ASAI-COM-7"], declared_payload_count=3),
            LaunchEvent(id="L-1993-036", name="Cosmos 2251 launch", launch_date=date(1993, 6, 16), site="Plesetsk 132/1", booster="Kosmos-3M", cospar_launch_id="1993-036", payload_ids=["COSMOS-2251"], declared_payload_count=1),
        ],
        ground_stations=[
            GroundStation(id="GS-LEOLABS-KSR", name="LeoLabs Kiwi Space Radar", latitude_deg=-45.04, longitude_deg=170.09, altitude_m=390, operator_id=None, sensor_type="radar", min_elevation_deg=5.0),
            GroundStation(id="GS-ORBITLINK-NOC", name="OrbitLink Redmond TT&C", latitude_deg=47.67, longitude_deg=-122.12, sensor_type="telemetry", operator_id="OP-ORBITLINK"),
        ],
        edges=[
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="ASAI-COM-7", target="ASAI-COM-6", attrs={"shell_km": 0.4}),
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="ASAI-COM-6", target="ASAI-COM-5", attrs={"shell_km": 0.4}),
            EdgeFixture(type="CO_LAUNCHED_WITH", source="ASAI-COM-7", target="ASAI-COM-6"),
            EdgeFixture(type="CO_LAUNCHED_WITH", source="ASAI-COM-7", target="ASAI-COM-5"),
            EdgeFixture(type="IN_CONJUNCTION_WITH", source="ASAI-COM-7", target="COSMOS-2251-DEB-34459", attrs={"tca": tca.isoformat(), "miss_km": round(pre.miss_distance_km, 4), "pc": pre.pc}),
            EdgeFixture(type="TRACKED_BY", source="ASAI-COM-7", target="GS-ORBITLINK-NOC"),
            EdgeFixture(type="TRACKED_BY", source="COSMOS-2251-DEB-34459", target="GS-LEOLABS-KSR"),
        ],
    )
    return ScenarioFixture(
        id="S1",
        title="High-Covariance Conjunction -> False Alarm Cleared (Fuel Saved)",
        trigger_type="CDM",
        summary="CDM reports a 120 m miss with a debris fragment, but the fragment's cross-track sigma is 4.5 km. The agent must refuse to manoeuvre, task radar, collapse covariance and re-screen.",
        objects=objects,
        graph=graph,
        conjunction=ConjunctionFixture(
            primary_id="ASAI-COM-7", secondary_id="COSMOS-2251-DEB-34459", tca=tca,
            states_pre={"ASAI-COM-7": sf(p_state), "COSMOS-2251-DEB-34459": sf(s_pre)},
            covariance_pre=cov_pre,
            states_post={"COSMOS-2251-DEB-34459": sf(s_post)},
            covariance_post=cov_post,
            cdm_reported_miss_km=round(pre.miss_distance_km, 4), cdm_reported_pc=pre.pc,
        ),
        observations=[
            ObservationFixture(sensor_type="radar", sensor_id="GS-LEOLABS-KSR", epoch=tca - timedelta(hours=6), governance_level="L0",
                               description="Dedicated tracking pass on COSMOS 2251 DEB; 3 detections, 90 s arc.",
                               effect="conjunction.states_post", payload={"detections": 3, "arc_s": 90, "range_noise_m": 15}),
            ObservationFixture(sensor_type="stc", sensor_id="OP-ORBITLINK", epoch=tca - timedelta(hours=9), governance_level="L1",
                               description="Operator confirms no planned manoeuvre for ASAI-COM-7 in the window.", effect="none", payload={"planned_burn": False}),
        ],
        expected=ExpectedOutcome(
            classification="debris_collision_risk_false_alarm", severity="ADVISORY", governance_level="L0",
            recommended_action="CANCEL_CAM",
            key_numbers={"pc_pre": pre.pc, "pc_post": post.pc, "miss_pre_km": pre.miss_distance_km, "miss_post_km": post.miss_distance_km, "sigma_cross_pre_km": 4.5},
        ),
    )


def conj_geom(tca, pid, sid, p: StateVector, s: StateVector, c: CovariancePair, hp: float, hs: float) -> ConjunctionGeometry:
    return ConjunctionGeometry(tca=tca, primary_id=pid, secondary_id=sid, r_primary=p.r, v_primary=p.v, r_secondary=s.r, v_secondary=s.v,
                               cov_primary_rtn=c.primary_rtn, cov_secondary_rtn=c.secondary_rtn, hbr_primary_m=hp, hbr_secondary_m=hs)


# --------------------------------------------------------------------------- #
# Scenario 2 : covert RPO / shadowing in GEO
# --------------------------------------------------------------------------- #
def scenario_2() -> ScenarioFixture:
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
    p_tle, p_state = make_primary(norad=59210, name="AEGIS-COM 3", epoch=t0, alt_km=None, mean_motion=GEO_MEAN_MOTION, inc=0.05, raan=95.0, argp=180.0, ma=42.3, ecc=0.0002, intl="24044A", bstar=0.0)
    # inspector trails by 35 km along-track with a slow closing drift
    s_tle, s_state = offset_secondary(p_state, [-1.5, -35.0, 0.8], [0.0, 0.0008, 0.0004], norad=61988, name="OBJECT 2025-088C", intl="25088C")
    civ_tle, civ_state = offset_secondary(p_state, [3.0, 1450.0, -2.0], [0.0, 0.0, 0.0], norad=61986, name="ZENOBIASAT-9", intl="25088A")

    cov_pair = CovariancePair(primary_rtn=cov(0.5, 1.0, 0.5), secondary_rtn=cov(1.0, 3.0, 1.0), note="GEO optical-only orbits")
    a = assess_conjunction(conj_geom(t0, "AEGIS-COM-3", "OBJ-2025-088C", p_state, s_state, cov_pair, 8.0, 3.0))

    objects = [
        ObjectFixture(id="AEGIS-COM-3", name="AEGIS-COM 3", norad_id=59210, kind="satellite", cospar_id="2024-044A", operator_id="OP-ALLIED-DEFENSE-COMMS", launch_id="L-2024-044", tle=tf(p_tle), hbr_m=8.0, mass_kg=4200, bus_type="Sentinel-GEO", country="Allied Coalition", orbit_regime="GEO", launch_date=date(2024, 3, 2), role="primary"),
        ObjectFixture(id="OBJ-2025-088C", name="OBJECT 2025-088C (unregistered)", norad_id=61988, kind="satellite", cospar_id="2025-088C", operator_id="OP-ZENOBIA-MAI", launch_id="L-2025-088", tle=tf(s_tle), hbr_m=3.0, mass_kg=650, bus_type="unknown (registry: 'adapter debris')", country="Zenobia", orbit_regime="GEO", registered=False, launch_date=date(2025, 5, 19), role="secondary"),
        ObjectFixture(id="ZENOBIASAT-9", name="ZENOBIASAT-9", norad_id=61986, kind="satellite", cospar_id="2025-088A", operator_id="OP-ZENOBIA-TELECOM", launch_id="L-2025-088", tle=tf(civ_tle), hbr_m=10.0, mass_kg=5100, bus_type="ZS-5000", country="Zenobia", orbit_regime="GEO", launch_date=date(2025, 5, 19), role="neighbor"),
    ]
    graph = GraphFixture(
        operators=[
            Operator(id="OP-ALLIED-DEFENSE-COMMS", name="Allied Defense Communications Agency", country="Allied Coalition", designation="military", contact_protocol="classified SATOPS desk", stc_responsive=True),
            Operator(id="OP-ZENOBIA-MAI", name="Zenobia Ministry of Aerospace Industry", country="Zenobia", designation="military", contact_protocol="none published", stc_responsive=False),
            Operator(id="OP-ZENOBIA-TELECOM", name="Zenobia National Telecom", country="Zenobia", designation="commercial", contact_protocol="ITU filing contact", stc_responsive=True),
        ],
        launch_events=[
            LaunchEvent(id="L-2024-044", name="AEGIS-COM 3 launch", launch_date=date(2024, 3, 2), site="Cape Canaveral SLC-41", booster="Vulcan Centaur", cospar_launch_id="2024-044", payload_ids=["AEGIS-COM-3"], declared_payload_count=1),
            LaunchEvent(id="L-2025-088", name="ZenobiaSat-9 launch", launch_date=date(2025, 5, 19), site="Zenobia Eastern Range", booster="ZL-5 heavy", cospar_launch_id="2025-088", payload_ids=["ZENOBIASAT-9", "OBJ-2025-088C"], declared_payload_count=1),
        ],
        ground_stations=[
            GroundStation(id="GS-GEO-OPTICAL-1", name="Allied GEO Optical Fence, Site 1", latitude_deg=-31.27, longitude_deg=149.07, altitude_m=1165, sensor_type="optical", min_elevation_deg=20.0),
        ],
        edges=[
            EdgeFixture(type="CO_LAUNCHED_WITH", source="OBJ-2025-088C", target="ZENOBIASAT-9"),
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="AEGIS-COM-3", target="OBJ-2025-088C", attrs={"shell_km": 1.5, "longitude_sep_deg": 0.05}),
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="AEGIS-COM-3", target="ZENOBIASAT-9", attrs={"shell_km": 3.0, "longitude_sep_deg": 1.97}),
            EdgeFixture(type="IN_CONJUNCTION_WITH", source="AEGIS-COM-3", target="OBJ-2025-088C", attrs={"tca": t0.isoformat(), "miss_km": round(a.miss_distance_km, 3), "pc": a.pc, "regime": a.encounter_regime}),
            EdgeFixture(type="TRACKED_BY", source="OBJ-2025-088C", target="GS-GEO-OPTICAL-1"),
        ],
    )
    hist = [ElementSnapshot(epoch=t0 - timedelta(days=d), a_km=a_, e=0.0003, i_deg=i_, raan_deg=95.0 + 0.01 * d, source="TLE") for d, a_, i_ in
            [(14, 42124.0, 0.62), (10, 42124.0, 0.62), (8, 42131.0, 0.55), (6, 42141.0, 0.41), (4, 42152.0, 0.24), (2, 42160.0, 0.11), (0, 42164.5, 0.05)]]
    burns = [ManeuverRecord(epoch=t0 - timedelta(days=d, hours=h), dv_mps=dv, direction=dirn) for d, h, dv, dirn in
             [(8, 3, 5.4, "normal_plus"), (7, 3, 5.6, "normal_plus"), (6, 2, 5.5, "normal_plus"), (5, 4, 5.2, "normal_plus"), (4, 3, 4.9, "normal_plus"), (3, 2, 4.8, "normal_plus"), (2, 1, 1.5, "prograde")]]
    return ScenarioFixture(
        id="S2",
        title="Non-Cooperative Inspector Satellite -> Covert RPO Detection",
        trigger_type="KINEMATIC",
        summary="An unregistered co-passenger from a foreign launch executes a 0.57 deg low-thrust plane change and drifts into the GEO slot of a defense comsat, holding 35 km trailing.",
        objects=objects,
        graph=graph,
        conjunction=ConjunctionFixture(primary_id="AEGIS-COM-3", secondary_id="OBJ-2025-088C", tca=t0,
                                       states_pre={"AEGIS-COM-3": sf(p_state), "OBJ-2025-088C": sf(s_state), "ZENOBIASAT-9": sf(civ_state)},
                                       covariance_pre=cov_pair),
        observations=[
            ObservationFixture(sensor_type="optical", sensor_id="GS-GEO-OPTICAL-1", epoch=t0 - timedelta(hours=10), governance_level="L0",
                               description="Astrometric arc confirms OBJ-2025-088C co-located within 40 km of AEGIS-COM 3 for 6 consecutive nights.", effect="kinematics.range_history_km", payload={"nights": 6}),
            ObservationFixture(sensor_type="stc", sensor_id="OP-ZENOBIA-MAI", epoch=t0 - timedelta(hours=8), governance_level="L1",
                               description="Coordination query sent to registered launch operator; no response in 8 h.", effect="none", payload={"responded": False}),
        ],
        kinematics=KinematicsFixture(
            object_id="OBJ-2025-088C", element_history=hist,
            residuals={"delta_i_deg": -0.57, "delta_a_km": 40.5, "dv_plane_change_mps": 30.6, "dv_altitude_mps": 1.5, "dv_total_estimated_mps": 32.9},
            maneuver_history=burns,
            range_history_km=[(-14, 1520.0), (-10, 1480.0), (-8, 1100.0), (-6, 620.0), (-4, 210.0), (-3, 96.0), (-2, 48.0), (-1, 38.0), (0, round(a.miss_distance_km, 1))],
            notes="Registry lists 2025-088C as launch adapter debris; observed dv budget (~33 m/s) is inconsistent with an inert object.",
        ),
        expected=ExpectedOutcome(classification="covert_rpo_shadowing", severity="CRITICAL", governance_level="L2",
                                 recommended_action="L1_OPERATOR_ADVISORY + L2_UN_REGISTRY_REPORT",
                                 key_numbers={"range_km": a.miss_distance_km, "dv_total_mps": 32.9, "delta_i_deg": 0.57, "pc": a.pc}),
    )


# --------------------------------------------------------------------------- #
# Scenario 3 : drag anomaly + tumbling
# --------------------------------------------------------------------------- #
def scenario_3() -> ScenarioFixture:
    t0 = datetime(2026, 9, 20, 22, 30, 0, tzinfo=UTC)
    p_tle, p_state = make_primary(norad=58311, name="TERRA-EYE 4", epoch=t0, alt_km=498.2, inc=97.42, raan=210.7, argp=95.0, ma=265.0, intl="23170B", bstar=4.2e-4)
    n1_tle, _ = make_primary(norad=58310, name="TERRA-EYE 3", epoch=t0, alt_km=505.0, inc=97.42, raan=210.7, argp=95.0, ma=290.0, intl="23170A", bstar=1.2e-4)
    n2_tle, _ = make_primary(norad=57600, name="POLARVIEW-2", epoch=t0, alt_km=512.0, inc=97.45, raan=214.0, argp=90.0, ma=20.0, intl="23120C", bstar=1.0e-4)

    fs, dur = 10.0, 60.0
    t = np.arange(0.0, dur, 1.0 / fs)
    rng = np.random.default_rng(42)
    mags = 7.5 + 0.6 * np.sin(2 * math.pi * 0.4 * t + 0.3) + 0.2 * np.sin(2 * math.pi * 0.8 * t + 1.1) + rng.normal(0.0, 0.05, t.size)
    lc = LightCurveFixture(sample_rate_hz=fs, duration_s=dur, dominant_frequency_hz=0.4, harmonic_frequency_hz=0.8, amplitude_mag=0.6, mean_mag=7.5, noise_sigma_mag=0.05, seed=42, samples_mag=[round(float(m), 4) for m in mags])

    hist = [ElementSnapshot(epoch=t0 - timedelta(days=d), a_km=a_, e=0.0011, i_deg=97.42, raan_deg=210.7 - 0.98 * d * 0 + 0.0) for d, a_ in
            [(6, 6878.62), (5, 6878.54), (4, 6878.46), (3, 6878.14), (2, 6877.71), (1, 6877.04), (0, 6876.34)]]
    objects = [
        ObjectFixture(id="TERRA-EYE-4", name="TERRA-EYE 4", norad_id=58311, kind="satellite", cospar_id="2023-170B", operator_id="OP-GEOVISTA", launch_id="L-2023-170", tle=tf(p_tle), hbr_m=4.0, mass_kg=180, bus_type="GV-180 EO", country="Canada", status="degraded", launch_date=date(2023, 11, 11), role="primary"),
        ObjectFixture(id="TERRA-EYE-3", name="TERRA-EYE 3", norad_id=58310, kind="satellite", cospar_id="2023-170A", operator_id="OP-GEOVISTA", launch_id="L-2023-170", tle=tf(n1_tle), hbr_m=4.0, mass_kg=180, bus_type="GV-180 EO", country="Canada", launch_date=date(2023, 11, 11), role="neighbor"),
        ObjectFixture(id="POLARVIEW-2", name="POLARVIEW-2", norad_id=57600, kind="satellite", cospar_id="2023-120C", operator_id="OP-POLARVIEW", launch_id="L-2023-120", tle=tf(n2_tle), hbr_m=3.0, mass_kg=120, bus_type="PV-120", country="Finland", launch_date=date(2023, 8, 3), role="neighbor"),
    ]
    graph = GraphFixture(
        operators=[
            Operator(id="OP-GEOVISTA", name="GeoVista Imaging", country="Canada", designation="commercial", contact_protocol="24/7 MOC phone + STC", stc_responsive=True),
            Operator(id="OP-POLARVIEW", name="PolarView Oy", country="Finland", designation="commercial", contact_protocol="STC email", stc_responsive=True),
        ],
        launch_events=[
            LaunchEvent(id="L-2023-170", name="Transporter-9 rideshare", launch_date=date(2023, 11, 11), site="Vandenberg SLC-4E", booster="Falcon 9 B5", cospar_launch_id="2023-170", payload_ids=["TERRA-EYE-3", "TERRA-EYE-4"], declared_payload_count=90),
            LaunchEvent(id="L-2023-120", name="Electron 'We Love The Nightlife'", launch_date=date(2023, 8, 3), site="Mahia LC-1B", booster="Electron", cospar_launch_id="2023-120", payload_ids=["POLARVIEW-2"], declared_payload_count=1),
        ],
        ground_stations=[
            GroundStation(id="GS-FTN-3", name="Falcon Telescope Network, Site 3", latitude_deg=38.99, longitude_deg=-104.86, altitude_m=2000, sensor_type="optical", min_elevation_deg=15.0),
            GroundStation(id="GS-SVALBARD", name="Svalbard SvalSat", latitude_deg=78.23, longitude_deg=15.39, altitude_m=460, operator_id="OP-GEOVISTA", sensor_type="telemetry"),
        ],
        edges=[
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="TERRA-EYE-4", target="TERRA-EYE-3", attrs={"shell_km": 6.8}),
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="TERRA-EYE-3", target="POLARVIEW-2", attrs={"shell_km": 7.0}),
            EdgeFixture(type="CO_LAUNCHED_WITH", source="TERRA-EYE-4", target="TERRA-EYE-3"),
            EdgeFixture(type="TRACKED_BY", source="TERRA-EYE-4", target="GS-SVALBARD"),
            EdgeFixture(type="TRACKED_BY", source="TERRA-EYE-4", target="GS-FTN-3"),
        ],
    )
    return ScenarioFixture(
        id="S3",
        title="Sudden Drag Anomaly -> Tumbling Spacecraft Triage",
        trigger_type="SIGNAL",
        summary="TERRA-EYE 4 stops downlinking and its semi-major axis decays 4x faster than its neighbours. Optical photometry must decide between a controlled safe-mode and an uncontrolled tumble.",
        objects=objects,
        graph=graph,
        observations=[
            ObservationFixture(sensor_type="telemetry", sensor_id="GS-SVALBARD", epoch=t0 - timedelta(hours=31), governance_level="L0", description="Last valid frame; carrier lost on next 3 passes.", effect="none", payload={"passes_missed": 3}),
            ObservationFixture(sensor_type="optical", sensor_id="GS-FTN-3", epoch=t0 - timedelta(hours=2), governance_level="L0", description="60 s photometric light curve at 10 Hz.", effect="light_curve", payload={"filter": "clear", "exposure_ms": 100}),
        ],
        light_curve=lc,
        kinematics=KinematicsFixture(
            object_id="TERRA-EYE-4", element_history=hist,
            residuals={"decay_rate_observed_km_day": -0.70, "decay_rate_nominal_km_day": -0.08, "ballistic_coefficient_ratio": 3.1, "hours_since_last_telemetry": 31},
            notes="Neighbour TERRA-EYE 3 (same bus, same shell) decays at 0.09 km/day over the same window; space weather Kp <= 3, so atmosphere is not the cause.",
        ),
        expected=ExpectedOutcome(classification="adcs_failure_tumbling", severity="WARNING", governance_level="L1",
                                 recommended_action="RECLASSIFY_UNCONTROLLED + NEIGHBOR_ADVISORY",
                                 key_numbers={"tumble_frequency_hz": 0.4, "decay_ratio": 8.75, "ballistic_coefficient_ratio": 3.1}),
    )


# --------------------------------------------------------------------------- #
# Scenario 4 : tight conjunction -> human-approved CAM
# --------------------------------------------------------------------------- #
def scenario_4() -> ScenarioFixture:
    tca = datetime(2026, 9, 23, 14, 52, 30, tzinfo=UTC)
    p_tle, p_state = make_primary(norad=60021, name="METEOSTAR-2", epoch=tca, alt_km=820.0, inc=98.7, raan=302.6, argp=88.0, ma=147.0, intl="24188A", bstar=2.1e-5)
    vrel = [0.0, -7.0, 8.0]
    miss_rtn = [0.020, 0.030, 0.028]  # |miss| ~ 46 m
    s_tle, s_state = offset_secondary(p_state, miss_rtn, vrel, norad=90211, name="SL-16 R/B", intl="92093B")
    hbr_p, hbr_s = 8.0, 17.0

    def pc_for(s: float) -> float:
        c = CovariancePair(primary_rtn=cov(0.03, 0.08, 0.03), secondary_rtn=cov(s, 2 * s, s))
        return assess_conjunction(conj_geom(tca, "METEOSTAR-2", "SL16-RB-90211", p_state, s_state, c, hbr_p, hbr_s)).pc

    s_star = brentq(lambda s: pc_for(s) - 5.2e-3, 0.02, 1.0, xtol=1e-9)
    cov_pair = CovariancePair(primary_rtn=cov(0.03, 0.08, 0.03), secondary_rtn=cov(s_star, 2 * s_star, s_star),
                              note=f"tight covariance from 4 radar tracks over 48 h; secondary sigma_R={s_star:.4f} km solved so Pc=5.2e-3")
    a = assess_conjunction(conj_geom(tca, "METEOSTAR-2", "SL16-RB-90211", p_state, s_state, cov_pair, hbr_p, hbr_s))

    cat_a, _ = make_primary(norad=60022, name="METEOSTAR-1", epoch=tca, alt_km=820.0, inc=98.7, raan=302.6, argp=88.0, ma=327.0, intl="24188B", bstar=2.1e-5)
    cat_b, _ = make_primary(norad=43010, name="SENTINEL-CLASS EO", epoch=tca, alt_km=786.0, inc=98.6, raan=310.2, argp=90.0, ma=200.0, intl="17064A", bstar=1.5e-5)

    objects = [
        ObjectFixture(id="METEOSTAR-2", name="METEOSTAR-2", norad_id=60021, kind="satellite", cospar_id="2024-188A", operator_id="OP-EUROMET", launch_id="L-2024-188", tle=tf(p_tle), hbr_m=hbr_p, mass_kg=1200, isp_s=220, bus_type="MetBus-1200 (hydrazine)", country="EU", launch_date=date(2024, 10, 4), role="primary"),
        ObjectFixture(id="SL16-RB-90211", name="SL-16 R/B (Zenit-2 stage 2)", norad_id=90211, kind="debris", parent_object_id=None, tle=tf(s_tle), hbr_m=hbr_s, rcs_category="large", country="Russia/Ukraine", launch_id="L-1992-093", role="secondary"),
        ObjectFixture(id="METEOSTAR-1", name="METEOSTAR-1", norad_id=60022, kind="satellite", cospar_id="2024-188B", operator_id="OP-EUROMET", launch_id="L-2024-188", tle=tf(cat_a), hbr_m=hbr_p, mass_kg=1200, bus_type="MetBus-1200", country="EU", launch_date=date(2024, 10, 4), role="catalog"),
        ObjectFixture(id="SENTINEL-CLASS-EO", name="SENTINEL-CLASS EO", norad_id=43010, kind="satellite", cospar_id="2017-064A", operator_id="OP-EUROMET", launch_id=None, tle=tf(cat_b), hbr_m=6.0, mass_kg=1100, bus_type="Astrobus-L", country="EU", launch_date=date(2017, 10, 13), role="catalog"),
    ]
    graph = GraphFixture(
        operators=[Operator(id="OP-EUROMET", name="EuroMet Agency", country="EU", designation="civil_government", contact_protocol="FDO desk + STC", stc_responsive=True),
                   Operator(id="OP-UKR-LEGACY", name="Yuzhnoye / legacy Zenit operator", country="Ukraine", designation="unknown", contact_protocol="none (spent stage)", stc_responsive=False)],
        launch_events=[LaunchEvent(id="L-2024-188", name="MeteoStar dual launch", launch_date=date(2024, 10, 4), site="Kourou ELA-4", booster="Ariane 6", cospar_launch_id="2024-188", payload_ids=["METEOSTAR-1", "METEOSTAR-2"], declared_payload_count=2),
                       LaunchEvent(id="L-1992-093", name="Tselina-2 launch", launch_date=date(1992, 12, 25), site="Baikonur 45/1", booster="Zenit-2", cospar_launch_id="1992-093", payload_ids=["SL16-RB-90211"], declared_payload_count=1)],
        ground_stations=[GroundStation(id="GS-TIRA", name="TIRA radar (Wachtberg)", latitude_deg=50.62, longitude_deg=7.13, altitude_m=250, sensor_type="radar", min_elevation_deg=5.0),
                         GroundStation(id="GS-EUROMET-FDO", name="EuroMet Flight Dynamics Centre", latitude_deg=49.87, longitude_deg=8.65, sensor_type="telemetry", operator_id="OP-EUROMET")],
        edges=[
            EdgeFixture(type="ORBITAL_NEIGHBOR", source="METEOSTAR-2", target="METEOSTAR-1", attrs={"shell_km": 0.2}),
            EdgeFixture(type="CO_LAUNCHED_WITH", source="METEOSTAR-2", target="METEOSTAR-1"),
            EdgeFixture(type="IN_CONJUNCTION_WITH", source="METEOSTAR-2", target="SL16-RB-90211", attrs={"tca": tca.isoformat(), "miss_km": round(a.miss_distance_km, 4), "pc": a.pc}),
            EdgeFixture(type="TRACKED_BY", source="SL16-RB-90211", target="GS-TIRA"),
            EdgeFixture(type="TRACKED_BY", source="METEOSTAR-2", target="GS-EUROMET-FDO"),
        ],
    )
    return ScenarioFixture(
        id="S4",
        title="Imminent Debris Conjunction -> Human-Approved CAM Burn",
        trigger_type="CDM",
        summary="A 46 m miss with a spent Zenit stage under tight covariance (Pc 5.2e-3). No sensing can lower the risk; the agent must plan a minimum-dv burn, screen it, and route it to the FDO.",
        objects=objects,
        graph=graph,
        conjunction=ConjunctionFixture(primary_id="METEOSTAR-2", secondary_id="SL16-RB-90211", tca=tca,
                                       states_pre={"METEOSTAR-2": sf(p_state), "SL16-RB-90211": sf(s_state)},
                                       covariance_pre=cov_pair, cdm_reported_miss_km=round(a.miss_distance_km, 4), cdm_reported_pc=a.pc),
        observations=[
            ObservationFixture(sensor_type="radar", sensor_id="GS-TIRA", epoch=tca - timedelta(hours=14), governance_level="L0", description="4th consecutive TIRA track; covariance already converged, further sensing yields <2% volume reduction.", effect="none", payload={"tracks": 4, "volume_reduction_pct": 1.8}),
        ],
        cam_policy=CamPolicyFixture(lead_time_s=2400.0, target_pc=1e-6, min_standoff_km=1.8, spacecraft_mass_kg=1200.0, isp_s=220.0,
                                    screening_window_s=6000.0, screening_catalog_ids=["METEOSTAR-1", "SENTINEL-CLASS-EO"]),
        expected=ExpectedOutcome(classification="debris_collision_risk_confirmed", severity="CRITICAL", governance_level="L2",
                                 recommended_action="EXECUTE_CAM_AFTER_FDO_SIGNOFF",
                                 key_numbers={"pc": a.pc, "miss_km": a.miss_distance_km, "dv_nominal_mps": 0.35, "lead_time_min": 40}),
    )


def build() -> FixtureFile:
    return FixtureFile(version="1.0.0", generated_by="data/build_fixtures.py", generated_at=datetime(2026, 9, 20, 0, 0, 0, tzinfo=UTC),
                       scenarios=[scenario_1(), scenario_2(), scenario_3(), scenario_4()])


def main(out: Path = Path(__file__).with_name("fixtures.json")) -> Path:
    ff = build()
    out.write_text(json.dumps(ff.model_dump(mode="json"), indent=1), encoding="utf-8")
    for sc in ff.scenarios:
        print(f"{sc.id}: {sc.title}  ->  {sc.expected.key_numbers}")
    print(f"wrote {out} ({out.stat().st_size/1024:.1f} KiB)")
    return out


if __name__ == "__main__":
    main()
