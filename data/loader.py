"""Load ``fixtures.json`` and materialise graph / astrodynamics objects from it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from astrodynamics import TLE, ConjunctionGeometry
from graph import Debris, EdgeType, OrbitalGraph, Satellite

from .schema import FixtureFile, ObjectFixture, ScenarioFixture

FIXTURES_PATH = Path(__file__).with_name("fixtures.json")


def load_fixtures(path: Path = FIXTURES_PATH) -> FixtureFile:
    with open(path, "r", encoding="utf-8") as fh:
        return FixtureFile.model_validate(json.load(fh))


def tle_of(obj: ObjectFixture) -> TLE:
    if obj.tle is None:
        raise ValueError(f"object {obj.id} has no TLE")
    return TLE(name=obj.name, line1=obj.tle.line1, line2=obj.tle.line2)


def _node_for(obj: ObjectFixture):
    common = dict(id=obj.id, name=obj.name, norad_id=obj.norad_id, orbit_regime=obj.orbit_regime, hbr_m=obj.hbr_m)
    tle = dict(tle_line1=obj.tle.line1, tle_line2=obj.tle.line2) if obj.tle else {}
    if obj.kind == "satellite":
        return Satellite(
            cospar_id=obj.cospar_id,
            bus_type=obj.bus_type,
            dry_mass_kg=obj.mass_kg,
            operator_id=obj.operator_id,
            country=obj.country,
            launch_date=obj.launch_date,
            status=obj.status,
            registered=obj.registered,
            **common,
            **tle,
        )
    return Debris(
        parent_object_id=obj.parent_object_id,
        rcs_category=obj.rcs_category,
        fragmentation_event_id=obj.fragmentation_event_id,
        **common,
        **tle,
    )


def build_graph(scenarios: Iterable[ScenarioFixture], graph: Optional[OrbitalGraph] = None) -> OrbitalGraph:
    """Populate an OrbitalGraph from one or more scenarios.

    Implicit edges are derived from object fields (OPERATED_BY, LAUNCHED_IN,
    PARENT_OF); explicit edges come from ``graph.edges``.
    """
    g = graph or OrbitalGraph()
    scenarios = list(scenarios)
    for sc in scenarios:
        for op in sc.graph.operators:
            if not g.has(op.id):
                g.add_node(op)
        for le in sc.graph.launch_events:
            if not g.has(le.id):
                g.add_node(le)
        for gs in sc.graph.ground_stations:
            if not g.has(gs.id):
                g.add_node(gs)
        for obj in sc.objects:
            if not g.has(obj.id):
                g.add_node(_node_for(obj))
    for sc in scenarios:
        for obj in sc.objects:
            if obj.operator_id and g.has(obj.operator_id):
                g.add_edge(EdgeType.OPERATED_BY, obj.id, obj.operator_id)
            if obj.launch_id and g.has(obj.launch_id):
                g.add_edge(EdgeType.LAUNCHED_IN, obj.id, obj.launch_id)
            if obj.parent_object_id and g.has(obj.parent_object_id):
                g.add_edge(EdgeType.PARENT_OF, obj.parent_object_id, obj.id)
        for e in sc.graph.edges:
            g.add_edge(EdgeType(e.type), e.source, e.target, **e.attrs)
    return g


def conjunction_geometry(sc: ScenarioFixture, stage: str = "pre") -> ConjunctionGeometry:
    """Build the ConjunctionGeometry for a scenario at the 'pre' or 'post' sensing stage."""
    if sc.conjunction is None:
        raise ValueError(f"scenario {sc.id} has no conjunction block")
    c = sc.conjunction
    if stage == "pre":
        states, cov = c.states_pre, c.covariance_pre
    elif stage == "post":
        if c.states_post is None or c.covariance_post is None:
            raise ValueError(f"scenario {sc.id} has no post-sensing data")
        states = {**c.states_pre, **c.states_post}
        cov = c.covariance_post
    else:
        raise ValueError("stage must be 'pre' or 'post'")
    p, s = states[c.primary_id], states[c.secondary_id]
    return ConjunctionGeometry(
        tca=c.tca,
        primary_id=c.primary_id,
        secondary_id=c.secondary_id,
        r_primary=p.r,
        v_primary=p.v,
        r_secondary=s.r,
        v_secondary=s.v,
        cov_primary_rtn=cov.primary_rtn,
        cov_secondary_rtn=cov.secondary_rtn,
        hbr_primary_m=sc.object(c.primary_id).hbr_m,
        hbr_secondary_m=sc.object(c.secondary_id).hbr_m,
        frame=c.frame,
    )


def light_curve_arrays(sc: ScenarioFixture) -> Tuple[np.ndarray, np.ndarray]:
    if sc.light_curve is None:
        raise ValueError(f"scenario {sc.id} has no light curve")
    lc = sc.light_curve
    mags = np.asarray(lc.samples_mag, float)
    t = np.arange(len(mags)) / lc.sample_rate_hz
    return t, mags
