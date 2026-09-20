"""Run all four fixture scenarios through the supervisor offline and print the trace."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ASAI_OFFLINE", "1")

from agent import FixtureSensorSimulator, Supervisor, trigger_from_scenario  # noqa: E402
from data.loader import build_graph, load_fixtures  # noqa: E402


def main() -> None:
    fx = load_fixtures()
    graph = build_graph(fx.scenarios)
    for sc in fx.scenarios:
        kwargs = {}
        if sc.id == "S4":
            kwargs["now"] = sc.conjunction.tca - timedelta(hours=3)
        trig = trigger_from_scenario(sc, **kwargs)
        sup = Supervisor(graph, sensor_simulator=FixtureSensorSimulator(sc))
        st = sup.open_case(trig, case_id=f"CASE-{sc.id}")
        sup.run(st)
        print("=" * 100)
        print(f"{sc.id}: {sc.title}")
        for ev in st.log:
            print(f"  [{ev.phase.value:<20}] {ev.message}")
        print(f"  classification={st.classification} severity={st.severity} conf={st.confidence:.3f}")
        print(f"  status={st.status.value} route={st.approval_route.value if st.approval_route else None}")
        if st.initial_action:
            print(f"  initial: {st.initial_action.action_type.value} dv={st.initial_action.delta_v_mps} dir={st.initial_action.burn_direction}")
        if st.final_action:
            a = st.final_action
            print(f"  final:   {'+'.join(x.value for x in a.all_actions)} dv={a.delta_v_mps:.4f} dir={a.burn_direction} prop={a.propellant_kg} pc={a.predicted_pc} miss={a.predicted_miss_km}")
        print("  what_changed:\n    " + st.what_changed.replace("\n", "\n    "))
        print("  citations: " + " | ".join(st.policy_citations))
        print(f"  narrative[{st.narrative_source}]: {st.narrative[:600]}")
    print("=" * 100)
    print(graph.stats())
    for c in graph.nodes_of_kind(graph.get('CASE-S1').kind):
        print(" case", c.id, c.outcome, "investigated:", [e[0] for e in graph.edges_of(c.id, direction="in")], "similar:", [e[1] for e in graph.edges_of(c.id, direction="out")])


if __name__ == "__main__":
    main()
