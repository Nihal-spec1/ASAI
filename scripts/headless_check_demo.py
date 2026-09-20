import os, time
os.environ["ASAI_OFFLINE"] = "1"
from streamlit.testing.v1 import AppTest

def show_exceptions(at, label):
    if at.exception:
        print(f"!! {label}: {len(at.exception)} exception(s)")
        for e in at.exception:
            print("   ", e.value)
            print("   ", e.stack_trace[-1200:] if hasattr(e, "stack_trace") else "")
        return True
    return False

t0 = time.time()
at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "demo.py"), default_timeout=120)
at.run()
assert not show_exceptions(at, "initial render"), "initial render failed"
print(f"initial render OK ({time.time()-t0:.1f}s), markdown blocks={len(at.markdown)}, buttons={[b.label for b in at.button]}")

for sid in ["S1", "S2", "S3", "S4"]:
    at.session_state["scenario_pills"] = sid
    at.run()
    assert not show_exceptions(at, f"{sid} pre-run"), sid
    at.button(key="run_btn").click().run()
    assert not show_exceptions(at, f"{sid} run"), sid
    st_ = at.session_state["cases"][sid]["state"]
    fa = st_.final_action
    print(f"{sid}: status={st_.status.value} route={st_.approval_route.value} initial={st_.initial_action.action_type.value} "
          f"final={'+'.join(a.value for a in fa.all_actions)} dv={fa.delta_v_mps:.3f} conf={st_.confidence:.2f} narrative={st_.narrative_source} tasks={len(st_.sensor_tasking_requests)}")
    auth = [b for b in at.button if b.key and b.key.startswith("auth_")]
    if st_.status.value == "READY_FOR_HUMAN":
        assert auth, f"{sid}: gate buttons missing"
        auth[0].click().run()
        assert not show_exceptions(at, f"{sid} approve"), sid
        st_ = at.session_state["cases"][sid]["state"]
        print(f"   L2 gate: approved={st_.human_decision.approved} by {st_.human_decision.approver} -> status={st_.status.value} node={st_.case_node_id}")
    else:
        assert not auth, f"{sid}: gate rendered without L2"

# second S1 run should recall the first case from graph memory
at.session_state["scenario_pills"] = "S1"; at.run()
at.button(key="run_btn").click().run()
assert not show_exceptions(at, "S1 rerun")
gf = at.session_state["cases"]["S1"]["state"].findings["graph"]
print("S1 rerun graph flags:", gf.flags, "| similar:", [s["id"] for s in gf.details["similar_cases"]])
print("graph:", at.session_state["graph"].stats())

# reset
at.button(key="reset_btn").click().run()
assert not show_exceptions(at, "reset")
print("after reset cases:", len(at.session_state["cases"]), "graph:", at.session_state["graph"].stats())
print(f"ALL HEADLESS CHECKS PASSED in {time.time()-t0:.1f}s")
