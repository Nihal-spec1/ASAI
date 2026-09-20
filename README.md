# ASAI — Autonomous Space Anomaly Investigator

A deterministic, policy-gated agent that investigates orbital anomalies (conjunctions, covert rendezvous, attitude failures), decides when to task sensors instead of guessing, and prepares Flight Director decision packages it is never allowed to execute itself.

Every number the system reports comes from closed-form or numerically integrated astrodynamics. The language model, when present, only narrates.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows   |   source .venv/bin/activate on macOS/Linux
pip install streamlit pydeck plotly networkx sgp4 skyfield scipy numpy pandas pydantic groq python-dotenv pytest

cp .env.example .env              # optional: add GROQ_API_KEY for live narration
streamlit run demo.py             # mission control console
```

Without a Groq key the console runs 100 % offline on the bundled fixtures with a deterministic narrative template.

## What it does

Four verified scenarios ship in `data/fixtures.json`:

| ID | Situation | Outcome |
|----|-----------|---------|
| S1 | 120 m nominal miss under 4.5 km covariance | Radar tasked, covariance collapses ~5,400×, true miss 4.2 km, CAM cancelled (L0) |
| S2 | Unregistered satellite matching planes with a defence comsat | Shadowing confirmed, operator unresponsive, advisories + UN registry report drafted (L2) |
| S3 | Sudden drag anomaly and loss of signal | Optical photometry shows 0.400 Hz tumble, asset reclassified uncontrolled (L1) |
| S4 | Pc 5.2e-3 at 46 m with tight covariance, 3 h to TCA | 0.355 m/s prograde CAM package, post-burn screening clear, awaits human sign-off (L2) |

Each investigation follows a two-stage protocol: a **baseline** recommendation before sensing, an **active sensing** loop when the evidence is not defensible, then a **definitive** recommendation with a `what_changed` diff.

## Governance

| Level | Agent may | Examples |
|-------|-----------|----------|
| L0 | act autonomously | catalog lookups, Pc computation, sensor tasking requests |
| L1 | send advisories | operator and neighbour bulletins, STC coordination |
| L2 | only prepare a package | any thruster burn, UN registry filing, ITU dispute |

A propulsive action routed below L2 is rejected at the schema level (rule R6). The policy engine (`agent/policy_engine.py`) encodes rules R1–R10.

## Layout

```
astrodynamics/   SGP4 propagation, frames, Foster 2D/3D Pc, min-Δv CAM planner, catalog screening
graph/           NetworkX orbital knowledge graph: operators, launches, neighbours, case memory
data/            fixture schema, loader, and the four bundled scenarios
agent/           investigation state, specialist swarm, active tasking, policy engine, supervisor, narrative
demo.py          Streamlit mission control console (3D globe, graph inspector, decision timeline, L2 gate)
scripts/         smoke_stage2.py (CLI trace of all scenarios), headless_check_demo.py (drives the UI headlessly)
tests/           Stage 1 and Stage 2 suites
```

## Verify

```bash
pytest tests -q                              # 78 tests
python scripts/smoke_stage2.py               # full trace of S1–S4 in the terminal
python scripts/headless_check_demo.py        # runs the console through all scenarios without a browser
```

## Stack

Python 3.11 · Pydantic v2 · NumPy / SciPy · sgp4 + Skyfield · NetworkX · Streamlit + Plotly · Groq (`llama-3.3-70b-versatile`, optional)
