"""ASAI Mission Control Console - Stage 3 (Streamlit).

A commercial-grade space mission control console over the Stage 1-2 stack:

    data.loader          -> fixtures + Orbital Knowledge Graph (NetworkX)
    agent.supervisor     -> deterministic state machine, specialist swarm, policy gate
    agent.sensor_sim     -> fixture-backed radar / optical / STC sensors
    astrodynamics        -> SGP4 orbit tracks, covariance -> ECI, encounter plane

Every number rendered here is read from the investigation state; nothing is
recomputed by hand and nothing comes from a language model. The console runs
100 % offline: with no usable GROQ_API_KEY the narrative layer falls back to
the deterministic template (agent.narrative).

Launch:  streamlit run demo.py
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv()

from agent import (  # noqa: E402
    ActionType,
    ApprovalRoute,
    CaseStatus,
    FixtureSensorSimulator,
    OrbitalInvestigationState,
    Phase,
    RULES,
    SensorTaskType,
    Supervisor,
    TaskStatus,
    key_is_usable,
    trigger_from_scenario,
)
from agent.specialists.photometric_specialist import dominant_peak, periodogram  # noqa: E402
from astrodynamics import R_EARTH_KM, TLE, propagate_range, propagate_tle, rtn_covariance_to_eci  # noqa: E402
from astrodynamics.foster_pc import project_to_encounter_plane  # noqa: E402
from data.loader import build_graph, load_fixtures  # noqa: E402
from data.schema import ScenarioFixture  # noqa: E402
from graph import CaseNode, Debris, EdgeType, NodeKind, OrbitalGraph, Satellite  # noqa: E402

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

VOID = "#050811"
INDIGO = "#0B1120"
CYAN = "#00E5FF"
VIOLET = "#7C4DFF"
AMBER = "#FFB020"
RED = "#FF3D5A"
EMERALD = "#22E39A"
TEXT = "#E6EEF8"
MUTED = "#8A9BB8"

SEVERITY_TONE = {"CRITICAL": "red", "WARNING": "amber", "ADVISORY": "cyan", "INFO": "grey"}
ROUTE_TONE = {"L0_AUTO": "emerald", "L1_ADVISORY": "cyan", "L2_HUMAN_MANDATORY": "red"}
ROUTE_LABEL = {"L0_AUTO": "L0 - AUTONOMOUS", "L1_ADVISORY": "L1 - OPERATOR ADVISORY", "L2_HUMAN_MANDATORY": "L2 - HUMAN MANDATORY"}
TASK_ICON = {"RADAR_REOBSERVATION": "📡", "OPTICAL_PHOTOMETRY": "🔭", "STC_OPERATOR_QUERY": "📨"}
TASK_LABEL = {"RADAR_REOBSERVATION": "RADAR RE-OBSERVATION", "OPTICAL_PHOTOMETRY": "OPTICAL PHOTOMETRY", "STC_OPERATOR_QUERY": "STC OPERATOR QUERY"}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "S1": {"pill": "[S1] False Alarm Conjunction", "sub": "Radar Covariance Collapse", "hours_before_tca": 36.0},
    "S2": {"pill": "[S2] Covert Inspector Satellite", "sub": "Non-Cooperative RPO", "hours_before_tca": 36.0},
    "S3": {"pill": "[S3] Attitude Failure Triage", "sub": "0.40 Hz Tumbling Photometry", "hours_before_tca": 36.0},
    "S4": {"pill": "[S4] High-Risk Collision", "sub": "Human-in-the-Loop CAM Burn", "hours_before_tca": 3.0},
}
PHASE_ORDER = [p.value for p in Phase]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
:root{--void:#050811;--indigo:#0B1120;--cyan:#00E5FF;--violet:#7C4DFF;--amber:#FFB020;--red:#FF3D5A;--emerald:#22E39A;--text:#E6EEF8;--muted:#8A9BB8;}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{
  background:
    radial-gradient(1200px 720px at 6% -12%, rgba(0,229,255,0.16), transparent 62%),
    radial-gradient(1000px 640px at 96% 8%, rgba(124,77,255,0.22), transparent 62%),
    radial-gradient(800px 520px at 48% 112%, rgba(0,229,255,0.09), transparent 62%),
    linear-gradient(180deg,#050811 0%,#070b17 55%,#050811 100%);
  color:var(--text);
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1840px;}
[data-testid="stSidebar"]{display:none;}
h1,h2,h3{color:var(--text);}
p,li,span,div{color:inherit;}

.glass{
  background:rgba(11,17,32,0.78);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:18px;padding:16px 18px;margin-bottom:14px;
  box-shadow:0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
  position:relative;overflow:hidden;transition:border-color .35s ease, box-shadow .35s ease, transform .35s ease;
}
.glass::before{content:"";position:absolute;inset:0;border-radius:18px;pointer-events:none;
  background:linear-gradient(135deg,rgba(255,255,255,0.07),transparent 38%);}
.glass:hover{border-color:rgba(0,229,255,0.35);
  box-shadow:0 22px 64px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(0,229,255,0.16), 0 0 34px rgba(0,229,255,0.10);}
.glass.red:hover{border-color:rgba(255,61,90,0.45);box-shadow:0 22px 64px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,61,90,0.22), 0 0 34px rgba(255,61,90,0.14);}
.glass.emerald:hover{border-color:rgba(34,227,154,0.45);}
.glass.amber:hover{border-color:rgba(255,176,32,0.45);}
.glass-title{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);
  display:flex;align-items:center;gap:8px;margin-bottom:10px;}
.glass-title .dot{width:6px;height:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan);flex:none;}
.glass-title .dot.red{background:var(--red);box-shadow:0 0 10px var(--red);}
.glass-title .dot.amber{background:var(--amber);box-shadow:0 0 10px var(--amber);}
.glass-title .dot.emerald{background:var(--emerald);box-shadow:0 0 10px var(--emerald);}
.glass-title .dot.violet{background:var(--violet);box-shadow:0 0 10px var(--violet);}
.glass-title .spacer{flex:1;}
.mono{font-family:'JetBrains Mono',monospace;}
.muted{color:var(--muted);}
.small{font-size:12px;}

.pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;border:1px solid;
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.08em;white-space:nowrap;line-height:1.4;}
.pill.cyan{color:var(--cyan);border-color:rgba(0,229,255,.38);background:rgba(0,229,255,.08);}
.pill.violet{color:#B39DFF;border-color:rgba(124,77,255,.45);background:rgba(124,77,255,.12);}
.pill.amber{color:var(--amber);border-color:rgba(255,176,32,.45);background:rgba(255,176,32,.10);}
.pill.red{color:var(--red);border-color:rgba(255,61,90,.5);background:rgba(255,61,90,.12);}
.pill.emerald{color:var(--emerald);border-color:rgba(34,227,154,.45);background:rgba(34,227,154,.10);}
.pill.grey{color:var(--muted);border-color:rgba(138,155,184,.35);background:rgba(138,155,184,.08);}
.pulse{width:8px;height:8px;border-radius:50%;background:currentColor;animation:pulse 1.6s ease-in-out infinite;flex:none;}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor;opacity:1}70%{box-shadow:0 0 0 9px transparent;opacity:.55}100%{box-shadow:0 0 0 0 transparent;opacity:1}}

.hud{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
.hud-title{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:19px;letter-spacing:.14em;
  background:linear-gradient(90deg,#FFFFFF 0%,#00E5FF 55%,#7C4DFF 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hud-sub{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.24em;color:var(--muted);}

.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px 14px;}
.kv .cell{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:8px 10px;}
.kv .k{font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.kv .v{font-size:17px;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text);margin-top:2px;}
.kv .v.cyan{color:var(--cyan);} .kv .v.amber{color:var(--amber);} .kv .v.red{color:var(--red);} .kv .v.emerald{color:var(--emerald);}
.kv .u{font-size:10px;color:var(--muted);margin-left:3px;font-weight:400;}

.terminal{background:rgba(3,6,13,.88);border:1px solid rgba(0,229,255,.18);border-left:3px solid var(--cyan);border-radius:10px;
  padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.75;color:#B8F5FF;white-space:pre-wrap;word-break:break-word;}
.terminal.violet{border-color:rgba(124,77,255,.28);border-left-color:var(--violet);color:#E2D9FF;}
.terminal .prompt{color:var(--emerald);}

.timeline{position:relative;padding-left:28px;}
.timeline::before{content:"";position:absolute;left:8px;top:8px;bottom:8px;width:2px;
  background:linear-gradient(180deg,var(--cyan),var(--violet) 55%,var(--emerald));opacity:.55;}
.tl-node{position:relative;margin-bottom:14px;}
.tl-node::before{content:"";position:absolute;left:-25px;top:16px;width:12px;height:12px;border-radius:50%;background:var(--void);
  border:2px solid var(--cyan);box-shadow:0 0 12px var(--cyan);}
.tl-node.amber::before{border-color:var(--amber);box-shadow:0 0 12px var(--amber);}
.tl-node.emerald::before{border-color:var(--emerald);box-shadow:0 0 12px var(--emerald);}
.tl-node.violet::before{border-color:var(--violet);box-shadow:0 0 12px var(--violet);}
.tl-node.red::before{border-color:var(--red);box-shadow:0 0 12px var(--red);}
.tl-node .glass{margin-bottom:0;}
.tl-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;}
.tl-step{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.2em;color:var(--muted);}
.tl-action{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;letter-spacing:.04em;}
.tl-action.cyan{color:var(--cyan);text-shadow:0 0 18px rgba(0,229,255,.45);}
.tl-action.emerald{color:var(--emerald);text-shadow:0 0 18px rgba(34,227,154,.45);}
.tl-action.amber{color:var(--amber);text-shadow:0 0 18px rgba(255,176,32,.45);}
.tl-action.red{color:var(--red);text-shadow:0 0 18px rgba(255,61,90,.5);}
.rationale{font-size:12.5px;color:#C5D2E6;line-height:1.55;margin-top:8px;}

.radar{position:relative;width:56px;height:56px;border-radius:50%;flex:none;border:1px solid rgba(0,229,255,.35);
  background:radial-gradient(circle,rgba(0,229,255,.16),transparent 70%);overflow:hidden;}
.radar::before{content:"";position:absolute;inset:14px;border-radius:50%;border:1px solid rgba(0,229,255,.25);}
.radar::after{content:"";position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 0deg,rgba(0,229,255,.6),transparent 28%);animation:sweep 2.4s linear infinite;}
.radar.done::after{animation:none;background:conic-gradient(from 0deg,rgba(34,227,154,.35),rgba(34,227,154,.35));}
@keyframes sweep{to{transform:rotate(360deg)}}
.task-row{display:flex;align-items:center;gap:14px;padding:10px 0;border-top:1px dashed rgba(255,255,255,.07);}
.task-row:first-of-type{border-top:none;}
.task-main{flex:1;min-width:0;}
.task-title{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--text);}
.task-meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);margin-top:3px;}
.shrink{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:var(--emerald);text-shadow:0 0 16px rgba(34,227,154,.5);}

.phase-strip{display:flex;gap:6px;flex-wrap:wrap;}
.phase{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:.12em;padding:4px 8px;border-radius:6px;
  border:1px solid rgba(255,255,255,.07);color:var(--muted);background:rgba(255,255,255,.02);}
.phase.done{color:var(--emerald);border-color:rgba(34,227,154,.35);}
.phase.now{color:#03060d;background:var(--cyan);border-color:var(--cyan);box-shadow:0 0 14px rgba(0,229,255,.5);font-weight:700;}

.chain{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.chip{font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.03);}
.chip .lbl{display:block;font-size:9px;letter-spacing:.16em;color:var(--muted);text-transform:uppercase;}
.chip.cyan{border-color:rgba(0,229,255,.35);} .chip.violet{border-color:rgba(124,77,255,.45);} .chip.amber{border-color:rgba(255,176,32,.45);} .chip.red{border-color:rgba(255,61,90,.5);}
.arrow{color:var(--muted);font-size:14px;}

.gate{border:1px solid rgba(255,61,90,.45);box-shadow:0 0 0 1px rgba(255,61,90,.15), 0 24px 70px rgba(0,0,0,.6), 0 0 48px rgba(255,61,90,.12);}
.gate-banner{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.3em;color:var(--red);text-align:center;
  border-top:1px solid rgba(255,61,90,.35);border-bottom:1px solid rgba(255,61,90,.35);padding:6px 0;margin:6px 0 12px 0;
  background:repeating-linear-gradient(90deg,rgba(255,61,90,.08) 0 12px,transparent 12px 24px);}
.pkg{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;}
.pkg .row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed rgba(255,255,255,.08);padding:7px 0;font-family:'JetBrains Mono',monospace;font-size:12px;}
.pkg .row .k{color:var(--muted);letter-spacing:.1em;font-size:10.5px;text-transform:uppercase;}
.pkg .row .v{color:var(--text);font-weight:600;text-align:right;}
.pkg .row .v.red{color:var(--red);} .pkg .row .v.emerald{color:var(--emerald);} .pkg .row .v.cyan{color:var(--cyan);} .pkg .row .v.amber{color:var(--amber);}
.stamp{border:2px solid var(--emerald);color:var(--emerald);border-radius:10px;padding:10px 14px;font-family:'JetBrains Mono',monospace;
  letter-spacing:.2em;font-size:13px;font-weight:700;display:inline-block;transform:rotate(-1.5deg);box-shadow:0 0 24px rgba(34,227,154,.35);}
.stamp.red{border-color:var(--amber);color:var(--amber);box-shadow:0 0 24px rgba(255,176,32,.35);}

.log-line{font-family:'JetBrains Mono',monospace;font-size:11px;color:#B8C7DE;line-height:1.7;}
.log-line .ph{color:var(--violet);}
.log-line .t{color:var(--muted);}

.stButton>button{font-family:'JetBrains Mono',monospace;letter-spacing:.08em;border-radius:12px;border:1px solid rgba(255,255,255,.12);
  background:rgba(11,17,32,.72);color:var(--text);transition:all .25s ease;}
.stButton>button:hover{border-color:rgba(0,229,255,.5);box-shadow:0 0 18px rgba(0,229,255,.25);color:var(--cyan);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,rgba(0,229,255,.95),rgba(124,77,255,.95));color:#03060d;border:none;font-weight:700;
  box-shadow:0 0 26px rgba(0,229,255,.38);}
.stButton>button[kind="primary"]:hover{color:#03060d;box-shadow:0 0 36px rgba(0,229,255,.6);transform:translateY(-1px);}
.st-key-gate_authorize .stButton>button{background:linear-gradient(135deg,rgba(34,227,154,.95),rgba(0,229,255,.9));color:#03060d;border:none;font-weight:700;
  padding:14px 10px;box-shadow:0 0 30px rgba(34,227,154,.45);font-size:13px;}
.st-key-gate_authorize .stButton>button:hover{color:#03060d;box-shadow:0 0 44px rgba(34,227,154,.7);transform:translateY(-1px);}
.st-key-gate_hold .stButton>button{border:1px solid rgba(255,61,90,.55);color:var(--red);padding:14px 10px;font-weight:700;font-size:13px;background:rgba(255,61,90,.07);}
.st-key-gate_hold .stButton>button:hover{box-shadow:0 0 30px rgba(255,61,90,.4);color:var(--red);}

[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button{
  font-family:'JetBrains Mono',monospace !important;font-size:11px !important;letter-spacing:.06em;border-radius:999px !important;
  border:1px solid rgba(255,255,255,.12) !important;background:rgba(11,17,32,.72) !important;color:var(--muted) !important;}
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stPills"] button[kind="pillsActive"]{
  color:#03060d !important;background:linear-gradient(135deg,var(--cyan),#7C4DFF) !important;border-color:transparent !important;
  box-shadow:0 0 20px rgba(0,229,255,.45);font-weight:700;}
[data-testid="stPills"] button p, [data-testid="stPills"] button span{color:inherit !important;}
.stTextInput input{font-family:'JetBrains Mono',monospace;background:rgba(3,6,13,.7);border:1px solid rgba(255,255,255,.1);color:var(--text);}
.stSlider [data-baseweb="slider"] div{color:var(--cyan);}
div[data-testid="stExpander"]{background:rgba(11,17,32,.6);border:1px solid rgba(255,255,255,.07);border-radius:14px;}
div[data-testid="stExpander"] summary{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.12em;color:var(--muted);border-radius:8px;padding:6px 10px;}
.stTabs [aria-selected="true"]{color:var(--cyan) !important;background:rgba(0,229,255,.08);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--cyan);}
.stTabs [data-baseweb="tab-border"]{background:rgba(255,255,255,.06);}
</style>
"""

PLOT_FONT = dict(family="JetBrains Mono, monospace", color=MUTED, size=11)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #


def H(*parts: str) -> str:
    """Join HTML fragments without newlines (Markdown would otherwise reflow them)."""
    return "".join(p for p in parts if p)


def esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pill(text: str, tone: str = "cyan", pulse: bool = False) -> str:
    return H(f'<span class="pill {tone}">', '<span class="pulse"></span>' if pulse else "", esc(text), "</span>")


def glass(title: str, body: str, *, tone: str = "cyan", right: str = "", extra_class: str = "") -> str:
    return H(
        f'<div class="glass {extra_class}">',
        f'<div class="glass-title"><span class="dot {tone}"></span>{esc(title)}<span class="spacer"></span>{right}</div>',
        body,
        "</div>",
    )


def kv(cells: Sequence[Tuple[str, str, str, str]]) -> str:
    """cells: (label, value, unit, tone)"""
    out = ['<div class="kv">']
    for k, v, u, tone in cells:
        out.append(f'<div class="cell"><div class="k">{esc(k)}</div><div class="v {tone}">{esc(v)}<span class="u">{esc(u)}</span></div></div>')
    out.append("</div>")
    return "".join(out)


def terminal(text: str, *, tone: str = "", prompt: str = "asai>") -> str:
    return H(f'<div class="terminal {tone}">', f'<span class="prompt">{esc(prompt)}</span> ', esc(text), "</div>")


def fmt_sci(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.3g}"
    return f"{x:.3f}"


def fmt_km(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_hours(s: Optional[float]) -> str:
    if s is None:
        return "n/a"
    return f"{s / 3600.0:.1f} h"


def utc(dt: Optional[datetime]) -> str:
    return "n/a" if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Session bootstrap and investigation control
# --------------------------------------------------------------------------- #


def boot() -> None:
    ss = st.session_state
    if "fixtures" not in ss:
        ss.fixtures = load_fixtures()
        ss.graph = build_graph(ss.fixtures.scenarios)
        ss.cases = {}
        ss.scenario_pills = "S1"  # widget key is the single source of truth for the active scenario
        ss.ellipsoid_scale = 60
        ss.groq_live = key_is_usable(os.environ.get("GROQ_API_KEY")) and os.environ.get("ASAI_OFFLINE", "0") not in ("1", "true", "True")


def reset_all() -> None:
    ss = st.session_state
    ss.graph = build_graph(ss.fixtures.scenarios)
    ss.cases = {}


def run_investigation(sid: str) -> None:
    ss = st.session_state
    sc: ScenarioFixture = ss.fixtures.scenario(sid)
    meta = SCENARIOS[sid]
    kwargs: Dict[str, Any] = {}
    if sc.conjunction is not None:
        kwargs["now"] = sc.conjunction.tca - timedelta(hours=meta["hours_before_tca"])
    trig = trigger_from_scenario(sc, **kwargs)
    sup = Supervisor(ss.graph, sensor_simulator=FixtureSensorSimulator(sc))
    state = sup.open_case(trig, case_id=f"CASE-{sid}-{uuid.uuid4().hex[:6].upper()}")
    sup.run(state)
    ss.cases[sid] = {"state": state, "sup": sup, "sc": sc, "ran_at": datetime.now(timezone.utc)}


# --------------------------------------------------------------------------- #
# Deterministic geometry helpers for the 3D view
# --------------------------------------------------------------------------- #


def _jd(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    return 2440587.5 + dt.timestamp() / 86400.0


def sun_unit_vector(epoch: datetime) -> np.ndarray:
    """Low-precision solar direction in the equatorial inertial frame (Astronomical Almanac approximation)."""
    n = _jd(epoch) - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])


def gmst_deg(epoch: datetime) -> float:
    d = _jd(epoch) - 2451545.0
    return (280.46061837 + 360.98564736629 * d) % 360.0


@st.cache_data(show_spinner=False)
def orbit_track(line1: str, line2: str, epoch_iso: str, n_points: int = 240) -> Tuple[List[float], List[float], List[float]]:
    """One full SGP4 revolution centred on ``epoch`` (J2000 frame), cached per TLE."""
    tle = TLE(line1=line1, line2=line2)
    epoch = datetime.fromisoformat(epoch_iso)
    period_s = tle.period_minutes * 60.0
    step = period_s / n_points
    states = propagate_range(tle, epoch - timedelta(seconds=period_s / 2), epoch + timedelta(seconds=period_s / 2), step)
    xs = [s.r[0] for s in states]
    ys = [s.r[1] for s in states]
    zs = [s.r[2] for s in states]
    return xs, ys, zs


def ellipsoid_mesh(cov_eci: np.ndarray, center: np.ndarray, scale: float, n: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(cov_eci, float))
    vals = np.clip(vals, 0.0, None)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    ux = np.outer(np.cos(u), np.sin(v))
    uy = np.outer(np.sin(u), np.sin(v))
    uz = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([ux, uy, uz], axis=-1) * np.sqrt(vals) * scale  # (n, n/2, 3) scaled along eigen axes
    pts = pts @ vecs.T + center
    return pts[..., 0], pts[..., 1], pts[..., 2]


def combined_cov_eci(state: OrbitalInvestigationState, cov_p, cov_s) -> np.ndarray:
    p = state.current_state_vectors[state.primary_id]
    s = state.current_state_vectors[state.secondary_id]  # type: ignore[index]
    return rtn_covariance_to_eci(cov_p, p.r, p.v) + rtn_covariance_to_eci(cov_s, s.r, s.v)


def build_globe(sc: ScenarioFixture, state: Optional[OrbitalInvestigationState], scale: float) -> go.Figure:
    epoch = (state.tca if state and state.tca else (state.now if state else datetime.now(timezone.utc)))
    fig = go.Figure()

    # ---- Earth: dark sphere with solar terminator shading
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    X = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    Y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    Z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    sun = sun_unit_vector(epoch)
    shade = (X * sun[0] + Y * sun[1] + Z * sun[2]) / R_EARTH_KM  # cos(zenith)
    shade = np.clip(shade, -0.25, 1.0)
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z, surfacecolor=shade, cmin=-0.25, cmax=1.0, showscale=False,
            colorscale=[[0.0, "#02040a"], [0.2, "#050a16"], [0.45, "#0a1a33"], [0.75, "#123a66"], [1.0, "#1c5a8f"]],
            lighting=dict(ambient=0.55, diffuse=0.6, specular=0.25, roughness=0.6, fresnel=0.35),
            lightposition=dict(x=float(sun[0] * 5e4), y=float(sun[1] * 5e4), z=float(sun[2] * 5e4)),
            hoverinfo="skip", name="Earth",
        )
    )
    # ---- atmospheric limb glow: two translucent shells
    for dr, op, col in ((90.0, 0.10, CYAN), (220.0, 0.045, "#5CC8FF")):
        k = (R_EARTH_KM + dr) / R_EARTH_KM
        fig.add_trace(go.Surface(x=X * k, y=Y * k, z=Z * k, surfacecolor=np.zeros_like(X), showscale=False, opacity=op,
                                 colorscale=[[0, col], [1, col]], hoverinfo="skip", lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)))
    # ---- graticule (rotated by GMST so the prime meridian sits where Earth actually is at the epoch)
    g0 = math.radians(gmst_deg(epoch))
    lat_lines = np.radians(np.arange(-60, 61, 30))
    lon_lines = np.radians(np.arange(0, 360, 30)) + g0
    gx, gy, gz = [], [], []
    for lat in lat_lines:
        lon = np.linspace(0, 2 * np.pi, 90)
        gx += list(R_EARTH_KM * 1.002 * np.cos(lat) * np.cos(lon)) + [None]
        gy += list(R_EARTH_KM * 1.002 * np.cos(lat) * np.sin(lon)) + [None]
        gz += list(R_EARTH_KM * 1.002 * np.sin(lat) * np.ones_like(lon)) + [None]
    for lon in lon_lines:
        lat = np.linspace(-np.pi / 2, np.pi / 2, 60)
        gx += list(R_EARTH_KM * 1.002 * np.cos(lat) * np.cos(lon)) + [None]
        gy += list(R_EARTH_KM * 1.002 * np.cos(lat) * np.sin(lon)) + [None]
        gz += list(R_EARTH_KM * 1.002 * np.sin(lat)) + [None]
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="rgba(0,229,255,0.14)", width=1), hoverinfo="skip", showlegend=False))

    # ---- orbit tracks
    secondary_id = state.secondary_id if state else (sc.conjunction.secondary_id if sc.conjunction else None)
    critical = bool(state and state.severity == "CRITICAL")
    for obj in sc.objects:
        if obj.tle is None:
            continue
        if obj.id == sc.primary.id:
            color, width, name = CYAN, 4, f"{obj.name} (primary)"
        elif obj.id == secondary_id:
            color, width, name = (RED if critical else AMBER), 3, f"{obj.name} (secondary)"
        else:
            color, width, name = "rgba(179,157,255,0.45)", 1.5, f"{obj.name} (neighbour)"
        try:
            xs, ys, zs = orbit_track(obj.tle.line1, obj.tle.line2, epoch.isoformat())
        except Exception:  # SGP4 decay or bad element set: skip the track, never crash the console
            continue
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width), name=name, hoverinfo="name"))
        try:
            here = propagate_tle(TLE(line1=obj.tle.line1, line2=obj.tle.line2), epoch)
            fig.add_trace(go.Scatter3d(x=[here.r[0]], y=[here.r[1]], z=[here.r[2]], mode="markers",
                                       marker=dict(size=4 if width > 2 else 3, color=color, opacity=0.95), name=obj.name, hoverinfo="name", showlegend=False))
        except Exception:
            pass

    # ---- conjunction point and covariance ellipsoids
    if state is not None and state.has_conjunction:
        p = state.current_state_vectors[state.primary_id]
        c = np.asarray(p.r, float)
        fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", marker=dict(size=22, color=CYAN, opacity=0.18), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers+text", marker=dict(size=7, color="#FFFFFF", line=dict(color=CYAN, width=2)),
                                   text=["TCA"], textposition="top center", textfont=dict(color=CYAN, family="JetBrains Mono", size=11), name="conjunction point", hoverinfo="name"))
        conj_fx = sc.conjunction
        pre_cov = combined_cov_eci(state, conj_fx.covariance_pre.primary_rtn, conj_fx.covariance_pre.secondary_rtn) if conj_fx else None
        post_cov = combined_cov_eci(state, state.covariance_matrix[state.primary_id], state.covariance_matrix[state.secondary_id]) if state.observations else None  # type: ignore[index]
        if pre_cov is not None:
            ex, ey, ez = ellipsoid_mesh(pre_cov, c, scale)
            fig.add_trace(go.Surface(x=ex, y=ey, z=ez, surfacecolor=np.zeros_like(ex), showscale=False, opacity=0.28 if post_cov is None else 0.16,
                                     colorscale=[[0, AMBER], [1, AMBER]], name="covariance (pre-sensing)", hoverinfo="name",
                                     lighting=dict(ambient=0.9, diffuse=0.3, specular=0.6, roughness=0.2)))
        if post_cov is not None:
            ex, ey, ez = ellipsoid_mesh(post_cov, c, scale)
            fig.add_trace(go.Surface(x=ex, y=ey, z=ez, surfacecolor=np.zeros_like(ex), showscale=False, opacity=0.55,
                                     colorscale=[[0, EMERALD], [1, EMERALD]], name="covariance (post-radar)", hoverinfo="name",
                                     lighting=dict(ambient=0.9, diffuse=0.3, specular=0.8, roughness=0.15)))
        eye = c / np.linalg.norm(c) * 1.75
        eye = eye + np.array([0.35, 0.25, 0.45])
    else:
        eye = np.array([1.35, 1.15, 0.9])

    axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])), center=dict(x=0, y=0, z=0))),
        margin=dict(l=0, r=0, t=0, b=0), height=560, font=PLOT_FONT,
        legend=dict(orientation="h", y=0.02, x=0.02, bgcolor="rgba(11,17,32,0.6)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1, font=dict(size=10)),
    )
    return fig


def build_encounter_plane(sc: ScenarioFixture, state: OrbitalInvestigationState) -> Optional[go.Figure]:
    """1/2/3-sigma covariance ellipses in the encounter plane, pre vs post sensing."""
    if not state.has_conjunction or sc.conjunction is None:
        return None
    geom = state.geometry()
    miss = geom.miss_vector_eci()
    v_rel = geom.relative_velocity_eci()
    if np.linalg.norm(v_rel) <= 0.1:
        return None  # low-velocity regime: encounter plane is not the right projection
    panels: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
    pre_cov = combined_cov_eci(state, sc.conjunction.covariance_pre.primary_rtn, sc.conjunction.covariance_pre.secondary_rtn)
    if state.observations and sc.conjunction.states_post:
        pre_p = sc.conjunction.states_pre[state.primary_id]
        pre_s = sc.conjunction.states_pre[state.secondary_id]  # type: ignore[index]
        pre_miss = np.asarray(pre_s.r) - np.asarray(pre_p.r)
        pre_vrel = np.asarray(pre_s.v) - np.asarray(pre_p.v)
        m2, c2 = project_to_encounter_plane(pre_miss, pre_vrel, rtn_covariance_to_eci(sc.conjunction.covariance_pre.primary_rtn, pre_p.r, pre_p.v)
                                            + rtn_covariance_to_eci(sc.conjunction.covariance_pre.secondary_rtn, pre_s.r, pre_s.v))
        panels.append(("PRE-SENSING (CDM)", m2, c2, AMBER))
        m2b, c2b = project_to_encounter_plane(miss, v_rel, geom.combined_covariance_eci())
        panels.append(("POST-RADAR", m2b, c2b, EMERALD))
    else:
        m2, c2 = project_to_encounter_plane(miss, v_rel, pre_cov)
        panels.append(("CURRENT BELIEF", m2, c2, AMBER if state.severity != "CRITICAL" else RED))
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels], horizontal_spacing=0.12)
    t = np.linspace(0, 2 * np.pi, 181)
    for i, (title, m2, c2, col) in enumerate(panels, start=1):
        vals, vecs = np.linalg.eigh(c2)
        vals = np.clip(vals, 0, None)
        for k, op in ((1, 0.9), (2, 0.55), (3, 0.3)):
            ring = (vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))) * k
            fig.add_trace(go.Scatter(x=ring[0], y=ring[1], mode="lines", line=dict(color=col, width=1.6), opacity=op, name=f"{k}σ", showlegend=(i == 1), hoverinfo="name"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=7, color=CYAN, symbol="cross"), name="primary", showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=[m2[0]], y=[m2[1]], mode="markers+text", marker=dict(size=10, color="#FFFFFF", line=dict(color=col, width=2)),
                                 text=[f"miss {np.hypot(*m2):.3f} km"], textposition="top right", textfont=dict(color=col, size=10, family="JetBrains Mono"),
                                 name="secondary (nominal)", showlegend=(i == 1)), row=1, col=i)
        lim = float(max(3.2 * math.sqrt(vals.max()), 1.4 * np.hypot(*m2), 0.2))
        fig.update_xaxes(range=[-lim, lim], row=1, col=i, title_text="in-plane [km]", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.12)")
        fig.update_yaxes(range=[-lim, lim], row=1, col=i, title_text="cross [km]", scaleanchor=f"x{i if i > 1 else ''}", scaleratio=1,
                         gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(3,6,13,0.5)", height=430, margin=dict(l=10, r=10, t=40, b=10), font=PLOT_FONT,
                      legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"))
    for ann in fig.layout.annotations:
        ann.font = dict(family="JetBrains Mono", size=11, color=MUTED)
    return fig


# --------------------------------------------------------------------------- #
# Knowledge graph rendering
# --------------------------------------------------------------------------- #

KIND_COLOR = {"Satellite": CYAN, "Debris": AMBER, "Operator": VIOLET, "LaunchEvent": "#5CC8FF", "GroundStation": "#8A9BB8", "Case": EMERALD}
KIND_SYMBOL = {"Satellite": "circle", "Debris": "diamond", "Operator": "square", "LaunchEvent": "triangle-up", "GroundStation": "cross", "Case": "star"}


def build_graph_figure(graph: OrbitalGraph, focus: List[str], height: int = 420) -> go.Figure:
    focus = [f for f in focus if graph.has(f)]
    sub = graph.subgraph_around(focus, hops=1) if focus else graph.g.copy()
    # pull in the cases that involve the focus objects so memory shows up
    for f in focus:
        for _, v, d in graph.g.out_edges(f, data=True):
            if d["type"] == EdgeType.INVESTIGATED_IN and v not in sub:
                sub.add_node(v, **graph.g.nodes[v])
                sub.add_edge(f, v, **d)
    simple = nx.Graph()
    simple.add_nodes_from(sub.nodes(data=True))
    for u, v, d in sub.edges(data=True):
        if simple.has_edge(u, v):
            simple[u][v]["types"].add(d["type"].value)
        else:
            simple.add_edge(u, v, types={d["type"].value})
    pos = nx.spring_layout(simple, seed=7, k=1.35 / math.sqrt(max(len(simple), 1)), iterations=200)
    fig = go.Figure()
    ex, ey, hover_x, hover_y, hover_t = [], [], [], [], []
    for u, v, d in simple.edges(data=True):
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
        hover_x.append((pos[u][0] + pos[v][0]) / 2)
        hover_y.append((pos[u][1] + pos[v][1]) / 2)
        hover_t.append(" / ".join(sorted(d["types"])))
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(0,229,255,0.22)", width=1.2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=6, color="rgba(0,0,0,0)"), text=hover_t, hoverinfo="text", showlegend=False))
    for kind, col in KIND_COLOR.items():
        ids = [n for n, d in simple.nodes(data=True) if d.get("kind") == kind]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids], y=[pos[n][1] for n in ids], mode="markers+text",
            marker=dict(size=[20 if n in focus else 13 for n in ids], color=col, symbol=KIND_SYMBOL[kind], opacity=0.95,
                        line=dict(color="#FFFFFF" if kind != "Case" else EMERALD, width=[2 if n in focus else 0.8 for n in ids])),
            text=[simple.nodes[n].get("name", n) if len(simple.nodes[n].get("name", n)) < 26 else n for n in ids],
            textposition="bottom center", textfont=dict(size=9, color="#C5D2E6", family="JetBrains Mono"), name=kind, hovertext=ids, hoverinfo="text",
        ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height, margin=dict(l=0, r=0, t=6, b=0), font=PLOT_FONT,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=9.5)))
    return fig


def relational_chain(graph: OrbitalGraph, object_id: str) -> str:
    """Operator country -> parent launch vehicle -> co-planar neighbours -> historical anomaly links."""
    if not graph.has(object_id):
        return '<div class="muted small mono">object not catalogued</div>'
    node = graph.get(object_id)
    op = graph.get_operator_of(object_id)
    lineage = graph.trace_parent_launch(object_id)
    cluster = graph.get_co_orbital_cluster(object_id, max_hops=2)
    cases = [v for _, v, d in graph.g.out_edges(object_id, data=True) if d["type"] == EdgeType.INVESTIGATED_IN]
    chips = [
        f'<div class="chip cyan"><span class="lbl">object</span>{esc(node.name)}<br><span class="muted">{esc(object_id)}</span></div>',
        '<span class="arrow">→</span>',
        f'<div class="chip violet"><span class="lbl">operator · country</span>{esc(op.name if op else "no operator of record")}<br><span class="muted">{esc(op.country if op else "—")} · {esc(op.designation if op else "—")}</span></div>',
        '<span class="arrow">→</span>',
    ]
    if lineage.launch_event:
        le = lineage.launch_event
        chips.append(f'<div class="chip"><span class="lbl">parent launch</span>{esc(le.booster)}<br><span class="muted">{esc(le.id)} · {esc(le.launch_date.isoformat())} · {esc(le.site)}</span></div>')
    else:
        chips.append('<div class="chip"><span class="lbl">parent launch</span>unknown</div>')
    if lineage.parent_object_id:
        chips.append(f'<div class="chip amber"><span class="lbl">fragment of</span>{esc(lineage.parent_object_id)}</div>')
    chips.append('<span class="arrow">→</span>')
    neigh = ", ".join(f"{n} (h{h})" for n, h in sorted(cluster.items(), key=lambda kv_: kv_[1])) or "none"
    chips.append(f'<div class="chip"><span class="lbl">co-planar neighbours</span>{esc(neigh)}</div>')
    chips.append('<span class="arrow">→</span>')
    if cases:
        rows = []
        for cid in cases:
            cn: CaseNode = graph.get(cid)  # type: ignore[assignment]
            rows.append(f"{cid} · {cn.classification} · {cn.outcome}")
        chips.append(f'<div class="chip red"><span class="lbl">historical anomaly links</span>{"<br>".join(esc(r) for r in rows)}</div>')
    else:
        chips.append('<div class="chip"><span class="lbl">historical anomaly links</span>none on record</div>')
    return H('<div class="chain">', *chips, "</div>")


# --------------------------------------------------------------------------- #
# Telemetry charts
# --------------------------------------------------------------------------- #


def build_light_curve_figure(lc: Dict[str, Any]) -> go.Figure:
    fs = float(lc["sample_rate_hz"])
    mags = np.asarray(lc["samples_mag"], float)
    t = np.arange(len(mags)) / fs
    freqs, power = periodogram(t, mags)
    f0, p0, snr = dominant_peak(freqs, power)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["OPTICAL LIGHT CURVE", "FFT POWER SPECTRUM"], horizontal_spacing=0.09)
    fig.add_trace(go.Scatter(x=t, y=mags, mode="lines", line=dict(color=CYAN, width=1.2), name="magnitude"), row=1, col=1)
    fig.update_yaxes(autorange="reversed", title_text="mag", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(title_text="time [s]", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    mask = freqs > 0
    fig.add_trace(go.Scatter(x=freqs[mask], y=power[mask] / p0, mode="lines", line=dict(color=VIOLET, width=1.5), fill="tozeroy",
                             fillcolor="rgba(124,77,255,0.18)", name="normalised power"), row=1, col=2)
    fig.add_vline(x=f0, line=dict(color=AMBER, width=1, dash="dot"), row=1, col=2)
    fig.add_annotation(x=f0, y=1.0, text=f"{f0:.3f} Hz · SNR {snr:.0f}", showarrow=True, arrowhead=2, arrowcolor=AMBER, ax=60, ay=-30,
                       font=dict(color=AMBER, family="JetBrains Mono", size=11), row=1, col=2)
    fig.update_xaxes(title_text="frequency [Hz]", range=[0, min(2.5, float(freqs[-1]))], row=1, col=2, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="P / P_peak", row=1, col=2, gridcolor="rgba(255,255,255,0.05)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(3,6,13,0.5)", height=320, margin=dict(l=10, r=10, t=40, b=10), font=PLOT_FONT, showlegend=False)
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(family="JetBrains Mono", size=11, color=MUTED)
    return fig


def build_range_history_figure(rh: List[List[float]]) -> go.Figure:
    days = [r[0] for r in rh]
    rng = [r[1] for r in rh]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=rng, mode="lines+markers", line=dict(color=AMBER, width=2), marker=dict(size=6, color=AMBER), name="range to asset"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(3,6,13,0.5)", height=260, margin=dict(l=10, r=10, t=10, b=10), font=PLOT_FONT,
                      xaxis=dict(title="day", gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(title="range [km]", gridcolor="rgba(255,255,255,0.05)"), showlegend=False)
    return fig


def build_decay_figure(kin: Dict[str, Any]) -> go.Figure:
    hist = kin.get("element_history", [])
    x = [datetime.fromisoformat(h["epoch"]) for h in hist]
    a = [h["a_km"] - R_EARTH_KM for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=a, mode="lines+markers", line=dict(color=CYAN, width=2), marker=dict(size=6), name="mean altitude"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(3,6,13,0.5)", height=240, margin=dict(l=10, r=10, t=10, b=10), font=PLOT_FONT,
                      xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(title="a - R_E [km]", gridcolor="rgba(255,255,255,0.05)"), showlegend=False)
    return fig


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #


def system_pill(state: Optional[OrbitalInvestigationState]) -> str:
    if state is None:
        return pill("● SYSTEM OPERATIONAL", "emerald", pulse=True)
    if state.status == CaseStatus.AWAITING_SENSING:
        return pill("RADAR TASKING ACTIVE", "cyan", pulse=True)
    if state.status == CaseStatus.READY_FOR_HUMAN:
        return pill("L2 GATE OPEN - FLIGHT DIRECTOR ACTION REQUIRED", "red", pulse=True)
    if state.final_action and state.final_action.action_type == ActionType.CLEARED:
        return pill("CLEARED", "emerald", pulse=True)
    if state.severity == "CRITICAL":
        return pill("CRITICAL CONJUNCTION", "red", pulse=True)
    return pill("● SYSTEM OPERATIONAL", "emerald", pulse=True)


def mission_clock() -> None:
    components.html(
        """
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
        <div id="clk" style="font-family:'JetBrains Mono',monospace;color:#8A9BB8;font-size:11px;letter-spacing:.2em;margin-top:2px;"></div>
        <script>
        function tick(){const d=new Date();const s=d.toISOString().replace('T',' ').slice(0,19);
          document.getElementById('clk').innerHTML='MISSION CLOCK <span style="color:#00E5FF">'+s+' UTC</span>';}
        tick(); setInterval(tick,1000);
        </script>
        """,
        height=24,
    )


def render_hud() -> None:
    ss = st.session_state
    case = ss.cases.get(ss.scenario)
    state = case["state"] if case else None
    left, mid, right = st.columns([3.1, 5.2, 2.2], vertical_alignment="center")
    with left:
        llm = pill("GROQ · llama-3.3-70b", "violet") if ss.groq_live else pill("OFFLINE · DETERMINISTIC NARRATIVE", "grey")
        st.markdown(H('<div class="hud"><div><div class="hud-title">AUTONOMOUS SPACE ANOMALY INVESTIGATOR</div>',
                      '<div class="hud-sub">ASAI · MISSION CONTROL CONSOLE · STAGE 3</div></div></div>',
                      '<div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">', system_pill(state), llm, "</div>"), unsafe_allow_html=True)
        mission_clock()
    with mid:
        st.markdown('<div class="hud-sub" style="margin-bottom:4px">SCENARIO SWITCHER</div>', unsafe_allow_html=True)
        choice = st.pills("scenario", options=list(SCENARIOS), format_func=lambda k: f"{SCENARIOS[k]['pill']} · {SCENARIOS[k]['sub']}",
                          selection_mode="single", label_visibility="collapsed", key="scenario_pills")
        ss.scenario = choice or ss.scenario  # deselecting every pill keeps the last scenario
    with right:
        b1, b2 = st.columns(2)
        if b1.button("🚀 RUN INVESTIGATION", type="primary", width="stretch", key="run_btn"):
            with st.spinner(f"Running {ss.scenario} through the supervisor..."):
                run_investigation(ss.scenario)
            st.rerun()
        if b2.button("↺ RESET", width="stretch", key="reset_btn", help="Clears every case and rebuilds graph memory from fixtures"):
            reset_all()
            st.rerun()


# --------------------------------------------------------------------------- #
# Left column
# --------------------------------------------------------------------------- #


def render_left(sc: ScenarioFixture, state: Optional[OrbitalInvestigationState]) -> None:
    ss = st.session_state
    graph: OrbitalGraph = ss.graph

    # ---------------- 3D orbital intelligence
    legend = H(
        f'<span class="pill cyan">PRIMARY ORBIT</span> ',
        f'<span class="pill {"red" if state and state.severity == "CRITICAL" else "amber"}">SECONDARY / DEBRIS</span> ',
        '<span class="pill amber">σ PRE-SENSING</span> ',
        '<span class="pill emerald">σ POST-RADAR</span>',
    )
    st.markdown(glass("3D Orbital Intelligence · SGP4 tracks · J2000 frame", "", right=legend), unsafe_allow_html=True)
    tabs = st.tabs(["◉ ORBITAL GLOBE", "⊕ ENCOUNTER PLANE", "◬ COVARIANCE"])
    with tabs[0]:
        st.plotly_chart(build_globe(sc, state, float(ss.ellipsoid_scale)), width="stretch", config={"displayModeBar": False})
        c1, c2 = st.columns([3, 2])
        with c1:
            ss.ellipsoid_scale = st.slider("uncertainty ellipsoid visual scale (×)", 1, 300, int(ss.ellipsoid_scale), key="scale_slider",
                                           help="Real 1-sigma ellipsoids are a few km across on a 6371 km globe; this factor only scales the rendering, never the numbers.")
        with c2:
            epoch = state.tca if state and state.tca else (state.now if state else None)
            st.markdown(H('<div class="mono small muted" style="margin-top:26px">',
                          f"render epoch {esc(utc(epoch))}<br>terminator from solar ephemeris · graticule rotated by GMST {gmst_deg(epoch):.1f}°" if epoch else "",
                          "</div>"), unsafe_allow_html=True)
    with tabs[1]:
        fig = build_encounter_plane(sc, state) if state else None
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown('<div class="mono small muted">Encounter-plane projection of the combined RTN covariance (Foster/Alfano basis). The white marker is the nominal secondary position; rings are 1σ/2σ/3σ.</div>', unsafe_allow_html=True)
        elif state and state.has_conjunction:
            st.markdown(terminal("low-velocity regime (relative speed < 0.1 km/s): 3-D instantaneous Pc is used; encounter-plane projection is not meaningful here.", tone="violet"), unsafe_allow_html=True)
        else:
            st.markdown(terminal("no conjunction geometry in this case." if state else "run the investigation to populate the encounter plane."), unsafe_allow_html=True)
    with tabs[2]:
        if state and state.has_conjunction:
            conj = state.findings.get("conjunction")
            snap0 = state.aux_evidence.get("_snapshot_initial", {})
            rows = []
            if conj:
                m = conj.metrics
                rows = [
                    ("Pc (current)", fmt_sci(m.get("pc")), "", "red" if m.get("pc", 0) >= 1e-4 else "emerald"),
                    ("miss distance", fmt_km(m.get("miss_km")), "km", "cyan"),
                    ("B-plane σ major", fmt_km(m.get("bplane_sigma_major_km"), 3), "km", "amber" if m.get("bplane_sigma_major_km", 0) > 2 else "emerald"),
                    ("covariance volume", fmt_sci(m.get("covariance_volume_km3")), "km³", ""),
                    ("Mahalanobis", fmt_km(m.get("mahalanobis_distance"), 2), "σ", ""),
                    ("σ / miss", fmt_km(m.get("sigma_to_miss_ratio"), 2), "", ""),
                ]
                if snap0 and "covariance_volume_km3" in snap0 and m.get("covariance_volume_km3"):
                    rows.append(("baseline volume", fmt_sci(snap0["covariance_volume_km3"]), "km³", ""))
            st.markdown(kv(rows), unsafe_allow_html=True)
            for oid in (state.primary_id, state.secondary_id):
                cov = np.asarray(state.covariance_matrix.get(oid, np.zeros((3, 3))), float)
                sig = np.sqrt(np.clip(np.diag(cov), 0, None))
                st.markdown(H(f'<div class="mono small" style="margin-top:8px"><span class="muted">{esc(oid)} · RTN 1σ</span> ',
                              f'R <span style="color:{CYAN}">{sig[0]:.3f}</span> km · T <span style="color:{CYAN}">{sig[1]:.3f}</span> km · N <span style="color:{CYAN}">{sig[2]:.3f}</span> km</div>'), unsafe_allow_html=True)
        else:
            st.markdown(terminal("no conjunction covariance in this case." if state else "run the investigation to populate covariance telemetry."), unsafe_allow_html=True)

    # ---------------- knowledge graph inspector
    focus = [sc.primary.id] + ([state.secondary_id] if state and state.secondary_id else ([sc.conjunction.secondary_id] if sc.conjunction else []))
    stats = graph.stats()
    right = H(pill(f"{len(graph)} NODES · {stats.get('edges', 0)} EDGES", "violet"), " ", pill(f"{stats.get('Case', 0)} CASES IN MEMORY", "emerald" if stats.get("Case", 0) else "grey"))
    body = H(relational_chain(graph, sc.primary.id))
    if len(focus) > 1 and focus[1] != sc.primary.id:
        body += H('<div style="height:10px"></div>', relational_chain(graph, focus[1]))
    st.markdown(glass("Knowledge Graph Inspector · multi-hop relational web", body, tone="violet", right=right), unsafe_allow_html=True)
    st.plotly_chart(build_graph_figure(graph, focus), width="stretch", config={"displayModeBar": False})
    gf = state.findings.get("graph") if state else None
    if gf:
        flags = " ".join(pill(f, "amber" if f in ("NON_RESPONSIVE_OPERATOR", "MILITARY_OPERATOR", "FOREIGN_STATE_OPERATOR", "UNDECLARED_PAYLOAD") else "grey") for f in gf.flags) or pill("NO GRAPH FLAGS", "grey")
        sims = gf.details.get("similar_cases", [])
        sim_html = ""
        if sims:
            sim_html = H('<div class="mono small" style="margin-top:8px"><span class="muted">EPISODIC RECALL ·</span> ',
                         " · ".join(f'<span style="color:{EMERALD}">{esc(s["id"])}</span> ({s["score"]:.2f}: {esc("; ".join(s["reasons"]))})' for s in sims), "</div>")
        st.markdown(H('<div style="margin:-6px 0 14px 0">', flags, sim_html, "</div>"), unsafe_allow_html=True)

    # ---------------- scenario-specific telemetry
    if sc.light_curve is not None:
        lc = state.aux_evidence.get("light_curve") if state else None
        photo = state.findings.get("photometric") if state else None
        if lc:
            right = H(pill(f"{photo.metrics['dominant_frequency_hz']:.3f} Hz", "amber"), " ", pill("TUMBLING" if photo and photo.has("TUMBLING") else "ATTITUDE STABLE", "red" if photo and photo.has("TUMBLING") else "emerald")) if photo else ""
            st.markdown(glass("Photometric Light Curve & FFT Spectrum · optical tasking product", "", tone="amber", right=right), unsafe_allow_html=True)
            st.plotly_chart(build_light_curve_figure(lc), width="stretch", config={"displayModeBar": False})
            if photo:
                st.markdown(kv([
                    ("dominant frequency", f"{photo.metrics['dominant_frequency_hz']:.3f}", "Hz", "amber"),
                    ("spin period", f"{photo.metrics['spin_period_s']:.2f}", "s", ""),
                    ("amplitude", f"{photo.metrics['amplitude_mag']:.2f}", "mag", ""),
                    ("peak SNR", f"{photo.metrics['peak_snr']:.0f}", "", "cyan"),
                    ("2f harmonic ratio", f"{photo.metrics['harmonic_power_ratio']:.2f}", "", ""),
                ]), unsafe_allow_html=True)
        else:
            st.markdown(glass("Photometric Light Curve & FFT Spectrum", terminal("optical photometry not yet unlocked: the light curve is delivered only after the supervisor tasks the optical network (L0)."), tone="amber"), unsafe_allow_html=True)
    if state and "kinematics" in state.aux_evidence:
        kin = state.aux_evidence["kinematics"]
        kf = state.findings.get("kinematics")
        right = " ".join(pill(f, "red" if f in ("UNANNOUNCED_DV", "SHADOWING_PATTERN", "ANOMALOUS_DECAY") else "grey") for f in (kf.flags if kf else [])[:5])
        st.markdown(glass("Kinematic Residuals · element history", "", tone="cyan", right=right), unsafe_allow_html=True)
        cols = st.columns(2 if kin.get("range_history_km") else 1)
        with cols[0]:
            st.plotly_chart(build_decay_figure(kin), width="stretch", config={"displayModeBar": False})
        if kin.get("range_history_km"):
            with cols[1]:
                st.plotly_chart(build_range_history_figure(kin["range_history_km"]), width="stretch", config={"displayModeBar": False})
        if kf:
            cells = []
            for k, label, unit in (("dv_unannounced_mps", "unannounced Δv", "m/s"), ("delta_i_deg", "plane change", "°"), ("range_km", "range to asset", "km"),
                                   ("recent_closing_rate_km_day", "closing rate", "km/d"), ("decay_ratio", "decay ratio", "× nominal"), ("hours_since_last_telemetry", "since last telemetry", "h")):
                if k in kf.metrics:
                    cells.append((label, f"{kf.metrics[k]:.2f}", unit, "amber" if k in ("dv_unannounced_mps", "decay_ratio") else ""))
            if cells:
                st.markdown(kv(cells), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Right column
# --------------------------------------------------------------------------- #


def action_label(a) -> str:
    return " + ".join(x.value for x in a.all_actions)


def action_tone(a) -> str:
    if a.action_type == ActionType.CLEARED:
        return "emerald"
    if a.action_type in (ActionType.EXECUTE_CAM, ActionType.ESCALATE_FDO) or ActionType.UN_REGISTRY_REPORT in a.all_actions:
        return "red"
    if a.action_type == ActionType.HOLD_MONITOR:
        return "amber"
    return "cyan"


def phase_strip(state: OrbitalInvestigationState) -> str:
    cur = PHASE_ORDER.index(state.phase.value)
    chips = []
    for i, p in enumerate(PHASE_ORDER):
        cls = "now" if i == cur else ("done" if i < cur or state.phase == Phase.COMMITTED else "")
        chips.append(f'<span class="phase {cls}">{esc(p)}</span>')
    return H('<div class="phase-strip">', *chips, "</div>")


def render_case_header(sc: ScenarioFixture, state: Optional[OrbitalInvestigationState]) -> None:
    prim = sc.primary
    if state is None:
        body = H(
            kv([("case", "NOT OPENED", "", ""), ("target", prim.name, "", "cyan"), ("orbit regime", prim.orbit_regime, "", ""), ("trigger", sc.trigger_type, "", "")]),
            f'<div class="rationale">{esc(sc.summary)}</div>',
            '<div class="mono small muted" style="margin-top:10px">press RUN INVESTIGATION to open the case and start the supervisor state machine.</div>',
        )
        st.markdown(glass("Active Case", body, tone="grey", right=pill("STANDBY", "grey")), unsafe_allow_html=True)
        return
    sev_tone = SEVERITY_TONE.get(state.severity, "grey")
    right = H(pill(state.severity, sev_tone, pulse=state.severity == "CRITICAL"), " ", pill(state.status.value.replace("_", " "), "red" if state.status == CaseStatus.READY_FOR_HUMAN else ("emerald" if state.status == CaseStatus.RESOLVED else "cyan")))
    cells = [
        ("case uuid", state.case_id, "", "cyan"),
        ("target", prim.name, "", ""),
        ("orbit regime", prim.orbit_regime, "", ""),
        ("secondary", state.secondary_id or "—", "", ""),
        ("classification", state.classification.replace("_", " "), "", sev_tone),
        ("confidence", f"{state.confidence:.2f}", "", "emerald" if state.confidence >= 0.65 else "amber"),
        ("time to TCA", fmt_hours(state.time_to_tca_s), "", ""),
        ("approval route", state.approval_route.short if state.approval_route else "pending", "", ROUTE_TONE.get(state.approval_route.value, "grey") if state.approval_route else "grey"),
    ]
    body = H(kv(cells), '<div style="height:10px"></div>', phase_strip(state),
             f'<div class="mono small muted" style="margin-top:8px">opened {esc(utc(state.opened_epoch))} · sim clock {esc(utc(state.now))}' + (f" · TCA {esc(utc(state.tca))}" if state.tca else "") + "</div>")
    st.markdown(glass("Active Case", body, tone=sev_tone, right=right, extra_class=sev_tone if sev_tone == "red" else ""), unsafe_allow_html=True)


def key_number_cells(a) -> str:
    cells = []
    order = [("pc", "Pc", ""), ("miss_km", "miss", "km"), ("bplane_sigma_major_km", "σ major", "km"), ("covariance_volume_km3", "cov volume", "km³"),
             ("dv_unannounced_mps", "unannounced Δv", "m/s"), ("range_km", "range", "km"), ("dominant_frequency_hz", "tumble f", "Hz"), ("decay_ratio", "decay", "×"),
             ("dv_mag_mps", "CAM Δv", "m/s"), ("predicted_miss_km", "post-burn miss", "km"), ("predicted_pc", "post-burn Pc", ""), ("propellant_kg", "propellant", "kg")]
    for k, label, unit in order:
        if k in a.key_numbers:
            v = a.key_numbers[k]
            cells.append((label, fmt_sci(v) if k in ("pc", "predicted_pc", "covariance_volume_km3") else f"{v:.3f}", unit, "amber" if k == "bplane_sigma_major_km" and v > 2 else ""))
    return kv(cells[:6]) if cells else ""


def render_timeline(sc: ScenarioFixture, state: OrbitalInvestigationState) -> None:
    snap0 = state.aux_evidence.get("_snapshot_initial", {})
    nodes: List[str] = []

    # 1. baseline
    a0 = state.initial_action
    if a0:
        flags = []
        if snap0.get("dilution_region"):
            flags.append(pill("DILUTION FLAG: ACTIVE", "amber"))
        conj = state.findings.get("conjunction")
        if conj and snap0.get("bplane_sigma_major_km", 0) > 2:
            flags.append(pill("HIGH COVARIANCE", "amber"))
        head = H('<div class="tl-head"><span class="tl-step">01 · BASELINE ASSESSMENT · PRE-SENSING</span>', pill(f"confidence {a0.confidence:.2f}", "grey"), *flags, "</div>")
        body = H(head, f'<div class="tl-action {action_tone(a0)}">{esc(action_label(a0))}</div>', key_number_cells(a0), f'<div class="rationale">{esc(a0.rationale)}</div>',
                 f'<div class="mono small muted" style="margin-top:6px">policy: {esc(" · ".join(a0.policy_refs) or "—")}</div>')
        nodes.append(H('<div class="tl-node amber">', glass("Baseline recommendation", body, tone="amber"), "</div>"))

    # 2. active sensing
    reqs = state.sensor_tasking_requests
    if reqs:
        rows = []
        for r in reqs:
            done = r.status != TaskStatus.DISPATCHED
            st_tone = "emerald" if r.status == TaskStatus.FULFILLED else ("amber" if r.status == TaskStatus.UNANSWERED else "cyan")
            rows.append(H(
                '<div class="task-row">', f'<div class="radar {"done" if done else ""}"></div>',
                '<div class="task-main">',
                f'<div class="task-title">{TASK_ICON.get(r.task_type.value, "")} [{esc(TASK_LABEL.get(r.task_type.value, r.task_type.value))} {"DISPATCHED" if not done else r.status.value}] → {esc(r.target_id)}</div>',
                f'<div class="task-meta">via {esc(r.sensor_id)} · latency {fmt_hours(r.latency_s)} · expected gain {r.expected_information_gain:.2f} · {esc(r.governance_level.short)} · {esc(r.rationale)}</div>',
                "</div>", pill(r.status.value, st_tone, pulse=not done), "</div>",
            ))
        shrink_html = ""
        conj = state.findings.get("conjunction")
        if conj and snap0.get("covariance_volume_km3") and conj.metrics.get("covariance_volume_km3"):
            factor = snap0["covariance_volume_km3"] / max(conj.metrics["covariance_volume_km3"], 1e-300)
            if factor > 1.05:
                shrink_html = H('<div style="display:flex;align-items:baseline;gap:10px;margin-top:10px;flex-wrap:wrap">',
                                f'<span class="shrink">→ Covariance volume shrunk {factor:,.0f}×</span>',
                                f'<span class="mono small muted">{fmt_sci(snap0["covariance_volume_km3"])} → {fmt_sci(conj.metrics["covariance_volume_km3"])} km³ · σ {snap0.get("bplane_sigma_major_km", 0):.2f} → {conj.metrics["bplane_sigma_major_km"]:.2f} km</span></div>')
        stc = state.aux_evidence.get("stc_response")
        stc_html = ""
        if stc is not None:
            stc_html = H('<div class="mono small" style="margin-top:8px"><span class="muted">STC CHANNEL ·</span> ',
                         pill("OPERATOR RESPONDED" if stc.get("responded") else "OPERATOR UNANSWERED", "emerald" if stc.get("responded") else "red"),
                         f' <span class="muted">{esc(stc.get("message", ""))}</span></div>')
        head = H('<div class="tl-head"><span class="tl-step">02 · ACTIVE SENSOR TASKING</span>', pill(f"round {state.sensing_rounds} of {state.max_sensing_rounds}", "grey"), "</div>")
        nodes.append(H('<div class="tl-node">', glass("Active sensing loop", H(head, *rows, shrink_html, stc_html), tone="cyan"), "</div>"))
    else:
        msg = "No active sensing required: evidence already defensible at baseline; definitive recommendation formulated immediately."
        if state.deadline_forced:
            msg = "Sensing was needed but the TCA deadline leaves no time for a sensor pass (R3): the decision is deadline-forced."
        head = H('<div class="tl-head"><span class="tl-step">02 · ACTIVE SENSOR TASKING</span>', pill("DEADLINE-FORCED" if state.deadline_forced else "SKIPPED", "red" if state.deadline_forced else "grey"), "</div>")
        nodes.append(H('<div class="tl-node violet">', glass("Active sensing loop", H(head, terminal(msg, tone="violet")), tone="violet"), "</div>"))

    # 3. definitive
    a1 = state.final_action
    if a1:
        route = state.approval_route
        rp = pill(ROUTE_LABEL.get(route.value, route.value), ROUTE_TONE.get(route.value, "grey")) if route else ""
        head = H('<div class="tl-head"><span class="tl-step">03 · DEFINITIVE RECOMMENDATION · POST-SENSING</span>', pill(f"confidence {a1.confidence:.2f}", "grey"), rp, "</div>")
        dv_line = H('<div class="mono" style="margin:4px 0 8px 0;font-size:13px">',
                    f'Δv = <span style="color:{EMERALD if a1.delta_v_mps == 0 else RED}">{a1.delta_v_mps:.3f} m/s</span> · direction {esc(a1.burn_direction)}',
                    f' · predicted miss {a1.predicted_miss_km:.3f} km' if a1.predicted_miss_km is not None else "",
                    f' · predicted Pc {fmt_sci(a1.predicted_pc)}' if a1.predicted_pc is not None else "",
                    "</div>")
        targets = f'<div class="mono small muted" style="margin-top:6px">outbound: {esc(", ".join(a1.outbound_targets))}</div>' if a1.outbound_targets else ""
        body = H(head, f'<div class="tl-action {action_tone(a1)}">{esc(action_label(a1))}</div>', dv_line, key_number_cells(a1), f'<div class="rationale">{esc(a1.rationale)}</div>', targets,
                 f'<div class="mono small muted" style="margin-top:6px">policy: {esc(" · ".join(a1.policy_refs) or "—")}</div>')
        nodes.append(H(f'<div class="tl-node {action_tone(a1)}">', glass("Definitive recommendation", body, tone=action_tone(a1), extra_class=action_tone(a1)), "</div>"))

    # 4. what changed + narrative
    if state.what_changed:
        src = state.narrative_source or "pending"
        head = H('<div class="tl-head"><span class="tl-step">04 · WHAT CHANGED · EXPLAINABILITY AUDIT</span>',
                 pill("NARRATIVE: GROQ llama-3.3-70b" if src == "groq" else "NARRATIVE: OFFLINE TEMPLATE", "violet" if src == "groq" else "grey"), "</div>")
        body = H(head, terminal(state.what_changed, prompt="what_changed>"), '<div style="height:8px"></div>',
                 terminal(state.narrative or "narrative pending", tone="violet", prompt="narrative>") if state.narrative else "")
        nodes.append(H('<div class="tl-node emerald">', glass("Two-stage diff", body, tone="emerald"), "</div>"))

    st.markdown(H('<div class="timeline">', *nodes, "</div>"), unsafe_allow_html=True)


def render_gate(case: Dict[str, Any]) -> None:
    state: OrbitalInvestigationState = case["state"]
    sup: Supervisor = case["sup"]
    a = state.final_action
    if a is None or state.approval_route != ApprovalRoute.L2_HUMAN_MANDATORY:
        return
    man = state.findings.get("maneuver")
    rows: List[Tuple[str, str, str]] = [("recommended action", action_label(a).replace("EXECUTE_CAM", "EXECUTE_CAM_BURN"), "red")]
    if a.action_type == ActionType.EXECUTE_CAM:
        rows += [
            ("maneuver vector", f"{a.delta_v_mps:.3f} m/s {a.burn_direction} burn", "cyan"),
            ("Δv RTN [m/s]", "  ".join(f"{x:+.3f}" for x in (a.dv_rtn_mps or ())), ""),
            ("burn epoch", utc(a.burn_epoch), ""),
            ("propellant expenditure", f"{a.propellant_kg:.3f} kg hydrazine" if a.propellant_kg is not None else "n/a", "amber"),
            ("predicted miss at TCA", f"{a.predicted_miss_km:.3f} km" if a.predicted_miss_km is not None else "n/a", "emerald"),
            ("predicted Pc", fmt_sci(a.predicted_pc), "emerald"),
        ]
        if man:
            n_hits = int(man.metrics.get("screening_objects", 0))
            viol = [h for h in man.details.get("screening", []) if h.get("violates_standoff")]
            rows.append(("catalog clearance", f"POST-BURN SCREENED: {len(viol)} SECONDARY CONJUNCTION{'S' if len(viol) != 1 else ''} · {n_hits} objects · min {man.metrics.get('screening_min_distance_km', float('inf')):.1f} km", "emerald" if man.has("SCREENING_CLEAR") else "red"))
            rows.append(("directions studied", f"{len(man.details.get('alternatives', []))} · lead time TCA-{man.metrics.get('lead_time_s', 0) / 60:.0f} min", ""))
    else:
        for k, label in (("dv_unannounced_mps", "unannounced Δv observed"), ("range_km", "stand-off range"), ("delta_i_deg", "plane change")):
            if k in a.key_numbers:
                rows.append((label, f"{a.key_numbers[k]:.2f} " + {"dv_unannounced_mps": "m/s", "range_km": "km", "delta_i_deg": "°"}[k], "amber"))
        rows.append(("outbound recipients", ", ".join(a.outbound_targets) or "—", "cyan"))
        rows.append(("classification", state.classification.replace("_", " "), "red"))
    rows.append(("authority level", "L2 - HUMAN MANDATORY", "red"))
    rows.append(("policy basis", " · ".join(a.policy_refs), ""))
    pkg = H('<div class="pkg">', *(f'<div class="row"><span class="k">{esc(k)}</span><span class="v {tone}">{esc(v)}</span></div>' for k, v, tone in rows), "</div>")
    head = '<div class="gate-banner">FLIGHT DIRECTOR DECISION PACKAGE · L2 AUTHORIZATION GATE</div>'
    if state.human_decision:
        hd = state.human_decision
        stamp = H(f'<div style="margin-top:14px;display:flex;align-items:center;gap:16px;flex-wrap:wrap"><span class="stamp {"" if hd.approved else "red"}">{"AUTHORIZED" if hd.approved else "HELD / DEFERRED"} · {esc(hd.approver)}</span>',
                  f'<span class="mono small muted">{esc(utc(hd.decided_at))} · {esc(hd.note or "no note")} · case node {esc(state.case_node_id or "")} committed to graph memory · status {esc(state.status.value)}</span></div>')
        st.markdown(glass("L2 Flight Director Authorization Gate", H(head, pkg, stamp), tone="emerald" if hd.approved else "amber", extra_class="gate"), unsafe_allow_html=True)
        return
    st.markdown(glass("L2 Flight Director Authorization Gate", H(head, pkg, '<div class="mono small muted" style="margin-top:10px">The agent has prepared this package and halted. Nothing is executed autonomously (R6). Your decision is recorded, the case is closed and anchored in graph memory via write_case_node().</div>'),
                      tone="red", extra_class="gate red"), unsafe_allow_html=True)
    approver = st.text_input("Flight Director callsign", value="FD-ON-DUTY", key=f"approver_{state.case_id}")
    note = st.text_input("Decision note (optional)", value="", key=f"note_{state.case_id}", placeholder="e.g. burn window confirmed with propulsion")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(key="gate_authorize"):
            label = "✅ AUTHORIZE THRUSTER BURN (L2 SIGN-OFF)" if a.action_type == ActionType.EXECUTE_CAM else "✅ AUTHORIZE RELEASE (L2 SIGN-OFF)"
            if st.button(label, width="stretch", key=f"auth_{state.case_id}"):
                sup.approve(state, approver or "FD-ON-DUTY", approved=True, note=note)
                st.rerun()
    with c2:
        with st.container(key="gate_hold"):
            if st.button("❌ HOLD / DEFER", width="stretch", key=f"hold_{state.case_id}"):
                sup.approve(state, approver or "FD-ON-DUTY", approved=False, note=note or "held by Flight Director")
                st.rerun()


def render_policy_and_log(state: OrbitalInvestigationState) -> None:
    with st.expander("policy citations & governance trace", expanded=False):
        cites = state.policy_citations
        if cites:
            st.markdown(H(*(f'<div class="log-line"><span class="ph">§</span> {esc(c)}</div>' for c in cites)), unsafe_allow_html=True)
        else:
            st.markdown('<div class="log-line muted">routing pending</div>', unsafe_allow_html=True)
        st.markdown('<div class="mono small muted" style="margin-top:8px">rule book</div>', unsafe_allow_html=True)
        st.markdown(H(*(f'<div class="log-line"><span class="t">{esc(r.id)}</span> · {esc(r.title)}</div>' for r in RULES.values())), unsafe_allow_html=True)
    with st.expander("supervisor event log", expanded=False):
        st.markdown(H(*(f'<div class="log-line"><span class="t">{esc(ev.at.strftime("%m-%d %H:%M:%S"))}</span> <span class="ph">[{esc(ev.phase.value)}]</span> {esc(ev.message)}</div>' for ev in state.log)), unsafe_allow_html=True)
    with st.expander("specialist findings", expanded=False):
        for name, f in state.findings.items():
            flags = " ".join(pill(x, "grey") for x in f.flags[:8])
            st.markdown(H(f'<div class="log-line"><span class="ph">{esc(name)}</span> <span class="t">conf {f.confidence:.2f}</span> · {esc(f.summary)}</div>', f'<div style="margin:4px 0 10px 0">{flags}</div>'), unsafe_allow_html=True)


def render_memory() -> None:
    graph: OrbitalGraph = st.session_state.graph
    cases = graph.nodes_of_kind(NodeKind.CASE)
    if not cases:
        return
    rows = []
    for c in sorted(cases, key=lambda c: c.opened_at):
        sims = [(v, d.get("score", 0.0)) for _, v, d in graph.g.out_edges(c.id, data=True) if d["type"] == EdgeType.SIMILAR_TO]
        rows.append(H(f'<div class="log-line"><span class="ph">★ {esc(c.id)}</span> · {esc(c.classification)} · <span style="color:{EMERALD}">{esc(c.outcome)}</span>',
                      f' · similar → {esc(", ".join(f"{v} ({s:.2f})" for v, s in sims))}' if sims else "", "</div>"))
    st.markdown(glass("Graph memory · anchored cases", H(*rows), tone="emerald", right=pill(f"{len(cases)} CASE NODE{'S' if len(cases) != 1 else ''}", "emerald")), unsafe_allow_html=True)


def render_right(sc: ScenarioFixture, case: Optional[Dict[str, Any]]) -> None:
    state = case["state"] if case else None
    render_case_header(sc, state)
    if state is None:
        st.markdown(glass("Two-Stage Decision Timeline", terminal("awaiting run. The supervisor will emit a baseline recommendation, decide whether to task sensors, fold observations, then emit the definitive recommendation and a what_changed diff."), tone="grey"), unsafe_allow_html=True)
        render_memory()
        return
    if state.status == CaseStatus.READY_FOR_HUMAN or state.human_decision:
        render_gate(case)  # type: ignore[arg-type]
    render_timeline(sc, state)
    render_policy_and_log(state)
    render_memory()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="ASAI · Mission Control", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    boot()
    ss = st.session_state
    ss.scenario = ss.get("scenario_pills") or ss.get("scenario") or "S1"
    render_hud()
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    sc: ScenarioFixture = ss.fixtures.scenario(ss.scenario)
    case = ss.cases.get(ss.scenario)
    left, right = st.columns([55, 45], gap="medium")
    with left:
        render_left(sc, case["state"] if case else None)
    with right:
        render_right(sc, case)
    st.markdown(H('<div class="mono small muted" style="text-align:center;margin-top:18px;letter-spacing:.14em">',
                  "ASAI · deterministic astrodynamics (SGP4 · Foster Pc · min-Δv planner) · NetworkX orbital knowledge graph · L0/L1/L2 policy gate · ",
                  "GROQ narrative online" if ss.groq_live else "offline narrative", "</div>"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
