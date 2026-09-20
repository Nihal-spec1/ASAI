# AGENTS.md — Operational Rules & Strict Invariants for ASAI

This document defines the non-negotiable operational rules, negative constraints, and architectural invariants for the Autonomous Space Anomaly Investigator (ASAI). Any AI agent, developer, or automated pipeline modifying this repository MUST strictly abide by these rules.

---

## 1. The Core Negative Constraints (What You Must NEVER Do)

### RULE 1: Never Fabricate or Approximate Orbital Mechanics
- **FORBIDDEN:** Never ask an LLM (Grok or Claude) to calculate miss distances, delta-v (Δv), or probability of collision ($P_c$) in prompt text.
- **MANDATORY:** All orbital mechanics MUST be computed deterministically using SGP4 propagation (`astrodynamics/propagation.py`) and the Foster 2D/3D collision probability algorithm (`astrodynamics/foster_pc.py`). The LLM is strictly reserved for hypothesis generation, relational graph interpretation, and explainability.

### RULE 2: Never Execute Autonomous Propulsive Maneuvers (CAM)
- **FORBIDDEN:** The agent is NEVER permitted to autonomously execute thruster burns or commit spacecraft propellant.
- **MANDATORY:** Collision Avoidance Maneuver (CAM) burn plans strictly require **L2 (Human Flight Director) Authorization**. The agent may only prepare the defensible decision package and await human sign-off.

### RULE 3: Never Over-Maneuver on High Covariance (Innocence-Testing First)
- **FORBIDDEN:** Never recommend a thruster burn when the orbital covariance ellipsoid is large ($> 2\,\text{km}$) and nominal miss distance is safe ($> 1.5\,\text{km}$).
- **MANDATORY:** When covariance is high, the agent must rule **benign / passive monitoring** and trigger active sensing (tasking radar or optical tracking) to collapse uncertainty before ever considering a maneuver.

### RULE 4: Zero-Burn Invariant for Cleared Cases
- **MANDATORY:** If a conjunction alert is cleared ($P_c < 10^{-4}$ or miss distance $> 5\,\text{km}$), the output schema must strictly enforce `delta_v = 0.0`, `burn_direction = "NONE"`, and `burn_epoch = null`. Residual burn parameters on cleared cases constitute an immediate invariant failure.

### RULE 5: Zero-Key Offline Fallback Mode
- **MANDATORY:** The system MUST function 100% offline using bundled fixture data in `data/fixtures.json`. If `XAI_API_KEY` is missing or the internet drops out, the system must gracefully fall back to deterministic local rule heuristics and cached summaries without crashing.

---

## 2. Runtime Model Integration (Grok / xAI)

- **Provider:** xAI Grok API (`https://api.x.ai/v1`) using the standard `openai` Python SDK.
- **Model String:** `grok-2-latest` (or `grok-beta`).
- **Role of Grok in ASAI:**
  1. Synthesizing evidence across the Knowledge Graph (linking operators, launch manifests, and orbital neighbors).
  2. Generating natural language flight safety advisories and UN/ITU regulatory filings.
  3. Formulating the `what_changed` explainability audit trail when sensor observations arrive.

---

## 3. Approval Routing Hierarchy

Every recommended action MUST be mapped to one of three deterministic approval levels:
- **`L0_AUTO` (Autonomous):** Passive sensor queries, orbital catalog lookups, internal case timeline logging.
- **`L1_ADVISORY` (Operator Notice):** Situational bulletins to neighboring satellites, informal operator coordination pings.
- **`L2_HUMAN_MANDATORY` (Flight Director Authorization):** Thruster burns (CAM), diplomatic ITU interference complaints, and emergency orbital status declarations.

---

## 4. Two-Stage Decision Protocol

Every investigation case must capture two distinct decision snapshots:
1. **`initial_action`:** Baseline recommendation formulated before active sensor tasking.
2. **`final_action`:** Definitive recommendation formulated after sensor observations collapse the covariance ellipsoid.
3. **`what_changed`:** A clear, mathematically justified explanation detailing how the new observations changed the threat assessment.