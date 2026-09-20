"""Graph specialist: operator fleet, co-orbital cluster, parent-launch lineage, case recall."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graph import CaseNode, Debris, EdgeType, LaunchEvent, NodeKind, Operator, OrbitalGraph, Satellite

from ..state import Finding, OrbitalInvestigationState
from .base import clamp01


class GraphSpecialist:
    name = "graph"

    def applicable(self, state: OrbitalInvestigationState) -> bool:
        return True

    def run(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph]) -> Finding:
        if graph is None:
            return Finding(specialist=self.name, produced_at=state.now, summary="no graph attached", confidence=0.0)

        details: Dict[str, Any] = {}
        flags: List[str] = []
        metrics: Dict[str, float] = {}
        parts: List[str] = []
        evidence: List[str] = []

        subjects = [state.primary_id] + ([state.secondary_id] if state.secondary_id else [])
        primary_country = self._country(graph, state.primary_id)
        for role, oid in zip(("primary", "secondary"), subjects):
            if not graph.has(oid):
                details[role] = {"id": oid, "known": False}
                flags.append(f"{role.upper()}_UNCATALOGUED")
                continue
            node = graph.get(oid)
            info: Dict[str, Any] = {"id": oid, "known": True, "kind": node.kind.value, "name": node.name}
            op = graph.get_operator_of(oid)
            if op is not None:
                fleet = graph.get_operator_fleet(op.id)
                info["operator"] = {
                    "id": op.id,
                    "name": op.name,
                    "country": op.country,
                    "designation": op.designation,
                    "stc_responsive": op.stc_responsive,
                    "fleet_size": len(fleet),
                    "fleet_ids": [f.id for f in fleet],
                }
                metrics[f"{role}_fleet_size"] = float(len(fleet))
                evidence.append(f"graph:OPERATED_BY:{oid}->{op.id}")
                if role == "secondary":
                    if not op.stc_responsive:
                        flags.append("NON_RESPONSIVE_OPERATOR")
                        parts.append(f"{oid} operator {op.name} is not STC-responsive")
                    if op.designation == "military":
                        flags.append("MILITARY_OPERATOR")
                    if primary_country and op.country and op.country != primary_country:
                        flags.append("FOREIGN_STATE_OPERATOR")
            else:
                info["operator"] = None
                if role == "secondary":
                    flags.append("NO_OPERATOR_OF_RECORD")

            cluster = graph.get_co_orbital_cluster(oid, max_hops=2)
            info["co_orbital_cluster"] = cluster
            metrics[f"{role}_cluster_size"] = float(len(cluster))
            info["neighbours"] = [n for n, h in cluster.items() if h == 1]
            # advisory audience = every operator across the 2-hop co-orbital cluster
            for n in cluster:
                nop = graph.get_operator_of(n)
                if nop is not None:
                    info.setdefault("neighbour_operators", [])
                    if nop.id not in info["neighbour_operators"]:
                        info["neighbour_operators"].append(nop.id)

            lineage = graph.trace_parent_launch(oid)
            info["lineage"] = {
                "parent_object_id": lineage.parent_object_id,
                "launch_id": lineage.launch_event.id if lineage.launch_event else None,
                "launch_date": lineage.launch_event.launch_date.isoformat() if lineage.launch_event else None,
                "co_manifested_ids": lineage.co_manifested_ids,
                "hops": lineage.hops,
            }
            if lineage.launch_event:
                evidence.append(f"graph:LAUNCHED_IN:{lineage.hops[-2] if len(lineage.hops) >= 2 else oid}->{lineage.launch_event.id}")
                le: LaunchEvent = lineage.launch_event
                tracked = len(set(le.payload_ids) | set(lineage.co_manifested_ids) | {lineage.hops[-2] if len(lineage.hops) >= 2 else oid})
                if le.declared_payload_count is not None and tracked > le.declared_payload_count:
                    info["lineage"]["declared_payload_count"] = le.declared_payload_count
                    info["lineage"]["tracked_payload_count"] = tracked
                    if role == "secondary":
                        flags.append("UNDECLARED_PAYLOAD")
                        parts.append(
                            f"launch {le.id} declared {le.declared_payload_count} payload(s) but {tracked} tracked objects are attributed to it"
                        )
            if isinstance(node, Debris):
                if role == "secondary":
                    flags.append("SECONDARY_IS_DEBRIS")
                if lineage.parent_object_id:
                    parts.append(f"{oid} is a fragment of {lineage.parent_object_id}")
            if isinstance(node, Satellite) and not node.registered:
                flags.append(f"{role.upper()}_UNREGISTERED")
            details[role] = info

        # conjunction edge context
        if state.secondary_id and graph.has(state.primary_id):
            for other, attrs in graph.get_conjunctions(state.primary_id):
                if other == state.secondary_id:
                    details["conjunction_edge"] = attrs

        # episodic recall (zero-shot: probe need not be stored)
        probe = CaseNode(
            id=f"probe-{state.case_id}",
            name="probe",
            opened_at=state.opened_epoch,
            classification=state.classification,
            involved_object_ids=subjects,
            operator_ids=[d["operator"]["id"] for d in details.values() if isinstance(d, dict) and d.get("operator")],
        )
        similar = graph.find_similar_cases(probe, top_k=3)
        details["similar_cases"] = [{"id": s.case.id, "score": s.score, "reasons": s.reasons, "outcome": s.case.outcome} for s in similar]
        metrics["similar_case_count"] = float(len(similar))
        if similar:
            flags.append("PRIOR_CASES_FOUND")
            parts.append(f"{len(similar)} similar prior case(s) recalled")

        known = sum(1 for k in ("primary", "secondary") if isinstance(details.get(k), dict) and details[k].get("known"))
        confidence = clamp01(0.4 + 0.3 * known)
        if not parts:
            parts.append("neighbourhood resolved without anomalies")
        return Finding(
            specialist=self.name,
            produced_at=state.now,
            summary="; ".join(parts),
            confidence=confidence,
            metrics=metrics,
            flags=flags,
            details=details,
            evidence_refs=evidence,
        )

    @staticmethod
    def _country(graph: OrbitalGraph, oid: str) -> str:
        if not graph.has(oid):
            return ""
        node = graph.get(oid)
        if isinstance(node, Satellite) and node.country:
            return node.country
        op = graph.get_operator_of(oid)
        return op.country if op else ""
