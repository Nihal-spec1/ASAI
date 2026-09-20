"""The Orbital Knowledge Graph: a typed wrapper over ``networkx.MultiDiGraph``.

A *multi* digraph (a ``DiGraph`` subclass) is required because two objects are
routinely related in several ways at once (CO_LAUNCHED_WITH and
ORBITAL_NEIGHBOR and IN_CONJUNCTION_WITH); a plain DiGraph keeps only one
edge per node pair and would silently overwrite relations.

Symmetric relations (co-launch, orbital neighbour, conjunction, case
similarity) are stored as a pair of directed edges so that every traversal
can use out-edges uniformly. Every node carries its pydantic model under
the ``model`` attribute and its ``kind`` under ``kind``; every edge carries
``type`` (an :class:`EdgeType`) plus free-form attributes.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .models import (
    CaseNode,
    Debris,
    EdgeType,
    GroundStation,
    LaunchEvent,
    LaunchLineage,
    NodeKind,
    NodeModel,
    Operator,
    Satellite,
    SimilarCase,
)


class OrbitalGraph:
    def __init__(self) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------ nodes
    def add_node(self, model: NodeModel) -> str:
        self.g.add_node(model.id, kind=model.kind.value, model=model, name=model.name)
        return model.id

    def add_nodes(self, models: Iterable[NodeModel]) -> None:
        for m in models:
            self.add_node(m)

    def has(self, node_id: str) -> bool:
        return node_id in self.g

    def get(self, node_id: str) -> NodeModel:
        if node_id not in self.g:
            raise KeyError(f"unknown node {node_id!r}")
        return self.g.nodes[node_id]["model"]

    def kind_of(self, node_id: str) -> NodeKind:
        return NodeKind(self.g.nodes[node_id]["kind"])

    def nodes_of_kind(self, kind: NodeKind) -> List[NodeModel]:
        return [d["model"] for _, d in self.g.nodes(data=True) if d["kind"] == kind.value]

    # ------------------------------------------------------------------ edges
    def add_edge(self, edge_type: EdgeType, source: str, target: str, **attrs) -> None:
        for nid in (source, target):
            if nid not in self.g:
                raise KeyError(f"cannot add edge to unknown node {nid!r}")
        # one edge per (source, target, type): re-adding the same relation updates its attributes
        self.g.add_edge(source, target, key=edge_type.value, type=edge_type, **attrs)
        if edge_type.symmetric:
            self.g.add_edge(target, source, key=edge_type.value, type=edge_type, **attrs)

    def edges_of(self, node_id: str, edge_type: Optional[EdgeType] = None, direction: str = "out") -> List[Tuple[str, str, dict]]:
        out: List[Tuple[str, str, dict]] = []
        if direction in ("out", "both"):
            out += [(u, v, d) for u, v, d in self.g.out_edges(node_id, data=True) if edge_type is None or d["type"] == edge_type]
        if direction in ("in", "both"):
            out += [(u, v, d) for u, v, d in self.g.in_edges(node_id, data=True) if edge_type is None or d["type"] == edge_type]
        return out

    def neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None, direction: str = "out") -> List[str]:
        seen: List[str] = []
        for u, v, _ in self.edges_of(node_id, edge_type, direction):
            other = v if u == node_id else u
            if other not in seen:
                seen.append(other)
        return seen

    # --------------------------------------------------------- multi-hop API
    def get_operator_of(self, object_id: str) -> Optional[Operator]:
        ops = self.neighbors(object_id, EdgeType.OPERATED_BY, "out")
        return self.get(ops[0]) if ops else None  # type: ignore[return-value]

    def get_operator_fleet(self, operator_id: str) -> List[NodeModel]:
        """All objects with an OPERATED_BY edge into ``operator_id``."""
        if self.kind_of(operator_id) != NodeKind.OPERATOR:
            raise ValueError(f"{operator_id!r} is not an Operator node")
        return [self.get(n) for n in self.neighbors(operator_id, EdgeType.OPERATED_BY, "in")]

    def get_co_orbital_cluster(
        self,
        object_id: str,
        *,
        max_hops: int = 2,
        max_shell_km: Optional[float] = None,
        include_conjunctions: bool = False,
    ) -> Dict[str, int]:
        """BFS over ORBITAL_NEIGHBOR (optionally IN_CONJUNCTION_WITH) edges.

        Returns ``{node_id: hop_distance}`` excluding the seed. ``max_shell_km``
        filters ORBITAL_NEIGHBOR edges by their ``shell_km`` attribute.
        """
        types = {EdgeType.ORBITAL_NEIGHBOR} | ({EdgeType.IN_CONJUNCTION_WITH} if include_conjunctions else set())
        dist: Dict[str, int] = {object_id: 0}
        q = deque([object_id])
        while q:
            cur = q.popleft()
            if dist[cur] >= max_hops:
                continue
            for _, nxt, d in self.g.out_edges(cur, data=True):
                if d["type"] not in types:
                    continue
                if d["type"] == EdgeType.ORBITAL_NEIGHBOR and max_shell_km is not None and d.get("shell_km", 0.0) > max_shell_km:
                    continue
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        dist.pop(object_id)
        return dist

    def trace_parent_launch(self, object_id: str) -> LaunchLineage:
        """Walk debris -> parent -> launch event and collect co-manifested payloads."""
        hops = [object_id]
        parent_id: Optional[str] = None
        cur = object_id
        # follow PARENT_OF (incoming) chains, e.g. fragment -> rocket body -> ...
        guard = 0
        while guard < 10:
            parents = self.neighbors(cur, EdgeType.PARENT_OF, "in")
            if not parents:
                break
            parent_id = parents[0]
            hops.append(parent_id)
            cur = parent_id
            guard += 1
        launches = self.neighbors(cur, EdgeType.LAUNCHED_IN, "out")
        launch: Optional[LaunchEvent] = None
        co: List[str] = []
        if launches:
            launch = self.get(launches[0])  # type: ignore[assignment]
            hops.append(launch.id)
            co = [n for n in self.neighbors(launch.id, EdgeType.LAUNCHED_IN, "in") if n != cur]
            # also union explicit CO_LAUNCHED_WITH edges
            for n in self.neighbors(cur, EdgeType.CO_LAUNCHED_WITH, "out"):
                if n not in co and n != cur:
                    co.append(n)
        else:
            co = [n for n in self.neighbors(cur, EdgeType.CO_LAUNCHED_WITH, "out") if n != cur]
        return LaunchLineage(
            object_id=object_id,
            parent_object_id=parent_id if parent_id != object_id else None,
            launch_event=launch,
            co_manifested_ids=co,
            hops=hops,
        )

    def get_conjunctions(self, object_id: str) -> List[Tuple[str, dict]]:
        return [(v, {k: val for k, val in d.items() if k != "type"}) for _, v, d in self.edges_of(object_id, EdgeType.IN_CONJUNCTION_WITH, "out")]

    # -------------------------------------------------------------- memory
    def write_case_node(self, case: CaseNode, *, link_similar: bool = True, similarity_floor: float = 0.3) -> str:
        """Anchor a case in graph memory (Phase 7).

        Adds the case node, INVESTIGATED_IN edges from every involved object
        and operator, and SIMILAR_TO edges to prior cases whose similarity
        score exceeds ``similarity_floor``.
        """
        prior = [c for c in self.nodes_of_kind(NodeKind.CASE) if c.id != case.id]
        self.add_node(case)
        for oid in case.involved_object_ids + case.operator_ids:
            if oid in self.g:
                self.add_edge(EdgeType.INVESTIGATED_IN, oid, case.id)
        if link_similar:
            for sim in self._score_similar(case, prior):
                if sim.score >= similarity_floor:
                    self.add_edge(EdgeType.SIMILAR_TO, case.id, sim.case.id, score=sim.score, reasons=sim.reasons)
        return case.id

    def find_similar_cases(self, probe: CaseNode, *, top_k: int = 5) -> List[SimilarCase]:
        """Zero-shot recall of prior cases resembling ``probe`` (need not be stored)."""
        prior = [c for c in self.nodes_of_kind(NodeKind.CASE) if c.id != probe.id]
        return self._score_similar(probe, prior)[:top_k]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    def _score_similar(self, probe: CaseNode, candidates: List[CaseNode]) -> List[SimilarCase]:
        out: List[SimilarCase] = []
        p_objs, p_ops = set(probe.involved_object_ids), set(probe.operator_ids)
        for c in candidates:
            score, reasons = 0.0, []
            if c.classification == probe.classification:
                score += 0.5
                reasons.append(f"same classification '{c.classification}'")
            shared_objs = p_objs & set(c.involved_object_ids)
            if shared_objs:
                score += 0.3
                reasons.append(f"shared objects {sorted(shared_objs)}")
            shared_ops = p_ops & set(c.operator_ids)
            if shared_ops:
                score += 0.2
                reasons.append(f"shared operators {sorted(shared_ops)}")
            if probe.embedding and c.embedding:
                cos = self._cosine(probe.embedding, c.embedding)
                score = 0.6 * score + 0.4 * max(cos, 0.0)
                reasons.append(f"embedding cosine {cos:.2f}")
            if score > 0:
                out.append(SimilarCase(case=c, score=round(score, 4), reasons=reasons))
        return sorted(out, key=lambda s: s.score, reverse=True)

    # ------------------------------------------------------------ utilities
    def subgraph_around(self, node_ids: Iterable[str], hops: int = 1) -> nx.MultiDiGraph:
        keep: Set[str] = set()
        frontier = set(node_ids)
        for _ in range(hops + 1):
            keep |= frontier
            nxt: Set[str] = set()
            for n in frontier:
                nxt |= set(self.g.successors(n)) | set(self.g.predecessors(n))
            frontier = nxt - keep
        return self.g.subgraph(keep).copy()

    def edge_records(self) -> List[dict]:
        seen = set()
        recs = []
        for u, v, d in self.g.edges(data=True):
            key = (min(u, v), max(u, v), d["type"]) if d["type"].symmetric else (u, v, d["type"])
            if key in seen:
                continue
            seen.add(key)
            recs.append({"source": u, "target": v, "type": d["type"].value, **{k: val for k, val in d.items() if k != "type"}})
        return recs

    def node_records(self) -> List[dict]:
        return [{"id": n, "kind": d["kind"], "name": d["name"]} for n, d in self.g.nodes(data=True)]

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            counts[d["kind"]] = counts.get(d["kind"], 0) + 1
        counts["edges"] = len(self.edge_records())
        return counts

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    def __repr__(self) -> str:
        return f"OrbitalGraph({self.stats()})"
