"""
Memory Graph built on NetworkX.
Integrates entities, claims, and evidence into a queryable graph structure.
Backed by SQLite for persistence.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from extraction.schema import Entity, Claim, EntityType, ClaimType, RelationType, TemporalStatus
from graph.store import MemoryStore


class MemoryGraph:
    """
    NetworkX-based memory graph with SQLite persistence.

    Nodes = Entities
    Edges = Claims/Relations between entities
    Node attributes include all entity properties.
    Edge attributes include claim content, evidence, confidence, temporal status.
    """

    def __init__(self, store: MemoryStore = None):
        self.store = store or MemoryStore()
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity):
        """Add an entity as a node."""
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            aliases=entity.aliases,
            properties=entity.properties,
            first_seen=entity.first_seen,
            last_seen=entity.last_seen,
        )
        self.store.upsert_entity(entity)

    def add_claim(self, claim: Claim):
        """Add a claim. If it's a relation, also add an edge."""
        for eid in [claim.subject_entity_id, claim.object_entity_id]:
            if eid and not self.store.entity_exists(eid):
                placeholder = Entity(
                    id=eid,
                    name=eid,
                    entity_type=EntityType.COMPONENT,
                    aliases=[],
                    properties={"placeholder": True},
                )
                self.add_entity(placeholder)

        self.store.upsert_claim(claim)

        if claim.object_entity_id and claim.subject_entity_id != claim.object_entity_id:
            if not self.graph.has_node(claim.subject_entity_id):
                self.graph.add_node(claim.subject_entity_id, name=claim.subject_entity_id)
            if not self.graph.has_node(claim.object_entity_id):
                self.graph.add_node(claim.object_entity_id, name=claim.object_entity_id)

            self.graph.add_edge(
                claim.subject_entity_id,
                claim.object_entity_id,
                key=claim.id,
                claim_id=claim.id,
                claim_type=claim.claim_type.value,
                relation_type=claim.relation_type.value if claim.relation_type else None,
                content=claim.content,
                confidence=claim.confidence.value,
                temporal_status=claim.temporal_status.value,
                valid_from=claim.valid_from,
                valid_until=claim.valid_until,
            )

    def build_from_store(self):
        """Rebuild the in-memory graph from the SQLite store."""
        self.graph.clear()

        entities = self.store.get_all_entities()
        for e in entities:
            self.graph.add_node(
                e["id"],
                name=e["name"],
                entity_type=e["entity_type"],
                aliases=e["aliases"],
                properties=e["properties"],
                first_seen=e.get("first_seen"),
                last_seen=e.get("last_seen"),
            )

        claims = self.store.get_all_claims()
        for c in claims:
            if c.get("object_entity_id") and c["subject_entity_id"] != c.get("object_entity_id"):
                src = c["subject_entity_id"]
                tgt = c["object_entity_id"]
                if not self.graph.has_node(src):
                    self.graph.add_node(src, name=src)
                if not self.graph.has_node(tgt):
                    self.graph.add_node(tgt, name=tgt)
                self.graph.add_edge(
                    src, tgt,
                    key=c["id"],
                    claim_id=c["id"],
                    claim_type=c["claim_type"],
                    relation_type=c.get("relation_type"),
                    content=c["content"],
                    confidence=c.get("confidence", "medium"),
                    temporal_status=c.get("temporal_status", "current"),
                )

    def get_entity_neighborhood(self, entity_id: str, depth: int = 1) -> dict:
        """Get an entity and its neighbors up to a given depth."""
        if not self.graph.has_node(entity_id):
            return {"entity": None, "neighbors": [], "edges": []}

        entity = dict(self.graph.nodes[entity_id])
        entity["id"] = entity_id

        neighbors = set()
        edges = []
        frontier = {entity_id}
        visited = {entity_id}

        for d in range(depth):
            next_frontier = set()
            for node in frontier:
                for _, target, key, data in self.graph.out_edges(node, keys=True, data=True):
                    if target not in visited:
                        next_frontier.add(target)
                    edges.append({
                        "source": node,
                        "target": target,
                        "claim_id": data.get("claim_id"),
                        "relation_type": data.get("relation_type"),
                        "content": data.get("content"),
                        "confidence": data.get("confidence"),
                        "temporal_status": data.get("temporal_status"),
                    })
                for source, _, key, data in self.graph.in_edges(node, keys=True, data=True):
                    if source not in visited:
                        next_frontier.add(source)
                    edges.append({
                        "source": source,
                        "target": node,
                        "claim_id": data.get("claim_id"),
                        "relation_type": data.get("relation_type"),
                        "content": data.get("content"),
                        "confidence": data.get("confidence"),
                        "temporal_status": data.get("temporal_status"),
                    })
            visited |= next_frontier
            frontier = next_frontier

        neighbor_list = []
        for nid in visited - {entity_id}:
            if self.graph.has_node(nid):
                n = dict(self.graph.nodes[nid])
                n["id"] = nid
                neighbor_list.append(n)

        seen_edges = set()
        unique_edges = []
        for e in edges:
            edge_key = (e["source"], e["target"], e.get("claim_id"))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(e)

        return {
            "entity": entity,
            "neighbors": neighbor_list,
            "edges": unique_edges,
        }

    def get_graph_summary(self) -> dict:
        """Get summary statistics of the graph."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self._count_by_attribute("entity_type"),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "connected_components": (
                nx.number_weakly_connected_components(self.graph)
                if self.graph.number_of_nodes() > 0 else 0
            ),
            "top_degree_nodes": self._top_degree_nodes(10),
        }

    def _count_by_attribute(self, attr: str) -> dict:
        counts = {}
        for _, data in self.graph.nodes(data=True):
            val = data.get(attr, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts

    def _top_degree_nodes(self, n: int) -> list[dict]:
        degrees = sorted(
            self.graph.degree(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
        result = []
        for node_id, degree in degrees:
            data = self.graph.nodes.get(node_id, {})
            result.append({
                "id": node_id,
                "name": data.get("name", node_id),
                "entity_type": data.get("entity_type", "unknown"),
                "degree": degree,
            })
        return result

    def export_json(self, path: Path = None) -> Path:
        """Export graph to JSON for visualization / portability."""
        path = path or config.GRAPH_PATH
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def import_json(self, path: Path = None):
        """Import graph from JSON."""
        path = path or config.GRAPH_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True, multigraph=True)

    def commit(self):
        """Commit all pending changes to SQLite."""
        self.store.commit()
