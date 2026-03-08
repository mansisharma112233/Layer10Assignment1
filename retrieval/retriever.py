"""
Retrieval and grounding engine.
Maps questions to entities/claims and returns grounded context packs
with cited evidence.

Uses hybrid retrieval: keyword matching + semantic embedding similarity.
"""

from __future__ import annotations
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from graph.store import MemoryStore
from graph.memory_graph import MemoryGraph


class EmbeddingIndex:
    """Simple embedding index using sentence-transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def add(self, text: str, item_id: str, meta: dict = None):
        """Add a text to the index."""
        self.texts.append(text)
        self.ids.append(item_id)
        self.metadata.append(meta or {})

    def build(self):
        """Build the embedding index."""
        if not self.texts:
            self.embeddings = np.array([])
            return
        print(f"Building embedding index for {len(self.texts)} items...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search the index by semantic similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] < 0.1:
                break
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx],
            })
        return results

    def save(self, path: Path):
        """Save index to disk."""
        np.savez(
            str(path),
            embeddings=self.embeddings if self.embeddings is not None else np.array([]),
            texts=np.array(self.texts, dtype=object),
            ids=np.array(self.ids, dtype=object),
            metadata=np.array([json.dumps(m) for m in self.metadata], dtype=object),
        )

    def load(self, path: Path):
        """Load index from disk."""
        data = np.load(str(path), allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.texts = data["texts"].tolist()
        self.ids = data["ids"].tolist()
        self.metadata = [json.loads(m) for m in data["metadata"].tolist()]


class ContextPack:
    """A grounded context pack returned by the retrieval engine."""

    def __init__(self, question: str):
        self.question = question
        self.entities: list[dict] = []
        self.claims: list[dict] = []
        self.evidence_snippets: list[dict] = []
        self.conflicts: list[dict] = []
        self.summary: str = ""
        self.generated_at: str = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "entities": self.entities,
            "claims": self.claims,
            "evidence_snippets": self.evidence_snippets,
            "conflicts": self.conflicts,
            "summary": self.summary,
            "generated_at": self.generated_at,
        }

    def to_formatted_text(self) -> str:
        """Format context pack as human-readable text with citations."""
        parts = []
        parts.append(f"Question: {self.question}")
        parts.append(f"Generated: {self.generated_at}")
        parts.append("")

        if self.entities:
            parts.append("=== Relevant Entities ===")
            for e in self.entities:
                parts.append(f"  [{e.get('entity_type', '?')}] {e.get('name', e.get('id'))}")
                if e.get("aliases"):
                    parts.append(f"    Aliases: {', '.join(e['aliases'])}")
            parts.append("")

        if self.claims:
            parts.append("=== Relevant Claims ===")
            for i, c in enumerate(self.claims, 1):
                status = c.get("temporal_status", "current")
                confidence = c.get("confidence", "medium")
                parts.append(f"  [{i}] ({status}, confidence={confidence}) {c.get('content', '')}")
                for ev in c.get("evidence", []):
                    parts.append(f"      Source: {ev.get('source_id', '?')}")
                    parts.append(f"      Excerpt: \"{ev.get('excerpt', '')[:200]}\"")
                    if ev.get("url"):
                        parts.append(f"      URL: {ev['url']}")
                parts.append("")

        if self.conflicts:
            parts.append("=== Conflicting Information ===")
            for conf in self.conflicts:
                parts.append(f"  Current: {conf.get('current', '')}")
                parts.append(f"  Historical: {conf.get('historical', '')}")
            parts.append("")

        return "\n".join(parts)


class Retriever:
    """
    Hybrid retrieval engine combining keyword search and semantic embeddings.
    Returns grounded context packs with evidence citations.
    """

    def __init__(self, store: MemoryStore = None, graph: MemoryGraph = None):
        self.store = store or MemoryStore()
        self.graph = graph or MemoryGraph(self.store)
        self.entity_index = EmbeddingIndex()
        self.claim_index = EmbeddingIndex()
        self._index_built = False

    def build_index(self):
        """Build embedding indices from the store."""
        for e in self.store.get_all_entities():
            text = f"{e['name']} ({e['entity_type']})"
            if e.get("aliases"):
                text += f" aliases: {', '.join(e['aliases'])}"
            self.entity_index.add(text, e["id"], {"entity_type": e["entity_type"]})

        for c in self.store.get_all_claims():
            text = f"{c['content']} [{c['claim_type']}]"
            self.claim_index.add(text, c["id"], {
                "claim_type": c["claim_type"],
                "subject": c["subject_entity_id"],
                "temporal_status": c.get("temporal_status", "current"),
                "confidence": c.get("confidence", "medium"),
            })

        self.entity_index.build()
        self.claim_index.build()
        self._index_built = True

    def retrieve(self, question: str, top_k: int = 10,
                 include_historical: bool = True) -> ContextPack:
        """
        Retrieve a grounded context pack for a question.

        Strategy:
          1. Keyword search for entity mentions
          2. Semantic search over entities and claims
          3. Graph expansion (neighbors of matched entities)
          4. Rank and diversify results
          5. Attach evidence citations
        """
        if not self._index_built:
            self.build_index()

        pack = ContextPack(question)

        keyword_entities = self._keyword_entity_search(question)

        semantic_entities = self.entity_index.search(question, top_k=top_k)
        semantic_claims = self.claim_index.search(question, top_k=top_k * 2)

        entity_ids = set()
        entity_scores = {}

        for e in keyword_entities:
            entity_ids.add(e["id"])
            entity_scores[e["id"]] = entity_scores.get(e["id"], 0) + 1.5

        for e in semantic_entities:
            entity_ids.add(e["id"])
            entity_scores[e["id"]] = entity_scores.get(e["id"], 0) + e["score"]

        top_entity_ids = sorted(entity_ids, key=lambda x: entity_scores.get(x, 0), reverse=True)[:top_k]

        for eid in top_entity_ids:
            entity_data = self.store.get_entity(eid)
            if entity_data:
                entity_data["relevance_score"] = entity_scores.get(eid, 0)
                pack.entities.append(entity_data)

        claim_ids = set()
        claim_scores = {}

        for c in semantic_claims:
            claim_ids.add(c["id"])
            claim_scores[c["id"]] = c["score"]

        for eid in top_entity_ids[:5]:
            entity_claims = self.store.get_claims_for_entity(eid)
            for ec in entity_claims:
                if ec["id"] not in claim_ids:
                    claim_ids.add(ec["id"])
                    claim_scores[ec["id"]] = 0.5

        ranked_claim_ids = sorted(claim_ids, key=lambda x: claim_scores.get(x, 0), reverse=True)

        for cid in ranked_claim_ids[:top_k * 2]:
            claims = self.store.search_claims("")
            claim_data = None
            for c in self.store.get_all_claims():
                if c["id"] == cid:
                    claim_data = c
                    break
            if claim_data:
                if not include_historical and claim_data.get("temporal_status") == "historical":
                    continue
                claim_data["relevance_score"] = claim_scores.get(cid, 0)
                pack.claims.append(claim_data)

                for ev in claim_data.get("evidence", []):
                    pack.evidence_snippets.append({
                        "claim_id": cid,
                        "claim_content": claim_data["content"],
                        "source_id": ev.get("source_id"),
                        "excerpt": ev.get("excerpt"),
                        "url": ev.get("url"),
                        "timestamp": ev.get("timestamp"),
                    })

        pack.conflicts = self._find_conflicts_in_results(pack.claims)

        pack.entities = pack.entities[:top_k]
        pack.claims = pack.claims[:top_k]
        pack.evidence_snippets = pack.evidence_snippets[:top_k * 3]

        pack.summary = self._generate_summary(pack)

        return pack

    def _keyword_entity_search(self, question: str) -> list[dict]:
        """Search for entities mentioned in the question by keyword."""
        mentions = set()

        mentions.update(re.findall(r"@(\w+)", question))

        mentions.update(re.findall(r'"([^"]+)"', question))
        mentions.update(re.findall(r"'([^']+)'", question))

        mentions.update(re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", question))

        results = []
        for mention in mentions:
            if len(mention) < 2:
                continue
            found = self.store.search_entities(mention)
            results.extend(found)

        words = re.findall(r"\b\w{4,}\b", question.lower())
        for word in words[:5]:
            found = self.store.search_entities(word)
            results.extend(found)

        seen = set()
        unique = []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)
        return unique

    def _find_conflicts_in_results(self, claims: list[dict]) -> list[dict]:
        """Find conflicting claims in the result set."""
        conflicts = []
        by_subject_type = {}
        for c in claims:
            key = (c.get("subject_entity_id"), c.get("claim_type"))
            by_subject_type.setdefault(key, []).append(c)

        for key, group in by_subject_type.items():
            current = [c for c in group if c.get("temporal_status") == "current"]
            historical = [c for c in group if c.get("temporal_status") == "historical"]
            if current and historical:
                conflicts.append({
                    "subject": key[0],
                    "claim_type": key[1],
                    "current": current[0].get("content"),
                    "historical": historical[0].get("content"),
                })

        return conflicts

    def _generate_summary(self, pack: ContextPack) -> str:
        """Generate a brief summary of the context pack."""
        parts = []
        if pack.entities:
            entity_names = [e.get("name", "") for e in pack.entities[:5]]
            parts.append(f"Found {len(pack.entities)} relevant entities: {', '.join(entity_names)}")
        if pack.claims:
            parts.append(f"Retrieved {len(pack.claims)} claims with {len(pack.evidence_snippets)} evidence snippets")
        if pack.conflicts:
            parts.append(f"Detected {len(pack.conflicts)} conflicting claim(s)")
        return ". ".join(parts) + "." if parts else "No relevant information found."

    def save_index(self, output_dir: Path = None):
        """Save embedding indices to disk."""
        output_dir = output_dir or config.GRAPH_DIR
        self.entity_index.save(output_dir / "entity_embeddings.npz")
        self.claim_index.save(output_dir / "claim_embeddings.npz")

    def load_index(self, input_dir: Path = None):
        """Load embedding indices from disk."""
        input_dir = input_dir or config.GRAPH_DIR
        entity_path = input_dir / "entity_embeddings.npz"
        claim_path = input_dir / "claim_embeddings.npz"
        if entity_path.exists() and claim_path.exists():
            self.entity_index.load(entity_path)
            self.claim_index.load(claim_path)
            self._index_built = True
            return True
        return False
