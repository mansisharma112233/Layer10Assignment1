"""
FastAPI-based retrieval API.
Provides REST endpoints for querying the memory graph.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from graph.store import MemoryStore
from graph.memory_graph import MemoryGraph
from retrieval.retriever import Retriever


app = FastAPI(
    title="Layer10 Memory Graph API",
    description="Retrieval and grounding API for the memory graph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_store: Optional[MemoryStore] = None
_graph: Optional[MemoryGraph] = None
_retriever: Optional[Retriever] = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def get_graph() -> MemoryGraph:
    global _graph
    if _graph is None:
        _graph = MemoryGraph(get_store())
        _graph.build_from_store()
    return _graph


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_store(), get_graph())
        if not _retriever.load_index():
            _retriever.build_index()
    return _retriever


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 10
    include_historical: bool = True


@app.get("/")
async def root():
    return {"message": "Layer10 Memory Graph API", "version": "1.0.0"}


@app.get("/stats")
async def get_stats():
    """Get memory graph statistics."""
    store = get_store()
    graph = get_graph()
    return {
        "store_stats": store.get_stats(),
        "graph_summary": graph.get_graph_summary(),
    }


@app.post("/retrieve")
async def retrieve(request: QuestionRequest):
    """
    Retrieve a grounded context pack for a question.
    Returns entities, claims, and evidence with citations.
    """
    retriever = get_retriever()
    pack = retriever.retrieve(
        question=request.question,
        top_k=request.top_k,
        include_historical=request.include_historical,
    )
    return pack.to_dict()


@app.get("/entities")
async def list_entities(
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """List or search entities."""
    store = get_store()
    if search:
        entities = store.search_entities(search, entity_type)
    else:
        entities = store.get_all_entities()
        if entity_type:
            entities = [e for e in entities if e["entity_type"] == entity_type]
    return {"entities": entities[:limit], "total": len(entities)}


@app.get("/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get entity details with claims."""
    store = get_store()
    entity = store.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    claims = store.get_claims_for_entity(entity_id)
    graph = get_graph()
    neighborhood = graph.get_entity_neighborhood(entity_id, depth=1)

    return {
        "entity": entity,
        "claims": claims,
        "neighborhood": neighborhood,
    }


@app.get("/claims")
async def list_claims(
    claim_type: Optional[str] = None,
    temporal_status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """List or search claims."""
    store = get_store()
    if search:
        claims = store.search_claims(search, claim_type, temporal_status)
    else:
        claims = store.get_all_claims()
        if claim_type:
            claims = [c for c in claims if c["claim_type"] == claim_type]
        if temporal_status:
            claims = [c for c in claims if c.get("temporal_status") == temporal_status]
    return {"claims": claims[:limit], "total": len(claims)}


@app.get("/claims/{claim_id}/evidence")
async def get_evidence(claim_id: str):
    """Get evidence for a specific claim."""
    store = get_store()
    evidence = store.get_evidence_for_claim(claim_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Claim not found or no evidence")
    return {"claim_id": claim_id, "evidence": evidence}


@app.get("/merges")
async def list_merges(limit: int = Query(50, ge=1, le=500)):
    """Get merge history for auditing."""
    store = get_store()
    merges = store.get_merge_log()
    return {"merges": merges[:limit], "total": len(merges)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
