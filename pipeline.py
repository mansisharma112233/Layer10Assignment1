"""
End-to-end pipeline orchestrator for Layer10 Memory Graph.

Runs:
  1. Corpus fetching (GitHub Issues/PRs)
  2. Structured extraction (Gemini LLM)
  3. Artifact deduplication
  4. Entity canonicalization
  5. Claim deduplication
  6. Memory graph construction
  7. Embedding index building
  8. Example retrievals

Usage:
    python pipeline.py
    python pipeline.py --skip-fetch
    python pipeline.py --skip-extract
"""

from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from corpus.fetch_github_issues import GitHubCorpusFetcher, load_corpus
from extraction.extractor import GeminiExtractor, load_extractions
from extraction.schema import ExtractionResult, Entity, Claim
from extraction.validator import ExtractionValidator
from dedup.artifact_dedup import ArtifactDeduplicator
from dedup.entity_canon import EntityCanonicalizer
from dedup.claim_dedup import ClaimDeduplicator
from graph.memory_graph import MemoryGraph
from graph.store import MemoryStore
from retrieval.retriever import Retriever


def run_pipeline(
    skip_fetch: bool = False,
    skip_extract: bool = False,
    max_issues: int = None,
):
    """Run the full pipeline end-to-end."""
    start = time.time()
    max_issues = max_issues or config.MAX_ISSUES

    print("=" * 60)
    print("  Layer10 Memory Graph Pipeline")
    print("=" * 60)

    corpus_path = config.RAW_DIR / "corpus.json"
    if skip_fetch and corpus_path.exists():
        print("\n[1/7] Loading cached corpus...")
        corpus = load_corpus()
    else:
        print("\n[1/7] Fetching corpus from GitHub...")
        fetcher = GitHubCorpusFetcher()
        corpus = fetcher.fetch_and_save(max_issues=max_issues)

    issues = corpus["issues"]
    comments = corpus["comments"]
    print(f"  -> {len(issues)} issues, {len(comments)} comments")

    print("\n[2/7] Deduplicating artifacts...")
    artifact_dedup = ArtifactDeduplicator()
    issues = artifact_dedup.deduplicate_issues(issues)
    comments = artifact_dedup.deduplicate_comments(comments)
    print(f"  -> After dedup: {len(issues)} issues, {len(comments)} comments")
    print(f"  -> {len(artifact_dedup.get_merge_log())} duplicates removed")

    comments_by_issue = {}
    for c in comments:
        comments_by_issue.setdefault(c["issue_id"], []).append(c)

    extractions_path = config.EXTRACTED_DIR / "extractions.json"
    if skip_extract and extractions_path.exists():
        print("\n[3/7] Loading cached extractions...")
        results = load_extractions()
    else:
        print("\n[3/7] Running structured extraction (Gemini)...")
        extractor = GeminiExtractor()
        results = extractor.extract_batch(issues, comments_by_issue)
        extractor.save_results(results)

    total_entities_raw = sum(len(r.entities) for r in results)
    total_claims_raw = sum(len(r.claims) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    print(f"  -> {total_entities_raw} raw entities, {total_claims_raw} raw claims, {total_errors} errors")

    print("\n[4/7] Canonicalizing entities...")
    all_entities = []
    for r in results:
        all_entities.extend(r.entities)

    entity_canon = EntityCanonicalizer()
    canonical_entities = entity_canon.canonicalize(all_entities)
    print(f"  -> {len(all_entities)} raw -> {len(canonical_entities)} canonical entities")
    print(f"  -> {len(entity_canon.get_merge_log())} entity merges")

    print("\n[5/7] Deduplicating claims...")
    all_claims = []
    for r in results:
        for claim in r.claims:
            claim.subject_entity_id = entity_canon.resolve_id(claim.subject_entity_id)
            if claim.object_entity_id:
                claim.object_entity_id = entity_canon.resolve_id(claim.object_entity_id)
            all_claims.append(claim)

    claim_dedup = ClaimDeduplicator()
    canonical_claims = claim_dedup.deduplicate(all_claims)
    print(f"  -> {len(all_claims)} raw -> {len(canonical_claims)} canonical claims")
    print(f"  -> {len(claim_dedup.get_merge_log())} claim merges")
    print(f"  -> {len(claim_dedup.get_conflicts())} conflicts detected")

    print("\n[6/7] Building memory graph...")
    store = MemoryStore()
    graph = MemoryGraph(store)

    for entity in canonical_entities:
        graph.add_entity(entity)

    for claim in canonical_claims:
        graph.add_claim(claim)

    for record in artifact_dedup.get_merge_log():
        store.log_merge(record)
    for record in entity_canon.get_merge_log():
        store.log_merge(record)
    for record in claim_dedup.get_merge_log():
        store.log_merge(record)

    for r in results:
        store.log_ingestion(
            source_id=r.source_id,
            extraction_version=r.extraction_version,
            model=r.model,
            num_entities=len(r.entities),
            num_claims=len(r.claims),
            errors=r.errors,
        )

    graph.commit()

    graph_path = graph.export_json()
    print(f"  -> Graph exported to {graph_path}")

    summary = graph.get_graph_summary()
    print(f"  -> Nodes: {summary['num_nodes']}, Edges: {summary['num_edges']}")
    print(f"  -> Components: {summary['connected_components']}")

    print("\n[7/7] Building retrieval index and running example queries...")
    retriever = Retriever(store, graph)
    retriever.build_index()
    retriever.save_index()

    example_questions = [
        "What is React Suspense and how does it work?",
        "Who are the main contributors and what do they work on?",
        "What bugs have been reported and resolved?",
        "What decisions were made about concurrent rendering?",
        "What components does React have?",
    ]

    context_packs = []
    for question in example_questions:
        print(f"\n  Q: {question}")
        pack = retriever.retrieve(question, top_k=5)
        print(f"    -> {pack.summary}")
        context_packs.append(pack.to_dict())

    output_path = config.GRAPH_DIR / "example_context_packs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(context_packs, f, indent=2, default=str)
    print(f"\n  Example context packs saved to {output_path}")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Entities: {len(canonical_entities)}")
    print(f"  Claims: {len(canonical_claims)}")
    print(f"  Graph nodes: {summary['num_nodes']}, edges: {summary['num_edges']}")
    print(f"\n  Next steps:")
    print(f"    1. streamlit run visualization/app.py")
    print(f"    2. uvicorn retrieval.api:app --reload")
    print("=" * 60)

    return {
        "entities": len(canonical_entities),
        "claims": len(canonical_claims),
        "graph_nodes": summary["num_nodes"],
        "graph_edges": summary["num_edges"],
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer10 Memory Graph Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip corpus fetching")
    parser.add_argument("--skip-extract", action="store_true", help="Skip LLM extraction")
    parser.add_argument("--max-issues", type=int, default=None, help="Max issues to fetch")
    args = parser.parse_args()

    run_pipeline(
        skip_fetch=args.skip_fetch,
        skip_extract=args.skip_extract,
        max_issues=args.max_issues,
    )
