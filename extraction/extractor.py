"""
LLM-based structured extraction pipeline using Google Gemini.
Extracts entities, claims, and evidence from GitHub issues/comments.
"""

from __future__ import annotations
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from extraction.schema import (
    Entity, Claim, Evidence, ExtractionResult,
    EntityType, ClaimType, RelationType,
    get_extraction_version,
)
from extraction.validator import ExtractionValidator, parse_llm_json


EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction system for a software project memory graph.
Your job is to extract structured entities, relationships, and claims from GitHub issues and comments.

ONTOLOGY:
Entity types: person, component, feature, bug, release, label, repository, team
Relation types: authored, assigned_to, reviewed, mentioned, collaborated_with,
    affects_component, implements_feature, fixes_bug, depends_on, related_to,
    part_of, supersedes, labeled_as, released_in
Claim types: decision, status_change, assignment, bug_report, feature_request,
    technical_fact, dependency, proposal, agreement, disagreement, workaround,
    root_cause, resolution

RULES:
1. Every claim MUST have at least one evidence with an exact excerpt from the source text.
2. Entity IDs should be lowercase slugs: "person:username" or "component:component-name".
3. Use the source_id provided for evidence references.
4. Extract ALL meaningful entities and claims, not just the obvious ones.
5. For temporal claims, set valid_from to the source timestamp.
6. Mark confidence as "high" for explicit statements, "medium" for inferences, "low" for uncertain.
7. If an issue is closed or resolved, create a status_change claim with temporal_status "current"
   and a corresponding historical claim for the open state.

OUTPUT FORMAT (strict JSON):
{
  "entities": [
    {
      "name": "Entity Name",
      "entity_type": "person|component|feature|bug|release|label|repository|team",
      "aliases": ["alias1"],
      "properties": {"key": "value"}
    }
  ],
  "claims": [
    {
      "claim_type": "decision|status_change|bug_report|...",
      "subject_entity_id": "type:entity-slug",
      "object_entity_id": "type:entity-slug or null",
      "relation_type": "relation or null",
      "content": "Natural language claim statement",
      "confidence": "high|medium|low",
      "temporal_status": "current|historical|disputed",
      "valid_from": "ISO timestamp or null",
      "evidence": [
        {
          "source_id": "provided source ID",
          "source_type": "issue|comment",
          "excerpt": "Exact quote from the source text"
        }
      ]
    }
  ]
}"""


def build_issue_prompt(issue: dict, comments: list[dict]) -> str:
    """Build the user prompt for extracting from an issue + its comments."""
    parts = []
    parts.append(f"=== GitHub Issue #{issue['number']}: {issue['title']} ===")
    parts.append(f"Source ID: {issue['id']}")
    parts.append(f"Author: {issue['author']}")
    parts.append(f"State: {issue['state']}")
    parts.append(f"Labels: {', '.join(issue['labels']) if issue['labels'] else 'none'}")
    parts.append(f"Assignees: {', '.join(issue['assignees']) if issue['assignees'] else 'none'}")
    parts.append(f"Created: {issue['created_at']}")
    parts.append(f"Updated: {issue['updated_at']}")
    if issue['closed_at']:
        parts.append(f"Closed: {issue['closed_at']}")
    parts.append(f"URL: {issue['html_url']}")
    parts.append(f"\n--- Body ---\n{issue['body'][:3000]}")

    for i, comment in enumerate(comments[:15]):
        parts.append(f"\n--- Comment {i+1} by {comment['author']} ({comment['created_at']}) ---")
        parts.append(f"Source ID: {comment['id']}")
        parts.append(f"URL: {comment['html_url']}")
        parts.append(comment['body'][:1500])

    parts.append("\n\nExtract all entities and claims from this issue and its comments. Return ONLY valid JSON.")
    return "\n".join(parts)


class GeminiExtractor:
    """Extracts structured data from corpus artifacts using Google Gemini."""

    def __init__(self, api_key: str = None, model: str = None):
        import google.generativeai as genai
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Get one free at https://aistudio.google.com/app/apikey")
        genai.configure(api_key=self.api_key)
        self.model_name = model or config.EXTRACTION_MODEL
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        self.validator = ExtractionValidator()
        self.extraction_version = get_extraction_version()

    def extract_from_issue(self, issue: dict, comments: list[dict],
                           max_retries: int = 2) -> ExtractionResult:
        """Extract entities and claims from a single issue + comments."""
        prompt = build_issue_prompt(issue, comments)

        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                raw_text = response.text

                raw_data = parse_llm_json(raw_text)
                raw_data["source_id"] = issue["id"]
                raw_data["extraction_version"] = self.extraction_version
                raw_data["model"] = self.model_name

                for claim in raw_data.get("claims", []):
                    for ev in claim.get("evidence", []):
                        if not ev.get("source_id"):
                            ev["source_id"] = issue["id"]
                        if not ev.get("url"):
                            ev["url"] = issue["html_url"]
                        if not ev.get("timestamp"):
                            ev["timestamp"] = issue["created_at"]

                result = self.validator.validate_extraction_result(raw_data)
                return result

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return ExtractionResult(
                    source_id=issue["id"],
                    extraction_version=self.extraction_version,
                    model=self.model_name,
                    extracted_at=datetime.utcnow().isoformat(),
                    errors=[f"Extraction failed after {max_retries + 1} attempts: {str(e)}"],
                )

    def extract_batch(self, issues: list[dict], comments_by_issue: dict[str, list[dict]],
                      batch_size: int = None) -> list[ExtractionResult]:
        """Extract from a batch of issues with rate limiting and incremental saving."""
        batch_size = batch_size or config.EXTRACTION_BATCH_SIZE
        results = []

        partial_path = config.EXTRACTED_DIR / "extractions_partial.json"
        done_ids = set()
        if partial_path.exists():
            try:
                with open(partial_path, "r", encoding="utf-8") as f:
                    partial_data = json.load(f)
                for r in partial_data.get("results", []):
                    results.append(ExtractionResult(**r))
                    done_ids.add(r["source_id"])
                print(f"Resuming: {len(done_ids)} issues already extracted")
            except Exception:
                pass

        remaining = [iss for iss in issues if iss["id"] not in done_ids]
        print(f"Extracting from {len(remaining)} issues ({len(done_ids)} cached)...")
        for i, issue in enumerate(tqdm(remaining, desc="Extraction")):
            issue_comments = comments_by_issue.get(issue["id"], [])
            result = self.extract_from_issue(issue, issue_comments)
            results.append(result)

            self._save_partial(results, partial_path)

            if (i + 1) % batch_size == 0:
                time.sleep(4)

        total_entities = sum(len(r.entities) for r in results)
        total_claims = sum(len(r.claims) for r in results)
        total_errors = sum(len(r.errors) for r in results)
        print(f"Extraction complete: {total_entities} entities, {total_claims} claims, {total_errors} errors")

        if partial_path.exists():
            partial_path.unlink()

        return results

    def _save_partial(self, results: list[ExtractionResult], path: Path):
        """Save partial extraction results for resume capability."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "results": [r.model_dump() for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def save_results(self, results: list[ExtractionResult], output_dir: Path = None):
        """Save extraction results to disk."""
        output_dir = output_dir or config.EXTRACTED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "extraction_version": self.extraction_version,
                "model": self.model_name,
                "extracted_at": datetime.utcnow().isoformat(),
                "num_results": len(results),
                "total_entities": sum(len(r.entities) for r in results),
                "total_claims": sum(len(r.claims) for r in results),
            },
            "results": [r.model_dump() for r in results],
        }

        output_path = output_dir / "extractions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Extraction results saved to {output_path}")
        return output_path


def load_extractions(path: Path = None) -> list[ExtractionResult]:
    """Load extraction results from disk."""
    path = path or (config.EXTRACTED_DIR / "extractions.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ExtractionResult(**r) for r in data["results"]]


if __name__ == "__main__":
    from corpus.fetch_github_issues import load_corpus
    corpus = load_corpus()

    comments_by_issue = {}
    for c in corpus["comments"]:
        comments_by_issue.setdefault(c["issue_id"], []).append(c)

    extractor = GeminiExtractor()
    results = extractor.extract_batch(corpus["issues"], comments_by_issue)
    extractor.save_results(results)
