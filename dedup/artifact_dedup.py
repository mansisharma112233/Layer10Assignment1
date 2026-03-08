"""
Artifact-level deduplication.
Detects and handles duplicate/near-duplicate source artifacts
(issues, comments) before they pollute the memory graph.
"""

from __future__ import annotations
import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional


class ArtifactDeduplicator:
    """
    Deduplicates source artifacts (issues, comments).
    Handles:
      - Exact duplicates (same content hash)
      - Near-duplicates (high text similarity, e.g., quoted replies)
      - Cross-references (issues that reference each other)
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: dict[str, str] = {}
        self.merge_log: list[dict] = []

    def deduplicate_issues(self, issues: list[dict]) -> list[dict]:
        """Remove duplicate issues, keeping the canonical one."""
        unique = []
        for issue in issues:
            content_hash = issue.get("content_hash") or self._hash_content(
                issue["title"] + issue["body"]
            )
            if content_hash in self.seen_hashes:
                self.merge_log.append({
                    "action": "artifact_dedup",
                    "duplicate_id": issue["id"],
                    "canonical_id": self.seen_hashes[content_hash],
                    "method": "exact_hash",
                    "reason": "Identical content hash",
                })
                continue
            self.seen_hashes[content_hash] = issue["id"]
            unique.append(issue)
        return unique

    def deduplicate_comments(self, comments: list[dict]) -> list[dict]:
        """
        Remove duplicate comments.
        Handles GitHub's quoting pattern (lines starting with '>').
        """
        unique = []
        seen_bodies: dict[str, str] = {}

        for comment in comments:
            body = comment.get("body", "")
            normalized = self._normalize_comment(body)

            if not normalized:
                continue

            body_hash = self._hash_content(normalized)
            if body_hash in seen_bodies:
                self.merge_log.append({
                    "action": "artifact_dedup",
                    "duplicate_id": comment["id"],
                    "canonical_id": seen_bodies[body_hash],
                    "method": "normalized_hash",
                    "reason": "Identical normalized content",
                })
                continue

            is_dup = False
            for seen_norm, seen_id in list(seen_bodies.items()):
                if self._similarity(normalized, seen_norm) > self.similarity_threshold:
                    self.merge_log.append({
                        "action": "artifact_dedup",
                        "duplicate_id": comment["id"],
                        "canonical_id": seen_id,
                        "method": "similarity",
                        "similarity": self._similarity(normalized, seen_norm),
                        "reason": "Near-duplicate content",
                    })
                    is_dup = True
                    break

            if not is_dup:
                seen_bodies[body_hash] = comment["id"]
                unique.append(comment)

        return unique

    def _normalize_comment(self, body: str) -> str:
        """Normalize a comment body for comparison."""
        if not body:
            return ""
        lines = body.split("\n")
        non_quoted = [l for l in lines if not l.strip().startswith(">")]
        text = "\n".join(non_quoted).strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    def _hash_content(self, text: str) -> str:
        """Create a content hash."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _similarity(self, a: str, b: str) -> float:
        """Compute text similarity ratio."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a[:500], b[:500]).ratio()

    def get_merge_log(self) -> list[dict]:
        """Return the merge log for auditing/reversibility."""
        return self.merge_log
