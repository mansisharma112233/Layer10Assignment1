"""
Claim deduplication.
Merges repeated statements of the same fact while preserving
all supporting evidence. Handles conflicts and revisions.
"""

from __future__ import annotations
import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional

from extraction.schema import Claim, Evidence, TemporalStatus, ConfidenceLevel


class ClaimDeduplicator:
    """
    Deduplicates claims at multiple levels:
      1. Exact content match (same claim text)
      2. Semantic near-duplicates (same subject + type + similar content)
      3. Conflict detection (contradictory claims about same subject)
      4. Temporal revision tracking (claim A superseded by claim B)
    """

    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold
        self.canonical_claims: dict[str, Claim] = {}
        self.merge_log: list[dict] = []
        self.conflicts: list[dict] = []

    def deduplicate(self, claims: list[Claim]) -> list[Claim]:
        """
        Process claims, merging duplicates and detecting conflicts.
        Returns canonical claim list.
        """
        for claim in claims:
            self._add_or_merge(claim)

        self._detect_conflicts()
        return list(self.canonical_claims.values())

    def _add_or_merge(self, claim: Claim):
        """Add claim or merge into existing canonical claim."""
        content_key = self._content_key(claim)
        for can_id, can_claim in self.canonical_claims.items():
            if self._content_key(can_claim) == content_key:
                self._merge(can_claim, claim, "exact_content")
                return

        for can_id, can_claim in self.canonical_claims.items():
            if self._is_duplicate(claim, can_claim):
                self._merge(can_claim, claim, "semantic_match")
                return

        self.canonical_claims[claim.id] = claim

    def _merge(self, canonical: Claim, duplicate: Claim, method: str):
        """Merge duplicate claim into canonical, preserving evidence."""
        existing_excerpts = {e.excerpt[:100] for e in canonical.evidence}
        for ev in duplicate.evidence:
            if ev.excerpt[:100] not in existing_excerpts:
                canonical.evidence.append(ev)
                existing_excerpts.add(ev.excerpt[:100])

        confidence_order = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.HIGH: 2,
        }
        if confidence_order.get(duplicate.confidence, 0) > confidence_order.get(canonical.confidence, 0):
            canonical.confidence = duplicate.confidence

        canonical.merged_from.append(duplicate.id)
        merge_record = {
            "action": "claim_merge",
            "canonical_id": canonical.id,
            "merged_id": duplicate.id,
            "method": method,
            "evidence_added": len(duplicate.evidence),
        }
        self.merge_log.append(merge_record)

    def _detect_conflicts(self):
        """
        Detect conflicting claims about the same subject.
        E.g., "Issue X is open" vs "Issue X is closed"
        """
        claims_by_subject = {}
        for claim in self.canonical_claims.values():
            key = (claim.subject_entity_id, claim.claim_type)
            claims_by_subject.setdefault(key, []).append(claim)

        for (subject, ctype), group in claims_by_subject.items():
            if len(group) < 2:
                continue

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if self._is_conflicting(group[i], group[j]):
                        newer, older = self._resolve_temporal(group[i], group[j])
                        if newer and older:
                            older.temporal_status = TemporalStatus.HISTORICAL
                            if newer.valid_from:
                                older.valid_until = newer.valid_from
                            newer.temporal_status = TemporalStatus.CURRENT
                            older.superseded_by = newer.id

                            self.conflicts.append({
                                "subject": subject,
                                "claim_type": ctype.value if hasattr(ctype, 'value') else str(ctype),
                                "current_claim_id": newer.id,
                                "historical_claim_id": older.id,
                                "resolution": "temporal_ordering",
                            })

    def _is_duplicate(self, a: Claim, b: Claim) -> bool:
        """Check if two claims are semantic duplicates."""
        if a.claim_type != b.claim_type:
            return False
        if a.subject_entity_id != b.subject_entity_id:
            return False

        ratio = SequenceMatcher(
            None,
            a.content.lower()[:200],
            b.content.lower()[:200],
        ).ratio()

        return ratio >= self.similarity_threshold

    def _is_conflicting(self, a: Claim, b: Claim) -> bool:
        """Check if two claims contradict each other."""
        if a.subject_entity_id != b.subject_entity_id:
            return False
        if a.claim_type != b.claim_type:
            return False

        content_sim = SequenceMatcher(
            None,
            a.content.lower()[:200],
            b.content.lower()[:200],
        ).ratio()

        if 0.3 < content_sim < self.similarity_threshold:
            conflict_patterns = [
                (r"\bopen\b", r"\bclosed\b"),
                (r"\btrue\b", r"\bfalse\b"),
                (r"\benabled\b", r"\bdisabled\b"),
                (r"\bwill\b", r"\bwon'?t\b"),
                (r"\baccept\b", r"\breject\b"),
                (r"\bnot a bug\b", r"\bis a bug\b"),
                (r"\bfixed\b", r"\bnot fixed\b"),
            ]
            for pat_a, pat_b in conflict_patterns:
                a_match = re.search(pat_a, a.content, re.IGNORECASE)
                b_match = re.search(pat_b, b.content, re.IGNORECASE)
                if (a_match and b_match) or (
                    re.search(pat_b, a.content, re.IGNORECASE) and
                    re.search(pat_a, b.content, re.IGNORECASE)
                ):
                    return True

        return False

    def _resolve_temporal(self, a: Claim, b: Claim) -> tuple[Optional[Claim], Optional[Claim]]:
        """Determine which claim is newer. Returns (newer, older)."""
        ts_a = a.valid_from or ""
        ts_b = b.valid_from or ""

        if not ts_a and a.evidence:
            ts_a = max((e.timestamp for e in a.evidence if e.timestamp), default="")
        if not ts_b and b.evidence:
            ts_b = max((e.timestamp for e in b.evidence if e.timestamp), default="")

        if ts_a and ts_b:
            if ts_a >= ts_b:
                return a, b
            else:
                return b, a
        elif ts_a:
            return a, b
        elif ts_b:
            return b, a
        else:
            return None, None

    def _content_key(self, claim: Claim) -> str:
        """Create a comparable key from claim content."""
        text = f"{claim.subject_entity_id}:{claim.claim_type.value}:{claim.content.lower().strip()}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get_merge_log(self) -> list[dict]:
        return self.merge_log

    def get_conflicts(self) -> list[dict]:
        return self.conflicts
