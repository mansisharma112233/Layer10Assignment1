"""
Validation and repair for extraction outputs.
Ensures all extracted entities and claims conform to the schema,
have required fields, and pass quality gates.
"""

from __future__ import annotations
import re
import json
from datetime import datetime
from typing import Optional

from extraction.schema import (
    Entity, Claim, Evidence, ExtractionResult,
    EntityType, ClaimType, RelationType,
    ConfidenceLevel, TemporalStatus,
)


class ValidationError:
    def __init__(self, field: str, message: str, severity: str = "error"):
        self.field = field
        self.message = message
        self.severity = severity

    def __repr__(self):
        return f"[{self.severity}] {self.field}: {self.message}"


class ExtractionValidator:
    """Validates and repairs extraction outputs."""

    def validate_entity(self, entity: dict) -> tuple[Optional[Entity], list[ValidationError]]:
        """Validate and repair a raw entity dict."""
        errors = []

        if not entity.get("name"):
            errors.append(ValidationError("name", "Entity name is required"))
            return None, errors

        entity_type = entity.get("entity_type", "").lower().strip()
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            type_map = {
                "user": EntityType.PERSON,
                "developer": EntityType.PERSON,
                "contributor": EntityType.PERSON,
                "maintainer": EntityType.PERSON,
                "author": EntityType.PERSON,
                "module": EntityType.COMPONENT,
                "package": EntityType.COMPONENT,
                "library": EntityType.COMPONENT,
                "api": EntityType.COMPONENT,
                "project": EntityType.REPOSITORY,
                "repo": EntityType.REPOSITORY,
                "tag": EntityType.LABEL,
                "category": EntityType.LABEL,
                "version": EntityType.RELEASE,
                "milestone": EntityType.RELEASE,
                "issue": EntityType.BUG,
                "defect": EntityType.BUG,
                "error": EntityType.BUG,
            }
            entity_type = type_map.get(entity_type, EntityType.COMPONENT)
            errors.append(ValidationError(
                "entity_type",
                f"Unknown type '{entity.get('entity_type')}', mapped to '{entity_type.value}'",
                severity="warning"
            ))

        entity_id = entity.get("id") or self._make_entity_id(entity["name"], entity_type)

        name = entity["name"].strip()

        aliases = entity.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",")]
        aliases = [a for a in aliases if a and a != name]

        try:
            result = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                aliases=aliases,
                properties=entity.get("properties", {}),
                first_seen=entity.get("first_seen", ""),
                last_seen=entity.get("last_seen", ""),
            )
            return result, errors
        except Exception as e:
            errors.append(ValidationError("entity", str(e)))
            return None, errors

    def validate_claim(self, claim: dict, known_entities: set[str]) -> tuple[Optional[Claim], list[ValidationError]]:
        """Validate and repair a raw claim dict."""
        errors = []

        if not claim.get("content"):
            errors.append(ValidationError("content", "Claim content is required"))
            return None, errors

        subject_id = claim.get("subject_entity_id", "")
        if not subject_id:
            errors.append(ValidationError("subject_entity_id", "Subject entity is required"))
            return None, errors

        claim_type = claim.get("claim_type", "").lower().strip()
        try:
            claim_type = ClaimType(claim_type)
        except ValueError:
            type_map = {
                "bug": ClaimType.BUG_REPORT,
                "fix": ClaimType.RESOLUTION,
                "request": ClaimType.FEATURE_REQUEST,
                "suggestion": ClaimType.PROPOSAL,
                "agree": ClaimType.AGREEMENT,
                "disagree": ClaimType.DISAGREEMENT,
                "fact": ClaimType.TECHNICAL_FACT,
                "info": ClaimType.TECHNICAL_FACT,
                "change": ClaimType.STATUS_CHANGE,
                "closed": ClaimType.STATUS_CHANGE,
                "opened": ClaimType.STATUS_CHANGE,
                "assigned": ClaimType.ASSIGNMENT,
                "cause": ClaimType.ROOT_CAUSE,
                "solved": ClaimType.RESOLUTION,
                "hack": ClaimType.WORKAROUND,
            }
            claim_type = type_map.get(claim_type, ClaimType.TECHNICAL_FACT)
            errors.append(ValidationError(
                "claim_type",
                f"Unknown type '{claim.get('claim_type')}', mapped to '{claim_type.value}'",
                severity="warning"
            ))

        relation_type = None
        if claim.get("relation_type"):
            try:
                relation_type = RelationType(claim["relation_type"].lower().strip())
            except ValueError:
                errors.append(ValidationError(
                    "relation_type",
                    f"Unknown relation type '{claim['relation_type']}'",
                    severity="warning"
                ))

        confidence = claim.get("confidence", "medium").lower().strip()
        try:
            confidence = ConfidenceLevel(confidence)
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        temporal = claim.get("temporal_status", "current").lower().strip()
        try:
            temporal = TemporalStatus(temporal)
        except ValueError:
            temporal = TemporalStatus.CURRENT

        evidence_list = claim.get("evidence", [])
        validated_evidence = []
        for ev in evidence_list:
            validated_ev = self._validate_evidence(ev)
            if validated_ev:
                validated_evidence.append(validated_ev)

        if not validated_evidence:
            errors.append(ValidationError("evidence", "Claim has no valid evidence", severity="warning"))
            validated_evidence = [Evidence(
                source_id=claim.get("source_id", "unknown"),
                source_type="issue",
                excerpt=claim["content"][:200],
            )]

        claim_id = claim.get("id") or self._make_claim_id(subject_id, claim_type.value, claim["content"])

        try:
            result = Claim(
                id=claim_id,
                claim_type=claim_type,
                subject_entity_id=subject_id,
                object_entity_id=claim.get("object_entity_id"),
                relation_type=relation_type,
                content=claim["content"].strip(),
                confidence=confidence,
                temporal_status=temporal,
                valid_from=claim.get("valid_from"),
                valid_until=claim.get("valid_until"),
                evidence=validated_evidence,
                extraction_version=claim.get("extraction_version", "v1.0"),
                created_at=datetime.utcnow().isoformat(),
                superseded_by=claim.get("superseded_by"),
            )
            return result, errors
        except Exception as e:
            errors.append(ValidationError("claim", str(e)))
            return None, errors

    def _validate_evidence(self, ev: dict) -> Optional[Evidence]:
        """Validate a single evidence entry."""
        if not ev.get("source_id") and not ev.get("excerpt"):
            return None
        try:
            return Evidence(
                source_id=ev.get("source_id", "unknown"),
                source_type=ev.get("source_type", "issue"),
                excerpt=ev.get("excerpt", "")[:500],
                url=ev.get("url", ""),
                timestamp=ev.get("timestamp", ""),
                char_offset_start=ev.get("char_offset_start"),
                char_offset_end=ev.get("char_offset_end"),
            )
        except Exception:
            return None

    def _make_entity_id(self, name: str, entity_type: EntityType) -> str:
        """Generate a deterministic entity ID."""
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return f"{entity_type.value}:{slug}"

    def _make_claim_id(self, subject_id: str, claim_type: str, content: str) -> str:
        """Generate a deterministic claim ID."""
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"claim:{claim_type}:{subject_id}:{content_hash}"

    def validate_extraction_result(self, raw: dict) -> ExtractionResult:
        """Validate a full extraction result (from LLM output)."""
        entities = []
        claims = []
        all_errors = []
        known_entity_ids = set()

        for raw_entity in raw.get("entities", []):
            entity, errors = self.validate_entity(raw_entity)
            all_errors.extend(errors)
            if entity:
                entities.append(entity)
                known_entity_ids.add(entity.id)

        for raw_claim in raw.get("claims", []):
            claim, errors = self.validate_claim(raw_claim, known_entity_ids)
            all_errors.extend(errors)
            if claim:
                claims.append(claim)

        return ExtractionResult(
            source_id=raw.get("source_id", "unknown"),
            entities=entities,
            claims=claims,
            extraction_version=raw.get("extraction_version", "v1.0"),
            model=raw.get("model", ""),
            extracted_at=datetime.utcnow().isoformat(),
            errors=[str(e) for e in all_errors if e.severity == "error"],
        )


def parse_llm_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.
    Handles markdown code blocks, trailing commas, etc.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM output as JSON: {text[:200]}...")
