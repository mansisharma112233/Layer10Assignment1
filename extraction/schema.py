"""
Ontology / Schema definitions for the Layer10 Memory Graph.

Entity types, relationship types, and claim types designed for
GitHub Issues/PRs corpus. Extensible to email, Slack, Jira/Linear.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PERSON = "person"
    COMPONENT = "component"
    FEATURE = "feature"
    BUG = "bug"
    RELEASE = "release"
    LABEL = "label"
    REPOSITORY = "repository"
    TEAM = "team"


class RelationType(str, Enum):
    AUTHORED = "authored"
    ASSIGNED_TO = "assigned_to"
    REVIEWED = "reviewed"
    MENTIONED = "mentioned"
    COLLABORATED_WITH = "collaborated_with"

    AFFECTS_COMPONENT = "affects_component"
    IMPLEMENTS_FEATURE = "implements_feature"
    FIXES_BUG = "fixes_bug"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    SUPERSEDES = "supersedes"

    LABELED_AS = "labeled_as"
    RELEASED_IN = "released_in"


class ClaimType(str, Enum):
    DECISION = "decision"
    STATUS_CHANGE = "status_change"
    ASSIGNMENT = "assignment"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    TECHNICAL_FACT = "technical_fact"
    DEPENDENCY = "dependency"
    PROPOSAL = "proposal"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    WORKAROUND = "workaround"
    ROOT_CAUSE = "root_cause"
    RESOLUTION = "resolution"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TemporalStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    DISPUTED = "disputed"


class Evidence(BaseModel):
    """A pointer to the exact source supporting a claim."""
    source_id: str = Field(..., description="ID of the source artifact (issue/comment)")
    source_type: str = Field(..., description="Type: 'issue' or 'comment'")
    excerpt: str = Field(..., description="Exact text excerpt supporting the claim")
    url: str = Field("", description="URL to the source")
    timestamp: str = Field("", description="ISO timestamp of the source")
    char_offset_start: Optional[int] = Field(None, description="Start character offset in source body")
    char_offset_end: Optional[int] = Field(None, description="End character offset in source body")


class Entity(BaseModel):
    """A named entity in the memory graph."""
    id: str = Field(..., description="Unique canonical ID")
    name: str = Field(..., description="Display name")
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list, description="Known aliases")
    properties: dict = Field(default_factory=dict, description="Additional properties")
    first_seen: str = Field("", description="First appearance timestamp")
    last_seen: str = Field("", description="Last appearance timestamp")
    merge_history: list[dict] = Field(default_factory=list, description="Track of merges for reversibility")


class Claim(BaseModel):
    """A grounded claim extracted from the corpus."""
    id: str = Field(..., description="Unique claim ID")
    claim_type: ClaimType
    subject_entity_id: str = Field(..., description="Entity this claim is about")
    object_entity_id: Optional[str] = Field(None, description="Target entity (for relations)")
    relation_type: Optional[RelationType] = Field(None, description="Relation if this is a relation claim")
    content: str = Field(..., description="Natural language statement of the claim")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM)
    temporal_status: TemporalStatus = Field(TemporalStatus.CURRENT)
    valid_from: Optional[str] = Field(None, description="When this claim became true")
    valid_until: Optional[str] = Field(None, description="When this claim stopped being true (if historical)")
    evidence: list[Evidence] = Field(default_factory=list, min_length=1)
    extraction_version: str = Field("v1.0", description="Schema + model + prompt version")
    created_at: str = Field("", description="When this claim was extracted")
    superseded_by: Optional[str] = Field(None, description="ID of claim that supersedes this one")
    merged_from: list[str] = Field(default_factory=list, description="IDs of claims merged into this one")


class ExtractionResult(BaseModel):
    """Result of extracting from a single source artifact."""
    source_id: str
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    extraction_version: str = "v1.0"
    model: str = ""
    extracted_at: str = ""
    errors: list[str] = Field(default_factory=list)


SCHEMA_VERSION = "v1.0"
EXTRACTION_PROMPT_VERSION = "v1.0"

def get_extraction_version() -> str:
    """Composite version string for tracking extraction lineage."""
    return f"schema={SCHEMA_VERSION}/prompt={EXTRACTION_PROMPT_VERSION}"
