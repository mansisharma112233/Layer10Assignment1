"""
Entity canonicalization.
Resolves aliases, renames, and collisions to produce canonical entity IDs.
All merges are logged for reversibility.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Optional

from extraction.schema import Entity, EntityType


class EntityCanonicalizer:
    """
    Merges duplicate entities, resolves aliases, and maintains a canonical registry.

    Strategy:
      1. GitHub username normalization (case-insensitive, handle bots)
      2. Component/feature name normalization (slug matching)
      3. Fuzzy matching for near-duplicates (Levenshtein)
      4. All merges logged for reversibility
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.canonical: dict[str, Entity] = {}
        self.alias_map: dict[str, str] = {}
        self.merge_log: list[dict] = []

    def canonicalize(self, entities: list[Entity]) -> list[Entity]:
        """
        Process a list of entities, merging duplicates.
        Returns the list of canonical entities.
        """
        for entity in entities:
            self._add_or_merge(entity)
        return list(self.canonical.values())

    def _add_or_merge(self, entity: Entity):
        """Add entity to registry, merging if duplicate found."""
        normalized_id = self._normalize_id(entity.name, entity.entity_type)

        if normalized_id in self.canonical:
            self._merge(self.canonical[normalized_id], entity, "exact_id_match")
            return

        for alias in [entity.name.lower()] + [a.lower() for a in entity.aliases]:
            if alias in self.alias_map:
                canonical_id = self.alias_map[alias]
                if canonical_id in self.canonical:
                    self._merge(self.canonical[canonical_id], entity, "alias_match")
                    return

        for can_id, can_entity in self.canonical.items():
            if can_entity.entity_type != entity.entity_type:
                continue
            if self._is_similar(entity, can_entity):
                self._merge(can_entity, entity, "fuzzy_match")
                return

        entity.id = normalized_id
        self.canonical[normalized_id] = entity
        self._register_aliases(entity)

    def _merge(self, canonical: Entity, duplicate: Entity, method: str):
        """Merge duplicate into canonical entity."""
        new_aliases = set(canonical.aliases)
        new_aliases.add(duplicate.name)
        new_aliases.update(duplicate.aliases)
        new_aliases.discard(canonical.name)
        canonical.aliases = sorted(new_aliases)

        for k, v in duplicate.properties.items():
            if k not in canonical.properties:
                canonical.properties[k] = v

        if duplicate.first_seen and (not canonical.first_seen or duplicate.first_seen < canonical.first_seen):
            canonical.first_seen = duplicate.first_seen
        if duplicate.last_seen and (not canonical.last_seen or duplicate.last_seen > canonical.last_seen):
            canonical.last_seen = duplicate.last_seen

        merge_record = {
            "action": "entity_merge",
            "canonical_id": canonical.id,
            "merged_id": duplicate.id,
            "merged_name": duplicate.name,
            "method": method,
            "canonical_state_before": canonical.model_dump(),
        }
        canonical.merge_history.append(merge_record)
        self.merge_log.append(merge_record)

        self._register_aliases(canonical)

    def _register_aliases(self, entity: Entity):
        """Register all aliases in the lookup map."""
        self.alias_map[entity.name.lower()] = entity.id
        self.alias_map[entity.id] = entity.id
        for alias in entity.aliases:
            self.alias_map[alias.lower()] = entity.id

    def _normalize_id(self, name: str, entity_type: EntityType) -> str:
        """Create a normalized canonical ID."""
        if entity_type == EntityType.PERSON:
            slug = name.lower().strip().lstrip("@")
            return f"person:{slug}"

        slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
        return f"{entity_type.value}:{slug}"

    def _is_similar(self, a: Entity, b: Entity) -> bool:
        """Check if two entities are similar enough to merge."""
        if a.entity_type == EntityType.PERSON:
            return a.name.lower().strip("@") == b.name.lower().strip("@")

        name_a = re.sub(r"[^a-z0-9]", "", a.name.lower())
        name_b = re.sub(r"[^a-z0-9]", "", b.name.lower())

        if name_a == name_b:
            return True

        ratio = SequenceMatcher(None, name_a, name_b).ratio()
        if ratio >= self.similarity_threshold:
            return True

        all_names_a = {a.name.lower()} | {al.lower() for al in a.aliases}
        all_names_b = {b.name.lower()} | {al.lower() for al in b.aliases}
        if all_names_a & all_names_b:
            return True

        return False

    def resolve_id(self, entity_id: str) -> str:
        """Resolve an entity ID to its canonical form."""
        if entity_id in self.canonical:
            return entity_id
        if entity_id.lower() in self.alias_map:
            return self.alias_map[entity_id.lower()]
        parts = entity_id.split(":", 1)
        if len(parts) > 1 and parts[1].lower() in self.alias_map:
            return self.alias_map[parts[1].lower()]
        return entity_id

    def get_merge_log(self) -> list[dict]:
        """Return full merge history for auditing."""
        return self.merge_log

    def undo_merge(self, merge_record: dict) -> Optional[Entity]:
        """
        Undo a specific merge (reversibility).
        Returns the re-separated entity, or None if can't undo.
        """
        canonical_id = merge_record["canonical_id"]
        merged_name = merge_record["merged_name"]

        if canonical_id not in self.canonical:
            return None

        canonical = self.canonical[canonical_id]

        if merged_name in canonical.aliases:
            canonical.aliases.remove(merged_name)

        if merged_name.lower() in self.alias_map:
            del self.alias_map[merged_name.lower()]

        canonical.merge_history = [
            m for m in canonical.merge_history
            if m.get("merged_name") != merged_name
        ]

        return canonical
