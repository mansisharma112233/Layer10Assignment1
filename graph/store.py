"""
SQLite persistence layer for the memory graph.
Stores entities, claims, evidence, and merge history.
Supports incremental updates, idempotency, and audit trails.
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from extraction.schema import Entity, Claim, Evidence, EntityType, ClaimType, ConfidenceLevel, TemporalStatus


class MemoryStore:
    """SQLite-backed persistence for the memory graph."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DB_PATH
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases TEXT DEFAULT '[]',
                properties TEXT DEFAULT '{}',
                first_seen TEXT,
                last_seen TEXT,
                merge_history TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                claim_type TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                object_entity_id TEXT,
                relation_type TEXT,
                content TEXT NOT NULL,
                confidence TEXT DEFAULT 'medium',
                temporal_status TEXT DEFAULT 'current',
                valid_from TEXT,
                valid_until TEXT,
                extraction_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                superseded_by TEXT,
                merged_from TEXT DEFAULT '[]',
                FOREIGN KEY (subject_entity_id) REFERENCES entities(id),
                FOREIGN KEY (object_entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                excerpt TEXT NOT NULL,
                url TEXT,
                timestamp TEXT,
                char_offset_start INTEGER,
                char_offset_end INTEGER,
                FOREIGN KEY (claim_id) REFERENCES claims(id)
            );

            CREATE TABLE IF NOT EXISTS merge_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_type TEXT,
                canonical_id TEXT,
                merged_id TEXT,
                method TEXT,
                reason TEXT,
                details TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                extraction_version TEXT,
                model TEXT,
                num_entities INTEGER,
                num_claims INTEGER,
                errors TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
            CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
            CREATE INDEX IF NOT EXISTS idx_claims_temporal ON claims(temporal_status);
            CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        """)
        self.conn.commit()

    def entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists in the store."""
        row = self.conn.execute("SELECT 1 FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return row is not None

    def upsert_entity(self, entity: Entity):
        """Insert or update an entity (idempotent)."""
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            INSERT INTO entities (id, name, entity_type, aliases, properties,
                                  first_seen, last_seen, merge_history, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                aliases = excluded.aliases,
                properties = excluded.properties,
                first_seen = CASE WHEN excluded.first_seen < entities.first_seen
                             THEN excluded.first_seen ELSE entities.first_seen END,
                last_seen = CASE WHEN excluded.last_seen > entities.last_seen
                            THEN excluded.last_seen ELSE entities.last_seen END,
                merge_history = excluded.merge_history,
                updated_at = excluded.updated_at
        """, (
            entity.id, entity.name, entity.entity_type.value,
            json.dumps(entity.aliases), json.dumps(entity.properties),
            entity.first_seen, entity.last_seen,
            json.dumps([m for m in entity.merge_history]),
            now, now,
        ))

    def upsert_claim(self, claim: Claim):
        """Insert or update a claim with its evidence."""
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            INSERT INTO claims (id, claim_type, subject_entity_id, object_entity_id,
                                relation_type, content, confidence, temporal_status,
                                valid_from, valid_until, extraction_version,
                                created_at, updated_at, superseded_by, merged_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                temporal_status = excluded.temporal_status,
                valid_until = excluded.valid_until,
                superseded_by = excluded.superseded_by,
                merged_from = excluded.merged_from,
                updated_at = excluded.updated_at
        """, (
            claim.id, claim.claim_type.value, claim.subject_entity_id,
            claim.object_entity_id, claim.relation_type.value if claim.relation_type else None,
            claim.content, claim.confidence.value, claim.temporal_status.value,
            claim.valid_from, claim.valid_until, claim.extraction_version,
            now, now, claim.superseded_by, json.dumps(claim.merged_from),
        ))

        for ev in claim.evidence:
            self.conn.execute("""
                INSERT INTO evidence (claim_id, source_id, source_type, excerpt,
                                      url, timestamp, char_offset_start, char_offset_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.id, ev.source_id, ev.source_type, ev.excerpt,
                ev.url, ev.timestamp, ev.char_offset_start, ev.char_offset_end,
            ))

    def log_merge(self, record: dict):
        """Log a merge operation for audit/reversibility."""
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            INSERT INTO merge_log (action, entity_type, canonical_id, merged_id,
                                    method, reason, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("action"), record.get("entity_type"),
            record.get("canonical_id"), record.get("merged_id", record.get("duplicate_id")),
            record.get("method"), record.get("reason"),
            json.dumps(record), now,
        ))

    def log_ingestion(self, source_id: str, extraction_version: str, model: str,
                      num_entities: int, num_claims: int, errors: list[str]):
        """Log an ingestion event for observability."""
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            INSERT INTO ingestion_log (source_id, extraction_version, model,
                                        num_entities, num_claims, errors, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source_id, extraction_version, model, num_entities, num_claims,
              json.dumps(errors), now))

    def commit(self):
        self.conn.commit()

    def get_all_entities(self) -> list[dict]:
        """Get all entities."""
        rows = self.conn.execute("SELECT * FROM entities ORDER BY name").fetchall()
        return [self._row_to_entity_dict(r) for r in rows]

    def get_all_claims(self) -> list[dict]:
        """Get all claims with evidence."""
        rows = self.conn.execute("SELECT * FROM claims ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            claim_dict = dict(r)
            claim_dict["merged_from"] = json.loads(claim_dict.get("merged_from", "[]"))
            evidence_rows = self.conn.execute(
                "SELECT * FROM evidence WHERE claim_id = ?", (claim_dict["id"],)
            ).fetchall()
            claim_dict["evidence"] = [dict(e) for e in evidence_rows]
            result.append(claim_dict)
        return result

    def get_entity(self, entity_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity_dict(row) if row else None

    def get_claims_for_entity(self, entity_id: str) -> list[dict]:
        """Get all claims where entity is subject or object."""
        rows = self.conn.execute("""
            SELECT * FROM claims
            WHERE subject_entity_id = ? OR object_entity_id = ?
            ORDER BY created_at DESC
        """, (entity_id, entity_id)).fetchall()
        result = []
        for r in rows:
            claim_dict = dict(r)
            claim_dict["merged_from"] = json.loads(claim_dict.get("merged_from", "[]"))
            evidence_rows = self.conn.execute(
                "SELECT * FROM evidence WHERE claim_id = ?", (claim_dict["id"],)
            ).fetchall()
            claim_dict["evidence"] = [dict(e) for e in evidence_rows]
            result.append(claim_dict)
        return result

    def get_evidence_for_claim(self, claim_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM evidence WHERE claim_id = ?", (claim_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_merge_log(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM merge_log ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Get summary statistics for observability."""
        stats = {}
        stats["entities"] = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        stats["claims"] = self.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        stats["evidence"] = self.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        stats["merges"] = self.conn.execute("SELECT COUNT(*) FROM merge_log").fetchone()[0]
        stats["ingestions"] = self.conn.execute("SELECT COUNT(*) FROM ingestion_log").fetchone()[0]

        stats["entities_by_type"] = {}
        for row in self.conn.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type"
        ).fetchall():
            stats["entities_by_type"][row["entity_type"]] = row["cnt"]

        stats["claims_by_type"] = {}
        for row in self.conn.execute(
            "SELECT claim_type, COUNT(*) as cnt FROM claims GROUP BY claim_type"
        ).fetchall():
            stats["claims_by_type"][row["claim_type"]] = row["cnt"]

        stats["claims_by_status"] = {}
        for row in self.conn.execute(
            "SELECT temporal_status, COUNT(*) as cnt FROM claims GROUP BY temporal_status"
        ).fetchall():
            stats["claims_by_status"][row["temporal_status"]] = row["cnt"]

        stats["claims_by_confidence"] = {}
        for row in self.conn.execute(
            "SELECT confidence, COUNT(*) as cnt FROM claims GROUP BY confidence"
        ).fetchall():
            stats["claims_by_confidence"][row["confidence"]] = row["cnt"]

        return stats

    def search_entities(self, query: str, entity_type: str = None) -> list[dict]:
        """Search entities by name/alias."""
        params = [f"%{query}%", f"%{query}%"]
        sql = "SELECT * FROM entities WHERE (name LIKE ? OR aliases LIKE ?)"
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        sql += " ORDER BY name"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_entity_dict(r) for r in rows]

    def search_claims(self, query: str, claim_type: str = None,
                      temporal_status: str = None) -> list[dict]:
        """Search claims by content."""
        params = [f"%{query}%"]
        sql = "SELECT * FROM claims WHERE content LIKE ?"
        if claim_type:
            sql += " AND claim_type = ?"
            params.append(claim_type)
        if temporal_status:
            sql += " AND temporal_status = ?"
            params.append(temporal_status)
        sql += " ORDER BY created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            claim_dict = dict(r)
            claim_dict["merged_from"] = json.loads(claim_dict.get("merged_from", "[]"))
            evidence_rows = self.conn.execute(
                "SELECT * FROM evidence WHERE claim_id = ?", (claim_dict["id"],)
            ).fetchall()
            claim_dict["evidence"] = [dict(e) for e in evidence_rows]
            result.append(claim_dict)
        return result

    def _row_to_entity_dict(self, row) -> dict:
        d = dict(row)
        d["aliases"] = json.loads(d.get("aliases", "[]"))
        d["properties"] = json.loads(d.get("properties", "{}"))
        d["merge_history"] = json.loads(d.get("merge_history", "[]"))
        return d

    def close(self):
        self.conn.close()
