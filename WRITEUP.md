# Layer10 Memory Graph — Design Writeup

## Table of Contents

1. [Corpus Choice](#corpus-choice)
2. [Ontology & Schema Design](#ontology--schema-design)
3. [Structured Extraction](#structured-extraction)
4. [Deduplication & Canonicalization](#deduplication--canonicalization)
5. [Memory Graph Design](#memory-graph-design)
6. [Retrieval & Grounding](#retrieval--grounding)
7. [Visualization](#visualization)
8. [Example Retrieved Context Packs](#example-retrieved-context-packs)
9. [Layer10 Adaptation](#layer10-adaptation)
10. [Tradeoffs & Future Work](#tradeoffs--future-work)

---

## Corpus Choice

**Corpus:** GitHub Issues and Pull Requests from **facebook/react**

**Why React:**
- Rich mix of **structured data** (labels, assignees, state transitions, milestones) and **unstructured data** (issue descriptions, comment discussions, technical debates)
- Many **identity resolution challenges**: contributors use different names, bots interact, people are referenced by @-mentions
- Strong examples of **decisions, reversals, and evolving state**: issues opened → triaged → debated → resolved → reopened
- **Cross-referencing**: issues reference PRs, PRs reference issues, comments quote previous discussions
- Well-suited to demonstrate all required capabilities (extraction, dedup, revision tracking, grounding)

**Reproduction:**
```bash
# Fetched via GitHub REST API (see corpus/fetch_github_issues.py)
# Top 200 issues sorted by comment count for maximum signal
python -c "from corpus.fetch_github_issues import GitHubCorpusFetcher; GitHubCorpusFetcher().fetch_and_save()"
```

No manual download needed — the pipeline fetches directly from the GitHub REST API (works without authentication for public repos, optional token for higher rate limits).

---

## Ontology & Schema Design

### Design Principles
1. **Coherent and extensible:** Entity/claim types are domain-relevant but generalize to other corpora
2. **Grounding-first:** Every claim requires at least one evidence pointer with source ID + excerpt
3. **Temporal awareness:** Claims track validity windows (valid_from/valid_until) and temporal status (current/historical/disputed)
4. **Merge-safe:** All entities track merge history; all dedup operations are logged and reversible

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| `person` | GitHub user / contributor | @dan_abramov, @acdlite |
| `component` | Code module or subsystem | React Fiber, Reconciler, Scheduler |
| `feature` | User-facing capability | Suspense, Server Components, Hooks |
| `bug` | A specific defect or issue | "useEffect fires twice in StrictMode" |
| `release` | Version milestone | React 18, React 19 |
| `label` | GitHub label category | "Type: Bug", "Component: Reconciler" |
| `repository` | The repo itself | facebook/react |
| `team` | A group of people | React Core Team |

### Relationship Types

Relationships connect entities bidirectionally:
- **Person → Artifact:** `authored`, `assigned_to`, `reviewed`, `mentioned`
- **Technical:** `affects_component`, `implements_feature`, `fixes_bug`, `depends_on`, `related_to`, `part_of`, `supersedes`
- **State:** `labeled_as`, `released_in`

### Claim Types

Claims represent extractable facts with temporal semantics:

| Type | Description | Temporal Pattern |
|------|-------------|------------------|
| `decision` | Team decided X | Usually current unless reversed |
| `status_change` | Issue opened/closed/reopened | Creates current + historical pair |
| `assignment` | X assigned to Y | May change over time |
| `bug_report` | X is broken when Y | Current until resolved |
| `feature_request` | Users want X | May be implemented or rejected |
| `technical_fact` | Component X uses algorithm Y | Generally stable |
| `proposal` | "We should do X" | May be accepted, rejected, or superseded |
| `resolution` | "Fixed by X" / "Won't fix" | Terminal state |
| `workaround` | "Work around this by doing X" | Valid until proper fix |
| `root_cause` | "The root cause is X" | May be revised |

### Evidence Model

Every claim must point to evidence:
```
Evidence {
  source_id:         "github-issue-facebook/react-12345"
  source_type:       "issue" | "comment"
  excerpt:           "Exact quote from the source text"
  url:               "https://github.com/facebook/react/issues/12345"
  timestamp:         "2024-01-15T10:30:00Z"
  char_offset_start: 142   (optional)
  char_offset_end:   298   (optional)
}
```

---

## Structured Extraction

### Pipeline Architecture

```
Raw Issue/Comments → Prompt Construction → Gemini 2.0 Flash → JSON Parse → Validation/Repair → Typed Objects
```

### Prompt Engineering

The extraction prompt is a **structured system instruction** that:
1. Defines the complete ontology (entity types, relation types, claim types)
2. Specifies strict JSON output format with required fields
3. Requires exact excerpts for evidence (grounding)
4. Instructs confidence calibration (high/medium/low)
5. Handles temporal semantics (current vs. historical claims)

Temperature is set to **0.1** for consistent, deterministic extraction.

### Validation & Repair

The `ExtractionValidator` handles:

1. **Type normalization:** Fuzzy-maps unknown types to valid enum values (e.g., "developer" → "person", "module" → "component")
2. **ID generation:** Deterministic slugged IDs when the LLM doesn't provide them
3. **Evidence repair:** Creates minimal evidence from claim content if the LLM omits it
4. **Confidence defaulting:** Falls back to "medium" for unparseable confidence values
5. **JSON repair:** Handles markdown code fences, trailing commas, single quotes, and partial JSON extraction

### Retry Strategy

- Up to 2 retries per issue with exponential backoff
- Failed extractions produce an `ExtractionResult` with error details (not silently dropped)
- Rate limiting: 4-second pause every `EXTRACTION_BATCH_SIZE` issues (respects Gemini free tier of ~15 RPM)

### Versioning

Every extraction is tagged with:
```
extraction_version = "schema=v1.0/prompt=v1.0"
model = "gemini-2.5-flash"
extracted_at = "2026-03-08T..."
```

To backfill when the ontology changes:
1. Bump schema version
2. Re-extract from cached corpus (skip-fetch mode)
3. Old extractions retained in DB with their version tag
4. The ingestion log tracks which source_id was processed with which version

### Quality Gates

1. **Evidence requirement:** Claims without evidence are flagged (warning) and assigned minimal evidence from the source
2. **Confidence calibration:** Low-confidence claims are retained but ranked lower in retrieval
3. **Error tracking:** Extraction errors are logged per-source for observability
4. **Cross-evidence support:** Claims supported by multiple evidence snippets get higher effective confidence during retrieval

---

## Deduplication & Canonicalization

### Three-Level Dedup Strategy

#### Level 1: Artifact Dedup
Removes duplicate/near-duplicate source artifacts before extraction.

- **Exact hash match:** SHA-256 of (title + body) catches identical issues
- **Normalized text match:** Strips quoting patterns (`>` lines), collapses whitespace, then hashes
- **Similarity threshold (0.85):** SequenceMatcher catches near-duplicates (quoted replies, cross-posts)

#### Level 2: Entity Canonicalization

Resolves the same entity referred to by different names:

- **Person normalization:** Case-insensitive username matching, `@` stripping, bot detection
- **Component normalization:** Slug-based matching (e.g., "React Fiber" = "react-fiber" = "Fiber")
- **Fuzzy matching:** SequenceMatcher ratio > 0.85 for entities of the same type
- **Alias tracking:** All known names map to a single canonical ID

#### Level 3: Claim Dedup

Merges repeated statements of the same fact:

- **Content keying:** SHA-256 of (subject + type + normalized content)
- **Semantic matching:** Same claim_type + same subject + content similarity > 0.80
- **Evidence union:** Merged claims keep ALL supporting evidence from both sides
- **Confidence upgrading:** Higher-confidence evidence upgrades the canonical claim's confidence

### Conflict Detection & Temporal Revision

When two claims about the same subject+type contradict:
1. **Temporal ordering:** Compare `valid_from` timestamps (or evidence timestamps)
2. **Newer wins:** The newer claim becomes `current`, the older becomes `historical`
3. **Linking:** The older claim's `superseded_by` field points to the newer claim's ID
4. **Both preserved:** Historical claims are not deleted — they remain queryable with status "historical"

### Reversibility

**All merges are logged** with:
- Action type (artifact_dedup, entity_merge, claim_merge)
- Canonical ID and merged ID
- Method used (exact_hash, fuzzy_match, etc.)
- Pre-merge state snapshot (for entity merges)

The `EntityCanonicalizer.undo_merge()` method supports reverting a specific merge by:
1. Removing the merged name from aliases
2. Removing from the alias map
3. Removing from merge history

The full merge log is accessible via the Visualization UI and the REST API's `/merges` endpoint.

---

## Memory Graph Design

### Storage Architecture

**Hybrid approach:** NetworkX in-memory graph + SQLite persistent store

| Layer | Purpose |
|-------|---------|
| **NetworkX MultiDiGraph** | Fast in-memory traversal, neighborhood queries, graph algorithms |
| **SQLite** | Durable persistence, complex queries, full-text search, audit trail |

This gives us the best of both worlds: graph traversal speed + relational query power, with zero infrastructure requirements.

### Core Tables

| Table | Purpose |
|-------|---------|
| `entities` | All canonical entities with aliases, properties, merge history |
| `claims` | All claims with temporal status, confidence, evidence pointers |
| `evidence` | Individual evidence snippets linked to claims, with source metadata |
| `merge_log` | Full audit trail of all dedup/merge operations |
| `ingestion_log` | Per-source tracking of extraction runs for observability |

### Time Semantics

- **Event time:** When something happened in the real world (issue created, comment posted)
- **Validity time:** When a claim was true (`valid_from` / `valid_until`)
- **"Current" resolution:** A claim is current if `temporal_status = "current"` AND `valid_until IS NULL`

Example: If Issue #123 was opened on Jan 1 and closed on Feb 15:
- Claim A: "Issue #123 is open" → `valid_from=Jan 1, valid_until=Feb 15, temporal_status=historical`
- Claim B: "Issue #123 is closed" → `valid_from=Feb 15, valid_until=NULL, temporal_status=current`

### Incremental Updates

- **Idempotent upserts:** `INSERT ... ON CONFLICT DO UPDATE` for both entities and claims
- **Temporal bounds merging:** `first_seen` takes the MIN, `last_seen` takes the MAX
- **Reprocessing:** Can re-extract any source and upsert without duplicating data

### Handling Edits/Deletes/Redactions

- **Edits:** Re-fetch the source, re-extract, upsert. The evidence excerpt may change but the claim is linked by source_id.
- **Deletes:** If a source is deleted, its evidence is marked with a tombstone. Claims lose that evidence but retain others.
- **Redactions:** Evidence excerpts can be scrubbed while preserving the claim and its other evidence.

### Permissions (Conceptual)

Each evidence record carries a `source_id` linking to the original artifact. In a production system:
1. Each source has an ACL (access control list)
2. At retrieval time, filter evidence: only show evidence from sources the user can access
3. If a claim loses all accessible evidence, it's hidden from that user
4. This ensures **claims are only visible if the user can see the supporting evidence**

### Observability

The `ingestion_log` table tracks:
- Which source was processed and when
- Which extraction version/model was used
- How many entities/claims were extracted
- Any errors encountered

This enables monitoring for:
- Extraction quality degradation (rising error rates)
- Schema drift (unexpected types)
- Throughput regression

---

## Retrieval & Grounding

### Hybrid Retrieval Strategy

```
Question → [Keyword Search] ──┐
                               ├── Merge & Rank → Graph Expansion → Evidence → Context Pack
Question → [Semantic Search] ──┘
```

1. **Keyword matching:** Extract entity mentions from the question (@mentions, quoted terms, capitalized phrases)
2. **Semantic embedding:** Encode question with sentence-transformers (all-MiniLM-L6-v2), cosine similarity against entity + claim indices
3. **Score fusion:** Keyword matches get a 1.5x bonus; semantic scores are additive
4. **Graph expansion:** For the top-5 matched entities, retrieve all linked claims (1-hop)
5. **Ranking:** Score = semantic_similarity + keyword_bonus + graph_proximity
6. **Diversity:** Results include multiple claim types and temporal statuses

### Grounding Guarantees

Every item in the returned **Context Pack** includes:
- **Entity:** With full metadata (type, aliases, properties)
- **Claim:** With claim_type, confidence, temporal_status
- **Evidence:** Exact source_id, excerpt, URL, timestamp

The `to_formatted_text()` method produces human-readable output with numbered citations.

### Ambiguity & Conflicts

When conflicting claims are found in the result set:
- **Both are shown**, with current vs. historical labeling
- Conflicts are surfaced in a dedicated `conflicts` section
- The UI uses color coding: 🟢 current, 🟡 historical, 🔴 disputed

### Expansion Control

To prevent query explosion:
- Graph expansion limited to top-5 entities, 1-hop depth
- Claims limited to top_k × 2 candidates, filtered to top_k
- Evidence limited to top_k × 3 snippets
- Minimum similarity threshold of 0.1 for semantic results

---

## Visualization

### Streamlit App

A full-featured web UI with six views:

1. **Dashboard:** KPI cards, entity/claim breakdowns, graph topology stats, top connected entities
2. **Graph Explorer:** Interactive PyVis graph with filters by entity type, confidence, temporal status. Focus mode for individual entities with 2-hop neighborhood.
3. **Question Retrieval:** Natural language query interface with formatted results, evidence panel, and conflict detection
4. **Entity Browser:** Search and browse entities with linked claims and merge history
5. **Claim Browser:** Search and filter claims by type/status, with inline evidence
6. **Merge Audit:** Full dedup/merge history, grouped by action type, inspectable details

### Graph Visualization Features

- **Color-coded nodes** by entity type (persons=green, components=blue, features=orange, bugs=red, etc.)
- **Edge labels** show relationship types
- **Edge thickness** encodes confidence (high=thick, low=thin)
- **Historical edges** are dimmer
- **Click-through:** Clicking any claim shows the supporting evidence excerpts and source URLs

---

## Example Retrieved Context Packs

Below are 5 example queries and their retrieved context packs (generated from the pipeline output stored in `data/graph/example_context_packs.json`). Each pack contains relevant entities, grounded claims, and supporting evidence.

### Q1: "What is React Suspense and how does it work?"

| Entities | Claims | Conflicts |
|----------|--------|-----------|
| React (component), Fundamental improvements to React (feature), React Fire (feature) | 5 | 1 |

**Top claims:**
- [technical_fact] React changed the way cem2ran views UI programming and Open Source. (confidence: high)
- [technical_fact] The fragment API is a hard problem with React's current architecture. (confidence: high)
- [proposal] React Fire is an effort to modernize React DOM. (confidence: high)

**Evidence grounding:**
- `github-comment-354625915`: _"You have changed the way I view UI programming and Open Source <3"_
- `github-issue-facebook/react-2127`: _"We want this too but it is a hard problem with our current architecture."_

### Q2: "Who are the main contributors and what do they work on?"

| Entities | Claims | Conflicts |
|----------|--------|-----------|
| Open-Source Contributors (team), React maintainers (team), dmitrif (person) | 5 | 1 |

**Top claims:**
- [agreement] arthurdenner thanked all open-source contributors. (confidence: high)
- [feature_request] Gabssnake suggests there should be a notice in the React documentation about the wrapping requirement. (confidence: high)
- [feature_request] dmitrif asks for suggestions on how to make `electrode-react-ssr-caching` work with React v16. (confidence: high)

**Evidence grounding:**
- `github-comment-354625851`: _"Thank you! Not only the React team, but to all the open-source contributors."_
- `github-comment-65140776`: _"I feel like there should be a notice somewhere, maybe I missed it?"_

### Q3: "What bugs have been reported and resolved?"

| Entities | Claims | Conflicts |
|----------|--------|-----------|
| Countless Bugs (bug), doesHavePendingPassiveEffects (feature), Bug 7179 (bug) | 5 | 0 |

**Top claims:**
- [technical_fact] Issue 20463 has known bugs in its current state. (confidence: high)
- [resolution] React Native fixed countless bugs. (confidence: high)
- [technical_fact] There are quite a few known problems, and some of them are hard or impossible to fix without bigger internal changes. (confidence: high)

**Evidence grounding:**
- `github-comment-745302277`: _"since the current state has known bugs."_
- `github-issue-facebook/react-11940`: _"fixing countless bugs"_

### Q4: "What decisions were made about concurrent rendering?"

| Entities | Claims | Conflicts |
|----------|--------|-----------|
| concurrent mode (feature), Asynchronous rendering (feature), Server renderer (component) | 5 | 0 |

**Top claims:**
- [technical_fact] The rewrite aimed to enable experimenting with asynchronous rendering of components for better perceived performance. (confidence: high)
- [status_change] Asynchronous rendering is still being experimented on internally. (confidence: high)
- [technical_fact] Syranide sees hidden dangers in allowing composite components to return multiple components. (confidence: high)

**Evidence grounding:**
- `github-issue-facebook/react-10294`: _"Enable us to start experimenting with asynchronous rendering of components for better perceived performance."_
- `github-comment-318184274`: _"We're still experimenting on it internally."_

### Q5: "What components does React have?"

| Entities | Claims | Conflicts |
|----------|--------|-----------|
| React (component), ReactDOMComponent (component), React documentation (component) | 5 | 0 |

**Top claims:**
- [technical_fact] The React team has mostly been focused on fundamental improvements to React. (confidence: high)
- [proposal] React Fire is an effort to modernize React DOM. (confidence: high)
- [technical_fact] The goal of React Fire is to make React smaller and faster. (confidence: high)

**Evidence grounding:**
- `github-issue-facebook/react-13525`: _"This year, the React team has mostly been focused on fundamental improvements to React"_
- `github-issue-facebook/react-13525`: _"We're calling this effort 'React Fire'."_

---

## Layer10 Adaptation

### Adapting the Ontology

For Layer10's target environment (email, Slack/Teams, docs, Jira/Linear):

**New entity types:**
- `email_thread`, `slack_channel`, `slack_message`
- `jira_ticket`, `linear_issue`, `sprint`, `epic`
- `document`, `wiki_page`, `meeting`
- `customer`, `deal`, `product`

**New relation types:**
- `discussed_in` (topic → channel/thread)
- `escalated_to` (ticket → person/team)
- `blocked_by` (ticket → ticket)
- `decided_in` (decision → meeting/thread)

**New claim types:**
- `commitment` ("Team X committed to deliver by date Y")
- `customer_feedback` ("Customer reported X")
- `meeting_decision` ("Decided in standup to...")
- `policy` ("Going forward, we will...")

### Extraction Contract Changes

1. **Multi-source fusion:** A single "decision" may span an email thread + a Jira ticket + a Slack msg. Extraction needs cross-source evidence aggregation.
2. **Real-time extraction:** Slack/email arrive continuously — extraction must be (near) real-time, not batch-only. Use webhooks + a message queue.
3. **Thread awareness:** Email forwarding chains and Slack threads need thread-level context for accurate extraction (not just individual messages).

### Dedup Strategy Changes

1. **Cross-platform dedup:** The same decision might appear in Slack, email, AND Jira. Need a `discussion_id` that links all three.
2. **Person resolution:** People use different names across Slack (@dan), email (dan@company.com), and Jira (Dan Abramov). Need an identity resolution service backed by org directory.
3. **State synchronization:** When a Jira ticket changes state, any claims derived from its previous state must be updated.

### Long-term Memory vs. Ephemeral Context

| Durable Memory | Ephemeral |
|----------------|-----------|
| Decisions, policies, architectural facts | "Can someone review this PR?" |
| Bug root causes and resolutions | "I'm OOO tomorrow" |
| Customer commitments | Casual chat messages |
| Ownership/assignment facts | Duplicate status updates |

**Decay strategy:** Claims without supporting evidence refreshed within a configurable window (e.g., 90 days) get flagged for review. High-confidence claims with multiple evidence sources are more resistant to decay.

### Grounding & Safety

- **Provenance:** Every claim traces back to specific messages/tickets with timestamps. No "hallucinated" memory.
- **Deletions/Redactions:** When a Slack message is deleted or an email is redacted, the evidence is scrubbed but the claim may be retained if other evidence supports it. If all evidence is removed, the claim is tombstoned.
- **Citations:** Retrieval output always includes source URLs so users can verify.

### Permissions

- Claims inherit the access level of their most restrictive evidence source
- If a claim is supported by both a public Slack message and a private email, only users with email access see the email evidence; the claim is still visible but with redacted evidence for others
- Jira/Linear permissions map directly: only users with project access see those claims
- Implementation: At query time, join evidence → source → permission check, filter out unauthorized evidence, hide claims with zero visible evidence

### Operational Reality

| Concern | Approach |
|---------|----------|
| **Scaling** | Replace SQLite with PostgreSQL + pgvector; use Redis for hot entity cache |
| **Cost** | Batch extraction during off-peak; cache LLM results; use cheaper models for re-extraction |
| **Incremental updates** | Webhook-driven ingest with idempotent upserts; no full re-processing needed |
| **Evaluation** | Golden set of manually annotated issues; precision/recall on entity extraction; claim grounding accuracy audits |
| **Regression testing** | Snapshot current extractions; after schema/prompt changes, compare against known-good outputs |

---

## Tradeoffs & Future Work

### Key Tradeoffs Made

1. **NetworkX + SQLite vs. Neo4j:** Chose simplicity and zero-infrastructure reproducibility over the richer query language of Neo4j. For a production system, Neo4j or a graph-capable DB would be preferred.

2. **Gemini Flash vs. larger models:** Flash is free-tier and fast but occasionally misses subtle claims. A production system could use a two-pass approach: Flash for initial extraction, a larger model for validation.

3. **SequenceMatcher vs. learned embeddings for dedup:** String similarity is fast and interpretable but misses semantic duplicates. Embedding-based dedup would catch "React Suspense" ≈ "the suspense feature in React".

4. **In-memory embedding index vs. vector DB:** For 200 issues this is fine; for millions of artifacts, use pgvector, Pinecone, or Weaviate.

### Future Work

- **Streaming ingestion:** Webhook-based real-time extraction from live repos
- **Multi-hop retrieval:** Follow chains of evidence across multiple claims
- **Learned entity resolution:** Train a model on merge decisions to auto-resolve entities
- **Claim confidence calibration:** Use retrieval feedback to adjust confidence scores
- **Time-travel queries:** "What did we know about X as of date Y?"
- **Human-in-the-loop:** Quality review UI for flagged low-confidence claims
