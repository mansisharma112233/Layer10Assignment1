# Layer10 Memory Graph

## Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

Built on **GitHub Issues/PRs from facebook/react** as the public corpus.

---

### Prerequisites

- **Python 3.10+**
- **Gemini API key** (free tier: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey))
- ~2GB disk for dependencies + data

---

### Quick Start (End-to-End Reproduction)

```bash
# 1. Clone the repo
git clone https://github.com/mansisharma112233/Layer10Assignment1.git
cd Layer10Assignment1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Optionally add GITHUB_TOKEN for higher GitHub API rate limits

# 4. Run the full pipeline (fetches corpus, extracts, deduplicates, builds graph)
python pipeline.py

# 5. Or skip fetch/extraction if outputs already exist:
python pipeline.py --skip-fetch              # reuse cached corpus
python pipeline.py --skip-fetch --skip-extract  # reuse cached extractions
python pipeline.py --max-issues 20           # limit to 20 issues for quick test
```

The pipeline runs 7 steps:
1. **Fetch** corpus from GitHub REST API (facebook/react issues + comments)
2. **Deduplicate** raw artifacts (exact + near-duplicate detection)
3. **Extract** entities, claims, and evidence via Gemini 2.5 Flash
4. **Canonicalize** entities (merge duplicates, normalize names)
5. **Deduplicate** claims (content + semantic matching, conflict detection)
6. **Build** memory graph (NetworkX + SQLite)
7. **Index** embeddings and run example retrieval queries

---

### Launching the Visualization (Streamlit)

```bash
# From the project root directory:
python -m streamlit run visualization/app.py

# Or with a custom port:
python -m streamlit run visualization/app.py --server.port 8505
```

Then open **http://localhost:8501** (or your custom port) in a browser.

The app has **6 interactive pages**:

| Page | What it shows |
|---|---|
| **Dashboard** | KPI cards (entity/claim counts), type breakdowns, graph topology stats, top connected entities |
| **Graph Explorer** | Interactive PyVis graph — filter by entity type, confidence, temporal status. Click nodes for details. |
| **Question Retrieval** | Type a natural language question, get grounded context packs with evidence citations |
| **Entity Browser** | Search and browse all entities with linked claims and merge history |
| **Claim Browser** | Search and filter claims by type/status, inline evidence snippets |
| **Merge Audit** | Full dedup/merge history, grouped by action type, inspectable details |

---

### Launching the REST API (FastAPI)

```bash
# From the project root directory:
uvicorn retrieval.api:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/retrieve` | POST | Retrieve grounded context pack for a question |
| `/entities` | GET | List all entities (with optional search) |
| `/entities/{id}` | GET | Get entity details + linked claims |
| `/claims` | GET | List all claims (with optional type filter) |
| `/claims/{id}/evidence` | GET | Get evidence for a specific claim |
| `/merges` | GET | Get merge/dedup audit log |
| `/stats` | GET | Get graph statistics |

---

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Corpus     │────▶│  Extraction   │────▶│   Dedup &     │────▶│   Memory    │
│  (GitHub)    │     │  (Gemini LLM) │     │  Canon.       │     │   Graph     │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬──────┘
                                                                      │
                                                          ┌───────────┼───────────┐
                                                          ▼           ▼           ▼
                                                    ┌──────────┐ ┌──────────┐ ┌──────────┐
                                                    │ Retrieval│ │ Vis. UI  │ │ REST API │
                                                    │ Engine   │ │(Streamlit)│ │(FastAPI) │
                                                    └──────────┘ └──────────┘ └──────────┘
```

### Project Structure

| Component | Description |
|---|---|
| `config.py` | Central configuration (API keys, paths, model settings) |
| `pipeline.py` | End-to-end orchestrator (7 steps, CLI flags) |
| `corpus/` | GitHub REST API fetcher with rate-limit handling |
| `extraction/` | Gemini-based structured extraction + schema + JSON validation/repair |
| `dedup/` | 3-level dedup: artifact, entity canonicalization, claim dedup with conflict detection |
| `graph/` | NetworkX memory graph + SQLite persistence (entities, claims, evidence, merge log) |
| `retrieval/` | Hybrid keyword+semantic retrieval engine + FastAPI REST endpoints |
| `visualization/` | 6-page Streamlit app with PyVis graph, evidence panel, merge audit |
| `data/graph/` | Generated outputs: `memory.db`, `memory_graph.json`, `example_context_packs.json` |
| `WRITEUP.md` | Full design writeup (ontology, extraction, dedup, graph, retrieval, Layer10 adaptation) |

---

### Pre-built Outputs

The repo includes pre-built outputs from a run on 20 React issues (1700+ comments):

- **`data/graph/memory.db`** — SQLite database with 287 entities, 391 claims, evidence, and merge log
- **`data/graph/memory_graph.json`** — Serialized NetworkX graph (326 nodes, 266 edges)
- **`data/graph/example_context_packs.json`** — 5 example retrieved context packs with evidence
- **`data/graph/entity_embeddings.npz`** / **`claim_embeddings.npz`** — Precomputed embeddings for retrieval

You can launch the visualization immediately without re-running the pipeline.

---

See [WRITEUP.md](WRITEUP.md) for the full design documentation covering ontology, extraction, deduplication, graph design, retrieval, visualization, and Layer10 adaptation.
