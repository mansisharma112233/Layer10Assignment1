# Layer10 Memory Graph

## Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

Built on **GitHub Issues/PRs from facebook/react** as the public corpus.

### Quick Start

```bash
# 1. Clone and install
cd layer10-memory
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (free from https://aistudio.google.com/app/apikey)
# Optionally add GITHUB_TOKEN for higher rate limits

# 3. Run the full pipeline
python pipeline.py

# 4. Launch visualization
streamlit run visualization/app.py

# 5. Launch retrieval API (optional)
uvicorn retrieval.api:app --reload
```

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

### Components

| Component | Description |
|---|---|
| `corpus/` | Fetches GitHub issues, PRs, and comments via GitHub REST API |
| `extraction/` | LLM-based structured extraction with schema validation |
| `dedup/` | Artifact, entity, and claim deduplication + canonicalization |
| `graph/` | NetworkX-based memory graph with SQLite persistence |
| `retrieval/` | Hybrid retrieval (keyword + embedding) with grounded citations |
| `visualization/` | Streamlit app with interactive graph explorer + evidence panel |

See [WRITEUP.md](WRITEUP.md) for full design documentation.
