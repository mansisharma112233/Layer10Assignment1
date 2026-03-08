"""
Configuration module for Layer10 Memory Graph.
Loads settings from .env file and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
GRAPH_DIR = DATA_DIR / "graph"

for d in [RAW_DIR, EXTRACTED_DIR, GRAPH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

GITHUB_REPO = os.getenv("GITHUB_REPO", "facebook/react")
MAX_ISSUES = int(os.getenv("MAX_ISSUES", "200"))

EXTRACTION_BATCH_SIZE = int(os.getenv("EXTRACTION_BATCH_SIZE", "5"))
EXTRACTION_MODEL = "gemini-2.5-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DB_PATH = GRAPH_DIR / "memory.db"
GRAPH_PATH = GRAPH_DIR / "memory_graph.json"
