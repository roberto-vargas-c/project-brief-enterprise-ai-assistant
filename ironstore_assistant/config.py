"""Central configuration, read from environment variables (or a .env file).

Everything that you may want to tweak without touching code lives here.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT_DIR = Path(__file__).resolve().parent.parent

# Raw data: the zip shipped with the brief (or an already-extracted folder)
DOCUMENTS_ZIP = Path(os.getenv("IRONSTORE_DOCUMENTS_ZIP", ROOT_DIR / "documents" / "documents.zip"))

# Where ingestion writes the extracted PDFs, the FAISS index and the reports
DATA_DIR = Path(os.getenv("IRONSTORE_DATA_DIR", ROOT_DIR / "data"))
EXTRACTED_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "faiss_index"
PRODUCTS_CSV = DATA_DIR / "products.csv"
# Bump when ingestion/chunking changes, so an existing index (e.g. on a deployed app) is rebuilt.
# v2: chunks span page breaks.
INDEX_VERSION = 2
QUALITY_REPORT = DATA_DIR / "data_quality_report.md"

# LLM. The course notebooks use OpenAI chat models through langchain_openai.ChatOpenAI.
CHAT_MODEL = os.getenv("IRONSTORE_CHAT_MODEL", "gpt-6-luna")
# Only sent when set: some newer models accept only their default temperature.
TEMPERATURE = os.getenv("IRONSTORE_TEMPERATURE")

# Embeddings: "openai" (default), "huggingface" (local sentence-transformers, no key needed)
# or "tfidf" (pure scikit-learn, offline, used for tests and quick demos).
EMBEDDINGS_PROVIDER = os.getenv("IRONSTORE_EMBEDDINGS", "openai").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("IRONSTORE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBEDDING_MODEL = os.getenv("IRONSTORE_HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking for policy documents (catalogs are chunked one product per chunk)
CHUNK_SIZE = int(os.getenv("IRONSTORE_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("IRONSTORE_CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("IRONSTORE_TOP_K", "8"))
FETCH_K = int(os.getenv("IRONSTORE_FETCH_K", "25"))
