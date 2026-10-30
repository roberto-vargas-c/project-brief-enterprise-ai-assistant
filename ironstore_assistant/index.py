"""Embeddings + FAISS vector store (build once, load at query time).

FAISS (used in the course's generative-agents notebook) runs locally and needs no account,
which is the right size for ~4k chunks. Swapping it for Pinecone (notebook 03) only means
changing ``build_index``/``load_index``.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import config


class TfidfEmbeddings(Embeddings):
    """Offline embeddings with scikit-learn's TF-IDF + SVD (LSA).

    Not as good as OpenAI embeddings, but it needs no API key or model download, so the
    index, the tests and the retrieval evaluation can run anywhere.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.pipeline = None

    def fit(self, texts: list[str]) -> "TfidfEmbeddings":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        dim = min(self.dim, len(texts) - 1)
        self.pipeline = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=40_000),
            TruncatedSVD(n_components=dim, random_state=0),
        ).fit(texts)
        return self

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.pipeline.transform(texts)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs.astype("float32").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings(provider: str | None = None, index_dir: Path | None = None) -> Embeddings:
    provider = (provider or config.EMBEDDINGS_PROVIDER).lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL,
                                     encode_kwargs={"normalize_embeddings": True})
    if provider == "tfidf":
        path = (index_dir or config.INDEX_DIR) / "tfidf.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        return TfidfEmbeddings()
    raise ValueError(f"Unknown embeddings provider: {provider!r} (use openai, huggingface or tfidf)")


def build_index(documents: list[Document], index_dir: Path | None = None,
                provider: str | None = None, batch_size: int = 500) -> FAISS:
    index_dir = index_dir or config.INDEX_DIR
    provider = (provider or config.EMBEDDINGS_PROVIDER).lower()
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(provider, index_dir=index_dir / "__none__")
    if isinstance(embeddings, TfidfEmbeddings):
        embeddings.fit([d.page_content for d in documents])
        with (index_dir / "tfidf.pkl").open("wb") as f:
            pickle.dump(embeddings, f)

    # Embed in batches to stay below API request limits
    store = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if store is None:
            store = FAISS.from_documents(batch, embeddings, distance_strategy="MAX_INNER_PRODUCT")
        else:
            store.add_documents(batch)
        print(f"  embedded {min(i + batch_size, len(documents))}/{len(documents)} chunks")
    store.save_local(str(index_dir))
    (index_dir / "manifest.json").write_text(json.dumps({
        "provider": provider,
        "model": {"openai": config.OPENAI_EMBEDDING_MODEL, "huggingface": config.HF_EMBEDDING_MODEL}.get(provider, "tfidf"),
        "chunks": len(documents),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_version": config.INDEX_VERSION,
    }, indent=2))
    return store


def load_index(index_dir: Path | None = None) -> FAISS:
    index_dir = index_dir or config.INDEX_DIR
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Build it first with: python -m ironstore_assistant.build_index")
    provider = json.loads(manifest_path.read_text())["provider"]
    embeddings = get_embeddings(provider, index_dir=index_dir)
    # The index is a pickle we wrote ourselves in build_index, so loading it is safe.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True,
                            distance_strategy="MAX_INNER_PRODUCT")
