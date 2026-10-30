"""Shared fixtures. Tests run fully offline: TF-IDF embeddings and the scripted FakeLLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from ironstore_assistant import config
from ironstore_assistant.ingest import extract_zip, ingest


@pytest.fixture(scope="session")
def pdf_paths(tmp_path_factory) -> list[Path]:
    return extract_zip(config.DOCUMENTS_ZIP, tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def corpus(pdf_paths):
    return ingest(pdf_paths)


@pytest.fixture(scope="session")
def store(corpus, tmp_path_factory):
    from ironstore_assistant.index import build_index, load_index

    index_dir = tmp_path_factory.mktemp("index")
    build_index(corpus.documents, index_dir=index_dir, provider="tfidf")
    return load_index(index_dir)
