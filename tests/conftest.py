"""Shared fixtures. Tests run fully offline: TF-IDF embeddings and a scripted fake LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from ironstore_assistant import config
from ironstore_assistant.ingest import extract_zip, ingest


class FakeLLM(BaseChatModel):
    """Returns scripted replies. `structured` is what with_structured_output() returns,
    `tool_script` is a list of AIMessages returned one by one by the tool-bound model."""

    text: str = "rewritten question"
    structured: Any = None
    tool_script: list = []
    seen_prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_prompts.append(messages)
        if self.tool_script:
            return ChatResult(generations=[ChatGeneration(message=self.tool_script.pop(0))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.text))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(prompt_value):
            self.seen_prompts.append(prompt_value)
            return self.structured
        return RunnableLambda(run)


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
