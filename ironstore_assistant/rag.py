"""The assistant: retrieval-augmented generation with citations and confidence.

Pipeline for every question (all LangChain runnables, as in N_LangChain/03 and 04):

1. **Condense** the follow-up question with the chat history into a standalone question
   (conversation memory, so "and for contractors?" works).
2. **Retrieve** from FAISS, optionally with **multi-query** expansion (N_LangChain/11),
   skipping OUTDATED documents unless the user asks about old data, and boosting exact SKU
   matches.
3. **Generate** a structured answer with Pydantic (L_LLM_APIs/6): answer text with [n]
   markers, cited passage numbers, a confidence level and an ``answerable`` flag.
4. **Validate**: citations must point to passages that were really retrieved; they are
   turned into "document, page" sources. An answer without valid citations is downgraded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from . import config
from .prompts import ANSWER_PROMPT, CONDENSE_PROMPT, MULTI_QUERY_PROMPT

SKU_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+\b")
OLD_DATA_RE = re.compile(r"\b(20[01]\d|202[0-4]|old|older|outdated|previous|historic(al)?|former|used to)\b", re.I)
NOT_FOUND = ("I couldn't find this in IronStore's internal documents, so I'd rather not guess. "
             "Try rephrasing, or ask the team that owns the topic.")


class StructuredAnswer(BaseModel):
    """Schema the LLM must fill (structured output)."""

    answer: str = Field(description="Answer for the employee, with [n] citation markers after each fact.")
    citations: list[int] = Field(default_factory=list, description="Numbers of the passages used.")
    confidence: Literal["high", "medium", "low"] = Field(description="How well the passages support the answer.")
    answerable: bool = Field(description="False if the passages do not contain the answer.")


@dataclass
class Source:
    number: int
    title: str
    source: str  # file name
    page: int
    doc_id: str = ""
    status: str = "current"
    last_reviewed: str = ""
    snippet: str = ""

    def label(self) -> str:
        extra = " — OUTDATED" if self.status == "outdated" else ""
        return f"{self.title} (p. {self.page}){extra}"


@dataclass
class AssistantResponse:
    answer: str
    confidence: str
    answerable: bool
    sources: list[Source] = field(default_factory=list)
    standalone_question: str = ""
    queries: list[str] = field(default_factory=list)
    retrieved: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        header = f"[{i}] {m['title']}, page {m['page']}"
        if m.get("doc_id"):
            header += f" (ID {m['doc_id']}{' — shared with another document' if m.get('id_conflict') else ''})"
        if m.get("last_reviewed"):
            header += f", last reviewed {m['last_reviewed']}"
        if m.get("status") == "outdated":
            header += " — OUTDATED"
        body = d.page_content.split("\n", 1)[-1]  # drop the chunk label, the header says it all
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def to_messages(history: list[tuple[str, str]] | None, window: int = 6) -> list[BaseMessage]:
    """(role, text) pairs -> LangChain messages, keeping the last `window` turns
    (the idea of ConversationBufferWindowMemory from N_LangChain/04)."""
    msgs: list[BaseMessage] = []
    for role, text in (history or [])[-window * 2:]:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return msgs


class Retriever:
    """FAISS search + metadata filters + multi-query + exact SKU boost."""

    def __init__(self, store, products: pd.DataFrame | None = None):
        self.store = store
        self.products = products if products is not None else pd.DataFrame()
        self._by_sku: dict[str, list[Document]] = {}
        for doc in store.docstore._dict.values():
            if sku := doc.metadata.get("sku"):
                self._by_sku.setdefault(sku, []).append(doc)

    def docs_for_sku(self, sku: str, include_outdated: bool = True) -> list[Document]:
        return [d for d in self._by_sku.get(sku, [])
                if include_outdated or d.metadata.get("status") != "outdated"]

    def search(self, queries: list[str], k: int = config.TOP_K, include_outdated: bool = False,
               doc_type: str | None = None, area: str | None = None) -> list[Document]:
        def keep(md: dict) -> bool:
            return ((include_outdated or md.get("status") != "outdated")
                    and (doc_type is None or md.get("doc_type") == doc_type)
                    and (area is None or md.get("area") == area))

        # Reciprocal-rank fusion across the queries (what MultiQueryRetriever does, plus ranking)
        scores: dict[int, float] = {}
        docs: dict[int, Document] = {}
        for q in queries:
            hits = self.store.similarity_search_with_score(q, k=k * 2, fetch_k=config.FETCH_K * 4, filter=keep)
            for rank, (d, _) in enumerate(hits):
                cid = d.metadata["chunk_id"]
                docs[cid] = d
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)
        result = [docs[c] for c in ranked[:k]]

        # Exact SKUs typed by the user always make it into the context
        for sku in {s for q in queries for s in SKU_RE.findall(q)}:
            for d in reversed(self.docs_for_sku(sku, include_outdated)):
                if d not in result:
                    result.insert(0, d)
        return result


class Assistant:
    def __init__(self, store, llm: BaseChatModel, products: pd.DataFrame | None = None,
                 multi_query: bool = True, n_queries: int = 3):
        self.llm = llm
        self.retriever = Retriever(store, products)
        self.multi_query = multi_query
        self.n_queries = n_queries
        self.condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        self.multi_query_chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | llm.with_structured_output(StructuredAnswer)

    # -- steps -------------------------------------------------------------------------
    def condense(self, question: str, history: list[BaseMessage]) -> str:
        if not history:
            return question
        return self.condense_chain.invoke({"question": question, "history": history}).strip() or question

    def expand(self, question: str) -> list[str]:
        if not self.multi_query:
            return [question]
        raw = self.multi_query_chain.invoke({"question": question, "n": self.n_queries})
        extra = [q.strip(" -•\t") for q in raw.splitlines() if q.strip()]
        return [question] + extra[: self.n_queries]

    def generate(self, question: str, docs: list[Document], history: list[BaseMessage]) -> StructuredAnswer:
        return self.answer_chain.invoke({"question": question, "context": format_context(docs), "history": history})

    # -- public API --------------------------------------------------------------------
    def ask(self, question: str, history: list[tuple[str, str]] | None = None,
            include_outdated: bool | None = None, area: str | None = None,
            retrieve: Callable[[str, list[BaseMessage]], list[Document]] | None = None) -> AssistantResponse:
        """Answer a question. `retrieve` lets the agent supply its own evidence."""
        msgs = to_messages(history)
        standalone = self.condense(question, msgs)
        if include_outdated is None:
            include_outdated = bool(OLD_DATA_RE.search(standalone))

        queries = [standalone]
        if retrieve is not None:
            docs = retrieve(standalone, msgs)
        else:
            queries = self.expand(standalone)
            docs = self.retriever.search(queries, include_outdated=include_outdated, area=area)

        if not docs:
            return AssistantResponse(NOT_FOUND, "low", False, standalone_question=standalone, queries=queries)

        out = self.generate(standalone, docs, msgs)
        return self._finalise(out, docs, standalone, queries)

    def _finalise(self, out: StructuredAnswer, docs: list[Document], standalone: str,
                  queries: list[str]) -> AssistantResponse:
        resp = AssistantResponse(answer=out.answer.strip(), confidence=out.confidence, answerable=out.answerable,
                                 standalone_question=standalone, queries=queries, retrieved=docs)
        cited = sorted({n for n in out.citations if 1 <= n <= len(docs)}
                       | {int(n) for n in re.findall(r"\[(\d+)\]", out.answer) if 1 <= int(n) <= len(docs)})
        invalid = sorted({n for n in out.citations if not 1 <= n <= len(docs)})
        if invalid:
            resp.warnings.append(f"Ignored citations to passages that do not exist: {invalid}")
        for n in cited:
            m = docs[n - 1].metadata
            resp.sources.append(Source(number=n, title=m["title"], source=m["source"], page=m["page"],
                                       doc_id=m.get("doc_id", ""), status=m.get("status", "current"),
                                       last_reviewed=m.get("last_reviewed", ""),
                                       snippet=docs[n - 1].page_content.split("\n", 1)[-1][:600]))

        if not out.answerable:
            resp.confidence = "low"
        elif not resp.sources:
            resp.confidence = "low"
            resp.warnings.append("The answer has no valid citation, so treat it with caution.")
        if any(s.status == "outdated" for s in resp.sources):
            resp.warnings.append("Part of this answer comes from an OUTDATED document (2021 catalog). "
                                 "Check the current catalogs before using it.")
        return resp


def load_products() -> pd.DataFrame:
    if config.PRODUCTS_CSV.exists():
        return pd.read_csv(config.PRODUCTS_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame()


def get_llm(model: str | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs = {"temperature": float(config.TEMPERATURE)} if config.TEMPERATURE else {}
    return ChatOpenAI(model=model or config.CHAT_MODEL, **kwargs)


def build_assistant(multi_query: bool = True, llm: BaseChatModel | None = None) -> Assistant:
    from .build_index import ensure_index
    from .index import load_index

    ensure_index()
    return Assistant(load_index(), llm or get_llm(), load_products(), multi_query=multi_query)
