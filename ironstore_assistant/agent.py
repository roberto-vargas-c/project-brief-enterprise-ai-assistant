"""Bonus: a tool-calling agent that decides *where* to look before answering.

Plain RAG embeds the question once. That struggles with exact lookups ("price of SKU
FUR-SOF-NOR-LYSA-GRY") and with questions that mix topics ("can I expense the Lysa sofa
for my home office?"). Here the LLM gets tools (see L_LLM_APIs/5_tool_calling and
N_LangChain/10_langchain-building-tools) and calls as many as it needs. Everything the
tools return is collected and handed to the same grounded, cited answer step as the plain
RAG mode, so answers keep the same citations, confidence and validation.

The loop is written by hand (bind_tools -> run tool calls -> ToolMessage) like in the
tool-calling notebook, so every step is visible and easy to debug.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .prompts import AGENT_SYSTEM_PROMPT
from .rag import OLD_DATA_RE, Assistant, AssistantResponse, Retriever


def make_tools(retriever: Retriever, collected: list[Document], include_outdated: bool = False):
    """Tools close over `collected`, where every piece of evidence they return is stored."""

    def _remember(docs: list[Document]) -> str:
        if not docs:
            return "No matching documents."
        lines = []
        for d in docs:
            if d not in collected:
                collected.append(d)
            m = d.metadata
            lines.append(f"- {m['title']} p.{m['page']}: {d.page_content.split(chr(10), 1)[-1][:400]}")
        return "\n".join(lines)

    @tool
    def search_policies(query: str, area: str = "") -> str:
        """Search IronStore internal policies and procedures (HR, Finance, IT, Legal, Sales,
        Customer support, Deliveries). `area` optionally restricts to one department."""
        areas = {d.metadata["area"].lower(): d.metadata["area"] for d in retriever.store.docstore._dict.values()
                 if d.metadata["doc_type"] == "policy"}
        docs = retriever.search([query], k=5, doc_type="policy", area=areas.get(area.lower()) if area else None)
        return _remember(docs)

    @tool
    def search_products(query: str) -> str:
        """Semantic search over the current product catalogs (descriptions, specs, categories)."""
        return _remember(retriever.search([query], k=5, doc_type="catalog", include_outdated=include_outdated))

    @tool
    def lookup_product(sku_or_name: str) -> str:
        """Exact lookup of a product by SKU (e.g. FUR-SOF-NOR-LYSA-GRY) or by (part of) its
        name. Returns price, availability, warehouse and the catalog page."""
        df = retriever.products
        if df.empty:
            return "Product table not available."
        key = sku_or_name.strip()
        hits = df[df.sku.str.upper() == key.upper()]
        if hits.empty:
            hits = df[df.name.str.contains(key, case=False, regex=False)]
        if not include_outdated:
            hits = hits[hits.status != "outdated"]
        if hits.empty:
            return f"No product found for {key!r}."
        hits = hits.head(5)
        docs = [d for r in hits.itertuples() for d in retriever.docs_for_sku(r.sku, include_outdated)
                if d.metadata["source"] == r.source]
        note = ""
        if hits.sku.duplicated().any():
            note = ("WARNING: this SKU is used by different products in different catalogs; "
                    "tell the user and show both.\n")
        return note + _remember(docs)

    @tool
    def list_documents(area: str = "") -> str:
        """List the available internal documents (title, ID, department, last review)."""
        seen = {}
        for d in retriever.store.docstore._dict.values():
            m = d.metadata
            if m["doc_type"] == "policy" and (not area or m["area"].lower() == area.lower()):
                seen[m["source"]] = f"- {m['title']} (ID {m['doc_id'] or 'n/a'}, {m['area']}, last reviewed {m['last_reviewed']})"
        return "\n".join(sorted(seen.values())) or "No documents for that area."

    return [search_policies, search_products, lookup_product, list_documents]


class AgentAssistant:
    """Same interface as Assistant.ask, but evidence is gathered by a tool-calling LLM."""

    def __init__(self, assistant: Assistant, max_steps: int = 4):
        self.assistant = assistant
        self.max_steps = max_steps
        self.last_trace: list[str] = []

    def ask(self, question: str, history: list[tuple[str, str]] | None = None,
            include_outdated: bool | None = None, area: str | None = None,
            on_step: Callable[[str, dict], None] | None = None) -> AssistantResponse:
        self.last_trace = []
        step = on_step or (lambda name, data: None)

        def gather(standalone: str, msgs: list[BaseMessage]) -> list[Document]:
            collected: list[Document] = []
            old = include_outdated if include_outdated is not None else bool(OLD_DATA_RE.search(standalone))
            tools = make_tools(self.assistant.retriever, collected, old)
            by_name = {t.name: t for t in tools}
            llm = self.assistant.llm.bind_tools(tools)
            messages: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(standalone)]
            for _ in range(self.max_steps):
                ai = llm.invoke(messages)
                messages.append(ai)
                if not getattr(ai, "tool_calls", None):
                    break
                for call in ai.tool_calls:
                    self.last_trace.append(f"{call['name']}({call['args']})")
                    fn = by_name.get(call["name"])
                    result = fn.invoke(call["args"]) if fn else f"Unknown tool {call['name']}"
                    step("tool", {"name": call["name"], "args": call["args"], "result": str(result)})
                    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            if not collected:  # the model answered without tools: fall back to plain retrieval
                collected = self.assistant.retriever.search([standalone], include_outdated=old)
            return collected[:10]

        resp = self.assistant.ask(question, history, include_outdated=include_outdated, area=area, retrieve=gather,
                                  on_step=on_step)
        resp.queries = self.last_trace or resp.queries
        return resp
