"""Animated, step-by-step view of how the assistant finds an answer (used by app.py).

Each stage of the RAG pipeline reports what it did through ``Assistant.ask(on_step=...)``,
and this module draws it live: the rewritten queries, what semantic and keyword search
found, how rank fusion merged them, where the question lands on a map of all chunks,
which passages went to the model, and which ones it actually cited.
"""

from __future__ import annotations

import html
import time
from collections import defaultdict

import streamlit as st

from ironstore_assistant.rag import Retriever, RetrievalTrace
from live_view import replay_map
from live_view import doc_color as _color
from live_view import doc_label as _doc_label

METHOD_BADGE = {"semantic": "meaning", "keyword": "keywords", "sku": "exact SKU"}

CSS = """
<style>
@keyframes grow { from { width: 0; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 0 rgba(52,211,153,.0); } 50% { box-shadow: 0 0 0 4px rgba(52,211,153,.35); } }
.pv-row { display: grid; grid-template-columns: 1.6rem 1fr 7rem; gap: .5rem; align-items: center;
          font-size: .85rem; margin: .18rem 0; animation: fadeUp .4s ease both; }
.pv-rank { color: #8b97b5; text-align: right; font-variant-numeric: tabular-nums; }
.pv-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pv-track { background: rgba(139,151,181,.14); border-radius: 999px; height: .55rem; overflow: hidden; }
.pv-bar { height: 100%; border-radius: 999px; animation: grow .9s cubic-bezier(.2,.8,.2,1) both; }
.pv-card { background: rgba(10,16,32,.55); border: 1px solid rgba(139,151,181,.18); border-left: 5px solid var(--c); border-radius: .6rem;
           padding: .55rem .75rem; margin: .4rem 0; animation: fadeUp .45s ease both; font-size: .85rem; }
.pv-card.cited { border-color: #34d399; border-left-color: #34d399; background: rgba(52,211,153,.07); animation: fadeUp .45s ease both, glow 1.6s ease 2; }
.pv-card.unused { opacity: .55; }
.pv-card .meta { color: #8b97b5; font-size: .78rem; margin-top: .15rem; }
.pv-card .snip { margin-top: .3rem; color: inherit; opacity: .85; }
.pv-badge { display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 999px;
            background: rgba(139,92,246,.18); color: #ddd6fe; margin-left: .25rem; }
.pv-query { display: inline-block; padding: .25rem .6rem; margin: .2rem .25rem .2rem 0; border-radius: 999px;
            background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(34,211,238,.12));
            border: 1px solid rgba(139,92,246,.5); font-size: .85rem;
            animation: fadeUp .4s ease both; }
</style>
"""


def _pause(speed: float, seconds: float = 0.25) -> None:
    if speed:
        time.sleep(seconds * speed)


# -- small HTML widgets ----------------------------------------------------------------------
def score_bars(rows: list[tuple[str, float, str]], max_score: float) -> str:
    """rows: (label, score, color) -> animated horizontal bars."""
    out = [CSS]
    for i, (label, score, color) in enumerate(rows, start=1):
        pct = 100 * score / max_score if max_score else 0
        out.append(f'<div class="pv-row" style="animation-delay:{i * .07:.2f}s"><span class="pv-rank">{i}</span>'
                   f'<span class="pv-label" title="{html.escape(label)}">{html.escape(label)}</span>'
                   f'<span class="pv-track"><span class="pv-bar" style="display:block;width:{pct:.0f}%;'
                   f'background:{color};animation-delay:{i * .07:.2f}s"></span></span></div>')
    return "".join(out)


def passage_cards(docs, trace: RetrievalTrace, cited_numbers: set[int] | None = None) -> str:
    found_by = defaultdict(set)
    for h in trace.hits:
        found_by[h.chunk_id].add(h.method)
    fused = dict(trace.fused)
    out = [CSS]
    for n, d in enumerate(docs, start=1):
        cid = d.metadata.get("chunk_id")
        state = "" if cited_numbers is None else ("cited" if n in cited_numbers else "unused")
        badges = "".join(f'<span class="pv-badge">{METHOD_BADGE[m]}</span>' for m in sorted(found_by.get(cid, [])))
        if cited_numbers is not None:
            badges += '<span class="pv-badge">cited</span>' if n in cited_numbers else \
                '<span class="pv-badge">not used</span>'
        snippet = html.escape(d.page_content.split("\n", 1)[-1][:220]) + "…"
        score = f" · fusion score {fused[cid]:.4f}" if cid in fused else ""
        out.append(f'<div class="pv-card {state}" style="--c:{_color(d)};animation-delay:{n * .08:.2f}s">'
                   f'<b>[{n}]</b> {html.escape(_doc_label(d))} {badges}'
                   f'<div class="meta">{html.escape(d.metadata.get("area", ""))}{score}</div>'
                   f'<div class="snip">{snippet}</div></div>')
    return "".join(out)


# -- live pipeline ---------------------------------------------------------------------------
class PipelineView:
    """Callback for Assistant.ask(on_step=...) that renders each stage as it happens."""

    def __init__(self, retriever: Retriever, speed: float = 1.0, show_map: bool = True):
        self.retriever = retriever
        self.speed = speed
        self.show_map = show_map
        self.docs = []
        self.trace: RetrievalTrace | None = None
        self.question = ""
        self._gen_status = None
        self._tools = None
        self._map_slot = None

    def __call__(self, name: str, data: dict) -> None:
        getattr(self, f"on_{name}", lambda d: None)(data)

    def on_condense(self, d: dict) -> None:
        self.question = d["standalone"]
        with st.status("Understanding your question", expanded=True) as s:
            if d["standalone"] != d["question"]:
                st.markdown(f"Using the last {d['history_turns']} turn(s) of our chat, I rewrote your follow-up "
                            f"as a standalone question:")
                st.markdown(f"> ~~{d['question']}~~  \n> **{d['standalone']}**")
            else:
                st.markdown(f"> **{d['question']}**")
            _pause(self.speed, .4)
            s.update(label="Understood your question", state="complete", expanded=False)

    def on_expand(self, d: dict) -> None:
        queries = d["queries"]
        with st.status("Rewording the question to search better", expanded=True) as s:
            st.caption("Employees and policies rarely use the same words, so I search with several versions "
                       "(multi-query).")
            slot = st.empty()
            shown = []
            for q in queries:
                shown.append(f'<span class="pv-query">{html.escape(q)}</span>')
                slot.markdown(CSS + "".join(shown), unsafe_allow_html=True)
                _pause(self.speed, .35)
            s.update(label=f"Searching with {len(queries)} versions of the question", state="complete",
                     expanded=False)

    def on_tool(self, d: dict) -> None:
        if self._tools is None:
            self._tools = st.status("The agent is choosing tools", expanded=True)
        with self._tools:
            st.markdown(f"**{d['name']}**(`{', '.join(f'{k}={v!r}' for k, v in d['args'].items())}`)")
            st.caption(d["result"][:300] + ("…" if len(d["result"]) > 300 else ""))
        _pause(self.speed, .4)

    def on_retrieve(self, d: dict) -> None:
        self.docs, self.trace = d["docs"], d["trace"]
        if self._tools is not None:
            self._tools.update(label="The agent gathered its evidence", state="complete", expanded=False)
        n_chunks = len(self.retriever.docs)
        with st.status(f"Searching {n_chunks:,} chunks", expanded=True) as s:
            hits = self.trace.hits
            if hits and any(h.method in ("semantic", "keyword") for h in hits):
                st.caption("Two searches per query: **meaning** (embeddings in FAISS) finds paraphrases, "
                           "**keywords** (TF-IDF) finds exact words like *30 days* or a product name.")
                first_q = self.trace.queries[0]
                cols = st.columns(2)
                for col, method in zip(cols, ("semantic", "keyword")):
                    rows = [h for h in hits if h.query == first_q and h.method == method][:5]
                    with col:
                        st.markdown(f"**{METHOD_BADGE[method]}** · top 5 for the original question")
                        top = max((h.score for h in rows), default=1)
                        st.markdown(score_bars([(_doc_label(self.retriever.docs[h.chunk_id]), h.score,
                                                 _color(self.retriever.docs[h.chunk_id])) for h in rows], top),
                                    unsafe_allow_html=True)
                    _pause(self.speed, .5)
                if self.trace.excluded_outdated:
                    st.caption("The outdated 2021 catalog was filtered out.")
            s.update(label=f"Searched {n_chunks:,} chunks with {len(self.trace.queries)} queries",
                     state="complete", expanded=False)

        if self.trace.fused:
            with st.status("Merging the results (rank fusion)", expanded=True) as s:
                st.caption("Each search votes for its results: 1/(60 + rank) points per vote. Chunks found by "
                           "several queries and by both methods win. Max 3 chunks per document, for variety.")
                chosen = set(self.trace.selected)
                top = self.trace.fused[:10]
                rows = []
                for cid, score in top:
                    doc = self.retriever.docs[cid]
                    tag = "" if cid in chosen else "  (skipped: same document)" \
                        if cid in self.trace.dropped_for_variety else "  (not in top 8)"
                    rows.append((_doc_label(doc) + tag, score, _color(doc) if cid in chosen else "#334155"))
                st.markdown(score_bars(rows, top[0][1]), unsafe_allow_html=True)
                _pause(self.speed, .9)
                s.update(label=f"Picked the {len(self.docs)} best passages", state="complete", expanded=False)

        if self.show_map and self.trace.selected:
            with st.status("Where your question lands", expanded=True) as s:
                st.caption("Every dot is a chunk, placed in 3D by its embedding (PCA). Close dots talk about "
                           "similar things. The yellow Q is your question; the numbers are the passages I picked. "
                           "When the animation ends, drag to rotate, scroll to zoom, hover, click and go full screen.")
                self._map_slot = st.empty()
                with self._map_slot.container():
                    replay_map(self.retriever, self.question, self.trace, self.docs, speed=self.speed or 1)
                _pause(self.speed, 3.5)
                s.update(label="Your question on the knowledge map", state="complete", expanded=True)

        with st.status(f"Reading {len(self.docs)} passages", expanded=True) as s:
            st.markdown(passage_cards(self.docs, self.trace), unsafe_allow_html=True)
            _pause(self.speed, .8)
            s.update(label=f"Sent {len(self.docs)} numbered passages to the model", state="complete",
                     expanded=False)

    def on_generate(self, d: dict) -> None:
        self._gen_status = st.status("Writing an answer that cites its passages…", expanded=False)

    def on_validate(self, d: dict) -> None:
        resp = d["response"]
        cited = {s.number for s in resp.sources}
        with self._gen_status:
            st.markdown(f"The model cited **{len(cited)}** of {len(self.docs)} passages and rated its confidence "
                        f"**{resp.confidence}**. I checked every citation points to a real passage.")
            st.markdown(passage_cards(self.docs, self.trace, cited), unsafe_allow_html=True)
        self._gen_status.update(label=f"Answer written and citations checked ({len(cited)} cited)",
                                state="complete", expanded=False)
        if self._map_slot is not None and cited:
            with self._map_slot.container():  # replay the map with the cited passages in green
                replay_map(self.retriever, self.question, self.trace, self.docs, resp, speed=.35)


def typewriter(text: str, speed: float):
    """Stream the answer word by word (st.write_stream)."""
    for word in text.split(" "):
        yield word + " "
        if speed:
            time.sleep(0.018 * speed)
