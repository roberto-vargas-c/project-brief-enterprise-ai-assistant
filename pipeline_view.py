"""Animated, step-by-step view of how the assistant finds an answer (used by app.py).

Each stage of the RAG pipeline reports what it did through ``Assistant.ask(on_step=...)``,
and this module draws it live: the rewritten queries, what semantic and keyword search
found, how rank fusion merged them, where the question lands on a map of all chunks,
which passages went to the model, and which ones it actually cited.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

from ironstore_assistant.rag import Retriever, RetrievalTrace, page_label

AREA_COLORS = {
    "Customer support": "#fb7185", "Deliveries": "#fbbf24", "Finance": "#2dd4bf", "HR policies": "#a78bfa",
    "IT": "#60a5fa", "Legal Terms": "#f472b6", "Sales": "#4ade80", "Product catalog": "#94a3b8",
}
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


def _doc_label(d) -> str:
    m = d.metadata
    name = m.get("product_name") or m["title"].replace("IronStore Product Catalog — ", "Catalog: ")
    return f"{name} · {page_label(m['page'], m.get('page_end'))}"


def _color(d) -> str:
    return AREA_COLORS.get(d.metadata.get("area", ""), "#94a3b8")


# -- knowledge map ---------------------------------------------------------------------------
MAP_TEMPLATE = Path(__file__).parent / "assets" / "knowledge_map.html"
MAP_AREAS = list(AREA_COLORS) + ["Other"]


@st.cache_resource(show_spinner="Drawing the knowledge map…")
def _map_data(_retriever: Retriever, retriever_id: int, index_size: int):
    """2D PCA projection of every chunk vector stored in FAISS (computed once)."""
    from sklearn.decomposition import PCA

    store = _retriever.store
    X = store.index.reconstruct_n(0, store.index.ntotal)
    pca = PCA(n_components=2, random_state=0).fit(X)
    xy = pca.transform(X)
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    chunk_xy = {}
    for pos, doc_id in store.index_to_docstore_id.items():
        d = store.docstore.search(doc_id)
        area = d.metadata.get("area", "")
        chunk_xy[d.metadata["chunk_id"]] = (xy[pos], MAP_AREAS.index(area) if area in AREA_COLORS else
                                            len(MAP_AREAS) - 1, _doc_label(d))
    return pca, lo, hi, chunk_xy


def _unwrap(text: str) -> str:
    """Join lines that the PDF broke in the middle of a sentence."""
    return re.sub(r"(?<![.:;!?])\n(?=[a-z(])", " ", text)


def knowledge_map(retriever: Retriever, question: str, selected: list[int], cited: set[int] | None = None,
                  height: int = 420) -> None:
    """Interactive map of every chunk (assets/knowledge_map.html). It opens with an animation: the
    chunks appear, the question drops in, the map zooms on it and lines shoot out to the picked
    passages (green when the answer cited them). Then the user can zoom, pan, hover, click and go
    full screen."""
    pca, lo, hi, chunk_xy = _map_data(retriever, id(retriever), retriever.store.index.ntotal)
    W, H, pad = 1000, height - 20, 24

    def proj(v):
        x = pad + (np.clip(v[0], lo[0], hi[0]) - lo[0]) / (hi[0] - lo[0] + 1e-9) * (W - 2 * pad)
        y = pad + (1 - (np.clip(v[1], lo[1], hi[1]) - lo[1]) / (hi[1] - lo[1] + 1e-9)) * (H - 2 * pad)
        return round(float(x), 1), round(float(y), 1)

    ids = list(chunk_xy)
    pos = {cid: i for i, cid in enumerate(ids)}
    points = [proj(v) for v, _, _ in chunk_xy.values()]
    qx, qy = proj(pca.transform(np.array([retriever.store.embeddings.embed_query(question)]))[0])
    hits = []
    for rank, cid in enumerate(selected, start=1):
        if cid not in pos:
            continue
        d = retriever.docs[cid]
        is_cited = bool(cited) and cid in cited
        x, y = points[pos[cid]]
        hits.append({"rank": rank, "i": pos[cid], "x": x, "y": y, "cited": is_cited,
                     "color": "#34d399" if is_cited else _color(d), "label": _doc_label(d),
                     "area": d.metadata.get("area", ""), "snippet": _unwrap(d.page_content.split("\n", 1)[-1])[:700]})

    # The opening animation zooms on the question and its passages
    xs, ys = [qx] + [h["x"] for h in hits], [qy] + [h["y"] for h in hits]
    bw, bh = max(xs) - min(xs) + 80, max(ys) - min(ys) + 80
    data = {"W": W, "H": H, "x": [p[0] for p in points], "y": [p[1] for p in points],
            "a": [a for _, a, _ in chunk_xy.values()], "labels": [lab for _, _, lab in chunk_xy.values()],
            "areas": MAP_AREAS, "colors": list(AREA_COLORS.values()) + ["#94a3b8"], "q": [qx, qy],
            "zoom": float(np.clip(min(W / bw, H / bh), 1, 3.5)),
            "center": [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2], "hits": hits, "answered": cited is not None}
    page = (MAP_TEMPLATE.read_text().replace("__W__", str(W)).replace("__H__", str(H))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    _embed_html(page, height=height + 30)


def _embed_html(content: str, height: int) -> None:
    """The map is built from our own index (all text is escaped in the page), so an iframe is safe here.
    st.iframe replaces st.components.v1.html in recent Streamlit versions."""
    if hasattr(st, "iframe"):
        st.iframe(content, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(content, height=height)


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
                st.caption("Every dot is a chunk, placed by its embedding (PCA to 2D). Close dots talk about "
                           "similar things. The yellow Q is your question; the numbers are the passages I picked. "
                           "When the animation ends you can zoom, drag, hover, click and go full screen.")
                self._map_slot = st.empty()
                with self._map_slot.container():
                    knowledge_map(self.retriever, self.question, self.trace.selected)
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
            cited_ids = {self.docs[n - 1].metadata.get("chunk_id") for n in cited}
            with self._map_slot.container():  # replay the map with the cited passages in green
                knowledge_map(self.retriever, self.question, self.trace.selected, cited_ids)


def typewriter(text: str, speed: float):
    """Stream the answer word by word (st.write_stream)."""
    for word in text.split(" "):
        yield word + " "
        if speed:
            time.sleep(0.018 * speed)
