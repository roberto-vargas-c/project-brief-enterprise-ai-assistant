"""Live 3D map of how the assistant finds an answer (used by app.py and pipeline_view.py).

The map (assets/live_map.html, three.js) plays a list of events: the question lands, each search
shoots dots out of it, rank fusion picks the passages, the model reads them, and the answer is
typed out in a caption bar while the cited passages turn green.

In live mode the events are written to a small JSON file under ``static/live/`` while
``Assistant.ask(on_step=...)`` runs, and the map polls that file (Streamlit static file serving),
so every step shows up the moment it happens. In replay mode the events are embedded in the page.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

from ironstore_assistant.rag import AssistantResponse, Retriever, RetrievalTrace, page_label

AREA_COLORS = {
    "Customer support": "#fb7185", "Deliveries": "#fbbf24", "Finance": "#2dd4bf", "HR policies": "#a78bfa",
    "IT": "#60a5fa", "Legal Terms": "#f472b6", "Sales": "#4ade80", "Product catalog": "#94a3b8",
}
MAP_AREAS = list(AREA_COLORS) + ["Other"]
TEMPLATE = Path(__file__).parent / "assets" / "live_map.html"
LIVE_DIR = Path(__file__).parent / "static" / "live"
LIVE_URL = "app/static/live"  # Streamlit serves ./static at app/static when enableStaticServing is on


def doc_label(d) -> str:
    m = d.metadata
    name = m.get("product_name") or m["title"].replace("IronStore Product Catalog — ", "Catalog: ")
    return f"{name} · {page_label(m['page'], m.get('page_end'))}"


def doc_color(d) -> str:
    return AREA_COLORS.get(d.metadata.get("area", ""), "#94a3b8")


def unwrap(text: str) -> str:
    """Join lines that the PDF broke in the middle of a sentence."""
    return re.sub(r"(?<![.:;!?])\n(?=[a-z(])", " ", text)


@st.cache_resource(show_spinner="Drawing the knowledge map…")
def _map_data(_retriever: Retriever, retriever_id: int, index_size: int):
    """3D PCA projection of every chunk vector stored in FAISS (computed once per index)."""
    from sklearn.decomposition import PCA

    store = _retriever.store
    X = store.index.reconstruct_n(0, store.index.ntotal)
    pca = PCA(n_components=3, random_state=0).fit(X)
    xyz = pca.transform(X)
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    pos, points, areas, labels = {}, [], [], []
    for row, doc_id in store.index_to_docstore_id.items():
        d = store.docstore.search(doc_id)
        area = d.metadata.get("area", "")
        pos[d.metadata["chunk_id"]] = len(points)
        points.append(_scale(xyz[row], lo, hi))
        areas.append(MAP_AREAS.index(area) if area in AREA_COLORS else len(MAP_AREAS) - 1)
        labels.append(doc_label(d))
    base = {"x": [p[0] for p in points], "y": [p[1] for p in points], "z": [p[2] for p in points],
            "a": areas, "labels": labels, "areas": MAP_AREAS, "colors": list(AREA_COLORS.values()) + ["#94a3b8"]}
    return pca, lo, hi, pos, base


def _scale(v, lo, hi, size: float = 100) -> list[float]:
    """Fit the cloud in a cube from -size/2 to size/2."""
    c = (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9) * size - size / 2
    return [round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2)]


class MapEvents:
    """Turns pipeline steps into map events."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.pca, self.lo, self.hi, self.pos, self.base = _map_data(retriever, id(retriever),
                                                                    retriever.store.index.ntotal)
        self.docs = []

    def where(self, text: str) -> list[float]:
        vec = self.retriever.store.embeddings.embed_query(text)
        return _scale(self.pca.transform(np.array([vec]))[0], self.lo, self.hi)

    def start(self, question: str) -> dict:
        return {"type": "start", "question": question, "q": self.where(question)}

    def condense(self, d: dict) -> dict:
        rewritten = d["standalone"].strip() != d["question"].strip()
        return {"type": "condense", "question": d["question"], "standalone": d["standalone"], "rewritten": rewritten,
                "q": self.where(d["standalone"]) if rewritten else None}

    @staticmethod
    def expand(d: dict) -> dict:
        return {"type": "expand", "queries": d["queries"]}

    @staticmethod
    def tool(d: dict) -> dict:
        args = ", ".join(f"{k}={v!r}" for k, v in d["args"].items())
        return {"type": "tool", "name": d["name"], "args": args[:200], "result": d["result"][:240]}

    def retrieve(self, trace: RetrievalTrace, docs) -> dict:
        self.docs = docs
        found = defaultdict(lambda: {"semantic": [], "keyword": [], "sku": []})
        for h in sorted(trace.hits, key=lambda h: h.rank):
            bucket = found[h.query][h.method]
            if len(bucket) < 6 and h.chunk_id in self.pos:
                bucket.append(self.pos[h.chunk_id])
        searches = [{"query": q, **found[q]} for q in trace.queries if q in found]
        selected = []
        for rank, d in enumerate(docs, start=1):
            cid = d.metadata.get("chunk_id")
            if cid in self.pos:
                selected.append({"rank": rank, "i": self.pos[cid], "label": doc_label(d), "color": doc_color(d),
                                 "area": d.metadata.get("area", ""),
                                 "snippet": unwrap(d.page_content.split("\n", 1)[-1])[:700]})
        return {"type": "retrieve", "searches": searches, "selected": selected,
                "fused": len(trace.fused), "excluded_outdated": trace.excluded_outdated}

    def generate(self) -> dict:
        return {"type": "generate", "n": len(self.docs)}

    @staticmethod
    def answer(resp: AssistantResponse) -> dict:
        return {"type": "answer", "text": resp.answer, "confidence": resp.confidence,
                "cited": [s.number for s in resp.sources], "warnings": resp.warnings}

    def payload(self, events: list[dict] | None, poll: str | None, speed: float) -> dict:
        return {**self.base, "events": events, "poll": poll, "speed": speed}


def _render(data: dict, height: int) -> None:
    page = (TEMPLATE.read_text().replace("__HEIGHT__", str(height))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    # The page is built from our own index and answer, and every text is escaped before it is shown.
    if hasattr(st, "iframe"):
        st.iframe(page, height=height + 70)
    else:
        import streamlit.components.v1 as components

        components.html(page, height=height + 70)


class LiveMap:
    """`on_step` callback that streams each pipeline step to the 3D map while the answer is built."""

    def __init__(self, retriever: Retriever, question: str, speed: float = 1.0, height: int = 600):
        self.ev = MapEvents(retriever)
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        self._cleanup()
        name = f"{uuid.uuid4().hex}.json"
        self.path = LIVE_DIR / name
        self.events: list[dict] = []
        self.arrivals: list[float] = []  # seconds after start, one per event
        self.speed, self.t0 = speed, time.monotonic()
        self._emit(self.ev.start(question))
        _render(self.ev.payload(None, f"{LIVE_URL}/{name}", speed), height)

    def __call__(self, name: str, data: dict) -> None:
        if name == "condense":
            self._emit(self.ev.condense(data))
        elif name == "expand":
            self._emit(self.ev.expand(data))
        elif name == "tool":
            self._emit(self.ev.tool(data))
        elif name == "retrieve":
            self._emit(self.ev.retrieve(data["trace"], data["docs"]))
        elif name == "generate":
            self._emit(self.ev.generate())
        elif name == "validate":
            self._emit(self.ev.answer(data["response"]))

    def finish(self, resp: AssistantResponse | None = None, error: str | None = None) -> None:
        if resp is not None and not any(e["type"] == "answer" for e in self.events):
            self._emit(self.ev.answer(resp))  # e.g. nothing was found, so there was no validate step
        self._emit({"type": "done", "error": error})

    def wait_for_animation(self, limit: float = 12) -> None:
        """Hold the text answer below the map until the map reaches the answer, so it doesn't give the
        ending away. The map plays each event when it arrives, or when it finishes the previous one if
        that is later; this replays that timeline with the pauses used in live_map.html."""
        clock = 1.0  # the page and three.js take about a second to start
        for (e, arrived) in zip(self.events, self.arrivals):
            if e["type"] == "answer":
                break
            clock = max(clock, arrived) + self._duration(e) * self.speed
        remaining = clock + 0.8 * self.speed - (time.monotonic() - self.t0)
        if remaining > 0:
            time.sleep(min(remaining, limit))

    @staticmethod
    def _duration(e: dict) -> float:
        t = e["type"]
        if t == "start":
            return 0.3 + len(e["question"]) / 2 * 0.018
        if t == "condense":
            return 0.9 + len(e["standalone"]) / 2 * 0.014 if e["rewritten"] else 0
        if t == "expand":
            return 0.3 * len(e["queries"]) + 0.2
        if t == "tool":
            return 0.9
        if t == "retrieve":
            searches = sum(max(0.8, (len(s["semantic"]) + len(s["keyword"]) + len(s["sku"])) * 0.045 + 0.6)
                           for s in e["searches"])
            return searches + 0.6 + 0.12 * len(e["selected"]) + 0.5 + 0.4  # + the two captions
        if t == "generate":
            return 0.6
        return 0

    def _emit(self, event: dict) -> None:
        self.events.append(event)
        self.arrivals.append(time.monotonic() - self.t0)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"events": self.events}))
        os.replace(tmp, self.path)  # atomic, so the map never reads half a file

    @staticmethod
    def _cleanup(max_age: float = 3600) -> None:
        now = time.time()
        for f in LIVE_DIR.glob("*.json"):
            try:
                if now - f.stat().st_mtime > max_age:
                    f.unlink()
            except OSError:
                pass


def replay_map(retriever: Retriever, question: str, trace: RetrievalTrace, docs,
               response: AssistantResponse | None = None, speed: float = 1.0, height: int = 460) -> None:
    """The same map, replaying a finished search (step-by-step view and chat history)."""
    ev = MapEvents(retriever)
    events = [ev.start(question), ev.retrieve(trace, docs)]
    if response is not None:
        events.append(ev.answer(response))
    events.append({"type": "done", "error": None})
    _render(ev.payload(events, None, speed), height)
