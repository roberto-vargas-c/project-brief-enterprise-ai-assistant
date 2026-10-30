"""Streamlit chat interface for the IronStore Assistant.

    streamlit run app.py

On Streamlit Community Cloud, put OPENAI_API_KEY in the app's Secrets. The index is built
automatically on the first visit.
"""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="IronStore Assistant", page_icon=":material/hub:", layout="centered")

# Streamlit secrets -> environment variables, before the config module reads them
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file when running locally: .env is used instead
    pass

from ironstore_assistant import config  # noqa: E402
from ironstore_assistant.agent import AgentAssistant  # noqa: E402
from ironstore_assistant.rag import build_assistant  # noqa: E402
from live_view import LiveMap  # noqa: E402
from pipeline_view import PipelineView, passage_cards, typewriter  # noqa: E402
import ui  # noqa: E402

ui.inject_css()


if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to `.env` locally, or to the app's **Secrets** on Streamlit Cloud.")
    st.stop()

EXAMPLES = [
    (":material/home_work:", "What's our remote work policy?"),
    (":material/beach_access:", "How do I request time off?"),
    (":material/assignment_return:", "What is the return window for customers?"),
    (":material/chair:", "How much does the Nordholm Lysa 3-Seater Sofa cost and is it in stock?"),
]
AVATARS = {"user": ":material/person:", "assistant": ":material/hub:"}


def code_version() -> str:
    """Hash of the assistant's source code. Streamlit keeps cached objects across a redeploy when
    the cached function itself is unchanged, so this key makes a new deploy build a fresh assistant
    (and rebuild the index if needed) instead of reusing one made by the old code."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.joinpath("ironstore_assistant").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Loading the knowledge base (the first visit builds the index, about 1-2 minutes)…")
def get_assistant(multi_query: bool, version: str):
    return build_assistant(multi_query=multi_query)


with st.sidebar:
    ui.brand()
    st.subheader("Search")
    mode = st.radio("Mode", ["Standard (RAG)", "Agent (tools)"],
                    help="Agent mode lets the model call tools such as an exact SKU/price lookup.")
    multi_query = st.toggle("Multi-query retrieval", value=True,
                            help="Rewrite the question several ways to find more relevant passages.")
    include_old = st.toggle("Include outdated documents", value=False,
                            help="The 2021 product catalog is excluded unless you turn this on "
                                 "or ask about old data explicitly.")
    areas = ["All", "Customer support", "Deliveries", "Finance", "HR policies", "IT", "Legal Terms", "Sales",
             "Product catalog"]
    area = st.selectbox("Department", areas)
    st.subheader("Show my work")
    show_work = st.toggle("Animate how I find the answer", value=True)
    view_mode = st.radio("View", ["Live 3D map", "Step by step"], disabled=not show_work,
                         help="Live 3D map: watch the search happen on a 3D map of every chunk, with the "
                              "answer typed at the bottom. Step by step: each stage as a panel.")
    speed_label = st.select_slider("Animation speed", ["Fast", "Normal", "Slow"], value="Normal",
                                   disabled=not show_work)
    show_map = st.toggle("Knowledge map in step-by-step view", value=True,
                         disabled=not show_work or view_mode != "Step by step",
                         help="Plot every chunk on a 3D map and show where your question lands.")
    st.write("")
    if st.button("New conversation", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = []
    manifest = config.INDEX_DIR / "manifest.json"
    embeddings = json.loads(manifest.read_text())["model"] if manifest.exists() else "no index yet"
    st.markdown(f'<div class="foot">Model <code>{config.CHAT_MODEL}</code><br>'
                f'Embeddings <code>{embeddings}</code></div>', unsafe_allow_html=True)

speed = {"Fast": 0.35, "Normal": 1.0, "Slow": 2.0}[speed_label] if show_work else 0.0

try:
    rag = get_assistant(multi_query, code_version())
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
bot = AgentAssistant(rag) if mode.startswith("Agent") else rag

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(msg: dict) -> None:
    if msg.get("confidence"):
        st.markdown(ui.confidence_badge(msg["confidence"]), unsafe_allow_html=True)
    for w in msg.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    if msg.get("sources"):
        st.markdown(ui.source_cards(msg["sources"]), unsafe_allow_html=True)
    if msg.get("trace") is not None and msg.get("docs"):
        with st.expander("How I found this", icon=":material/manage_search:"):
            trace = msg["trace"]
            st.markdown("**Searched with:** " + " · ".join(f"`{q}`" for q in trace.queries))
            st.markdown(passage_cards(msg["docs"], trace, {s.number for s in msg["sources"]}),
                        unsafe_allow_html=True)


question = st.chat_input("Ask about a policy, a procedure or a product…") or st.session_state.pop("pending", None)
welcome = st.empty()  # always rendered, so the welcome screen disappears as soon as a question is asked
if not st.session_state.messages and not question:
    with welcome.container():
        docs = rag.retriever.docs.values()
        ui.hero(len({d.metadata["source"] for d in docs}), len(rag.retriever.docs), len(rag.retriever.products))
        st.markdown('<div class="try">Try one of these</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, ex) in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, icon=icon, use_container_width=True, key=f"ex{i}"):
                st.session_state.pending = ex
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)

if question:
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        kwargs = {"history": history, "include_outdated": True if include_old else None,
                  "area": None if area == "All" else area}
        if show_work and view_mode == "Live 3D map":
            live = LiveMap(rag.retriever, question, speed=speed)
            try:
                resp = bot.ask(question, on_step=live, **kwargs)
            except Exception as e:
                live.finish(error="Something went wrong while answering. Please try again.")
                raise e
            live.finish(resp)  # the answer and its sources are shown inside the map
        elif show_work:
            view = PipelineView(rag.retriever, speed=speed, show_map=show_map)
            resp = bot.ask(question, on_step=view, **kwargs)
            st.write_stream(typewriter(resp.answer, speed))
        else:
            with st.spinner("Searching the documents…"):
                resp = bot.ask(question, **kwargs)
            st.markdown(resp.answer)
        msg = {"role": "assistant", "content": resp.answer, "confidence": resp.confidence,
               "warnings": resp.warnings, "sources": resp.sources, "trace": resp.trace, "docs": resp.retrieved}
        if not (show_work and view_mode == "Live 3D map"):
            render_sources(msg)
    st.session_state.messages.append(msg)
