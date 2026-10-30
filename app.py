"""Streamlit chat interface for the IronStore Assistant.

    streamlit run app.py
"""

import json

import streamlit as st

from ironstore_assistant import config
from ironstore_assistant.agent import AgentAssistant
from ironstore_assistant.rag import build_assistant

st.set_page_config(page_title="IronStore Assistant", page_icon="🤖", layout="wide")

CONFIDENCE_BADGE = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}
EXAMPLES = [
    "What's our remote work policy?",
    "How do I request time off?",
    "What is the return window for customers?",
    "How much does the Nordholm Lysa 3-Seater Sofa cost and is it in stock?",
]


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_assistant(multi_query: bool):
    return build_assistant(multi_query=multi_query)


with st.sidebar:
    st.title("🤖 IronStore Assistant")
    st.caption("Answers from IronStore's internal policies and product catalogs, with sources.")
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
    if st.button("New conversation"):
        st.session_state.messages = []
    st.divider()
    manifest = config.INDEX_DIR / "manifest.json"
    embeddings = json.loads(manifest.read_text())["model"] if manifest.exists() else "no index yet"
    st.caption(f"Model: `{config.CHAT_MODEL}` · Embeddings: `{embeddings}`")

try:
    rag = get_assistant(multi_query)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
bot = AgentAssistant(rag) if mode.startswith("Agent") else rag

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_response(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("confidence"):
        st.caption(CONFIDENCE_BADGE[msg["confidence"]])
    for w in msg.get("warnings", []):
        st.warning(w, icon="⚠️")
    if msg.get("sources"):
        st.markdown("**Sources**")
        for s in msg["sources"]:
            with st.expander(f"[{s.number}] {s.label()}"):
                st.caption(f"File: `{s.source}` · page {s.page}"
                           + (f" · ID {s.doc_id}" if s.doc_id else "")
                           + (f" · last reviewed {s.last_reviewed}" if s.last_reviewed else ""))
                st.text(s.snippet)
    if msg.get("trace"):
        with st.expander("How I searched"):
            for q in msg["trace"]:
                st.code(q, language=None)


st.header("Ask about IronStore policies, procedures and products")
if not st.session_state.messages:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_response(msg) if msg["role"] == "assistant" else st.markdown(msg["content"])

question = st.chat_input("Ask a question…") or st.session_state.pop("pending", None)
if question:
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            resp = bot.ask(question, history=history, include_outdated=True if include_old else None,
                           area=None if area == "All" else area)
        msg = {"role": "assistant", "content": resp.answer, "confidence": resp.confidence,
               "warnings": resp.warnings, "sources": resp.sources, "trace": resp.queries}
        render_response(msg)
    st.session_state.messages.append(msg)
