"""Evaluation page: run the evaluation harness from the app, with the app's own OpenAI key.

Same scoring as ``python -m ironstore_assistant.evaluate`` (see that module): code checks,
BLEU / ROUGE / similarity against reference answers, four LLM-as-judge graders, safety cases,
and seconds and tokens per answer, for plain RAG and for the agent.
"""

import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation · IronStore Assistant", page_icon=":material/fact_check:", layout="wide")

# Streamlit secrets -> environment variables, before the config module reads them (as in app.py)
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

from ironstore_assistant import config  # noqa: E402
from ironstore_assistant.eval_cases import all_cases  # noqa: E402
from ironstore_assistant import evaluate as ev  # noqa: E402
import ui  # noqa: E402

ui.inject_css()
LAST_RUN = config.DATA_DIR / "eval_results_app.md"

with st.sidebar:
    ui.brand()

st.title("Evaluation")
st.markdown("Ask the test questions to the assistant and score every answer, the way the course's "
            "**O_Evaluation** module does: rule checks, comparison with a reference answer, and an LLM acting "
            "as a judge. Each answer costs one assistant call plus up to four judge calls.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to `.env` locally, or to the app's **Secrets** on Streamlit Cloud.")
    st.stop()

cases = all_cases()
categories = sorted({c.category for c in cases})


def spread(pool):
    """Interleave categories, so a short run still covers policies, products and the traps."""
    groups = {}
    for c in pool:
        groups.setdefault(c.category, []).append(c)
    out, queues = [], list(groups.values())
    while queues:
        out += [q.pop(0) for q in queues]
        queues = [q for q in queues if q]
    return out


with st.form("settings"):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    modes = c1.segmented_control("Compare", ["RAG", "Agent"], selection_mode="multi", default=["RAG", "Agent"],
                                 help="Plain retrieval-augmented generation, the tool-calling agent, or both.")
    picked = c2.multiselect("Categories", categories, placeholder="All categories")
    use_judges = c3.toggle("LLM judges", value=True, help=f"Grade with `{config.JUDGE_MODEL}` "
                                                         "(set IRONSTORE_JUDGE_MODEL to change it).")
    pool = spread([c for c in cases if not picked or c.category in picked])
    limit = st.slider("Questions", 1, len(pool), min(10, len(pool)),
                      help=f"{len(cases)} cases in the set, mixed across categories so a short run still covers the "
                           "traps. Start small: a full run of both modes takes a while.")
    calls = limit * len(modes or []) * (1 + (4 if use_judges else 0))
    st.caption(f"About {calls} model calls. Keep this page open while it runs.")
    run = st.form_submit_button("Run evaluation", icon=":material/play_arrow:", type="primary")

retrieval = st.button("Quick retrieval check (no LLM, free)", icon=":material/search:")


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_rag():
    from ironstore_assistant.rag import build_assistant

    return build_assistant()


def show_results(results: dict[str, list[dict]]) -> None:
    summaries = {m: ev.summarize(rows) for m, rows in results.items()}
    names = list(results)

    st.subheader("Summary")
    for m in names:
        st.markdown(f"**{m}** · {summaries[m]['cases']} questions")
        for col, (label, key) in zip(st.columns(5), [("Rule checks passed", "code checks passed"),
                                                     ("Correct (judge)", "correctness"),
                                                     ("Grounded (judge)", "groundedness"),
                                                     ("Safety cases passed", "safety cases passed"),
                                                     ("Seconds per answer", "mean seconds")]):
            col.metric(label, ev._fmt(key, summaries[m][key]))

    rows = []
    for section, keys in ev.SECTIONS:
        for k in keys:
            rows.append({"family": section, "metric": k, **{m: ev._fmt(k, summaries[m][k]) for m in names}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Scores are 0–1. Judges: yes = 1, partly = 0.5, no = 0. BLEU and ROUGE only count shared words, "
               "so read them next to similarity and correctness.")

    tabs = st.tabs([f"What went wrong ({m})" for m in names] + [f"All answers ({m})" for m in names])
    for tab, m in zip(tabs[: len(names)], names):
        with tab:
            bad = [r for r in results[m] if not r["passed"] or any(
                r.get(j) is not None and r[j] < 1 for j in ("correctness", "groundedness"))]
            if not bad:
                st.success("Every case passed the rule checks and the judges.")
            for r in bad:
                failed = ", ".join(k for k in ev.CODE_CHECKS if not r[k])
                with st.expander(f"{r['question']}  ·  {r['category']}"):
                    if failed:
                        st.markdown(f"**Failed checks:** {failed}  \n**Cited:** {', '.join(r['cited']) or 'nothing'}")
                    for j in ("correctness", "groundedness", "doc_relevance", "answer_relevance"):
                        if r.get(j) is not None and r[j] < 1:
                            st.markdown(f"**{j} {r[j]:g}:** {r.get(f'{j}_why', '')}")
                    st.markdown("**Answer**")
                    st.markdown(r["answer"])
    for tab, m in zip(tabs[len(names):], names):
        with tab:
            cols = ["question", "category", "passed", "rougeL", "similarity", "correctness", "groundedness",
                    "doc_relevance", "answer_relevance", "seconds", "input_tokens", "output_tokens", "answer"]
            df = pd.DataFrame(results[m])
            st.dataframe(df[[c for c in cols if c in df]], hide_index=True, width="stretch")

    report = ev.to_report(results, "Answer evaluation (" + " vs ".join(names) + ")")
    st.download_button("Download the report (Markdown)", report, "eval_report.md", icon=":material/download:")


if retrieval:
    get_rag()  # makes sure the index exists
    ret = pd.DataFrame(ev.run_retrieval(cases=pool))
    st.subheader("Retrieval")
    a, b, c = st.columns(3)
    a.metric(f"Recall@{config.TOP_K}", f"{ret.recall.mean():.0%}")
    b.metric("Right document first", f"{ret.top1.mean():.0%}")
    c.metric("MRR", f"{ret.rr.mean():.2f}")
    st.dataframe(ret, hide_index=True, width="stretch")

if run:
    if not modes:
        st.warning("Pick RAG, Agent or both.")
        st.stop()
    from ironstore_assistant.agent import AgentAssistant

    rag = get_rag()
    subset = pool[:limit]
    results = {}
    for m in modes:
        bar = st.progress(0.0, text=f"{m}: starting…")
        bot = AgentAssistant(rag) if m == "Agent" else rag

        def progress(i, n, row, m=m, bar=bar):
            bar.progress(i / n, text=f"{m}: {i}/{n} · {'pass' if row['passed'] else 'FAIL'} · {row['question'][:70]}")

        results[m] = ev.run_full(m.lower(), subset, bot=bot, use_judges=use_judges, on_case=progress)
        bar.empty()
    st.session_state.eval_results = results
    try:
        LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN.write_text(ev.to_report(results), encoding="utf-8")
    except OSError:
        pass

if st.session_state.get("eval_results"):
    show_results(st.session_state.eval_results)
elif not retrieval and LAST_RUN.exists():
    with st.expander("Last run on this server", icon=":material/history:"):
        st.markdown(LAST_RUN.read_text(encoding="utf-8"))
