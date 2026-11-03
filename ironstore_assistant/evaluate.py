"""Evaluation harness: retrieval metrics, code checks, reference metrics and LLM judges.

    python -m ironstore_assistant.evaluate --retrieval-only        # no LLM calls, checks the retriever
    python -m ironstore_assistant.evaluate                         # full run, RAG mode
    python -m ironstore_assistant.evaluate --mode agent            # full run, agent mode
    python -m ironstore_assistant.evaluate --mode both             # RAG vs agent comparison report
    python -m ironstore_assistant.evaluate --no-judges --limit 10  # quick and cheap

Each answer is scored with every family of evals from O_Evaluation/0_intro-to-evals:

1. **Code-based checks** (deterministic): the ``answerable`` flag, the cited documents, required
   and forbidden keywords, and the outdated-data warning.
2. **Reference-based metrics**: BLEU, ROUGE and embedding similarity against the case's
   reference answer (metrics.py), plus an LLM-as-judge for correctness.
3. **LLM-as-judge without a reference**: groundedness (hallucination), document relevance and
   answer relevance (judges.py).
4. **Safety**: the traps, out-of-scope questions and prompt injections, reported on their own.
5. **Performance and cost**: latency and tokens per answer (and a cost estimate when prices are
   set in ``IRONSTORE_PRICE_INPUT_PER_M`` / ``IRONSTORE_PRICE_OUTPUT_PER_M``).

The cases live in eval_cases.py. With ``LANGSMITH_API_KEY`` set, every call is also traced to
LangSmith (project ``ironstore-eval-<mode>``), as in O_Evaluation/3.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

from . import config
from .eval_cases import CASES, Case, all_cases  # noqa: F401  (CASES is re-exported for notebook 06)

SAFETY_PREFIXES = ("trap", "out of scope", "not in documents", "prompt injection")
CODE_CHECKS = ("answerable_ok", "sources_ok", "keywords_ok", "no_forbidden", "outdated_ok")
REFERENCE_METRICS = ("bleu", "rouge1", "rouge2", "rougeL", "similarity")


# --------------------------------------------------------------------------------------------
# Retrieval only (no LLM)
# --------------------------------------------------------------------------------------------
def run_retrieval(k: int = config.TOP_K, cases: list[Case] | None = None) -> list[dict]:
    from .index import load_index
    from .rag import OLD_DATA_RE, Retriever, load_products

    retriever = Retriever(load_index(), load_products())
    rows = []
    for c in cases if cases is not None else CASES:
        if not c.expected_sources or c.history:
            continue
        docs = retriever.search([c.question], k=k, include_outdated=bool(OLD_DATA_RE.search(c.question)))
        ranked = [d.metadata["source"] for d in docs]
        hits = [s for s in c.expected_sources if s in ranked]
        first = next((i for i, s in enumerate(ranked) if s in c.expected_sources), None)
        rows.append({"question": c.question, "category": c.category,
                     "recall": len(hits) / len(c.expected_sources),
                     "top1": bool(ranked) and ranked[0] in c.expected_sources,
                     "rr": 0.0 if first is None else round(1 / (first + 1), 3)})
    return rows


# --------------------------------------------------------------------------------------------
# Full evaluation (assistant + judges)
# --------------------------------------------------------------------------------------------
def _tokens(usage: dict) -> tuple[int, int]:
    ins = sum(u.get("input_tokens", 0) for u in usage.values())
    outs = sum(u.get("output_tokens", 0) for u in usage.values())
    return ins, outs


def cost(input_tokens: int, output_tokens: int) -> float | None:
    if not (config.PRICE_INPUT_PER_M or config.PRICE_OUTPUT_PER_M):
        return None
    return round((input_tokens * config.PRICE_INPUT_PER_M + output_tokens * config.PRICE_OUTPUT_PER_M) / 1e6, 6)


def code_checks(c: Case, resp) -> dict[str, bool]:
    cited = {s.source for s in resp.sources}
    text = resp.answer.lower()
    return {
        "answerable_ok": resp.answerable == c.answerable,
        "sources_ok": all(s in cited for s in c.expected_sources),
        "keywords_ok": all(k.lower() in text for k in c.must_include),
        "no_forbidden": not any(k.lower() in text for k in c.must_not_include),
        "outdated_ok": (not c.expect_outdated_warning) or any("OUTDATED" in w for w in resp.warnings),
    }


def evaluate_case(c: Case, bot, judge=None, similarity=None) -> dict:
    """Ask one question and score the answer. Judge tokens are not counted as assistant cost."""
    from langchain_core.callbacks import get_usage_metadata_callback

    from .metrics import reference_scores
    from .rag import format_context

    t0 = time.time()
    with get_usage_metadata_callback() as cb:
        resp = bot.ask(c.question, history=c.history)
    seconds = round(time.time() - t0, 2)
    ins, outs = _tokens(cb.usage_metadata)

    checks = code_checks(c, resp)
    row = {"question": c.question, "category": c.category, "origin": c.origin, **checks,
           "passed": all(checks.values()), "confidence": resp.confidence, "seconds": seconds,
           "input_tokens": ins, "output_tokens": outs, "cost": cost(ins, outs),
           "cited": sorted({s.source for s in resp.sources}), "answer": resp.answer}
    if c.reference:
        row.update(reference_scores(resp.answer, c.reference, similarity))
    if judge is not None:
        verdicts = judge.grade_all(c.question, resp.answer, format_context(resp.retrieved) if resp.retrieved else "",
                                   reference=c.reference, with_doc_relevance=c.answerable)
        for name, v in verdicts.items():
            row[name] = v.score
            row[f"{name}_why"] = v.reasoning
    return row


def _tracing(mode: str):
    """Trace to LangSmith when a key is configured (O_Evaluation/3), otherwise do nothing."""
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return contextlib.nullcontext()
    from langchain_core.tracers.context import tracing_v2_enabled

    return tracing_v2_enabled(project_name=f"ironstore-eval-{mode}")


def run_full(mode: str = "rag", cases: list[Case] | None = None, bot=None, judge=None, use_judges: bool = True,
             similarity=None, on_case: Callable[[int, int, dict], None] | None = None) -> list[dict]:
    """Run every case through the assistant ("rag" or "agent") and score it.

    ``bot``, ``judge`` and ``similarity`` can be injected (the tests use a scripted FakeLLM);
    by default they are built from config (OpenAI key needed).
    """
    cases = cases if cases is not None else all_cases()
    if bot is None:
        from .agent import AgentAssistant
        from .rag import build_assistant

        rag = build_assistant()
        bot = AgentAssistant(rag) if mode == "agent" else rag
    if judge is None and use_judges:
        from .judges import Judge
        from .rag import get_llm

        judge = Judge(get_llm(config.JUDGE_MODEL))
    if similarity is None:
        from .metrics import SemanticSimilarity

        similarity = SemanticSimilarity()

    rows = []
    with _tracing(mode):
        for i, c in enumerate(cases, start=1):
            row = {"mode": mode, **evaluate_case(c, bot, judge if use_judges else None, similarity)}
            rows.append(row)
            if on_case:
                on_case(i, len(cases), row)
    return rows


# --------------------------------------------------------------------------------------------
# Summaries and reports
# --------------------------------------------------------------------------------------------
def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _p95(values) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))]


def is_safety(category: str) -> bool:
    return category.startswith(SAFETY_PREFIXES)


def summarize(rows: list[dict]) -> dict[str, float | int | None]:
    """One number per metric, grouped like the families in O_Evaluation/0."""
    safety = [r for r in rows if is_safety(r["category"])]
    s: dict[str, float | int | None] = {"cases": len(rows)}
    s["code checks passed"] = _mean(r["passed"] for r in rows)
    for k in CODE_CHECKS:
        s[k] = _mean(r[k] for r in rows)
    for k in REFERENCE_METRICS:
        s[k] = _mean(r.get(k) for r in rows)
    for k in ("correctness", "groundedness", "doc_relevance", "answer_relevance"):
        s[k] = _mean(r.get(k) for r in rows)
    s["safety cases passed"] = _mean(r["passed"] for r in safety)
    s["mean seconds"] = _mean(r["seconds"] for r in rows)
    s["p95 seconds"] = _p95(r["seconds"] for r in rows)
    s["mean tokens"] = _mean(r["input_tokens"] + r["output_tokens"] for r in rows)
    costs = [r.get("cost") for r in rows if r.get("cost") is not None]
    s["total cost"] = sum(costs) if costs else None
    return s


SECTIONS = [
    ("Code-based checks", ["code checks passed", *CODE_CHECKS]),
    ("Reference-based (vs ground truth)", [*REFERENCE_METRICS, "correctness"]),
    ("LLM-as-judge (no reference)", ["groundedness", "doc_relevance", "answer_relevance"]),
    ("Safety and traps", ["safety cases passed"]),
    ("Performance and cost", ["mean seconds", "p95 seconds", "mean tokens", "total cost"]),
]
PERCENT = {"code checks passed", *CODE_CHECKS, "correctness", "groundedness", "doc_relevance",
           "answer_relevance", "safety cases passed"}


def _fmt(key: str, v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if key in PERCENT:
        return f"{v:.0%}"
    if key == "total cost":
        return f"{v:.4f}"
    if key in REFERENCE_METRICS:
        return f"{v:.2f}"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def to_report(results: dict[str, list[dict]], title: str = "Answer evaluation") -> str:
    """Markdown report. ``results`` maps a mode ("rag", "agent") to its rows."""
    modes = list(results)
    summaries = {m: summarize(rows) for m, rows in results.items()}
    out = [f"# {title}", "",
           f"Cases: {', '.join(f'{m}: {summaries[m]['cases']}' for m in modes)} · "
           f"chat model `{config.CHAT_MODEL}` · judge `{config.JUDGE_MODEL}` · "
           f"embeddings `{config.EMBEDDINGS_PROVIDER}` · {time.strftime('%Y-%m-%d %H:%M')}", ""]

    out += ["## Summary", "", "| metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for section, keys in SECTIONS:
        out.append(f"| **{section}** |" + " |" * len(modes))
        for k in keys:
            out.append(f"| {k} | " + " | ".join(_fmt(k, summaries[m][k]) for m in modes) + " |")
    out += ["", "Scores are 0–1. Judges: yes = 1, partly = 0.5, no = 0. BLEU and ROUGE only count shared words, "
                "so read them next to similarity and correctness.", ""]

    out += ["## By category", ""]
    cats = sorted({r["category"] for rows in results.values() for r in rows})
    out += ["| category | " + " | ".join(f"{m} passed | {m} correct" for m in modes) + " |",
            "|---|" + "---|---|" * len(modes)]
    for cat in cats:
        cells = []
        for m in modes:
            rows = [r for r in results[m] if r["category"] == cat]
            cells += [_fmt("code checks passed", _mean(r["passed"] for r in rows)),
                      _fmt("correctness", _mean(r.get("correctness") for r in rows))]
        out.append(f"| {cat} | " + " | ".join(cells) + " |")

    for m in modes:
        out += ["", f"## Cases ({m})", "",
                "| question | category | passed | failed checks | rougeL | similarity | correct | grounded | "
                "docs | relevant | s | tokens |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in results[m]:
            failed = ", ".join(k for k in CODE_CHECKS if not r[k]) or ""
            out.append("| " + " | ".join([
                r["question"].replace("|", "/"), r["category"], _fmt("passed", r["passed"]), failed,
                _fmt("rougeL", r.get("rougeL")), _fmt("similarity", r.get("similarity")),
                _fmt("correctness", r.get("correctness")), _fmt("groundedness", r.get("groundedness")),
                _fmt("doc_relevance", r.get("doc_relevance")), _fmt("answer_relevance", r.get("answer_relevance")),
                f"{r['seconds']:.1f}", str(r["input_tokens"] + r["output_tokens"])]) + " |")

        bad = [r for r in results[m] if not r["passed"] or any(
            (r.get(j) is not None and r[j] < 1) for j in ("correctness", "groundedness"))]
        if bad:
            out += ["", f"### What went wrong ({m})", ""]
            for r in bad:
                failed = ", ".join(k for k in CODE_CHECKS if not r[k])
                out.append(f"- **{r['question']}** ({r['category']})")
                if failed:
                    out.append(f"  - failed checks: {failed}; cited: {', '.join(r['cited']) or 'nothing'}")
                for j in ("correctness", "groundedness"):
                    if r.get(j) is not None and r[j] < 1:
                        out.append(f"  - {j} {r[j]:g}: {r.get(f'{j}_why', '')}")
                out.append(f"  - answer: {r['answer'].replace(chr(10), ' ')[:400]}")
    return "\n".join(out)


def to_markdown(rows: list[dict], title: str) -> str:
    """Retrieval report (kept for notebook 06 and the --retrieval-only run)."""
    if not rows:
        return f"# {title}\n\nNo rows."
    if "passed" in rows[0]:
        return to_report({rows[0].get("mode", "rag"): rows}, title)
    cols = list(rows[0])
    out = [f"# {title}", "",
           f"**Recall@k: {_mean(r['recall'] for r in rows):.0%} · Top-1 hit: {_mean(r['top1'] for r in rows):.0%} · "
           f"MRR: {_mean(r['rr'] for r in rows):.2f}** ({len(rows)} cases with an expected source)", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(("yes" if v else "no") if isinstance(v, bool) else
                                     (f"{v:.0%}" if k == "recall" else str(v)) for k, v in r.items()) + " |")
    return "\n".join(out)


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="only check the retriever (no LLM)")
    parser.add_argument("--mode", choices=["rag", "agent", "both"], default="rag")
    parser.add_argument("--agent", action="store_true", help="same as --mode agent")
    parser.add_argument("--no-judges", action="store_true", help="skip the LLM-as-judge graders")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--category", help="only cases whose category contains this text")
    parser.add_argument("--no-generated", action="store_true", help="ignore evals/generated_cases.jsonl")
    parser.add_argument("--include-unreviewed", action="store_true", help="also use unreviewed generated cases")
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "eval_results.md")
    args = parser.parse_args()

    cases = all_cases(include_generated=not args.no_generated, include_unreviewed=args.include_unreviewed)
    if args.category:
        cases = [c for c in cases if args.category.lower() in c.category.lower()]
    cases = cases[: args.limit] if args.limit else cases

    if args.retrieval_only:
        md = to_markdown(run_retrieval(cases=cases), "Retrieval evaluation")
    else:
        modes = ["rag", "agent"] if args.mode == "both" else ["agent" if args.agent else args.mode]

        def progress(i, n, row):
            print(f"  [{i}/{n}] {'PASS' if row['passed'] else 'FAIL'} {row['question'][:70]}", flush=True)

        results = {}
        for m in modes:
            print(f"Evaluating {len(cases)} cases in {m} mode…")
            results[m] = run_full(m, cases, use_judges=not args.no_judges, on_case=progress)
            save_rows(results[m], args.out.with_name(f"{args.out.stem}_{m}.jsonl"))
        md = to_report(results, "Answer evaluation (" + " vs ".join(modes) + ")")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
