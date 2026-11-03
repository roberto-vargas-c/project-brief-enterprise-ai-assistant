"""Generate more test questions with Giskard RAGET, and run Giskard's own RAG report
(O_Evaluation/4_extra-evaluating-rag-giskard).

    # In a Python 3.10-3.12 environment, with an OPENAI_API_KEY:
    pip install "giskard[llm]>=2.15,<3" "pandas<3" "pyarrow<17" "datasets<3"
    python -m ironstore_assistant.testset generate -n 40   # writes evals/generated_cases.jsonl
    python -m ironstore_assistant.testset report           # writes data/giskard_report.html

Giskard reads the document chunks, groups them into topics and asks an LLM to write questions
with reference answers, of several types: simple, complex, distracting, situational, double,
conversational and out-of-scope. Generated questions can be wrong or ambiguous, so they are
saved with ``"reviewed": false``: read each one, fix or delete it, then set ``"reviewed": true``.
Only reviewed cases are picked up by ``python -m ironstore_assistant.evaluate``.

Giskard 3 dropped the ``giskard.rag`` module used in the course, hence the ``<3`` pin; the
other pins work around Giskard 2's incompatibilities with pandas 3 and recent pyarrow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import config
from .eval_cases import GENERATED_CASES, Case

AGENT_DESCRIPTION = ("An internal assistant for IronStore employees that answers questions about the company's "
                     "policies and procedures (HR, finance, IT, legal, sales, customer support, deliveries) and its "
                     "product catalogs, citing the documents it used.")
GISKARD_TESTSET = config.DATA_DIR / "giskard_testset.jsonl"
GISKARD_REPORT = config.DATA_DIR / "giskard_report.html"


def knowledge_frame(doc_types: tuple[str, ...] = ("policy",)) -> pd.DataFrame:
    """One row per indexed chunk (outdated ones left out), indexed by chunk id so every generated
    question can be traced back to the file it came from."""
    from .build_index import ensure_index
    from .index import load_index

    ensure_index()
    rows = [{"chunk_id": str(d.metadata["chunk_id"]), "text": d.page_content, "source": d.metadata["source"]}
            for d in load_index().docstore._dict.values()
            if d.metadata.get("doc_type") in doc_types and d.metadata.get("status") != "outdated"]
    return pd.DataFrame(rows).set_index("chunk_id")


def testset_to_cases(df: pd.DataFrame, source_by_chunk: dict[str, str]) -> list[dict]:
    """Giskard's testset.to_pandas() -> our case dicts (pure function, tested without Giskard)."""
    out = []
    for _, r in df.iterrows():
        meta = r.get("metadata") or {}
        qtype = meta.get("question_type", "unknown")
        out_of_scope = qtype == "out of scope"
        src = source_by_chunk.get(str(meta.get("seed_document_id", "")))
        history = [(m["role"], m["content"]) for m in (r.get("conversation_history") or [])]
        case = Case(question=r["question"], category=f"generated: {qtype}",
                    expected_sources=[] if out_of_scope or not src else [src], answerable=not out_of_scope,
                    history=history, reference=r["reference_answer"], origin=f"giskard:{qtype}")
        out.append(case.to_dict() | {"reviewed": False, "reference_context": r.get("reference_context", "")})
    return out


def _configure_giskard():
    os.environ.setdefault("GSK_DISABLE_ANALYTICS", "True")
    try:
        import giskard
        from giskard.rag import KnowledgeBase  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Giskard RAGET is not installed. In Python 3.10-3.12 run: pip install '
                         '"giskard[llm]>=2.15,<3" "pandas<3" "pyarrow<17" "datasets<3"') from exc
    giskard.llm.set_llm_model(os.getenv("IRONSTORE_GISKARD_MODEL", config.JUDGE_MODEL))
    giskard.llm.set_embedding_model(os.getenv("IRONSTORE_GISKARD_EMBEDDING_MODEL",
                                              f"openai/{config.OPENAI_EMBEDDING_MODEL}"))


def generate(num_questions: int = 40, out: Path = GENERATED_CASES, include_catalogs: bool = False) -> list[dict]:
    _configure_giskard()
    from giskard.rag import KnowledgeBase, generate_testset

    frame = knowledge_frame(("policy", "catalog") if include_catalogs else ("policy",))
    kb = KnowledgeBase(frame[["text"]], columns=["text"], seed=0)
    testset = generate_testset(kb, num_questions=num_questions, agent_description=AGENT_DESCRIPTION)
    GISKARD_TESTSET.parent.mkdir(parents=True, exist_ok=True)
    testset.save(str(GISKARD_TESTSET))

    new = testset_to_cases(testset.to_pandas(), frame["source"].to_dict())
    # Keep what was already generated and reviewed; append the new questions.
    existing = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if out.exists() else []
    seen = {r["question"] for r in existing}
    rows = existing + [r for r in new if r["question"] not in seen]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Added {len(rows) - len(existing)} questions to {out}. Review them and set \"reviewed\": true.")
    return rows


def report(mode: str = "rag", out: Path = GISKARD_REPORT, include_catalogs: bool = False):
    """Giskard's evaluate(): its own correctness judge plus a breakdown by RAG component."""
    _configure_giskard()
    from giskard.rag import AgentAnswer, KnowledgeBase, QATestset, evaluate

    from .agent import AgentAssistant
    from .rag import build_assistant

    if not GISKARD_TESTSET.exists():
        raise SystemExit("No Giskard testset yet. Run: python -m ironstore_assistant.testset generate")
    rag = build_assistant()
    bot = AgentAssistant(rag) if mode == "agent" else rag

    def answer_fn(question: str, history: list[dict] | None = None) -> AgentAnswer:
        resp = bot.ask(question, history=[(m["role"], m["content"]) for m in history or []])
        return AgentAnswer(message=resp.answer, documents=[d.page_content for d in resp.retrieved])

    frame = knowledge_frame(("policy", "catalog") if include_catalogs else ("policy",))
    kb = KnowledgeBase(frame[["text"]], columns=["text"], seed=0)
    result = evaluate(answer_fn, testset=QATestset.load(str(GISKARD_TESTSET)), knowledge_base=kb,
                      agent_description=AGENT_DESCRIPTION)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_html(str(out))
    print(result.correctness_by_question_type())
    print(result.component_scores())
    print(f"Saved to {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write new questions to evals/generated_cases.jsonl")
    g.add_argument("-n", "--num-questions", type=int, default=40)
    g.add_argument("--include-catalogs", action="store_true", help="also generate product questions")
    r = sub.add_parser("report", help="run Giskard's RAG report on the generated testset")
    r.add_argument("--mode", choices=["rag", "agent"], default="rag")
    r.add_argument("--include-catalogs", action="store_true")
    args = parser.parse_args()
    if args.cmd == "generate":
        generate(args.num_questions, include_catalogs=args.include_catalogs)
    else:
        report(args.mode, include_catalogs=args.include_catalogs)


if __name__ == "__main__":
    main()
