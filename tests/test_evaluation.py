"""The evaluation harness itself: cases, metrics, judges and the report (all offline)."""

import json

import pytest

from ironstore_assistant import evaluate
from ironstore_assistant.eval_cases import CASES, Case, load_generated
from ironstore_assistant.fake_llm import FakeLLM
from ironstore_assistant.judges import JUDGE_NAMES, Grade, Judge, judge_prompt
from ironstore_assistant.metrics import SemanticSimilarity, bleu, clean, reference_scores, rouge
from ironstore_assistant.rag import Assistant, StructuredAnswer


def test_cases_are_well_formed(corpus):
    files = {d.metadata["source"] for d in corpus.documents}
    assert len(CASES) >= 50
    assert len({c.question for c in CASES}) == len(CASES)
    for c in CASES:
        assert c.reference, c.question
        assert set(c.expected_sources) <= files, c.question
        assert c.answerable or not c.must_include or c.category == "prompt injection"


def test_every_trap_is_covered():
    cats = {c.category for c in CASES}
    for trap in ("trap: outdated only", "trap: outdated asked", "trap: shared ID", "trap: reused SKU",
                 "trap: conflicting documents", "out of scope", "prompt injection", "follow-up (memory)"):
        assert trap in cats


def test_lexical_metrics():
    ref = "Most hybrid employees attend the office two days per week."
    assert rouge(ref, ref)["rougeL"] == 1.0
    assert bleu(ref, ref) == pytest.approx(1.0)
    # citation markers and Markdown are ignored
    assert clean("**Two** days per week [1][2].") == "Two days per week."
    assert rouge("Most hybrid employees attend the office two days per week [1].", ref)["rouge1"] == 1.0
    # a paraphrase with the same meaning scores far below 1: the weakness of lexical metrics
    assert rouge("Hybrid staff should be on site 2 days a week.", ref)["rougeL"] < 0.5


def test_semantic_similarity_ranks_the_right_answer_higher(store):
    sim = SemanticSimilarity(store.embedding_function)
    ref = "Passwords must be at least 14 characters long."
    assert sim(ref, ref) == pytest.approx(1.0, abs=1e-3)
    assert sim("The minimum password length is 14 characters.", ref) > sim("Dinner is reimbursed up to €45.", ref)


def test_judge_prompts_keep_data_separate():
    for name in JUDGE_NAMES:
        msgs = judge_prompt(name).format_messages(question="Q?", reference="R", answer="A", context="C")
        assert "DATA to evaluate" in msgs[0].content
        assert "<" in msgs[1].content


def test_judge_maps_verdicts_and_survives_errors():
    judge = Judge(FakeLLM(structured=Grade(reasoning="Matches the reference.", verdict="partly")))
    out = judge.grade_all("Q?", "A", context="[1] passage", reference="R")
    assert set(out) == {"correctness", "groundedness", "doc_relevance", "answer_relevance"}
    assert all(v.score == 0.5 for v in out.values())
    assert "correctness" not in judge.grade_all("Q?", "A", context="", reference="")

    broken = Judge(FakeLLM(structured=None))  # .verdict on None raises
    v = broken.grade("answer_relevance", question="Q?", answer="A")
    assert v.score is None and v.reasoning.startswith("judge error")


def test_full_run_and_report_offline(store, corpus):
    llm = FakeLLM(structured=StructuredAnswer(
        answer="Hybrid employees attend the office two days per week [1].", citations=[1], confidence="high",
        answerable=True))
    bot = Assistant(store, llm, corpus.products, multi_query=False)
    judge = Judge(FakeLLM(structured=Grade(reasoning="ok", verdict="yes")))
    cases = [CASES[0], CASES[16]]  # a policy question and an out-of-scope one
    rows = evaluate.run_full("rag", cases, bot=bot, judge=judge, similarity=SemanticSimilarity(store.embedding_function))

    assert rows[0]["passed"] and rows[0]["rougeL"] > 0.3 and rows[0]["correctness"] == 1.0
    assert not rows[1]["passed"] and not rows[1]["answerable_ok"]  # it answered an out-of-scope question
    assert "doc_relevance" not in rows[1]  # not graded when nothing should be found

    s = evaluate.summarize(rows)
    assert s["code checks passed"] == 0.5 and s["safety cases passed"] == 0.0

    md = evaluate.to_report({"rag": rows, "agent": rows})
    for heading in ("## Summary", "Code-based checks", "Reference-based", "LLM-as-judge", "Safety",
                    "Performance and cost", "## By category", "### What went wrong (rag)"):
        assert heading in md


def test_cost_needs_prices(monkeypatch):
    assert evaluate.cost(1000, 1000) is None
    monkeypatch.setattr(evaluate.config, "PRICE_INPUT_PER_M", 1.0)
    monkeypatch.setattr(evaluate.config, "PRICE_OUTPUT_PER_M", 4.0)
    assert evaluate.cost(1_000_000, 500_000) == pytest.approx(3.0)


def test_only_reviewed_generated_cases_are_used(tmp_path):
    path = tmp_path / "generated.jsonl"
    rows = [Case("Q1?", "generated: simple", reference="A1").to_dict() | {"reviewed": True},
            Case("Q2?", "generated: complex", reference="A2").to_dict() | {"reviewed": False}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert [c.question for c in load_generated(path)] == ["Q1?"]
    assert len(load_generated(path, include_unreviewed=True)) == 2


def test_reference_scores_without_similarity():
    s = reference_scores("two days [1]", "two days")
    assert s["rougeL"] == 1.0 and "similarity" not in s


def test_giskard_testset_is_converted_to_unreviewed_cases():
    import pandas as pd

    from ironstore_assistant.testset import testset_to_cases

    df = pd.DataFrame([
        {"question": "How many office days?", "reference_answer": "Two.", "reference_context": "ctx",
         "conversation_history": [], "metadata": {"question_type": "simple", "seed_document_id": "7"}},
        {"question": "And for interns?", "reference_answer": "Not covered.", "reference_context": "",
         "conversation_history": [{"role": "user", "content": "Tell me about hybrid work"}],
         "metadata": {"question_type": "conversational", "seed_document_id": "7"}},
        {"question": "Who won the 2022 World Cup?", "reference_answer": "Out of scope.", "reference_context": "",
         "conversation_history": [], "metadata": {"question_type": "out of scope", "seed_document_id": "7"}},
    ])
    rows = testset_to_cases(df, {"7": "IronStore_Remote_Work_Policy.pdf"})
    assert [r["reviewed"] for r in rows] == [False] * 3
    assert rows[0]["expected_sources"] == ["IronStore_Remote_Work_Policy.pdf"] and rows[0]["answerable"]
    assert Case.from_dict(rows[1]).history == [("user", "Tell me about hybrid work")]
    assert rows[2]["expected_sources"] == [] and not rows[2]["answerable"]
    assert rows[2]["category"] == "generated: out of scope"


def test_evaluation_page_loads(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # the page only calls the model once you click Run
    at = AppTest.from_file("../pages/1_Evaluation.py", default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Evaluation"
    assert at.slider[0].value == 10 and at.slider[0].max == len(CASES)
