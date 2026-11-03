"""Reference-based metrics: compare a generated answer with a ground-truth answer.

From O_Evaluation:

* **Lexical similarity** (notebooks 1 and 2): BLEU counts n-gram *precision* (made for
  translation), ROUGE counts n-gram *recall* / longest common subsequence (made for
  summaries). Both are cheap and deterministic but only see words: "two days a week" and
  "2 days per week" score badly although they mean the same.
* **Semantic similarity** (0_intro-to-evals): cosine similarity of the two answers'
  embeddings, which sees meaning rather than exact words.

Citation markers like ``[1]`` and Markdown are removed first, since the reference has none.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_CITATION_RE = re.compile(r"\s*\[\d+\](?:\[\d+\])*")
_MARKDOWN_RE = re.compile(r"[*_`#>]|^\s*[-•]\s+", re.M)
_WORD_RE = re.compile(r"\w+(?:[.,]\d+)*", re.U)


def clean(text: str) -> str:
    """Drop [n] citation markers and Markdown so only the words are compared."""
    text = _CITATION_RE.sub("", text or "")
    text = _MARKDOWN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 (0-1), with stemming as in O_Evaluation/2."""
    scores = _rouge_scorer().score(clean(reference), clean(prediction))
    return {name: round(s.fmeasure, 4) for name, s in scores.items()}


def bleu(prediction: str, reference: str) -> float:
    """Sentence BLEU (0-1) with smoothing, so short answers without 4-gram matches don't get 0."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    pred = _WORD_RE.findall(clean(prediction).lower())
    ref = _WORD_RE.findall(clean(reference).lower())
    if not pred or not ref:
        return 0.0
    return round(float(sentence_bleu([ref], pred, smoothing_function=SmoothingFunction().method1)), 4)


class SemanticSimilarity:
    """Cosine similarity between embeddings (0-1 for normalised positive vectors).

    By default it uses the same embeddings as the index (OpenAI, Hugging Face or the offline
    TF-IDF model), so it works with or without an API key.
    """

    def __init__(self, embeddings=None):
        if embeddings is None:
            import json

            from . import config
            from .index import get_embeddings

            manifest = config.INDEX_DIR / "manifest.json"
            provider = json.loads(manifest.read_text())["provider"] if manifest.exists() else None
            embeddings = get_embeddings(provider)
        self.embeddings = embeddings

    def __call__(self, prediction: str, reference: str) -> float:
        a, b = (np.asarray(v, dtype=float) for v in self.embeddings.embed_documents([clean(prediction),
                                                                                       clean(reference)]))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return round(float(a @ b / denom), 4) if denom else 0.0


def reference_scores(prediction: str, reference: str, similarity: SemanticSimilarity | None = None) -> dict:
    out = {"bleu": bleu(prediction, reference), **rouge(prediction, reference)}
    if similarity is not None:
        out["similarity"] = similarity(prediction, reference)
    return out
