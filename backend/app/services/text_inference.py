from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from transformers import pipeline

_BASE_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]


@lru_cache(maxsize=1)
def _get_text_pipeline():
    # English emotion model (DistilRoBERTa) fine-tuned for emotions
    # https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )


def _map_text_label_to_base(label: str) -> str:
    l = label.lower()
    # Map model's labels to our base set
    if "anger" in l or l == "annoyance":
        return "anger"
    if l in {"disgust"}:
        return "disgust"
    if l in {"fear"}:
        return "fear"
    if l in {"joy", "happiness", "optimism"}:
        return "joy"
    if l in {"sadness", "grief"}:
        return "sadness"
    if l in {"surprise", "realization"}:
        return "surprise"
    # Default fallback
    return "neutral"


def predict_text_emotions(text: str) -> Dict[str, float]:
    """Return probabilities over base emotions from text input."""
    clf = _get_text_pipeline()
    # pipeline with top_k=None returns list of dicts for each label
    preds: List[dict] = clf(text, top_k=None)[0]
    probs: Dict[str, float] = {e: 0.0 for e in _BASE_EMOTIONS}
    for p in preds:
        mapped = _map_text_label_to_base(p["label"])  # type: ignore[index]
        probs[mapped] = max(probs.get(mapped, 0.0), float(p["score"]))  # type: ignore[index]
    # Renormalize
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s for k, v in probs.items()}
    return probs
