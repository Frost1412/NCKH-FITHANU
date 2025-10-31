from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
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
def _get_audio_pipeline():
    # Wav2Vec2 emotion recognition model from SUPERB ER task
    # https://huggingface.co/superb/wav2vec2-base-superb-er
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")


def _map_audio_label_to_base(label: str) -> str:
    l = label.lower()
    if l in {"angry", "anger"}:
        return "anger"
    if l in {"disgust"}:
        return "disgust"
    if l in {"fear"}:
        return "fear"
    if l in {"happy", "joy"}:
        return "joy"
    if l in {"sad", "sadness"}:
        return "sadness"
    if l in {"surprise"}:
        return "surprise"
    if l in {"neutral", "other"}:
        return "neutral"
    return "neutral"


def predict_audio_emotions(waveform: np.ndarray, sr: int) -> Tuple[Dict[str, float], List[dict]]:
    """Return probabilities over base emotions from audio waveform and raw model scores.

    Returns: (probs_dict, raw_topk)
    """
    clf = _get_audio_pipeline()
    # The HF pipeline can take numpy array with sampling rate
    preds: List[dict] = clf({"array": waveform, "sampling_rate": sr}, top_k=None)
    probs: Dict[str, float] = {e: 0.0 for e in _BASE_EMOTIONS}
    for p in preds:
        mapped = _map_audio_label_to_base(p["label"])  # type: ignore[index]
        probs[mapped] = max(probs.get(mapped, 0.0), float(p["score"]))  # type: ignore[index]
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s for k, v in probs.items()}
    # Sort raw preds by score desc, take top 5 for reporting
    preds_sorted = sorted(preds, key=lambda x: float(x["score"]), reverse=True)[:5]
    return probs, preds_sorted
