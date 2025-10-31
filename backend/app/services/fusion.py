from __future__ import annotations

from typing import Dict, Tuple

_BASE_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]


def fuse_probs(audio_probs: Dict[str, float], text_probs: Dict[str, float] | None, audio_weight: float = 0.7) -> Tuple[Dict[str, float], str]:
    if text_probs is None:
        fused = dict(audio_probs)
    else:
        aw = max(0.0, min(1.0, audio_weight))
        tw = 1.0 - aw
        fused = {e: aw * audio_probs.get(e, 0.0) + tw * text_probs.get(e, 0.0) for e in _BASE_EMOTIONS}
    # Normalize
    s = sum(fused.values())
    if s > 0:
        fused = {k: v / s for k, v in fused.items()}
    # Final label
    final_label = max(fused.items(), key=lambda kv: kv[1])[0]
    return fused, final_label
