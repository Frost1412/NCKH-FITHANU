from typing import List, Optional, Dict
from pydantic import BaseModel


class EmotionScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    final_label: str
    final_score: float
    top_k: List[EmotionScore]
    audio_top_k: List[EmotionScore]
    text_top_k: Optional[List[EmotionScore]] = None
    fusion_weights: Dict[str, float]
