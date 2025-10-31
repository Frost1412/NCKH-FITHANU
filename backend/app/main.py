from __future__ import annotations

from typing import Optional, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import EmotionScore, PredictResponse
from .services.audio_processing import load_audio_bytes
from .models.emotion_inference import predict_audio_emotions
from .services.text_inference import predict_text_emotions
from .services.fusion import fuse_probs


app = FastAPI(title="SER Classroom Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    audio_weight: float = Form(0.7),
):
    # Read audio bytes
    raw = await file.read()
    y, sr = load_audio_bytes(raw)

    # Audio model
    audio_probs, audio_raw = predict_audio_emotions(y, sr)

    # Optional text model
    text_probs = None
    if transcript and transcript.strip():
        text_probs = predict_text_emotions(transcript.strip())

    fused, final_label = fuse_probs(audio_probs, text_probs, audio_weight=audio_weight)

    # Convert to model for response
    def to_scores(d: dict) -> List[EmotionScore]:
        return [EmotionScore(label=k, score=float(v)) for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)]

    # Audio top-k from raw model outputs
    audio_top_k = [EmotionScore(label=it["label"], score=float(it["score"])) for it in audio_raw]

    text_top_k: Optional[List[EmotionScore]] = None
    if text_probs is not None:
        text_top_k = to_scores(text_probs)[:5]

    resp = PredictResponse(
        final_label=final_label,
        final_score=float(max(fused.values())),
        top_k=to_scores(fused)[:5],
        audio_top_k=audio_top_k,
        text_top_k=text_top_k,
        fusion_weights={"audio": float(audio_weight), "text": float(1.0 - audio_weight) if text_probs is not None else 0.0},
    )
    return resp
