# SER Classroom Backend (FastAPI)

A FastAPI backend for Speech Emotion Recognition (SER) in online classrooms. It combines:

- Audio emotion recognition (Wav2Vec2: `superb/wav2vec2-base-superb-er`)
- Optional text emotion estimation (DistilRoBERTa: `j-hartmann/emotion-english-distilroberta-base`)
- Fusion of audio+text probabilities (audio-weight default 0.7)

## Endpoints

- `GET /health` — health check
- `POST /predict` — multipart form:
  - `file`: audio file (wav/mp3)
  - `transcript` (optional): text transcript to improve accuracy
  - `audio_weight` (optional float): weight for audio in [0,1], default 0.7

Response example:
```
{
  "final_label": "joy",
  "final_score": 0.82,
  "top_k": [{"label": "joy", "score": 0.82}, ...],
  "audio_top_k": [{"label": "happy", "score": 0.77}, ...],
  "text_top_k": [{"label": "joy", "score": 0.65}, ...],
  "fusion_weights": {"audio": 0.7, "text": 0.3}
}
```

## Windows setup (PowerShell)

```
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt

# Run the API
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

Notes:
- First run will download Hugging Face models (~hundreds of MB). Please wait.
- If you use a corporate proxy, configure HTTP(S)_PROXY env vars before install.

## Run tests (optional)

```
pytest -q
```

## Project structure

```
backend/
  app/
    main.py
    schemas.py
    models/
      emotion_inference.py
    services/
      audio_processing.py
      text_inference.py
      fusion.py
  tests/
    test_audio_processing.py
  requirements.txt
```
