# Changelog

Tất cả các thay đổi quan trọng của dự án được ghi lại tại đây.

## [0.1.0] - 2025-11-01

### ✨ Added - Tính năng mới

**Backend (FastAPI):**
- API endpoint `/health` để health check
- API endpoint `/predict` để nhận diện cảm xúc từ audio
- Hỗ trợ multipart form upload (audio file + optional transcript)
- Tích hợp Wav2Vec2 model (`superb/wav2vec2-base-superb-er`) cho audio emotion
- Tích hợp DistilRoBERTa model (`j-hartmann/emotion-english-distilroberta-base`) cho text emotion
- Cơ chế fusion đa phương thức (audio + text) với trọng số điều chỉnh
- CORS middleware để support web frontend
- Pydantic schemas cho type-safe responses
- Audio processing utilities: load audio, MFCC, Mel-spectrogram
- Label mapping về 7 base emotions (anger, disgust, fear, joy, sadness, surprise, neutral)

**Web Frontend:**
- Upload audio files (.wav, .mp3, etc.)
- Audio player preview
- Optional transcript input (English)
- Adjustable audio weight slider (0-1)
- Display results: final label, score, top-k from 3 sources
- Dark theme responsive design
- Error handling với user-friendly messages
- Auto-scroll to results
- File info display (name, size)

**Documentation:**
- README.md với giới thiệu đầy đủ
- QUICKSTART.md với hướng dẫn chạy chi tiết
- COMMANDS.md với quick reference
- backend/README.md với API docs
- web/README.md với frontend usage
- Inline code comments

**Scripts & Tools:**
- `run_backend.ps1` - Script PowerShell tự động chạy backend
- `check_backend.py` - Health check script
- `create_sample_audio.py` - Tạo file audio test
- `.gitignore` - Ignore Python/IDE/cache files
- `requirements.txt` - Python dependencies

**Tests:**
- `test_audio_processing.py` - Unit test cho MFCC computation

### 🏗️ Architecture

**Cấu trúc dự án:**
```
NCKH2025/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app
│   │   ├── schemas.py                   # Pydantic models
│   │   ├── models/
│   │   │   └── emotion_inference.py     # Wav2Vec2 pipeline
│   │   ├── services/
│   │   │   ├── audio_processing.py      # Audio utils
│   │   │   ├── text_inference.py        # BERT pipeline
│   │   │   └── fusion.py                # Multi-modal fusion
│   │   └── utils/
│   ├── tests/
│   │   └── test_audio_processing.py
│   ├── requirements.txt
│   └── README.md
├── web/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── README.md
├── run_backend.ps1
├── check_backend.py
├── create_sample_audio.py
├── .gitignore
├── README.md
├── QUICKSTART.md
├── COMMANDS.md
└── CHANGELOG.md
```

**Tech Stack:**
- Python 3.12+
- FastAPI 0.115+
- Transformers 4.44+ (Hugging Face)
- PyTorch 2.3+
- Librosa 0.10+ (audio processing)
- Uvicorn (ASGI server)
- Vanilla JS + HTML5 + CSS3 (frontend)

### 📊 Models

**Audio Emotion Recognition:**
- Model: `superb/wav2vec2-base-superb-er`
- Type: Wav2Vec2 (Transformer-based)
- Task: Emotion Recognition from speech
- Source: SUPERB benchmark

**Text Emotion Classification:**
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Type: DistilRoBERTa (BERT-family)
- Task: Emotion classification from text
- Language: English

### 🎯 Emotions Supported

7 base emotions:
1. Anger (tức giận)
2. Disgust (ghê tởm)
3. Fear (sợ hãi)
4. Joy (vui vẻ)
5. Sadness (buồn bã)
6. Surprise (ngạc nhiên)
7. Neutral (trung tính)

### 📝 Notes

- MVP version - proof of concept
- Models được cache local sau lần tải đầu tiên
- Inference speed: 1-3s cho audio < 10s (CPU)
- Default fusion weights: audio 0.7, text 0.3
- Target sampling rate: 16kHz
- CORS enabled cho localhost development

### 🚧 Known Limitations

- Chỉ hỗ trợ tiếng Anh (models trained on English)
- Chưa có ASR tự động (transcript nhập tay)
- Chưa tối ưu cho production (single worker)
- Chưa có logging/monitoring
- Chưa có authentication
- Chưa test trên datasets chuẩn (RAVDESS, EMO-DB)
- Frontend đơn giản (no real-time, no batch)

---

## [Unreleased] - Roadmap

### 🔮 Planned Features

**Phase 2: Data & Evaluation**
- [ ] Tích hợp RAVDESS dataset
- [ ] Tích hợp EMO-DB dataset
- [ ] Module đánh giá: Accuracy, F1, Precision, Recall
- [ ] Confusion Matrix visualization
- [ ] Benchmark với baseline methods

**Phase 3: ASR Integration**
- [ ] Tích hợp Whisper cho speech-to-text
- [ ] Hoặc Vosk (offline ASR)
- [ ] Auto transcript generation

**Phase 4: Model Improvement**
- [ ] Fine-tune trên classroom domain
- [ ] Noise reduction pipeline
- [ ] Domain adaptation
- [ ] Multi-lingual support (Vietnamese)

**Phase 5: Production Ready**
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Logging (structlog)
- [ ] Monitoring (Prometheus)
- [ ] Authentication & Authorization
- [ ] Rate limiting
- [ ] Caching layer (Redis)

**Phase 6: Advanced Frontend**
- [ ] Real-time emotion tracking
- [ ] Dashboard analytics cho giảng viên
- [ ] Batch upload
- [ ] Waveform visualization
- [ ] Export reports (PDF, CSV)

**Phase 7: Research**
- [ ] Publish results
- [ ] Comparison study
- [ ] User evaluation (teachers, students)
- [ ] Qualitative analysis

---

**Format:** [version] - date  
**Types:** Added, Changed, Deprecated, Removed, Fixed, Security
