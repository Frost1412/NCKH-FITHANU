# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-11-01

### ✨ Added - Training Infrastructure

**Training Scripts:**
- `training/prepare_ravdess.py` - RAVDESS dataset preprocessing
- `training/prepare_custom.py` - Custom dataset preprocessing with validation
- `training/train.py` - Main training script with Wav2Vec2 fine-tuning
- `training/utils/audio_preprocessing.py` - Audio loading, feature extraction, dataset splits
- `training/utils/augmentation.py` - Data augmentation utilities

**Data Organization:**
- `data/raw/` - Original unprocessed audio files
- `data/processed/` - Preprocessed features and metadata
- `data/datasets/RAVDESS/` - RAVDESS dataset location
- `data/datasets/EMO-DB/` - EMO-DB dataset location
- `data/datasets/custom/` - Custom classroom recordings
- `models/checkpoints/` - Saved model weights

**Training Features:**
- Automated train/val/test splitting with stratification
- Audio validation (format, duration, quality checks)
- Data augmentation (time stretch, pitch shift, noise, volume, masking)
- Wav2Vec2 fine-tuning with early stopping
- Comprehensive evaluation (accuracy, F1, confusion matrix)
- Weights & Biases integration for experiment tracking
- Multiple pretrained model support

**Documentation:**
- `TRAINING_GUIDE.md` - Complete training tutorial (6.5KB)
- `TRAINING_QUICKSTART.md` - Quick start guide
- `training/README.md` - Training scripts reference
- `data/README.md` - Dataset organization guide
- `TRAINING_SUMMARY.md` - Infrastructure summary
- `PROJECT_STRUCTURE.md` - Complete project overview

**Learning Materials:**
- `notebooks/01_training_example.ipynb` - Interactive tutorial

## [1.0.0] - 2025-11-01

### ✨ Added - Initial Release

**Backend (FastAPI):**
- API endpoint `/health` for health check
- API endpoint `/predict` for emotion recognition from audio
- Multipart form upload support (audio file + optional transcript)
- Wav2Vec2 model integration (`superb/wav2vec2-base-superb-er`) for audio emotion
- DistilRoBERTa model integration (`j-hartmann/emotion-english-distilroberta-base`) for text emotion
- Multi-modal fusion mechanism (audio + text) with adjustable weights
- CORS middleware to support web frontend
- Pydantic schemas for type-safe responses
- Audio processing utilities: load audio, MFCC, Mel-spectrogram
- Label mapping to 7 base emotions (anger, disgust, fear, joy, sadness, surprise, neutral)

**Web Frontend:**
- Upload audio files (.wav, .mp3, etc.)
- Audio player preview
- Optional transcript input (English)
- Adjustable audio weight slider (0-1)
- Display results: final label, score, top-k from 3 sources
- Dark theme responsive design
- Error handling with user-friendly messages
- Auto-scroll to results
- File info display (name, size)

**Documentation:**
- README.md with complete introduction
- QUICKSTART.md with detailed setup guide
- COMMANDS.md with quick reference
- backend/README.md with API docs
- web/README.md with frontend usage
- Inline code comments

**Scripts & Tools:**
- `run_backend.ps1` - PowerShell script to auto-start backend
- `check_backend.py` - Health check script
- `create_sample_audio.py` - Create test audio files
- `.gitignore` - Ignore Python/IDE/cache files
- `requirements.txt` - Python dependencies

**Tests:**
- `test_audio_processing.py` - Unit test for MFCC computation

### 🏗️ Architecture

**Project Structure:**
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
1. Anger
2. Disgust
3. Fear
4. Joy
5. Sadness
6. Surprise
7. Neutral

### 📝 Notes

- MVP version - proof of concept
- Models are cached locally after first download
- Inference speed: 1-3s for audio < 10s (CPU)
- Default fusion weights: audio 0.7, text 0.3
- Target sampling rate: 16kHz
- CORS enabled for localhost development

### 🚧 Known Limitations

- English language only (models trained on English)
- No automatic ASR (transcript must be manually entered)
- Not optimized for production (single worker)
- No logging/monitoring
- No authentication
- Not yet tested on standard datasets (RAVDESS, EMO-DB) - now available in v1.1.0
- Simple frontend (no real-time, no batch)

---

## [Unreleased] - Roadmap

### 🔮 Planned Features

**Phase 2: Advanced Features**
- [ ] Automatic Speech Recognition (ASR) integration with Whisper
- [ ] Real-time audio streaming support
- [ ] Batch processing for multiple files
- [ ] Model ensemble methods
- [ ] Production optimization (multi-worker, caching)
- [ ] Logging and monitoring dashboard
- [ ] User authentication and API keys

**Phase 3: Research & Evaluation**
- [x] RAVDESS dataset integration
- [x] EMO-DB dataset support
- [x] Evaluation module: Accuracy, F1, Precision, Recall
- [ ] Confusion Matrix visualization in web UI
- [ ] Benchmark against baseline methods
- [ ] Cross-dataset evaluation
- [ ] Publish research results

**Phase 4: Deployment**
- [ ] Docker containerization
- [ ] Cloud deployment guide (AWS, Azure, GCP)
- [ ] REST API documentation (OpenAPI/Swagger)
- [ ] Client SDKs (Python, JavaScript)
- [ ] Performance optimization
- [ ] Scalability improvements

**Phase 5: Model Improvement**
- [ ] Whisper integration for speech-to-text
- [ ] Offline ASR option (Vosk)
- [ ] Auto transcript generation
- [ ] Fine-tune on classroom domain data
- [ ] Noise reduction pipeline
- [ ] Domain adaptation techniques
- [ ] Multi-lingual support (Vietnamese, etc.)

**Phase 6: Production Ready**
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Structured logging (structlog)
- [ ] Monitoring dashboard (Prometheus/Grafana)
- [ ] Authentication & Authorization (OAuth2/JWT)
- [ ] Rate limiting
- [ ] Caching layer (Redis)
- [ ] Load balancing
- [ ] Database integration for result storage
- [ ] API versioning

### 🐛 Bug Fixes

None reported yet - this is the initial release.

### 📚 Documentation Updates

- All documentation translated to English
- Comprehensive training guides added
- Interactive Jupyter notebook tutorial
- Complete API reference
- Deployment guides

---

## Release Notes

### Version 1.1.0 - Training Infrastructure
Major update adding complete training capabilities for custom model development.

### Version 1.0.0 - Initial Release
First public release with core emotion recognition functionality and web demo.

---

**Legend:**
- ✨ New features
- 🐛 Bug fixes
- 🏗️ Architecture changes
- 📚 Documentation
- 🚧 Known issues
- 🔮 Future plans

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
