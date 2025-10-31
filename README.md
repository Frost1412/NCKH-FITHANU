# NCKH-FITHANU

**Speech Emotion Recognition System for Online Classrooms**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.44%2B-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

> AI-powered emotion recognition combining Wav2Vec2 and DistilRoBERTa for analyzing student emotions in online learning environments.

## 🌟 Highlights

- **🎯 Multi-modal Fusion**: Combines audio (Wav2Vec2) and text (BERT) for improved accuracy
- **🚀 Real-time Processing**: Fast inference with pre-trained Transformer models
- **🌐 Web Interface**: User-friendly demo for quick testing
- **📊 RESTful API**: Easy integration with FastAPI backend
- **🔬 Research-ready**: Designed for academic evaluation and extension

## 📖 Overview

This system detects 7 emotions from speech:
- 😠 Anger
- 🤢 Disgust  
- 😨 Fear
- 😊 Joy
- 😢 Sadness
- 😲 Surprise
- 😐 Neutral

**Tech Stack:**
- **Backend**: FastAPI + Transformers (Hugging Face)
- **Models**: Wav2Vec2 (audio), DistilRoBERTa (text)
- **Audio Processing**: Librosa (MFCC, Mel-spectrogram)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 4GB+ RAM
- Internet connection (for first-time model download)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/NCKH-FITHANU.git
cd NCKH-FITHANU

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt
```

### Run Backend

```bash
# Option 1: Automated script (Windows)
.\run_backend.ps1

# Option 2: Manual
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

### Open Web Interface

1. Double-click `web/index.html` or
2. Use Live Server extension in VS Code (recommended)

Visit: http://127.0.0.1:8000/docs for API documentation

## 🎓 Train Your Own Models

Want to train custom emotion recognition models? We've got you covered!

### Quick Training Setup

```bash
# 1. Install training dependencies
pip install -r training/requirements.txt

# 2. Option A: Use RAVDESS dataset
# Download from https://zenodo.org/record/1188976
# Extract to data/datasets/RAVDESS/
python training/prepare_ravdess.py
python training/train.py --dataset ravdess --epochs 10

# 2. Option B: Use your custom data
# Organize audio files by emotion in data/datasets/custom/
python training/prepare_custom.py
python training/train.py --dataset custom --epochs 10
```

**📖 Learn More:**
- [Training Quickstart](TRAINING_QUICKSTART.md) - Get started in 5 minutes
- [Complete Training Guide](TRAINING_GUIDE.md) - Detailed tutorial
- [Example Notebook](notebooks/01_training_example.ipynb) - Interactive learning

## 📚 Documentation

### Getting Started
- [Quickstart Guide](QUICKSTART.md) - Detailed setup instructions
- [Command Reference](COMMANDS.md) - Useful commands and scripts
- [API Documentation](backend/README.md) - Backend API details
- [Web Guide](web/README.md) - Frontend usage

### Training Your Own Models
- [Training Quickstart](TRAINING_QUICKSTART.md) - ⚡ Start training in 5 minutes
- [Complete Training Guide](TRAINING_GUIDE.md) - 📖 In-depth training tutorial
- [Training Scripts Reference](training/README.md) - 🛠️ Script documentation
- [Example Notebook](notebooks/01_training_example.ipynb) - 📓 Interactive tutorial
- [Dataset Guide](data/README.md) - 📊 Dataset preparation

### Project Info
- [Changelog](CHANGELOG.md) - Version history and roadmap

## 🏗️ Architecture

```
┌─────────────┐
│  Audio File │
└──────┬──────┘
       │
       ├─────► Speech Processing ────► MFCC / Mel-Spec
       │
       ├─────► Wav2Vec2 Model ────────► Audio Emotions
       │                                      │
       │                                      │
       ▼                                      ▼
┌─────────────┐                        ┌───────────┐
│  Transcript │                        │  Fusion   │
│  (optional) │                        │  Module   │
└──────┬──────┘                        └─────┬─────┘
       │                                      │
       └─────► DistilRoBERTa ────────► Text Emotions
                                              │
                                              ▼
                                        Final Emotion
```

## 📊 API Usage

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Predict Emotion
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@audio.wav" \
  -F "transcript=I am happy" \
  -F "audio_weight=0.7"
```

**Response:**
```json
{
  "final_label": "joy",
  "final_score": 0.834,
  "top_k": [...],
  "audio_top_k": [...],
  "text_top_k": [...],
  "fusion_weights": {"audio": 0.7, "text": 0.3}
}
```

## 🔬 Research Context

This project is part of research on "AI Application for Speech Emotion Recognition in Online Classroom Environments" at Faculty of Information Technology, Hanoi University (FITHANU).

**Objectives:**
1. Investigate state-of-the-art speech emotion recognition methods
2. Develop multi-modal fusion approach (audio + text)
3. Evaluate on classroom environment data
4. Create practical application for online teaching

## 🛠️ Development

### Run Tests
```bash
pytest -v backend/tests/
```

### Create Sample Audio
```bash
python create_sample_audio.py
```

### Check Backend Health
```bash
python check_backend.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models
- [FastAPI](https://fastapi.tiangolo.com/) for excellent web framework
- [Librosa](https://librosa.org/) for audio processing tools
- SUPERB and emotion-english datasets communities

## 📧 Contact

For questions or collaboration:
- 📫 Create an issue in this repository
- 🏫 Faculty of Information Technology, Hanoi University

## 🗺️ Roadmap

- [x] MVP with Wav2Vec2 + BERT fusion
- [x] Web demo interface  
- [x] RESTful API
- [x] Training infrastructure and scripts
- [x] Data preprocessing utilities
- [x] RAVDESS dataset support
- [x] Custom dataset support
- [x] Training documentation and notebooks
- [ ] ASR integration (Whisper)
- [ ] RAVDESS/EMO-DB evaluation results
- [ ] Model ensemble methods
- [ ] Real-time audio streaming
- [ ] Deployment guide (Docker, cloud)
- [ ] Fine-tuning for classroom domain
- [ ] Real-time emotion tracking
- [ ] Docker deployment
- [ ] Multi-language support

---

**Version:** 0.1.0 (MVP)  
**Last Updated:** November 2025
