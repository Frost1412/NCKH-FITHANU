# Data Directory

This directory contains all datasets for training and evaluating the emotion recognition models.

## Structure

```
data/
├── raw/                    # Original, unprocessed data
│   └── (your raw audio files here)
├── processed/              # Preprocessed data ready for training
│   ├── features/          # Extracted features (MFCC, Mel-spec)
│   ├── splits/            # Train/val/test splits
│   └── metadata.csv       # Dataset metadata
├── datasets/              # Standard emotion recognition datasets
│   ├── RAVDESS/          # Ryerson Audio-Visual Database of Emotional Speech and Song
│   ├── EMO-DB/           # Berlin Database of Emotional Speech
│   └── custom/           # Your custom classroom recordings
└── README.md             # This file
```

## Supported Datasets

### 1. RAVDESS (Recommended)

**Download:** https://zenodo.org/record/1188976

**Description:**
- 24 professional actors (12 male, 12 female)
- 7,356 audio files
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- English speech
- High quality recordings

**How to use:**
1. Download and extract RAVDESS
2. Place audio files in `data/datasets/RAVDESS/`
3. Run preprocessing script: `python training/prepare_ravdess.py`

**File naming convention:**
```
03-01-01-01-01-01-01.wav
│  │  │  │  │  │  │
│  │  │  │  │  │  └─ Actor ID (01-24)
│  │  │  │  │  └──── Repetition (01 or 02)
│  │  │  │  └─────── Intensity (01=normal, 02=strong)
│  │  │  └────────── Statement ("Kids are talking by the door")
│  │  └───────────── Emotion (01-08)
│  └──────────────── Vocal channel (speech)
└─────────────────── Modality (Audio-only)

Emotions:
01 = neutral
02 = calm
03 = happy
04 = sad
05 = angry
06 = fearful
07 = disgust
08 = surprised
```

### 2. EMO-DB (German)

**Download:** http://www.emodb.bilderbar.info/

**Description:**
- 10 professional German speakers
- ~500 audio files
- 7 emotions: anger, boredom, disgust, fear, happiness, sadness, neutral
- German language

**How to use:**
1. Download EMO-DB
2. Place in `data/datasets/EMO-DB/`
3. Run: `python training/prepare_emodb.py`

### 3. Custom Dataset

For your own classroom recordings:

**Directory structure:**
```
data/datasets/custom/
├── anger/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── joy/
├── sadness/
├── fear/
├── disgust/
├── surprise/
└── neutral/
```

**Requirements:**
- Audio format: WAV, MP3, or FLAC
- Sample rate: 16kHz (will be resampled if different)
- Duration: 2-10 seconds recommended
- Clear speech with minimal background noise

## Quick Start

### 1. Prepare Your Data

Place your audio files in the appropriate folder:

```bash
# For RAVDESS
# Download and extract to data/datasets/RAVDESS/

# For custom recordings
data/datasets/custom/
  anger/your_audio.wav
  joy/your_audio.wav
  ...
```

### 2. Preprocess Data

```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Preprocess RAVDESS
python training/prepare_ravdess.py

# OR preprocess custom dataset
python training/prepare_custom.py --input data/datasets/custom --output data/processed
```

### 3. Train Model

```bash
# Basic training
python training/train.py --dataset RAVDESS --model wav2vec2

# With custom config
python training/train.py --config training/configs/wav2vec2_base.yaml
```

## Data Statistics

After preprocessing, check `data/processed/statistics.json` for:
- Total samples
- Samples per emotion
- Train/val/test split ratios
- Audio duration statistics
- Sample rate distribution

## Data Augmentation

Available augmentation techniques:
- Time stretching
- Pitch shifting
- Background noise injection
- Volume adjustment
- Time masking (SpecAugment)

Configure in `training/configs/augmentation.yaml`

## Notes

- **Do not commit large audio files to git!** (They're in `.gitignore`)
- Keep original data in `raw/` folder
- Processed data in `processed/` can be regenerated
- Use `data/processed/splits/` for reproducible train/val/test splits

## Troubleshooting

### Out of memory during preprocessing
- Process in smaller batches
- Reduce sample rate (e.g., 8kHz instead of 16kHz)
- Use data streaming instead of loading all at once

### Imbalanced dataset
- Use weighted sampling during training
- Apply data augmentation to minority classes
- Consider oversampling or undersampling

### Poor quality recordings
- Apply noise reduction: `training/utils/denoise.py`
- Filter out low-quality samples
- Use voice activity detection (VAD) to trim silence

## References

- [RAVDESS Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
- [EMO-DB Info](http://emodb.bilderbar.info/docu/)
- [Speech Emotion Recognition Survey](https://arxiv.org/abs/2104.03465)
