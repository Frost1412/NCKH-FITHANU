# Training Guide

This guide will help you train custom emotion recognition models using your own data.

## Prerequisites

1. **Install training dependencies**:
   ```bash
   pip install -r training/requirements.txt
   ```

2. **Prepare your dataset** in one of these formats:
   - RAVDESS dataset (download separately)
   - Custom dataset organized by emotion folders

## Quick Start

### Option 1: Train with RAVDESS Dataset

1. **Download RAVDESS**:
   - Visit: https://zenodo.org/record/1188976
   - Download "Audio_Speech_Actors_01-24.zip"
   - Extract to `data/datasets/RAVDESS/`

2. **Prepare the data**:
   ```bash
   python training/prepare_ravdess.py
   ```

3. **Train the model**:
   ```bash
   python training/train.py --dataset ravdess --epochs 10 --batch-size 8
   ```

### Option 2: Train with Custom Data

1. **Organize your audio files**:
   ```
   data/datasets/custom/
   ├── anger/
   │   ├── audio1.wav
   │   ├── audio2.wav
   │   └── ...
   ├── joy/
   │   ├── audio1.wav
   │   └── ...
   ├── sadness/
   ├── fear/
   ├── disgust/
   ├── surprise/
   └── neutral/
   ```

2. **Prepare the data**:
   ```bash
   python training/prepare_custom.py
   ```

3. **Train the model**:
   ```bash
   python training/train.py --dataset custom --epochs 10 --batch-size 8
   ```

## Detailed Steps

### 1. Data Preparation

#### RAVDESS Dataset

The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) contains 7,356 files from 24 actors.

```bash
# Basic usage
python training/prepare_ravdess.py

# With custom paths
python training/prepare_ravdess.py \
    --ravdess-dir data/datasets/RAVDESS \
    --output-dir data/processed \
    --test-size 0.2 \
    --val-size 0.1
```

**Expected output**:
- `data/processed/ravdess_train.csv`
- `data/processed/ravdess_val.csv`
- `data/processed/ravdess_test.csv`
- `data/processed/ravdess_statistics.json`

#### Custom Dataset

For classroom recordings or your own data:

```bash
# Basic usage
python training/prepare_custom.py

# With validation
python training/prepare_custom.py \
    --custom-dir data/datasets/custom \
    --output-dir data/processed \
    --test-size 0.2 \
    --val-size 0.1
```

**Audio requirements**:
- Format: WAV, MP3, FLAC, M4A, or OGG
- Sample rate: 16kHz (will be resampled automatically)
- Duration: 2-10 seconds recommended
- Quality: Clear speech, minimal background noise

### 2. Model Training

#### Basic Training

```bash
python training/train.py \
    --dataset ravdess \
    --model wav2vec2 \
    --epochs 10 \
    --batch-size 8
```

#### Advanced Training Options

```bash
python training/train.py \
    --dataset custom \
    --model wav2vec2 \
    --model-name facebook/wav2vec2-base \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --output-dir models/checkpoints/my_model \
    --use-wandb
```

**Parameters**:
- `--dataset`: Dataset to use (`ravdess` or `custom`)
- `--model`: Model architecture (currently only `wav2vec2`)
- `--model-name`: Pretrained model from HuggingFace
  - `facebook/wav2vec2-base` (95M params)
  - `facebook/wav2vec2-large` (317M params)
  - `superb/wav2vec2-base-superb-er` (pretrained for emotion)
- `--epochs`: Number of training epochs (10-20 recommended)
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--learning-rate`: Learning rate (3e-5 is a good default)
- `--output-dir`: Where to save model checkpoints
- `--use-wandb`: Enable Weights & Biases logging

#### GPU/CPU Configuration

The training script automatically uses GPU if available:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For CPU-only training (slower)
# The script will automatically fall back to CPU if no GPU is available
```

### 3. Monitor Training

Training progress will be displayed in the console:

```
Epoch 1/10
Train Loss: 1.234
Val Loss: 1.456
Val Accuracy: 0.65
Val F1 (Macro): 0.62

Epoch 2/10
Train Loss: 0.987
Val Loss: 1.234
Val Accuracy: 0.72
Val F1 (Macro): 0.70
...
```

If using `--use-wandb`, you can monitor training in real-time at https://wandb.ai

### 4. Evaluate Results

After training, the script will evaluate on the test set:

```
Test Accuracy: 0.78
Test F1 (Macro): 0.76
Test F1 (Weighted): 0.77

Classification Report:
              precision    recall  f1-score   support

       anger       0.82      0.79      0.80       120
     disgust       0.75      0.73      0.74       115
        fear       0.80      0.78      0.79       118
         joy       0.85      0.82      0.83       125
     neutral       0.70      0.75      0.72       110
     sadness       0.77      0.80      0.78       122
    surprise       0.73      0.71      0.72       108

Confusion Matrix:
...
```

### 5. Use Trained Model

To use your trained model in the web demo:

1. **Find the best model**:
   ```
   models/checkpoints/<dataset>_<timestamp>/best_model/
   ```

2. **Update backend code**:
   Edit `backend/app/models/emotion_inference.py`:
   ```python
   # Change this line:
   model="superb/wav2vec2-base-superb-er"
   
   # To:
   model="path/to/your/best_model"
   ```

3. **Restart backend**:
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```

## Tips for Better Results

### 1. Data Quality
- Use high-quality audio (16-bit, 16kHz minimum)
- Ensure clear speech with minimal background noise
- Balance classes (similar number of samples per emotion)
- Include diverse speakers and recording conditions

### 2. Data Augmentation

The training scripts support data augmentation. Edit `training/utils/augmentation.py` to customize:

```python
from training.utils.augmentation import apply_random_augmentations

# Apply augmentations during training
augmented_audio = apply_random_augmentations(
    y=audio_waveform,
    sr=16000,
    prob=0.5,  # 50% chance per augmentation
    time_stretch_range=(0.9, 1.1),
    pitch_shift_range=(-2, 2),
    noise_factor=0.005
)
```

### 3. Hyperparameter Tuning

Common parameters to tune:
- **Learning rate**: Try 1e-5, 3e-5, 5e-5
- **Batch size**: Increase if you have more GPU memory
- **Epochs**: 10-20 for most datasets
- **Model size**: Larger models (wav2vec2-large) may perform better but are slower

### 4. Handling Class Imbalance

If some emotions have fewer samples:

1. **Collect more data** for underrepresented classes
2. **Use data augmentation** more heavily for minority classes
3. **Adjust class weights** in training (edit `train.py`)

### 5. Troubleshooting

**Out of Memory (OOM) errors**:
```bash
# Reduce batch size
python training/train.py --dataset ravdess --batch-size 4

# Use a smaller model
python training/train.py --dataset ravdess --model-name facebook/wav2vec2-base
```

**Poor validation accuracy**:
- Check data quality and labeling
- Increase training epochs
- Try data augmentation
- Use a larger pretrained model
- Ensure balanced class distribution

**Overfitting (train accuracy >> val accuracy)**:
- Add more data
- Use data augmentation
- Reduce model size
- Add regularization (increase weight_decay)

## Advanced Usage

### Custom Emotion Labels

To train with different emotions, edit `training/train.py`:

```python
# Change this:
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# To your labels:
EMOTION_LABELS = ['happy', 'sad', 'angry', 'calm']
```

### Fine-tune from Checkpoint

To continue training from a saved checkpoint:

```bash
python training/train.py \
    --dataset custom \
    --model-name models/checkpoints/ravdess_20241201_120000/best_model \
    --epochs 10
```

### Export for Production

After training, convert the model for faster inference:

```bash
# Install ONNX runtime
pip install onnx onnxruntime

# Export model (TODO: add export script)
python training/export_onnx.py --model-dir models/checkpoints/my_model/best_model
```

## Next Steps

1. **Experiment with different datasets**: Try RAVDESS, EMO-DB, or your custom data
2. **Tune hyperparameters**: Adjust learning rate, batch size, epochs
3. **Try model ensembles**: Combine multiple models for better accuracy
4. **Deploy to production**: Use the best model in your web demo
5. **Continuous improvement**: Collect more data and retrain periodically

## Resources

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Weights & Biases](https://wandb.ai)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example notebook: `notebooks/01_training_example.ipynb`
3. Open an issue on GitHub
