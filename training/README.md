# Training Directory

This directory contains all the tools and scripts for training custom emotion recognition models.

## 📁 Directory Structure

```
training/
├── utils/
│   ├── __init__.py              # Package initialization
│   ├── audio_preprocessing.py   # Audio loading, feature extraction, dataset splits
│   └── augmentation.py          # Data augmentation utilities
├── prepare_ravdess.py           # Prepare RAVDESS dataset
├── prepare_custom.py            # Prepare custom dataset
├── train.py                     # Main training script
├── requirements.txt             # Training dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r training/requirements.txt
```

### 2. Prepare Your Dataset

**Option A: RAVDESS Dataset**
```bash
# Download RAVDESS from https://zenodo.org/record/1188976
# Extract to data/datasets/RAVDESS/
python training/prepare_ravdess.py
```

**Option B: Custom Dataset**
```bash
# Organize your audio files by emotion in data/datasets/custom/
python training/prepare_custom.py
```

### 3. Train the Model

```bash
python training/train.py --dataset ravdess --epochs 10 --batch-size 8
```

## 📋 Available Scripts

### `prepare_ravdess.py`

Preprocesses the RAVDESS dataset for training.

**Usage:**
```bash
python training/prepare_ravdess.py \
    --ravdess-dir data/datasets/RAVDESS \
    --output-dir data/processed \
    --test-size 0.2 \
    --val-size 0.1
```

**Arguments:**
- `--ravdess-dir`: Path to RAVDESS dataset (default: `data/datasets/RAVDESS`)
- `--output-dir`: Output directory for metadata (default: `data/processed`)
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion (default: 0.1)
- `--random-state`: Random seed (default: 42)

**Output:**
- `data/processed/ravdess_train.csv`
- `data/processed/ravdess_val.csv`
- `data/processed/ravdess_test.csv`
- `data/processed/ravdess_statistics.json`

---

### `prepare_custom.py`

Preprocesses custom audio data organized by emotion folders.

**Usage:**
```bash
python training/prepare_custom.py \
    --custom-dir data/datasets/custom \
    --output-dir data/processed \
    --test-size 0.2
```

**Arguments:**
- `--custom-dir`: Path to custom dataset (default: `data/datasets/custom`)
- `--output-dir`: Output directory (default: `data/processed`)
- `--no-validate`: Skip audio validation
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion (default: 0.1)

**Expected Directory Structure:**
```
data/datasets/custom/
├── anger/
│   ├── audio1.wav
│   └── audio2.wav
├── joy/
├── sadness/
├── fear/
├── disgust/
├── surprise/
└── neutral/
```

**Output:**
- `data/processed/custom_train.csv`
- `data/processed/custom_val.csv`
- `data/processed/custom_test.csv`
- `data/processed/custom_statistics.json`

---

### `train.py`

Main training script for emotion recognition models.

**Basic Usage:**
```bash
python training/train.py --dataset ravdess --model wav2vec2 --epochs 10
```

**Advanced Usage:**
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

**Arguments:**
- `--dataset`: Dataset to use (`ravdess` or `custom`) [required]
- `--model`: Model architecture (currently only `wav2vec2`)
- `--processed-dir`: Directory with processed metadata (default: `data/processed`)
- `--output-dir`: Output directory for checkpoints
- `--model-name`: Pretrained model name (default: `facebook/wav2vec2-base`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 3e-5)
- `--use-wandb`: Enable Weights & Biases logging

**Available Pretrained Models:**
- `facebook/wav2vec2-base` (95M params)
- `facebook/wav2vec2-large` (317M params)
- `superb/wav2vec2-base-superb-er` (pretrained for emotion recognition)

**Output:**
- Model checkpoints in `models/checkpoints/<dataset>_<timestamp>/`
- Best model saved to `best_model/` subdirectory
- Training logs and metrics
- Test evaluation results in `test_results.json`

## 🔧 Utilities

### Audio Preprocessing (`utils/audio_preprocessing.py`)

**Functions:**
- `load_and_preprocess_audio()` - Load and normalize audio
- `extract_features()` - Extract MFCC, mel-spectrogram, chroma, etc.
- `create_dataset_splits()` - Create train/val/test splits
- `save_metadata()` - Save dataset metadata to CSV
- `compute_statistics()` - Compute dataset statistics

**Example:**
```python
from training.utils.audio_preprocessing import load_and_preprocess_audio, extract_features

# Load audio
y, sr = load_and_preprocess_audio("audio.wav", target_sr=16000)

# Extract features
features = extract_features(y, sr)
print(features['mfcc_mean'])  # Mean MFCC coefficients
```

---

### Data Augmentation (`utils/augmentation.py`)

**Functions:**
- `time_stretch()` - Time stretching
- `pitch_shift()` - Pitch shifting
- `add_noise()` - Add Gaussian noise
- `change_volume()` - Volume adjustment
- `time_mask()` - Time masking
- `shift_time()` - Circular time shift
- `apply_random_augmentations()` - Apply multiple augmentations randomly

**Example:**
```python
from training.utils.augmentation import apply_random_augmentations

# Apply random augmentations
augmented = apply_random_augmentations(
    y=audio_waveform,
    sr=16000,
    prob=0.5,  # 50% chance per augmentation
    time_stretch_range=(0.9, 1.1),
    pitch_shift_range=(-2, 2)
)
```

## 📊 Training Workflow

1. **Prepare Data**
   ```bash
   python training/prepare_ravdess.py
   ```

2. **Verify Data**
   - Check `data/processed/ravdess_statistics.json`
   - Review train/val/test splits
   - Ensure balanced class distribution

3. **Train Model**
   ```bash
   python training/train.py --dataset ravdess --epochs 10 --batch-size 8
   ```

4. **Monitor Training**
   - Watch console output for loss and metrics
   - Use `--use-wandb` for web-based monitoring

5. **Evaluate Results**
   - Review test accuracy, F1-score
   - Check confusion matrix
   - Examine per-class performance

6. **Deploy Model**
   - Copy best model checkpoint
   - Update backend configuration
   - Test with web demo

## 💡 Tips

### For Better Performance

1. **Use GPU**: Training is much faster on GPU
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Increase Batch Size**: If you have enough GPU memory
   ```bash
   python training/train.py --batch-size 16
   ```

3. **Use Larger Model**: For better accuracy
   ```bash
   python training/train.py --model-name facebook/wav2vec2-large
   ```

4. **Data Augmentation**: Increase dataset diversity
   - Edit `utils/augmentation.py`
   - Apply augmentations in training loop

### Troubleshooting

**Out of Memory (OOM)**:
```bash
# Reduce batch size
python training/train.py --batch-size 4

# Use smaller model
python training/train.py --model-name facebook/wav2vec2-base
```

**Poor Validation Accuracy**:
- Check data quality and labels
- Increase training epochs
- Try a larger pretrained model
- Ensure balanced class distribution

**Slow Training**:
- Use GPU instead of CPU
- Increase batch size
- Use mixed precision training (edit train.py)

## 📚 Additional Resources

- **Full Training Guide**: See `TRAINING_GUIDE.md` in project root
- **Example Notebook**: See `notebooks/01_training_example.ipynb`
- **Data README**: See `data/README.md` for dataset instructions

## 🎯 Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Test different models**: Compare wav2vec2-base vs wav2vec2-large
3. **Fine-tune on custom data**: Use your classroom recordings
4. **Track experiments**: Use Weights & Biases for experiment tracking
5. **Deploy best model**: Integrate into the web demo

## 📝 Notes

- Training time depends on dataset size and hardware (typically 1-3 hours on GPU)
- Best model is automatically saved based on validation F1 score
- Early stopping prevents overfitting (patience=3 epochs)
- All random operations are seeded for reproducibility
