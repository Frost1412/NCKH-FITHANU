"""
Prepare custom emotion dataset for training.

Expected structure:
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
"""
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import json

from tqdm import tqdm
import numpy as np

from utils.audio_preprocessing import (
    load_and_preprocess_audio,
    create_dataset_splits,
    save_metadata,
    compute_statistics
)


SUPPORTED_EMOTIONS = [
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'
]

SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']


def collect_custom_files(custom_dir: str) -> List[Tuple[str, str]]:
    """
    Collect custom audio files organized by emotion folders.
    
    Args:
        custom_dir: Path to custom dataset directory
        
    Returns:
        List of (file_path, emotion_label) tuples
    """
    custom_path = Path(custom_dir)
    file_label_pairs = []
    
    for emotion in SUPPORTED_EMOTIONS:
        emotion_dir = custom_path / emotion
        
        if not emotion_dir.exists():
            print(f"Warning: Emotion folder '{emotion}' not found, skipping...")
            continue
        
        # Collect all audio files
        audio_files = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(emotion_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"Warning: No audio files found in '{emotion}' folder")
            continue
        
        print(f"Found {len(audio_files)} files for emotion: {emotion}")
        
        for audio_file in audio_files:
            file_label_pairs.append((str(audio_file), emotion))
    
    return file_label_pairs


def validate_audio_files(file_label_pairs: List[Tuple[str, str]], target_sr: int = 16000):
    """
    Validate audio files and filter out corrupted ones.
    
    Args:
        file_label_pairs: List of (file_path, label) tuples
        target_sr: Target sample rate
        
    Returns:
        Filtered list of valid files
    """
    valid_pairs = []
    
    print("\nValidating audio files...")
    for file_path, label in tqdm(file_label_pairs, desc="Validation"):
        try:
            # Try to load audio
            y, sr = load_and_preprocess_audio(file_path, target_sr=target_sr, duration=10.0)
            
            # Check duration (should be at least 0.5 seconds)
            duration = len(y) / sr
            if duration < 0.5:
                print(f"Skipping {file_path}: too short ({duration:.2f}s)")
                continue
            
            valid_pairs.append((file_path, label))
            
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    
    print(f"Valid files: {len(valid_pairs)} / {len(file_label_pairs)}")
    return valid_pairs


def process_custom_dataset(
    custom_dir: str,
    output_dir: str,
    validate: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Process custom emotion dataset.
    
    Args:
        custom_dir: Path to custom dataset
        output_dir: Path to save processed data
        validate: Whether to validate audio files
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
    """
    print("=" * 60)
    print("Custom Dataset Preprocessing")
    print("=" * 60)
    
    # Collect files
    print("\n[1/4] Collecting files...")
    file_label_pairs = collect_custom_files(custom_dir)
    
    if not file_label_pairs:
        print("\nError: No audio files found!")
        print(f"\nPlease organize your data in: {custom_dir}")
        print("Expected structure:")
        print("  custom/")
        print("  ├── anger/")
        print("  │   ├── audio1.wav")
        print("  │   └── audio2.wav")
        print("  ├── joy/")
        print("  │   └── ...")
        print("  └── ...")
        print(f"\nSupported emotions: {', '.join(SUPPORTED_EMOTIONS)}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\nTotal files collected: {len(file_label_pairs)}")
    
    # Validate files
    if validate:
        print("\n[2/4] Validating audio files...")
        file_label_pairs = validate_audio_files(file_label_pairs)
        
        if not file_label_pairs:
            print("Error: No valid audio files after validation!")
            return
    else:
        print("\n[2/4] Skipping validation...")
    
    file_paths = [fp for fp, _ in file_label_pairs]
    labels = [label for _, label in file_label_pairs]
    
    print(f"\nFinal dataset size: {len(file_paths)} files")
    print(f"Emotion distribution:")
    for emotion, count in sorted(zip(*np.unique(labels, return_counts=True))):
        print(f"  {emotion:12s}: {count:4d}")
    
    # Check for class imbalance
    label_counts = np.unique(labels, return_counts=True)[1]
    if max(label_counts) / min(label_counts) > 5:
        print("\nWarning: Significant class imbalance detected!")
        print("Consider collecting more samples for underrepresented classes.")
    
    # Create splits
    print("\n[3/4] Creating train/val/test splits...")
    splits = create_dataset_splits(
        file_paths, labels,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=True
    )
    
    # Save metadata
    print("\n[4/4] Saving metadata...")
    save_metadata(splits, output_dir, dataset_name="custom")
    
    # Compute statistics
    stats = compute_statistics(splits)
    
    # Save statistics
    stats_path = Path(output_dir) / "custom_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Statistics Summary")
    print("=" * 60)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {split_stats['total_samples']}")
        if 'label_distribution' in split_stats:
            print("  Label distribution:")
            for label, count in sorted(split_stats['label_distribution'].items()):
                print(f"    {label:12s}: {count:4d}")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review metadata files in: {output_dir}")
    print(f"2. Run training: python training/train.py --dataset custom --model wav2vec2")


def main():
    parser = argparse.ArgumentParser(description="Prepare custom emotion dataset")
    parser.add_argument(
        "--custom-dir",
        type=str,
        default="data/datasets/custom",
        help="Path to custom dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip audio validation"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    process_custom_dataset(
        custom_dir=args.custom_dir,
        output_dir=args.output_dir,
        validate=not args.no_validate,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
