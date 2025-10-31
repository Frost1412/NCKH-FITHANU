"""
Prepare RAVDESS dataset for training.

RAVDESS file naming convention:
Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav

Example: 03-01-06-01-02-01-12.wav
- 03: face-and-voice
- 01: speech
- 06: fearful
- 01: normal intensity
- 02: statement "dogs"
- 01: 1st repetition
- 12: actor 12

Emotion codes:
01 = neutral
02 = calm
03 = happy
04 = sad
05 = angry
06 = fearful
07 = disgust
08 = surprised
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
    extract_features,
    create_dataset_splits,
    save_metadata,
    compute_statistics
)


# RAVDESS emotion mapping
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",  # calm -> neutral
    "03": "joy",      # happy -> joy
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


def parse_ravdess_filename(filename: str) -> dict:
    """
    Parse RAVDESS filename to extract metadata.
    
    Args:
        filename: RAVDESS audio filename
        
    Returns:
        Dictionary with parsed information
    """
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {filename}")
    
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion_code': parts[2],
        'emotion': RAVDESS_EMOTION_MAP.get(parts[2], 'unknown'),
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6],
        'filename': filename
    }


def collect_ravdess_files(ravdess_dir: str) -> List[Tuple[str, str]]:
    """
    Collect all RAVDESS audio files with labels.
    
    Args:
        ravdess_dir: Path to RAVDESS dataset directory
        
    Returns:
        List of (file_path, emotion_label) tuples
    """
    ravdess_path = Path(ravdess_dir)
    file_label_pairs = []
    
    # RAVDESS is organized in Actor_XX folders
    actor_folders = sorted(ravdess_path.glob("Actor_*"))
    
    if not actor_folders:
        print(f"Warning: No Actor_* folders found in {ravdess_dir}")
        print("Expected structure: RAVDESS/Actor_01/, Actor_02/, etc.")
        return []
    
    print(f"Found {len(actor_folders)} actor folders")
    
    for actor_folder in actor_folders:
        wav_files = list(actor_folder.glob("*.wav"))
        
        for wav_file in wav_files:
            try:
                metadata = parse_ravdess_filename(wav_file.name)
                emotion = metadata['emotion']
                
                if emotion != 'unknown':
                    file_label_pairs.append((str(wav_file), emotion))
            except ValueError as e:
                print(f"Skipping file {wav_file.name}: {e}")
    
    return file_label_pairs


def process_ravdess_dataset(
    ravdess_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Process RAVDESS dataset and create train/val/test splits.
    
    Args:
        ravdess_dir: Path to RAVDESS dataset
        output_dir: Path to save processed data
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
    """
    print("=" * 60)
    print("RAVDESS Dataset Preprocessing")
    print("=" * 60)
    
    # Collect files
    print("\n[1/4] Collecting files...")
    file_label_pairs = collect_ravdess_files(ravdess_dir)
    
    if not file_label_pairs:
        print("Error: No valid RAVDESS files found!")
        print(f"Please check that {ravdess_dir} contains Actor_* folders with .wav files")
        return
    
    file_paths = [fp for fp, _ in file_label_pairs]
    labels = [label for _, label in file_label_pairs]
    
    print(f"Found {len(file_paths)} audio files")
    print(f"Emotion distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Create splits
    print("\n[2/4] Creating train/val/test splits...")
    splits = create_dataset_splits(
        file_paths, labels,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=True
    )
    
    # Save metadata
    print("\n[3/4] Saving metadata...")
    save_metadata(splits, output_dir, dataset_name="ravdess")
    
    # Compute statistics
    print("\n[4/4] Computing statistics...")
    stats = compute_statistics(splits)
    
    # Save statistics
    stats_path = Path(output_dir) / "ravdess_statistics.json"
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
    print(f"2. Run training: python training/train.py --dataset ravdess --model wav2vec2")


def main():
    parser = argparse.ArgumentParser(description="Prepare RAVDESS dataset")
    parser.add_argument(
        "--ravdess-dir",
        type=str,
        default="data/datasets/RAVDESS",
        help="Path to RAVDESS dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
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
    
    process_ravdess_dataset(
        ravdess_dir=args.ravdess_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
