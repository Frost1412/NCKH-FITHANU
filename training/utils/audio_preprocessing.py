"""
Data preprocessing utilities for emotion recognition datasets.
"""
import os
from pathlib import Path
from typing import Tuple, List, Dict

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    duration: float = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        duration: Maximum duration to load (None = full file)
        offset: Start reading after this time (in seconds)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    y, sr = librosa.load(audio_path, sr=target_sr, duration=duration, offset=offset, mono=True)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    return y, sr


def extract_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract various audio features.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCCs
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features['mfcc'] = mfcc
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    features['mel_spec'] = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # RMS energy
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    
    return features


def create_dataset_splits(
    file_paths: List[str],
    labels: List[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        file_paths: List of audio file paths
        labels: List of corresponding labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from train)
        random_state: Random seed
        stratify: Whether to stratify splits by label
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    stratify_param = labels if stratify else None
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_param_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_param_temp
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def save_metadata(
    splits: Dict[str, Tuple[List[str], List[str]]],
    output_dir: str,
    dataset_name: str = "emotion_dataset"
):
    """
    Save dataset metadata to CSV files.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Directory to save metadata
        dataset_name: Name of dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, (files, labels) in splits.items():
        df = pd.DataFrame({
            'file_path': files,
            'label': labels
        })
        
        csv_path = output_path / f"{dataset_name}_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} metadata to {csv_path} ({len(df)} samples)")


def compute_statistics(splits: Dict[str, Tuple[List[str], List[str]]]) -> Dict:
    """
    Compute dataset statistics.
    
    Args:
        splits: Dictionary with train/val/test splits
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for split_name, (files, labels) in splits.items():
        label_counts = pd.Series(labels).value_counts().to_dict()
        stats[split_name] = {
            'total_samples': len(files),
            'label_distribution': label_counts
        }
    
    # Overall statistics
    all_labels = []
    for _, labels in splits.values():
        all_labels.extend(labels)
    
    stats['overall'] = {
        'total_samples': len(all_labels),
        'label_distribution': pd.Series(all_labels).value_counts().to_dict(),
        'num_classes': len(set(all_labels))
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Audio preprocessing utilities loaded.")
    print("Import this module in your preprocessing scripts.")
