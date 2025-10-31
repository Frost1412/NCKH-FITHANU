"""
Data augmentation utilities for audio emotion recognition.
"""
import librosa
import numpy as np
from typing import Optional


def time_stretch(y: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Time-stretch an audio signal without changing pitch.
    
    Args:
        y: Audio waveform
        rate: Stretch factor (>1.0 faster, <1.0 slower)
        
    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = 0.0) -> np.ndarray:
    """
    Shift the pitch of an audio signal.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add random Gaussian noise to audio.
    
    Args:
        y: Audio waveform
        noise_factor: Standard deviation of noise
        
    Returns:
        Audio with added noise
    """
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    # Cast to same dtype as input
    return augmented.astype(y.dtype)


def change_volume(y: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    """
    Change volume of audio signal.
    
    Args:
        y: Audio waveform
        gain_db: Volume change in decibels
        
    Returns:
        Volume-adjusted audio
    """
    gain_linear = 10 ** (gain_db / 20.0)
    return y * gain_linear


def time_mask(y: np.ndarray, sr: int, max_mask_duration: float = 0.1) -> np.ndarray:
    """
    Apply time masking (set random segment to zero).
    
    Args:
        y: Audio waveform
        sr: Sample rate
        max_mask_duration: Maximum mask duration in seconds
        
    Returns:
        Time-masked audio
    """
    augmented = y.copy()
    mask_samples = int(sr * max_mask_duration)
    
    # Random start position
    start = np.random.randint(0, len(y) - mask_samples)
    augmented[start:start + mask_samples] = 0
    
    return augmented


def shift_time(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """
    Shift audio in time (circular shift).
    
    Args:
        y: Audio waveform
        shift_max: Maximum shift ratio (0.2 = 20% of length)
        
    Returns:
        Time-shifted audio
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def apply_random_augmentations(
    y: np.ndarray,
    sr: int,
    prob: float = 0.5,
    time_stretch_range: tuple = (0.9, 1.1),
    pitch_shift_range: tuple = (-2, 2),
    noise_factor: float = 0.005,
    volume_range: tuple = (-3, 3)
) -> np.ndarray:
    """
    Apply random augmentations with given probability.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        prob: Probability of applying each augmentation
        time_stretch_range: Range for time stretch rates
        pitch_shift_range: Range for pitch shift in semitones
        noise_factor: Noise standard deviation
        volume_range: Volume change range in dB
        
    Returns:
        Augmented audio
    """
    augmented = y.copy()
    
    # Time stretch
    if np.random.random() < prob:
        rate = np.random.uniform(*time_stretch_range)
        augmented = time_stretch(augmented, rate=rate)
    
    # Pitch shift
    if np.random.random() < prob:
        n_steps = np.random.uniform(*pitch_shift_range)
        augmented = pitch_shift(augmented, sr=sr, n_steps=n_steps)
    
    # Add noise
    if np.random.random() < prob:
        augmented = add_noise(augmented, noise_factor=noise_factor)
    
    # Change volume
    if np.random.random() < prob:
        gain = np.random.uniform(*volume_range)
        augmented = change_volume(augmented, gain_db=gain)
    
    # Time shift
    if np.random.random() < prob:
        augmented = shift_time(augmented, shift_max=0.2)
    
    return augmented


if __name__ == "__main__":
    # Example usage
    print("Audio augmentation utilities loaded.")
    print("Available functions:")
    print("  - time_stretch()")
    print("  - pitch_shift()")
    print("  - add_noise()")
    print("  - change_volume()")
    print("  - time_mask()")
    print("  - shift_time()")
    print("  - apply_random_augmentations()")
